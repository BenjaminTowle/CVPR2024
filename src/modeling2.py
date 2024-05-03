import monai
import numpy as np
import torch
import torch.nn.functional as F
import random

from abc import ABC, abstractmethod
from dataclasses import dataclass
from torch import nn
from torch.distributions import Normal, Independent, kl
from transformers.models.sam.modeling_sam import (
    SamModel, 
    SamPreTrainedModel,
    SamImageSegmentationOutput
)
from typing import Optional, Callable, Union
from torch.distributions.multivariate_normal import MultivariateNormal
from typing import Tuple
from scipy.optimize import linear_sum_assignment
from functools import partial
import math

from src.click import ClickStrategy
from src.metrics import iou, dice
from src.utils import RegistryMixin
from src.search import SearchStrategy


@dataclass
class SamMultimaskOutput(SamImageSegmentationOutput):
    loss: torch.FloatTensor = None
    input_points: torch.Tensor = None
    iou_targets: torch.Tensor = None
    iou_pred: torch.Tensor = None
    input_labels: torch.Tensor = None
    initial_pred_masks: torch.Tensor = None
    union: torch.Tensor = None
    intersection: torch.Tensor = None


class Model(ABC, SamPreTrainedModel):

    def __init__(self, config) -> None:
        super().__init__(config)
        self.sam = SamModel(config)

    @staticmethod
    def compute_loss(
        pred_masks: torch.Tensor, 
        labels: torch.Tensor, 
        loss_fn: Callable,
        return_dict: bool = False
    ):
        """
        pred_masks: (bsz, 1, num_multimask_outputs, H, W)
        labels: (bsz, H, W)
        """
        bsz, _, num_preds, H, W = pred_masks.size()

        loss = loss_fn(
            pred_masks.reshape(-1, H, W), 
            labels.repeat_interleave(num_preds, dim=0)
        ).reshape(bsz, num_preds, -1)

        loss = loss.mean(dim=2)

        if not return_dict:
            return loss.min(dim=1)[0].mean()

        _min = loss.min(dim=1)

        return {
            "loss": _min[0].mean(),
            "indices": _min[1],
        }

    @staticmethod
    def iou_loss(pred_masks, iou_pred, labels, return_iou=False):
        bsz, _, num_preds, H, W = pred_masks.size()
        iou_targets = torch.zeros(bsz, num_preds).to(pred_masks.device)
        pred_masks = pred_masks.squeeze(1).detach().cpu().numpy() > 0.5
        labels = labels.detach().cpu().numpy()
        for i in range(bsz):
            for j in range(num_preds):
                iou_targets[i, j] = iou(pred_masks[i, j], labels[i])
        iou_loss = F.mse_loss(iou_pred.squeeze(), iou_targets.squeeze())

        if return_iou:
            return iou_loss, iou_targets

        return iou_loss


class SamBaseline(Model):

    def __init__(self, config, processor, multimask_output: bool = True):
        super().__init__(config)
        self.sam = SamModel(config)
        self.processor = processor
        self.seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="none")
        self.multimask_output = multimask_output

    @property
    def mask_decoder(self):
        return self.sam.mask_decoder

    def forward(
        self, 
        pixel_values: Optional[torch.Tensor] = None,
        image_embeddings: Optional[torch.Tensor] = None,
        input_boxes: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        original_sizes: Optional[torch.Tensor] = None,
        reshaped_input_sizes: Optional[torch.Tensor] = None,
        label_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):

        new_labels = []
        for i in range(len(labels)):
            while True:
                rand_idx = random.randint(0, labels.shape[1] - 1) 
                if label_mask[i, rand_idx] > 0:
                    break
            new_labels.append(labels[i, rand_idx])
        labels = torch.stack(new_labels, dim=0).to(self.device)
        
        if image_embeddings is None:
            image_embeddings = self.sam.get_image_embeddings(pixel_values)

        outputs = self.sam(
            image_embeddings=image_embeddings, 
            input_boxes=input_boxes,
            multimask_output=self.multimask_output,
        )

        loss = self.compute_loss(
            pred_masks=outputs.pred_masks, 
            labels=labels, 
            loss_fn=self.seg_loss, 
            return_dict=True
        )        

        loss = loss["loss"]

        return SamMultimaskOutput(
            loss=loss,
            iou_scores=outputs.iou_scores,
            pred_masks=outputs.pred_masks,
        )




class SamAR(Model):

    def __init__(self, config, processor, num_samples: int = 4, ablation="none"):
        super().__init__(config)
        self.sam = SamModel(config)
        self.processor = processor
        self.seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="none")
        self.multimask_output = True
        self.num_samples = num_samples
        self.ablation = ablation

        self.unet = UNet(input_channels=1, num_classes=1, num_filters=[32, 64, 128, 192], apply_last_layer=True)
        self.cell = CRnnCell()

        self.search_strategy = SearchStrategy.create("mcts")()

    @property
    def mask_decoder(self):
        return self.sam.mask_decoder
    
    def _clustering(
        self,
        pred_masks,
    ):

        p_pred = (pred_masks > 0.0).squeeze(1).cpu().numpy()
        shape = (p_pred.shape[0], p_pred.shape[1], p_pred.shape[1])
        similarity_matrix = np.zeros(shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    similarity_matrix[i, j, k] = iou(p_pred[i, j], p_pred[i, k])
        
        chosen_preds = []
        all_chosen_idxs = []
        for i in range(pred_masks.size(0)):
            chosen_idxs = self.search_strategy.run_search(scores=similarity_matrix[i], k=3)
            chosen_preds.append(pred_masks[i, :, chosen_idxs])
            all_chosen_idxs.append(chosen_idxs)
        chosen_preds = torch.stack(chosen_preds, dim=0)
        
        try:
            all_chosen_idxs = torch.stack(all_chosen_idxs).to(self.device)
        except:
            all_chosen_idxs = torch.tensor(all_chosen_idxs).to(self.device)

        return chosen_preds

    def forward(
        self, 
        pixel_values: Optional[torch.Tensor] = None,
        image_embeddings: Optional[torch.Tensor] = None,
        input_boxes: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        original_sizes: Optional[torch.Tensor] = None,
        reshaped_input_sizes: Optional[torch.Tensor] = None,
        label_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        
        self.cell.reset()
        
        if image_embeddings is None:
            image_embeddings = self.sam.get_image_embeddings(pixel_values)

        input_masks = None
        total_loss = 0.0
        pred_masks = []

        # Ensures we have enough samples for the labels
        num_samples = max([labels.shape[1], self.num_samples])
        
        for i in range(num_samples):
            self.sam.mask_decoder.forward = partial(forward, self=self.sam.mask_decoder, cell=self.cell)
            
            outputs = self.sam(
                image_embeddings=image_embeddings, 
                input_boxes=input_boxes,
                input_masks=input_masks,
                multimask_output=False,
            )
            input_masks = outputs.pred_masks.squeeze(2)

            if self.ablation == "sg":
                input_masks = input_masks.detach()
                        
            pred_masks.append(outputs.pred_masks)

        if self.training:
            if self.ablation == "random":
                rand_idxs = random.sample(range(num_samples), k=labels.shape[1])
                pred_masks = [pred_masks[i] for i in rand_idxs]

            elif self.ablation == "sequential":
                pred_masks = pred_masks[:labels.shape[1]]

            else:
                num_labels = labels.shape[1]

                def ceildiv(a, b):
                    return -(a // -b)
                
                chunk_size = ceildiv(num_samples, num_labels)
                new_pred_masks = []
                for i in range(0, num_samples, chunk_size):
                    masks = pred_masks[i:i+chunk_size]
                    new_pred_masks.append(random.choice(masks))
                pred_masks = new_pred_masks
            
        total_loss = 0.0
        pred_masks = torch.cat(pred_masks, dim=2)
        for i in range(labels.shape[0]):
            losses = torch.zeros(pred_masks.shape[2], labels.shape[1]).to(labels.device)
            for j in range(pred_masks.shape[2]):
                for k in range(labels.shape[1]):
                    loss = self.seg_loss(pred_masks[i, 0, j], labels[i, k]).mean()
                    losses[j, k] = loss
            
            
            if self.ablation != "no_ha":
                row_ind, col_ind = linear_sum_assignment(losses.detach().cpu().numpy())
            else:
                row_ind = torch.arange(losses.shape[0])
                col_ind = torch.arange(losses.shape[1])
            loss = losses[row_ind, col_ind]
            if self.training:
                loss *= label_mask[i].float()
            total_loss += loss.sum()
        
        total_loss /= labels.shape[1]

        #if not self.training:
           # pred_masks = self._clustering(pred_masks)
                
        return SamMultimaskOutput(
            loss=total_loss,
            iou_scores=outputs.iou_scores,
            pred_masks=pred_masks[:, :, :self.num_samples],
        )







def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        truncated_normal_(m.bias, mean=0, std=0.001)

class DownConvBlock(nn.Module):
    """
    A block of three convolutional layers where each layer is followed by a non-linear activation function
    Between each block we add a pooling operation.
    """
    def __init__(self, input_dim, output_dim, initializers, padding, pool=True):
        super(DownConvBlock, self).__init__()
        layers = []

        if pool:
            layers.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True))

        layers.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1, padding=int(padding)))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=int(padding)))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=int(padding)))
        layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

        self.layers.apply(init_weights)

    def forward(self, patch):
        return self.layers(patch)


class UpConvBlock(nn.Module):
    """
    A block consists of an upsampling layer followed by a convolutional layer to reduce the amount of channels and then a DownConvBlock
    If bilinear is set to false, we do a transposed convolution instead of upsampling
    """
    def __init__(self, input_dim, output_dim, initializers, padding, bilinear=True):
        super(UpConvBlock, self).__init__()
        self.bilinear = bilinear

        if not self.bilinear:
            self.upconv_layer = nn.ConvTranspose2d(input_dim, output_dim, kernel_size=2, stride=2)
            self.upconv_layer.apply(init_weights)

        self.conv_block = DownConvBlock(input_dim, output_dim, initializers, padding, pool=False)

    def forward(self, x, bridge):
        if self.bilinear:
            up = nn.functional.interpolate(x, mode='bilinear', scale_factor=2, align_corners=True)
        else:
            up = self.upconv_layer(x)
        
        assert up.shape[3] == bridge.shape[3]
        out = torch.cat([up, bridge], 1)
        out =  self.conv_block(out)

        return out



class UNetCell(nn.Module):
    """
    A UNet (https://arxiv.org/abs/1505.04597) implementation.
    input_channels: the number of channels in the image (1 for greyscale and 3 for RGB)
    num_classes: the number of classes to predict
    num_filters: list with the amount of filters per layer
    apply_last_layer: boolean to apply last layer or not (not used in Probabilistic UNet)
    padidng: Boolean, if true we pad the images with 1 so that we keep the same dimensions
    """

    def __init__(self, input_channels=1, num_classes=1, num_filters=[32,64,128,192], initializers=None, apply_last_layer=True, padding=True):
        super(UNet, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.padding = padding
        self.activation_maps = []
        self.apply_last_layer = apply_last_layer
        self.contracting_path = nn.ModuleList()

        for i in range(len(self.num_filters)):
            input = self.input_channels if i == 0 else output
            output = self.num_filters[i]

            if i == 0:
                pool = False
            else:
                pool = True

            self.contracting_path.append(DownConvBlock(input, output, initializers, padding, pool=pool))

        self.upsampling_path = nn.ModuleList()

        n = len(self.num_filters) - 2
        for i in range(n, -1, -1):
            input = output + self.num_filters[i]
            output = self.num_filters[i]
            self.upsampling_path.append(UpConvBlock(input, output, initializers, padding))

        if self.apply_last_layer:
            self.last_layer = nn.Conv2d(output, num_classes, kernel_size=1)

    @property
    def device(self):
        return next(self.parameters()).device

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    @classmethod
    def from_pretrained(cls, model_path: str, device: str = "cuda"):
        model = cls()
        model.load_state_dict(torch.load(model_path, map_location=device))
        return model
    
    @classmethod
    def save_pretrained(cls, model, model_path: str):
        torch.save(model.state_dict(), model_path)

    def init_hidden(self, x):
        return torch.zeros(x.size(0), self.num_filters[-1], x.size(2), x.size(3), device=x.device)

    def forward(self, x: torch.Tensor, h=None):
        if h is None:
            h = self.init_hidden(x)
        
        blocks = []
        for i, down in enumerate(self.contracting_path):
            x = down(x)
            if i != len(self.contracting_path)-1:
                blocks.append(x)

        for i, up in enumerate(self.upsampling_path):
            x = up(x, blocks[-i-1])

        del blocks

        # Used for saving the activations and plotting
        if not self.training:
            self.activation_maps.append(x)
        
        if self.apply_last_layer:
            x =  self.last_layer(x)

        return x, h




class UNet(nn.Module):
    """
    A UNet (https://arxiv.org/abs/1505.04597) implementation.
    input_channels: the number of channels in the image (1 for greyscale and 3 for RGB)
    num_classes: the number of classes to predict
    num_filters: list with the amount of filters per layer
    apply_last_layer: boolean to apply last layer or not (not used in Probabilistic UNet)
    padidng: Boolean, if true we pad the images with 1 so that we keep the same dimensions
    """

    def __init__(self, input_channels=1, num_classes=1, num_filters=[32,64,128,192], initializers=None, apply_last_layer=True, padding=True):
        super(UNet, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.padding = padding
        self.activation_maps = []
        self.apply_last_layer = apply_last_layer
        self.contracting_path = nn.ModuleList()

        for i in range(len(self.num_filters)):
            input = self.input_channels if i == 0 else output
            output = self.num_filters[i]

            if i == 0:
                pool = False
            else:
                pool = True

            self.contracting_path.append(DownConvBlock(input, output, initializers, padding, pool=pool))

        self.upsampling_path = nn.ModuleList()

        n = len(self.num_filters) - 2
        for i in range(n, -1, -1):
            input = output + self.num_filters[i]
            output = self.num_filters[i]
            self.upsampling_path.append(UpConvBlock(input, output, initializers, padding))

        if self.apply_last_layer:
            self.last_layer = nn.Conv2d(output, num_classes, kernel_size=1)

    @property
    def device(self):
        return next(self.parameters()).device

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    @classmethod
    def from_pretrained(cls, model_path: str, device: str = "cuda"):
        model = cls()
        model.load_state_dict(torch.load(model_path, map_location=device))
        return model
    
    @classmethod
    def save_pretrained(cls, model, model_path: str):
        torch.save(model.state_dict(), model_path)


    def forward(self, pixel_values, labels=None):
        x = pixel_values
        blocks = []
        for i, down in enumerate(self.contracting_path):
            x = down(x)
            if i != len(self.contracting_path)-1:
                blocks.append(x)

        for i, up in enumerate(self.upsampling_path):
            x = up(x, blocks[-i-1])

        del blocks

        # Used for saving the activations and plotting
        if not self.training:
            self.activation_maps.append(x)
        
        if self.apply_last_layer:
            x =  self.last_layer(x)

        logits = x
        # Dummy iou_scores required for evaluation code
        iou_scores = torch.zeros(logits.shape[0], dtype=torch.float32, device=logits.device).unsqueeze(1).unsqueeze(1)
        loss = None
        if labels is not None:
            loss = nn.BCEWithLogitsLoss()(logits.squeeze(1), labels)
        
        return SamMultimaskOutput(loss=loss, iou_scores=iou_scores, pred_masks=logits.unsqueeze(1))


class CRnnCell(nn.Module):
    def __init__(self):
        # A cnn which goes from 512 x 64 x 64 -> 256 x 64 x 64
        super(CRnnCell, self).__init__()
        self.hidden_state = None
        self.conv_o = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv_h = nn.Conv2d(512, 256, kernel_size=3, padding=1)

    def forward(self, x):
        if self.hidden_state is None:
            self.hidden_state = torch.zeros(x.size(0), 256, x.size(2), x.size(3), device=x.device)
        combined = torch.cat([x, self.hidden_state], dim=1)
        o = self.conv_o(combined)
        h = self.conv_h(combined)
        self.hidden_state = h
        return o
    
    def reset(self):
        self.hidden_state = None


def forward(
        self,
        image_embeddings: torch.Tensor,
        image_positional_embeddings: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        output_attentions: Optional[bool] = None,
        attention_similarity: torch.Tensor = None,
        target_embedding: torch.Tensor = None,
        cell=None

    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Args:
            image_embeddings (`torch.Tensor`):
                the embeddings from the image encoder
            image_positional_embedding (`torch.Tensor`):
                positional encoding with the shape of image_embeddings
            sparse_prompt_embeddings (`torch.Tensor`):
                The embeddings of the points and boxes
            dense_prompt_embeddings (`torch.Tensor`):
                the embeddings of the mask inputs
            multimask_output (bool):
                Whether to return multiple masks or a single mask.
            output_attentions (bool, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
        """
        batch_size, num_channels, height, width = image_embeddings.shape
        point_batch_size = sparse_prompt_embeddings.shape[1]
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        
        output_tokens = output_tokens.repeat(batch_size, point_batch_size, 1, 1)

        if sparse_prompt_embeddings.sum().item() != 0:
            tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=2)
        else:
            tokens = output_tokens
        point_embeddings = tokens.to(self.iou_token.weight.dtype)

        # Expand per-image data in batch direction to be per-point
        dense_prompt_embeddings = cell(dense_prompt_embeddings)
        
        image_embeddings = image_embeddings + dense_prompt_embeddings
        image_embeddings = image_embeddings.repeat_interleave(point_batch_size, 0)
        image_positional_embeddings = image_positional_embeddings.repeat_interleave(point_batch_size, 0)

        # Run the transformer, image_positional_embedding are consumed
        point_embedding, image_embeddings, attentions = self.transformer(
            point_embeddings=point_embeddings,
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
            attention_similarity=attention_similarity,
            target_embedding=target_embedding,
            output_attentions=output_attentions,
        )
        iou_token_out = point_embedding[:, :, 0, :]
        mask_tokens_out = point_embedding[:, :, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        image_embeddings = image_embeddings.transpose(2, 3).reshape(
            batch_size * point_batch_size, num_channels, height, width
        )

        upscaled_embedding = self.upscale_conv1(image_embeddings)
        upscaled_embedding = self.activation(self.upscale_layer_norm(upscaled_embedding))
        upscaled_embedding = self.activation(self.upscale_conv2(upscaled_embedding))

        hyper_in_list = []
        for i in range(self.num_mask_tokens):
            current_mlp = self.output_hypernetworks_mlps[i]
            hyper_in_list += [current_mlp(mask_tokens_out[:, :, i, :])]
        hyper_in = torch.stack(hyper_in_list, dim=2)

        _, num_channels, height, width = upscaled_embedding.shape
        upscaled_embedding = upscaled_embedding.reshape(batch_size, point_batch_size, num_channels, height * width)
        masks = (hyper_in @ upscaled_embedding).reshape(batch_size, point_batch_size, -1, height, width)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, :, mask_slice, :, :]
        iou_pred = iou_pred[:, :, mask_slice]

        outputs = (masks, iou_pred)

        if output_attentions:
            outputs = outputs + (attentions,)
        else:
            outputs = outputs + (None,)

        return outputs


class Encoder(nn.Module):
    """
    A convolutional neural network, consisting of len(num_filters) times a block of no_convs_per_block convolutional layers,
    after each block a pooling operation is performed. And after each convolutional layer a non-linear (ReLU) activation function is applied.
    """
    def __init__(self, input_channels, num_filters, no_convs_per_block, initializers, padding=True, posterior=False):
        super(Encoder, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.num_filters = num_filters

        if posterior:
            #To accomodate for the mask that is concatenated at the channel axis, we increase the input_channels.
            self.input_channels += 1

        layers = []
        for i in range(len(self.num_filters)):
            """
            Determine input_dim and output_dim of conv layers in this block. The first layer is input x output,
            All the subsequent layers are output x output.
            """
            input_dim = self.input_channels if i == 0 else output_dim
            output_dim = num_filters[i]
            
            if i != 0:
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True))
            
            layers.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=int(padding)))
            layers.append(nn.ReLU(inplace=True))

            for _ in range(no_convs_per_block-1):
                layers.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=int(padding)))
                layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

        self.layers.apply(init_weights)

    def forward(self, input):
        output = self.layers(input)
        return output
    

class AxisAlignedConvGaussian(nn.Module):
    """
    A convolutional net that parametrizes a Gaussian distribution with axis aligned covariance matrix.
    """
    def __init__(self, input_channels, num_filters, no_convs_per_block, latent_dim, initializers, posterior=False):
        super(AxisAlignedConvGaussian, self).__init__()
        self.input_channels = input_channels
        self.channel_axis = 1
        self.num_filters = num_filters
        self.no_convs_per_block = no_convs_per_block
        self.latent_dim = latent_dim
        self.posterior = posterior
        if self.posterior:
            self.name = 'Posterior'
        else:
            self.name = 'Prior'
        self.encoder = Encoder(self.input_channels, self.num_filters, self.no_convs_per_block, initializers, posterior=self.posterior)
        self.conv_layer = nn.Conv2d(num_filters[-1], 2 * self.latent_dim, (1,1), stride=1)
        self.show_img = 0
        self.show_seg = 0
        self.show_concat = 0
        self.show_enc = 0
        self.sum_input = 0

        nn.init.kaiming_normal_(self.conv_layer.weight, mode='fan_in', nonlinearity='relu')
        nn.init.normal_(self.conv_layer.bias)

    def forward(self, input, segm=None):

        #If segmentation is not none, concatenate the mask to the channel axis of the input
        if segm is not None:
            self.show_img = input
            self.show_seg = segm
            input = torch.cat((input, segm), dim=1)
            self.show_concat = input
            self.sum_input = torch.sum(input)

        encoding = self.encoder(input)
        self.show_enc = encoding

        #We only want the mean of the resulting hxw image
        encoding = torch.mean(encoding, dim=2, keepdim=True)
        encoding = torch.mean(encoding, dim=3, keepdim=True)

        #Convert encoding to 2 x latent dim and split up for mu and log_sigma
        mu_log_sigma = self.conv_layer(encoding)

        #We squeeze the second dimension twice, since otherwise it won't work when batch size is equal to 1
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)

        mu = mu_log_sigma[:,:self.latent_dim]
        log_sigma = mu_log_sigma[:,self.latent_dim:]

        #This is a multivariate normal with diagonal covariance matrix sigma
        #https://github.com/pytorch/pytorch/pull/11178
        dist = Independent(Normal(loc=mu, scale=torch.exp(log_sigma)),1)
        return dist
