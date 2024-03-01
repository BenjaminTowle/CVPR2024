import monai
import numpy as np
import torch
import torch.nn.functional as F
import random

from abc import ABC, abstractmethod
from dataclasses import dataclass
from torch import nn
from transformers.models.sam.modeling_sam import (
    SamModel, 
    SamPreTrainedModel,
    SamImageSegmentationOutput
)
from typing import Optional, Callable, Union

from src.click import ClickStrategy
from src.metrics import iou
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


class SimilarityScorer(ABC, RegistryMixin):

    @abstractmethod
    def score(self, A: np.array, B: np.array) -> float:
        pass


@SimilarityScorer.register_subclass("iou")
class IOUScorer(SimilarityScorer):

    def score(self, A: np.array, B: np.array) -> float:
        return iou(A, B)


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
        **kwargs,
    ):

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
        _iou_loss= self.iou_loss(outputs.pred_masks, outputs.iou_scores, labels)

        loss = loss["loss"] + _iou_loss

        return SamMultimaskOutput(
            loss=loss,
            iou_scores=outputs.iou_scores,
            pred_masks=outputs.pred_masks,
        )


class SLIP(Model):

    """
    Wraps around SamModel to compute loss for multiple masks
    """
    def __init__(
        self, 
        config, 
        processor, 
        num_simulations: int = 50, 
        num_preds: int = 4,
        tau=0.7,
        theta_tau=0.2, 
        threshold=0.25,
        model_path: str = "data/theta",
        search_strategy: str = "mcts",
        click_strategy: str = "sampling",
        scorer: str = "iou",
        use_posterior: bool = False,
        do_reduce: bool = False,
        multiple_annotations: bool = True,
    ):
        super().__init__(config)
        self.sam = SamModel(config)
        self.seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="none")
        self.processor = processor
        self.num_simulations = num_simulations
        self.multiple_annotations = multiple_annotations

        self.num_preds = num_preds
        self.threshold = threshold
        self.tau = tau
        self.theta_tau = theta_tau
        self.search_strategy = SearchStrategy.create(search_strategy)()
        self.click_strategy = ClickStrategy.create(click_strategy)(
            temperature=tau, threshold=threshold, model_path=model_path,
            theta_tau=theta_tau,
        )
        self.scorer = SimilarityScorer.create(scorer)()
        self.use_posterior = use_posterior
        self.do_reduce = do_reduce

    def set_click_strategy(self, click_strategy: Union[str, ClickStrategy]):
        if isinstance(click_strategy, str):
            self.click_strategy = ClickStrategy.create(click_strategy)(
                temperature=self.click_strategy.temperature, 
                threshold=self.threshold,
                model_path=self.click_strategy.model_path,
                theta_tau=self.click_strategy.theta_tau,
            )
        else:
            self.click_strategy = click_strategy

    def _get_clicks(
        self,
        pred_masks: torch.Tensor,
        #binary_input_masks: torch.Tensor,
        original_sizes: torch.Tensor,
        reshaped_input_sizes: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        num_samples: int = 1,
        image_embeddings: Optional[torch.Tensor] = None,
        input_boxes: Optional[torch.Tensor] = None,
    ):

        #input_masks = self.processor.image_processor.post_process_masks(
            #pred_masks.cpu(), original_sizes.cpu(), 
            #reshaped_input_sizes.cpu(),
            #binarize=False,
        #)
        #input_masks = torch.stack(input_masks, dim=0).to(self.device)

        input_points, input_labels = zip(*map(lambda i: self.click_strategy.get_click(
            input_mask=pred_masks[i],
            label=labels[i, random.randint(0, 3)] if labels is not None else None,
            num_samples=num_samples,
            image_embeddings=image_embeddings[i] if image_embeddings is not None else None,
            input_boxes=input_boxes[i] if input_boxes is not None else None,
        ), range(pred_masks.size(0))))

        input_points = torch.tensor(input_points).to(self.device).unsqueeze(1)
        input_labels = torch.tensor(input_labels).to(self.device).unsqueeze(1)

        ratios = reshaped_input_sizes / original_sizes
        ratios = ratios.unsqueeze(1).unsqueeze(1).expand_as(input_points)

        return input_points * ratios, input_labels

    def _forward_train(
        self,
        pred_masks: torch.Tensor,
        image_embeddings: torch.Tensor, 
        input_boxes: torch.Tensor,
        original_sizes: torch.Tensor,
        reshaped_input_sizes: torch.Tensor, 
        labels: torch.Tensor,
    ):

        old_strategy = self.click_strategy
        self.set_click_strategy("training")
        
        input_points, input_labels = None, None

        loss = self.compute_loss(pred_masks, labels) 

        input_points, input_labels = self._get_clicks(
            pred_masks=pred_masks,
            original_sizes=original_sizes,
            reshaped_input_sizes=reshaped_input_sizes,
            labels=labels,
            num_samples=1,
        )

        outputs = self.sam(
            image_embeddings=image_embeddings,
            input_boxes=input_boxes,
            input_points=input_points,
            input_labels=input_labels,
            input_masks=pred_masks.squeeze(2),
            multimask_output=False,
        )
        pred_masks = outputs.pred_masks

        loss += self.compute_loss(pred_masks, labels)

        self.set_click_strategy(old_strategy)

        return SamMultimaskOutput(
            loss=loss,
            iou_scores=outputs.iou_scores,
            pred_masks=outputs.pred_masks,
            initial_pred_masks=pred_masks,
            input_points=input_points,
            input_labels=input_labels,
        )


    def _clustering(
        self,
        pred_masks,
        target_masks,
        labels=None,
        return_idxs=False,
        probs=None,
        iou_scores=None,
    ):

        p_pred = (pred_masks > 0.0).squeeze(1).cpu().numpy()
        shape = (p_pred.shape[0], p_pred.shape[1], p_pred.shape[1])
        similarity_matrix = np.zeros(shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    similarity_matrix[i, j, k] = self.scorer.score(p_pred[i, j], p_pred[i, k])
        
        chosen_preds = []
        all_chosen_idxs = []
        for i in range(pred_masks.size(0)):
            chosen_idxs = self.search_strategy.run_search(scores=similarity_matrix[i], k=self.num_preds)
            chosen_preds.append(pred_masks[i, :, chosen_idxs])
            all_chosen_idxs.append(chosen_idxs)
        chosen_preds = torch.stack(chosen_preds, dim=0)
        
        try:
            all_chosen_idxs = torch.stack(all_chosen_idxs).to(self.device)
        except:
            all_chosen_idxs = torch.tensor(all_chosen_idxs).to(self.device)

        if return_idxs:
            return chosen_preds, all_chosen_idxs, np.mean(similarity_matrix, axis=-1)

        return chosen_preds

    def _simulation(
        self,
        pred_masks,
        image_embeddings: torch.Tensor,
        input_boxes,
        original_sizes,
        reshaped_input_sizes,
        i=0
    ):

        input_points, input_labels = self._get_clicks(
            pred_masks=pred_masks, 
            original_sizes=original_sizes,
            reshaped_input_sizes=reshaped_input_sizes,
            num_samples=self.num_simulations if i == 0 else 1,
            image_embeddings=image_embeddings,
            input_boxes=input_boxes,
            labels=None,
        )

        bsz = input_points.size(0)
        CHUNK_SIZE = min(10, self.num_simulations)

        chunk_image_embeddings = image_embeddings.repeat_interleave(CHUNK_SIZE, dim=0)
        if input_boxes is not None:
            input_boxes = input_boxes.repeat_interleave(CHUNK_SIZE, dim=0)
        
        input_masks = (pred_masks > 0.0).repeat_interleave(CHUNK_SIZE, dim=0).float()
        all_pred_masks = []
        all_pred_iou = []

        for j in range(0, self.num_simulations, CHUNK_SIZE):

            chunk_input_points = input_points[:, :, j:j+CHUNK_SIZE].reshape(
                bsz * CHUNK_SIZE, 1, 1, 2)
            chunk_input_labels = input_labels[:, :, j:j+CHUNK_SIZE].reshape(
                bsz * CHUNK_SIZE, 1, 1)

            outputs = self.sam(
                image_embeddings=chunk_image_embeddings,
                input_points=chunk_input_points,
                input_labels=chunk_input_labels,
                input_boxes=input_boxes,
                input_masks=input_masks.squeeze(2).to(self.device),
                multimask_output=False
            )
            pred_masks = outputs.pred_masks.reshape(-1, 1, CHUNK_SIZE, *pred_masks.shape[-2:])
            pred_iou = outputs.iou_scores.reshape(-1, CHUNK_SIZE)
            all_pred_masks.append(pred_masks)
            all_pred_iou.append(pred_iou)
        
        new_pred_masks = torch.cat(all_pred_masks, dim=2)
        new_pred_iou = torch.cat(all_pred_iou, dim=1)

        return new_pred_masks, new_pred_iou

    def _forward_eval(
        self,
        pred_masks,
        image_embeddings: torch.Tensor,
        input_boxes,
        original_sizes,
        reshaped_input_sizes,
        labels
    ):
        assert pred_masks.size(0) == 1, "Only batch size 1 is supported for evaluation"
        
        new_pred_masks, new_pred_iou = self._simulation(
            pred_masks=pred_masks,
            image_embeddings=image_embeddings,
            input_boxes=input_boxes,
            original_sizes=original_sizes,
            reshaped_input_sizes=reshaped_input_sizes,
        )

        assert new_pred_masks.size(2) == self.num_simulations

        loss = self.compute_loss(new_pred_masks, labels)
        original_sizes = original_sizes.repeat_interleave(self.num_simulations, dim=0)
        reshaped_input_sizes = reshaped_input_sizes.repeat_interleave(self.num_simulations, dim=0)

        # Find intersection between all masks along dim 2
        pred_masks, idxs, pred_iou = self._clustering(new_pred_masks, pred_masks, labels=labels, return_idxs=True)
        iou_scores = torch.gather(torch.tensor(pred_iou).to(idxs.device), 1, idxs).unsqueeze(1)

        return SamMultimaskOutput(
            loss=loss,
            iou_scores=iou_scores,
            pred_masks=pred_masks,
        )

    @staticmethod
    def loss_fn(A, B):
        return nn.BCEWithLogitsLoss()(A, B) + monai.losses.DiceLoss(sigmoid=True)(A, B)
        #return monai.losses.DiceLoss(sigmoid=True)(A, B)

    def compute_loss(self, pred_masks, labels):
        #loss_fn = nn.BCEWithLogitsLoss()
        #loss_fn = monai.losses.DiceLoss(sigmoid=True)
        labels = labels.float()
        pred_masks = pred_masks[:, 0, 0, :, :]
        if self.multiple_annotations:
            loss = 0.0
            for i in range(labels.shape[1]):
                loss += self.loss_fn(pred_masks, labels[:, i])
            return loss / labels.shape[1]
        
        return loss_fn(pred_masks, labels[:, 0])

    def forward(
        self, 
        pixel_values=None,
        image_embeddings=None,
        input_boxes=None,
        input_masks=None,
        labels=None,
        original_sizes=None,
        reshaped_input_sizes=None,
    ):
        
        if image_embeddings is None:
            image_embeddings = self.sam.get_image_embeddings(pixel_values)

        outputs = self.sam(
            image_embeddings=image_embeddings, 
            input_boxes=input_boxes,
            input_masks=input_masks, 
            multimask_output=False
        )

        kwargs = {
            "pred_masks": outputs.pred_masks,
            "image_embeddings": image_embeddings,
            "input_boxes": input_boxes,
            "original_sizes": original_sizes,
            "reshaped_input_sizes": reshaped_input_sizes,
            "labels": labels,
        }
        
        if self.training:
            return self._forward_train(**kwargs)

        return self._forward_eval(**kwargs)


class SamThetaForTraining(Model):

    def __init__(self, config):
        super().__init__(config)
        self.env = None
        self.seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="none")

    def set_env(self, env: SLIP):
        self.env = env

    @torch.no_grad()
    def get_pretraining_targets(self, image_embeddings, input_boxes, labels):
        
        self.env.eval()
        outputs = self.env.sam(
            image_embeddings=image_embeddings,
            input_boxes=input_boxes,
            multimask_output=False
        )
        binary_input_mask = (F.sigmoid(outputs.pred_masks) > 0.5).squeeze(1).squeeze(1)
        fn_mask = torch.logical_and(binary_input_mask == 0, labels == 1)
        fp_mask = torch.logical_and(binary_input_mask == 1, labels == 0)
        click_mask = torch.logical_or(fn_mask, fp_mask).int()
        probs_mask = click_mask / click_mask.sum(dim=(-1, -2), keepdim=True)

        return probs_mask

    def forward(
        self, 
        pixel_values=None,
        input_boxes=None,
        labels=None,
        original_sizes=None,
        reshaped_input_sizes=None,
    ):

        image_embeddings = self.sam.get_image_embeddings(pixel_values)

        # Get targets for training
        new_labels = self.get_pretraining_targets(
            image_embeddings=image_embeddings,
            input_boxes=input_boxes,
            labels=labels
        )

        outputs = self.sam(
            image_embeddings=image_embeddings,
            input_boxes=input_boxes,
            multimask_output=False,
        )

        bsz = outputs.pred_masks.size(0)

        logits = outputs.pred_masks.reshape(bsz, -1)
        loss = (-1 * new_labels.reshape(bsz, -1) * torch.log_softmax(logits, dim=-1)).sum(-1).mean(0)

        return SamMultimaskOutput(
            loss=loss,
            iou_scores=outputs.iou_scores,
            pred_masks=outputs.pred_masks,
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
