import math
import torch
import torch.nn.functional as F
from functools import partial
from torch import nn
from torch.distributions import MultivariateNormal, Independent, Normal, kl
from transformers import SamModel, SamPreTrainedModel
from typing import Optional, Tuple

from .modeling_utils import SamMultimaskOutput


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
        #nn.init.normal_(m.weight, std=0.001)
        #nn.init.normal_(m.bias, std=0.001)
        truncated_normal_(m.bias, mean=0, std=0.001)


def l2_regularisation(m):
    l2_reg = None

    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    return l2_reg


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
        sampled_tokens: torch.Tensor = None,
        mean: nn.Linear = None,
        cov: nn.Linear = None,
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
        #output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        #output_tokens = torch.cat([self.iou_token.weight, sampled_tokens], dim=0)
        iou_token_weight = self.iou_token.weight.repeat(batch_size, point_batch_size, 1, 1)
        sampled_tokens = sampled_tokens.repeat(batch_size, point_batch_size, 1, 1)
        output_tokens = torch.cat([iou_token_weight, sampled_tokens], dim=2)
        #output_tokens = output_tokens.repeat(batch_size, point_batch_size, 1, 1)

        if sparse_prompt_embeddings.sum().item() != 0:
            tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=2)
        else:
            tokens = output_tokens
        point_embeddings = tokens.to(self.iou_token.weight.dtype)

        # Expand per-image data in batch direction to be per-point
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
            current_mlp = self.output_hypernetworks_mlps[0]
            hyper_in_list += [current_mlp(mask_tokens_out[:, :, 0, :])]
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


class ProbabilisticSam(SamPreTrainedModel):
    """
    Probabilistic UNet model which replaces Unet with a SAM backbone.
    Hyperparameters: https://github.com/MiguelMonteiro/PHiSeg-code/blob/master/phiseg/experiments/probunet_1annot.py
    """

    def __init__(
        self,
        config, 
        input_channels=3,
        num_filters=[32, 64, 128, 192, 192, 192, 192],
        latent_dim=256,
        beta=10.0,
        **kwargs
    ):

        super().__init__(config, **kwargs)

        self.input_channels = input_channels
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.no_convs_per_block = 3
        self.initializers = {'w':'he_normal', 'b':'normal'}
        self.beta = beta

        self.prior_latent_space = None
        self.posterior_latent_space = None
        self.reconstruction = None
        
        self.sam = SamModel(config)
        self.prior = AxisAlignedConvGaussian(self.input_channels, self.num_filters, self.no_convs_per_block, self.latent_dim,  self.initializers,)
        self.posterior = AxisAlignedConvGaussian(self.input_channels, self.num_filters, self.no_convs_per_block, self.latent_dim, self.initializers, posterior=True)

        #self.upscale = nn.Linear(self.latent_dim, 256) # Additional layer required to scale up the latent space to the same size as the image embeddings

    def sample(self, image_embeddings, input_boxes=None):
        z_prior = self.prior_latent_space.sample()
        self.sam.mask_decoder.forward = partial(forward, self=self.sam.mask_decoder, sampled_tokens=z_prior)
        outputs = self.sam(
            image_embeddings=image_embeddings,
            input_boxes=input_boxes,
            multimask_output=False,
        )

        return outputs.pred_masks
        
    
    def reconstruct(self, image_embeddings, input_boxes=None, use_posterior_mean=False, calculate_posterior=False, z_posterior=None):
        """
        Reconstruct a segmentation from a posterior sample (decoding a posterior sample) and UNet feature map
        use_posterior_mean: use posterior_mean instead of sampling z_q
        calculate_posterior: use a provided sample or sample from posterior latent space
        """
        if use_posterior_mean:
            z_posterior = self.posterior_latent_space.loc
        else:
            if calculate_posterior:
                z_posterior = self.posterior_latent_space.rsample()
        
        self.sam.mask_decoder.forward = partial(forward, self=self.sam.mask_decoder, sampled_tokens=z_posterior)
        outputs = self.sam(
            image_embeddings=image_embeddings,
            input_boxes=input_boxes,
            multimask_output=False,
        )

        return outputs.pred_masks
    
    def kl_divergence(self, analytic=True, calculate_posterior=False, z_posterior=None):
        """
        Calculate the KL divergence between the posterior and prior KL(Q||P)
        analytic: calculate KL analytically or via sampling from the posterior
        calculate_posterior: if we use samapling to approximate KL we can sample here or supply a sample
        """
        if analytic:
            #Neeed to add this to torch source code, see: https://github.com/pytorch/pytorch/issues/13545
            kl_div = kl.kl_divergence(self.posterior_latent_space, self.prior_latent_space)
        else:
            if calculate_posterior:
                z_posterior = self.posterior_latent_space.rsample()
            log_posterior_prob = self.posterior_latent_space.log_prob(z_posterior)
            log_prior_prob = self.prior_latent_space.log_prob(z_posterior)
            kl_div = log_posterior_prob - log_prior_prob
        return kl_div   

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

        labels = labels.float()[:, 0]
        loss = 0.0
        
        if image_embeddings is None:
            image_embeddings = self.sam.get_image_embeddings(pixel_values)

        self.prior_latent_space = self.prior.forward(pixel_values)

        if self.training:
            upsampled_labels = F.interpolate(labels.unsqueeze(1), size=pixel_values.shape[-2:], mode="nearest")
            self.posterior_latent_space = self.posterior.forward(pixel_values, upsampled_labels)
            z_posterior = self.posterior_latent_space.rsample()
            self.kl = torch.mean(self.kl_divergence(analytic=True, calculate_posterior=False, z_posterior=z_posterior))
            self.reconstruction = self.reconstruct(image_embeddings=image_embeddings, input_boxes=input_boxes, use_posterior_mean=False, calculate_posterior=False, z_posterior=z_posterior)
            
            criterion = nn.BCEWithLogitsLoss(size_average = False, reduce=False, reduction=None)
            reconstruction_loss = criterion(input=self.reconstruction.squeeze(1).squeeze(1), target=labels)
            self.reconstruction_loss = torch.sum(reconstruction_loss)
            self.mean_reconstruction_loss = torch.mean(reconstruction_loss)

            loss = -(self.reconstruction_loss + self.beta * self.kl)
            reg_loss = l2_regularisation(self.posterior) + l2_regularisation(self.prior)
            loss = -loss + 1e-5 * reg_loss
        else:
            pred_masks = self.sample(image_embeddings, input_boxes)
            for _ in range(5):
                pred_masks = torch.cat([pred_masks, self.sample(image_embeddings, input_boxes)], dim=2)
            self.reconstruction = pred_masks

            criterion = nn.BCEWithLogitsLoss(size_average = False, reduce=False, reduction=None)
            reconstruction_loss = criterion(input=self.reconstruction.squeeze(1).mean(1), target=labels)
            self.reconstruction_loss = torch.sum(reconstruction_loss)
            loss = self.reconstruction_loss

        return SamMultimaskOutput(
            loss=loss,
            pred_masks=self.reconstruction,
            iou_scores=torch.zeros(self.reconstruction.shape[0], dtype=torch.float32, device=self.reconstruction.device).unsqueeze(1).unsqueeze(1)
        )
