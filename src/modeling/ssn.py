import numpy as np
import torch
import torch.nn.functional as F
from transformers import SamPreTrainedModel, SamModel
from torch.distributions import LowRankMultivariateNormal
from torch import nn
from typing import Optional

from .modeling_utils import SamMultimaskOutput


def _batch_mv(bmat, bvec):
    bmat = tf.tile(tf.expand_dims(bmat, axis=0), (bvec.shape[0].value,) + (1,) * len(bmat.shape))
    bvec = tf.expand_dims(bvec, axis=-1)
    return tf.squeeze(tf.matmul(bmat, bvec), axis=-1)


# port from pytorch, fast and dirty no checks, not meant to ever be reused
class LowRankMultivariateNormal2(object):
    def __init__(self, loc, cov_factor, cov_diag):
        self.loc = loc
        self.cov_diag = cov_diag
        self.cov_factor = cov_factor
        self.batch_shape = loc.shape[0]
        self.event_shape = loc.shape[-1]

    @staticmethod
    def get_shape(shape):
        return tuple(s.value if s.value is not None else -1 for s in shape)

    def rsample(self, sample_shape):
        assert isinstance(sample_shape, tuple)
        shape = sample_shape + self.batch_shape + self.event_shape
        W_shape = shape[:-1] + (self.cov_factor.shape[-1].value,)
        eps_W = tf.random.normal(W_shape, dtype=self.loc.dtype)
        eps_D = tf.random.normal(shape, dtype=self.loc.dtype)
        return (self.loc + _batch_mv(self.cov_factor, eps_W) + tf.sqrt(self.cov_diag) * eps_D)


import math

class SsnSam(SamPreTrainedModel):
    def __init__(
            self, 
            config,
            rank=10,
            **kwargs
        ) -> None:
        super().__init__(config, **kwargs)
        self.sam = SamModel(config)

        self.mean = nn.Conv2d(1, 1, kernel_size=(1, 1))
        self.log_cov_diag = nn.Conv2d(1, 1, kernel_size=(1, 1))
        self.cov_factor = nn.Conv2d(1, rank, kernel_size=(1, 1))
        self.rank = rank

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
            multimask_output=False,
        )

        logits = outputs.pred_masks.squeeze(2)
        mean = self.mean(logits)
        epsilon = 1e-5
        log_cov_diag = torch.exp(self.log_cov_diag(logits)) + epsilon
        cov_factor = self.cov_factor(logits)
        
        num_pixels = np.prod(mean.shape[-2:])
        mean = mean.reshape(-1, num_pixels)
        log_cov_diag = log_cov_diag.reshape(-1, num_pixels)
        cov_factor = cov_factor.reshape(-1, num_pixels, self.rank)

        batch_size = mean.shape[0]
        num_mc_samples = 20 if self.training else 3
        m = LowRankMultivariateNormal(mean, cov_factor, log_cov_diag)
        sample = m.rsample([num_mc_samples]).reshape(-1, num_pixels)
        target = labels[:, :1].expand(-1, num_mc_samples, -1, -1).reshape((batch_size * num_mc_samples, -1)).float()
        log_prob = -F.cross_entropy(sample, target, reduction='none').reshape((batch_size, num_mc_samples, -1))
        loglikelihood = torch.mean(torch.logsumexp(torch.sum(log_prob, dim=-1), dim=1) - math.log(num_mc_samples))
        loss = -loglikelihood / (labels.shape[-1] * labels.shape[-2])
        
        #loss = nn.BCEWithLogitsLoss()(sample.reshape(-1, *logits.shape[-2:]), labels[:, 0].float())

        sample = sample.reshape(batch_size, 1, num_mc_samples, *logits.shape[-2:])

        return SamMultimaskOutput(
            loss=loss,
            pred_masks=sample,
            iou_scores=torch.zeros(sample.shape[0], dtype=torch.float32, device=sample.device).unsqueeze(1).unsqueeze(1)
        )