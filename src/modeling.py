import monai
import torch
import random

from dataclasses import dataclass
from enum import Enum, auto
from torch import nn
from transformers.models.sam.modeling_sam import (
    SamModel, 
    SamPreTrainedModel,
    SamImageSegmentationOutput
)
from typing import Optional, Callable
from scipy.optimize import linear_sum_assignment
from functools import partial

class Ablation(Enum):
    NONE = auto()
    RANDOM = auto()
    SEQUENTIAL = auto()
    STOP_GRADIENTS = auto()
    NO_HUNGARIAN_ALGORITHM = auto()

    @classmethod
    def from_str(cls, label):
        if label == "no_ha":
            return cls.NO_HUNGARIAN_ALGORITHM
        elif label == "sg":
            return cls.STOP_GRADIENTS
        else:
            label = label.upper()
            return cls[label]


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


class SamBaseline(SamPreTrainedModel):

    def __init__(self, config, multimask_output: bool = True):
        super().__init__(config)
        self.sam = SamModel(config)
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
        label_mask: Optional[torch.Tensor] = None
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


class SeqSam(SamPreTrainedModel):

    def __init__(self, config, num_samples: int = 4, ablation: str = "none"):
        super().__init__(config)
        self.sam = SamModel(config)
        self.seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="none")
        self.multimask_output = True
        self.num_samples = num_samples
        self.ablation = Ablation.from_str(ablation)

        self.cell = CRnnCell()
        self.sam.mask_decoder.forward = partial(
            wrap_forward, 
            cell=self.cell, 
            old_forward=self.sam.mask_decoder.forward
        )

    @property
    def mask_decoder(self):
        return self.sam.mask_decoder
    
    def forward(
        self, 
        pixel_values: Optional[torch.Tensor] = None,
        image_embeddings: Optional[torch.Tensor] = None,
        input_boxes: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        label_mask: Optional[torch.Tensor] = None
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
            
            outputs = self.sam(
                image_embeddings=image_embeddings, 
                input_boxes=input_boxes,
                input_masks=input_masks,
                multimask_output=False,
            )
            input_masks = outputs.pred_masks.squeeze(2)

            if self.ablation == Ablation.STOP_GRADIENTS:
                input_masks = input_masks.detach()
                        
            pred_masks.append(outputs.pred_masks)

        if self.training:
            if self.ablation == Ablation.RANDOM:
                rand_idxs = random.sample(range(num_samples), k=labels.shape[1])
                pred_masks = [pred_masks[i] for i in rand_idxs]

            elif self.ablation == Ablation.SEQUENTIAL:
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
            
            if self.ablation != Ablation.NO_HUNGARIAN_ALGORITHM:
                row_ind, col_ind = linear_sum_assignment(losses.detach().cpu().numpy())
            else:
                row_ind = torch.arange(losses.shape[0])
                col_ind = torch.arange(losses.shape[1])
            loss = losses[row_ind, col_ind]
            if self.training:
                loss *= label_mask[i].float()
            total_loss += loss.sum()
        
        total_loss /= labels.shape[1]
                
        return SamMultimaskOutput(
            loss=total_loss,
            iou_scores=outputs.iou_scores,
            pred_masks=pred_masks[:, :, :self.num_samples],
        )


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


def wrap_forward(
    old_forward: callable,
    cell: CRnnCell,
    dense_prompt_embeddings: torch.Tensor,
    **kwargs
):

    dense_prompt_embeddings = cell(dense_prompt_embeddings)

    return old_forward(
        dense_prompt_embeddings=dense_prompt_embeddings,
        **kwargs
    )
