import torch
from dataclasses import dataclass, field
from transformers.models.sam.modeling_sam import SamImageSegmentationOutput

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
