import numpy as np
import torch
import torch.nn.functional as F
import random

from abc import ABC, abstractmethod
from transformers import SamModel
from typing import Optional, Tuple

from src.utils import RegistryMixin

class ClickStrategy(ABC, RegistryMixin):

    def __init__(
        self, 
        threshold: float = 0.3, 
        temperature: float = 1.0, 
        theta_tau: float = 0.2, 
        model_path: str = "data/theta"
    ) -> None:

        super().__init__()
        self.threshold = threshold
        self.temperature = temperature
        self.theta_tau = theta_tau
        self.model_path = model_path  # Only instantiated for theta

    @staticmethod
    def get_click_random(mask: torch.Tensor, count: int, num_samples: int):
        idxs = np.random.choice(count, num_samples)
        y, x = torch.where(mask.squeeze())
        return [[x[idx], y[idx]] for idx in idxs]

    @abstractmethod
    def get_click(
        self, 
        input_mask: torch.Tensor, 
        binary_input_mask: torch.Tensor,
        label: Optional[torch.Tensor] = None, 
        num_samples: int = 1,
        image_embeddings: Optional[torch.Tensor] = None,
        input_boxes: Optional[torch.Tensor] = None
    ):
        pass


@ClickStrategy.register_subclass("training")
class TrainingClickStrategy(ClickStrategy):

    def get_click(
        self, 
        input_mask: torch.Tensor, 
        label: Optional[torch.Tensor] = None, 
        num_samples: int = 1,
        image_embeddings: Optional[torch.Tensor] = None,
        input_boxes: Optional[torch.Tensor] = None
    ):
        binary_input_mask = (input_mask > 0.0).float()
        
        clicks, labels = [], []
        
        fn_mask = torch.logical_and(binary_input_mask == 0, label == 1)
        fp_mask = torch.logical_and(binary_input_mask == 1, label == 0)
        fn_count = fn_mask.sum()
        fp_count = fp_mask.sum()

        if fn_count == 0 and fp_count == 0:
            return [[0, 0] for _ in range(num_samples)], [0 for _ in range(num_samples)]

        num_fn_samples = 0
        num_fp_samples = 0
        for _ in range(num_samples):
            if random.random() < (fn_count / (fn_count + fp_count)):
                num_fn_samples += 1
            else:
                num_fp_samples += 1

        if num_fn_samples > 0:
            clicks.extend(self.get_click_random(fn_mask, fn_count.item(), num_fn_samples))
        if num_fp_samples > 0:
            clicks.extend(self.get_click_random(fp_mask, fp_count.item(), num_fp_samples))
        labels.extend([1] * num_fn_samples)
        labels.extend([0] * num_fp_samples)

        return clicks, labels


@ClickStrategy.register_subclass("threshold")
class ThresholdClickStrategy(ClickStrategy):

    def fp_mask(self, input_mask: torch.Tensor) -> torch.Tensor:
        return torch.logical_and(F.sigmoid(input_mask) > 0.5, F.sigmoid(input_mask) < (0.5 + self.threshold))

    def fn_mask(self, input_mask: torch.Tensor) -> torch.Tensor:
        return torch.logical_and(F.sigmoid(input_mask) < 0.5, F.sigmoid(input_mask) > (0.5 - self.threshold))

    def _get_clicks_and_labels(self, fp_mask: torch.Tensor, fn_mask: torch.Tensor, num_samples: int) -> Tuple[list, list]:
        clicks = []
        labels = []
        
        fn_count = fn_mask.sum()
        fp_count = fp_mask.sum()

        if fn_count == 0 and fp_count == 0:
            return [[0, 0] for _ in range(num_samples)], [0 for _ in range(num_samples)]

        num_fn_samples = int(num_samples * (fn_count / (fn_count + fp_count)))
        num_fp_samples = num_samples - num_fn_samples

        clicks.extend(self.get_click_random(fn_mask, fn_count.item(), num_fn_samples))
        clicks.extend(self.get_click_random(fp_mask, fp_count.item(), num_fp_samples))
        labels.extend([1] * num_fn_samples)
        labels.extend([0] * num_fp_samples)

        return clicks, labels

    def get_click(
        self, 
        input_mask: torch.Tensor, 
        binary_input_mask: torch.Tensor,
        label: Optional[torch.Tensor] = None, 
        num_samples: int = 1,
        image_embeddings: Optional[torch.Tensor] = None,
        input_boxes: Optional[torch.Tensor] = None
    ):
        fp_mask, fn_mask = self.fp_mask(input_mask), self.fn_mask(input_mask)

        error_mask = torch.tensor(fp_mask + fn_mask).to("cuda")
        probs = (error_mask / error_mask.sum()).reshape(-1)

        binary_input_mask = binary_input_mask.squeeze().cpu().numpy()
        H, W = binary_input_mask.shape[-2:]

        idxs = torch.multinomial(probs, num_samples, replacement=True).cpu().numpy()

        clicks = [[idx % W, idx // W] for idx in idxs]
        labels = 1.0 - binary_input_mask.reshape(-1)[idxs]

        return clicks, labels


@ClickStrategy.register_subclass("random")
class RandomClickStrategy(ThresholdClickStrategy):
    """
    Ablation that just randomly samples clicks without strategy
    """

    def get_click(
        self, 
        input_mask: torch.Tensor, 
        binary_input_mask: torch.Tensor,
        label: Optional[torch.Tensor] = None, 
        num_samples: int = 1,
        image_embeddings: Optional[torch.Tensor] = None,
        input_boxes: Optional[torch.Tensor] = None
    ):

        error_mask = torch.ones(input_mask.shape).to("cuda")
        probs = (error_mask / error_mask.sum()).reshape(-1)

        binary_input_mask = binary_input_mask.squeeze().cpu().numpy()
        H, W = binary_input_mask.shape[-2:]

        idxs = torch.multinomial(probs, num_samples, replacement=True).cpu().numpy()

        clicks = [[idx % W, idx // W] for idx in idxs]
        labels = 1.0 - binary_input_mask.reshape(-1)[idxs]

        return clicks, labels


@ClickStrategy.register_subclass("theta")
class ThetaClickStrategy(ClickStrategy):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = SamModel.from_pretrained(self.model_path).to("cuda")

    def get_click(
        self, 
        input_mask: torch.Tensor, 
        binary_input_mask: torch.Tensor,
        label: Optional[torch.Tensor] = None, 
        num_samples: int = 1,
        image_embeddings: Optional[torch.Tensor] = None,
        input_boxes: Optional[torch.Tensor] = None
    ):
        self.model.eval()
        theta_mask = self.model(
            image_embeddings=image_embeddings.unsqueeze(0),
            input_boxes=input_boxes.unsqueeze(0),
            multimask_output=False,
        ).pred_masks.squeeze(0).squeeze(0)
        
        binary_input_mask = binary_input_mask.squeeze().cpu().numpy()
        H, W = theta_mask.shape[-2:]

        probs = F.softmax(theta_mask.reshape(-1) / self.theta_tau, dim=0)
        idxs = torch.multinomial(probs, num_samples, replacement=True).cpu().numpy()

        clicks = [[idx % W, idx // W] for idx in idxs]
        labels = 1.0 - binary_input_mask.reshape(-1)[idxs]
        
        return clicks, labels 


@ClickStrategy.register_subclass("sampling")
class SamplingClickStrategy(ClickStrategy):

    def get_click(
        self, 
        input_mask: torch.Tensor, 
        label: Optional[torch.Tensor] = None, 
        num_samples: int = 1,
        image_embeddings: Optional[torch.Tensor] = None,
        input_boxes: Optional[torch.Tensor] = None
    ):

        binary_input_mask = (input_mask > 0.0).float()
        
        fp_mask = 1.0 - F.sigmoid(input_mask).squeeze().cpu().numpy()
        fn_mask = F.sigmoid(input_mask).squeeze().cpu().numpy()

        # Set fp_mask to 0.0 where binary_input_mask is 0
        fp_mask = np.where(binary_input_mask.squeeze().cpu().numpy() == 0, 0.0, fp_mask)
        # Set fn_mask to 0.0 where binary_input_mask is 1
        fn_mask = np.where(binary_input_mask.squeeze().cpu().numpy() == 1, 0.0, fn_mask)

        error_mask = torch.tensor(fp_mask + fn_mask).to("cuda")
        topk = torch.topk(error_mask.reshape(-1), num_samples, largest=True)
        idxs = topk.indices.cpu().numpy()

        binary_input_mask = binary_input_mask.squeeze().cpu().numpy()
        _, W = binary_input_mask.shape[-2:]

        clicks = [[idx % W, idx // W] for idx in idxs]
        labels = 1.0 - binary_input_mask.reshape(-1)[idxs]
        
        
        return clicks, labels
