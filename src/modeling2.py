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
        #_iou_loss= self.iou_loss(outputs.pred_masks, outputs.iou_scores, labels)

        loss = loss["loss"] #+ _iou_loss

        return SamMultimaskOutput(
            loss=loss,
            iou_scores=outputs.iou_scores,
            pred_masks=outputs.pred_masks,
        )


from scipy.optimize import linear_sum_assignment

class SamAR(Model):

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
        
        if image_embeddings is None:
            image_embeddings = self.sam.get_image_embeddings(pixel_values)

        input_masks = None
        total_loss = 0.0
        pred_masks = []
        for i in range(labels.shape[1]):
            outputs = self.sam(
                image_embeddings=image_embeddings, 
                input_boxes=input_boxes,
                input_masks=input_masks,
                multimask_output=False,
            )
            input_masks = outputs.pred_masks.squeeze(2)
            pred_masks.append(outputs.pred_masks)

            #loss = self.seg_loss(outputs.pred_masks.squeeze(1).squeeze(1), labels[:, i]).mean(-1).mean(-1)
            #loss *= label_mask[:, i].float()
            #total_loss += loss.mean()
        
        total_loss = 0.0
        pred_masks = torch.cat(pred_masks, dim=2)
        for i in range(labels.shape[0]):
            losses = torch.zeros(labels.shape[1], labels.shape[1]).to(labels.device)
            for j in range(labels.shape[1]):
                for k in range(labels.shape[1]):
                    loss = self.seg_loss(pred_masks[i, 0, j], labels[i, k]).mean()
                    losses[j, k] = loss
            
            row_ind, col_ind = linear_sum_assignment(losses.detach().cpu().numpy())
            loss = losses[row_ind, col_ind]
            loss *= label_mask[i].float()
            total_loss += loss.sum()
        
        total_loss /= labels.shape[1]

        
            
        return SamMultimaskOutput(
            loss=total_loss,
            iou_scores=outputs.iou_scores,
            pred_masks=pred_masks,
        )


from torch.distributions.multivariate_normal import MultivariateNormal
from typing import Tuple


class BlankClassifier(Model):
    def __init__(
        self, 
        config
    ) -> None:
        super().__init__(config)
        self.sam = SamModel(config)

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

        # Create labels for the blank classifier
        # Should be 0 if no mask is present, 1 if mask is present
        new_labels = []
        num_labels = labels.shape[1]
        for i in range(labels.shape[0]):
            num_blanks = 0
            for j in range(num_labels):
                if labels[i, j].sum() == 0:
                    num_blanks += 1
            new_labels.append(1 - (num_blanks / num_labels))
        new_labels = torch.tensor(new_labels).to(labels.device).float()
        batch_size = labels.shape[0]
        loss = nn.BCEWithLogitsLoss()(outputs.iou_scores.reshape(batch_size), new_labels)

        return SamMultimaskOutput(
            loss=loss,
            pred_masks=outputs.pred_masks,
            iou_scores=outputs.iou_scores,
        )


class SLIP(Model):

    """
    The world model 
    """
    def __init__(
        self, 
        config, 
        processor, 
        num_simulations: int = 50, 
        num_preds: int = 3,
        num_iterations: int = 5,
        num_mc_samples: int = 1,
        tau=0.7,
        theta_tau=0.2, 
        threshold=0.25,
        model_path: str = "data/theta",
        search_strategy: str = "mcts",
        click_strategy: str = "sampling",
        scorer: str = "iou",
        use_posterior: bool = False,
        multiple_annotations: bool = True,
    ):
        super().__init__(config)
        self.sam = SamModel(config)
        self.processor = processor
        self.num_simulations = num_simulations
        self.multiple_annotations = multiple_annotations

        self.num_preds = num_preds
        self.num_iterations = num_iterations
        self.num_mc_samples = num_mc_samples
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
        self.i = 0
        self.gains = 0.0

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

    @staticmethod
    def get_click_random(mask: torch.Tensor, count: int, num_samples: int):
        idxs = np.random.choice(count, num_samples)
        y, x = torch.where(mask.squeeze())
        return [[x[idx], y[idx]] for idx in idxs]

    def get_click_training(
        self, 
        pred_mask: torch.Tensor, 
        label: Optional[torch.Tensor] = None,
        fn_mask: Optional[torch.Tensor] = None,
        fp_mask: Optional[torch.Tensor] = None, 
    ):

        if fn_mask is None or fp_mask is None:
            binary_input_mask = (pred_mask > 0.0).int()
            fn_mask = torch.logical_and(binary_input_mask == 0, label == 1)
            fp_mask = torch.logical_and(binary_input_mask == 1, label == 0)
        fn_mask = fn_mask.int()
        fp_mask = fp_mask.int()
        fn_count = fn_mask.sum()
        fp_count = fp_mask.sum()

        if fn_count == 0 and fp_count == 0:
            return [[0, 0]], [0]

        if fn_count > fp_count:
            return self.get_click_random(fn_mask, fn_count.item(), 1), [1]
        
        return self.get_click_random(fp_mask, fp_count.item(), 1), [0]

    def get_click_eval(self, fn_mask: torch.Tensor, fp_mask: torch.Tensor):
        H, W = fn_mask.shape[-2:]
        error_mask = torch.cat([
            fn_mask.reshape(-1),
            fp_mask.reshape(-1),
        ], dim=-1)
        tau = 0.75
        probs = F.softmax(error_mask / tau, dim=-1)
        idxs = torch.multinomial(probs, 1).cpu().numpy()
        label = [1] if idxs[0] < error_mask.size(0) // 2 else [0]
        clicks = [[idx % W, idx // W] for idx in idxs]

        return clicks, label
    

    def _postprocess_clicks(self, input_points, input_labels, original_sizes, reshaped_input_sizes):
        input_points = torch.tensor(input_points).to(self.device).unsqueeze(1)
        input_labels = torch.tensor(input_labels).to(self.device).unsqueeze(1)

        ratios = reshaped_input_sizes / original_sizes
        ratios = ratios.unsqueeze(1).unsqueeze(1).expand_as(input_points)

        return input_points * ratios, input_labels


    def _get_clicks(
        self,
        pred_masks: torch.Tensor,
        original_sizes: torch.Tensor,
        reshaped_input_sizes: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        fn_masks: Optional[torch.Tensor] = None,
        fp_masks: Optional[torch.Tensor] = None,
    ):

        if self.training:
            input_points, input_labels = zip(*map(lambda i: self.get_click_training(
                pred_mask=pred_masks[i],
                label=labels[i] if labels is not None else None,
            ), range(pred_masks.size(0))))

        else:
            input_points, input_labels = zip(*map(lambda i: self.get_click_eval(
                fn_mask=fn_masks[i],
                fp_mask=fp_masks[i],
            ), range(pred_masks.size(0))))

        return self._postprocess_clicks(
            input_points=input_points,
            input_labels=input_labels,
            original_sizes=original_sizes,
            reshaped_input_sizes=reshaped_input_sizes,
        )

    def _forward_train(
        self,
        image_embeddings: torch.Tensor, 
        input_boxes: torch.Tensor,
        original_sizes: torch.Tensor,
        reshaped_input_sizes: torch.Tensor, 
        labels: torch.Tensor,
        label_mask: torch.Tensor
    ):

        all_labels = labels
        new_labels = []
        for i in range(len(labels)):
            while True:
                rand_idx = random.randint(0, labels.shape[1] - 1) 
                if label_mask[i, rand_idx] > 0:
                    break
            new_labels.append(labels[i, rand_idx])
        labels = torch.stack(new_labels, dim=0).to(self.device)
        

        old_strategy = self.click_strategy
        self.set_click_strategy("training")
        
        input_points, input_labels, pred_masks = None, None, None
        losses, fn_loss, fp_loss = [], 0.0, 0.0

        batch_size = image_embeddings.size(0)
        self.num_mc_samples = 1 if self.training else 3

        image_embeddings = image_embeddings.repeat_interleave(self.num_mc_samples, dim=0)
        input_boxes = input_boxes.repeat_interleave(self.num_mc_samples, dim=0) if input_boxes is not None else None
        labels = labels.repeat_interleave(self.num_mc_samples, dim=0)
        original_sizes = original_sizes.repeat_interleave(self.num_mc_samples, dim=0)
        reshaped_input_sizes = reshaped_input_sizes.repeat_interleave(self.num_mc_samples, dim=0)

        for _ in range(self.num_iterations):

            world_outputs = self.sam(
                image_embeddings=image_embeddings, 
                input_boxes=input_boxes,
                input_points=input_points,
                input_labels=input_labels,
                input_masks=(pred_masks > 0.0).float() if pred_masks is not None else None,
                multimask_output=False
            )
            pred_masks = world_outputs.pred_masks.squeeze(2)

            policy_outputs = self.sam(
                image_embeddings=image_embeddings, 
                input_boxes=input_boxes,
                input_points=input_points,
                input_labels=input_labels,
                input_masks=pred_masks,
                multimask_output=True
            )

            binary_input_mask = (world_outputs.pred_masks > 0.0).squeeze(1).squeeze(1).detach()
            fn_masks = torch.logical_and(binary_input_mask == 0, labels == 1).float()
            fp_masks = torch.logical_and(binary_input_mask == 1, labels == 0).float()

            fn_pred = policy_outputs.pred_masks[:, 0, 0]
            fp_pred = policy_outputs.pred_masks[:, 0, 1]

            # Normalise the pred_mask to be consistent with false positive and false negative masks
            #if not self.training:
            #    pred_masks = pred_masks.masked_fill(fn_pred.unsqueeze(1) > 0.0, -1.0)
            #    pred_masks = pred_masks.masked_fill(fp_pred.unsqueeze(1) > 0.0, 1.0)

            fn_loss += nn.BCEWithLogitsLoss()(fn_pred, fn_masks)
            fp_loss += nn.BCEWithLogitsLoss()(fp_pred, fp_masks)

            input_points, input_labels = self._get_clicks(
                pred_masks=pred_masks,
                original_sizes=original_sizes,
                reshaped_input_sizes=reshaped_input_sizes,
                labels=labels,
                fn_masks=fn_pred if not self.training else None,
                fp_masks=fp_pred if not self.training else None,
            )

            losses.append(self.compute_loss(world_outputs.pred_masks, labels))

        total_loss = torch.stack(losses).sum() + fn_loss + fp_loss

        # Track information gain
        gains = []
        for i in range(1, len(losses)):
            gains.append((losses[i-1] - losses[i]) / losses[i-1])
        gains = torch.stack(gains).mean().detach().cpu().numpy().item()
        self.gains = self.gains * 0.9 + gains * 0.1
        self.i += 1

        pred_masks = world_outputs.pred_masks.reshape(batch_size, 1, self.num_mc_samples, *pred_masks.shape[-2:])
        iou_scores = world_outputs.iou_scores.reshape(batch_size, self.num_mc_samples)

        if not self.training:
            pred_masks, idxs, _ = self._clustering(pred_masks, return_idxs=True)
            #iou_scores = torch.gather(torch.tensor(pred_iou).to(idxs.device), 1, idxs).unsqueeze(1)

        
        # Create labels for the blank classifier
        # Should be 0 if no mask is present, 1 if mask is present
        
        if self.training:
            new_labels = []
            num_labels = all_labels.shape[1]
            for i in range(all_labels.shape[0]):
                num_blanks = 0
                for j in range(num_labels):
                    if all_labels[i, j].sum() == 0:
                        num_blanks += 1
                new_labels.append(1 - (num_blanks / num_labels))
            new_labels = torch.tensor(new_labels).to(labels.device).float()
            iou_loss = nn.BCEWithLogitsLoss()(world_outputs.iou_scores.reshape(all_labels.shape[0]), new_labels)

            #total_loss += iou_loss

        """
        if not self.training:
            iou_scores = world_outputs.iou_scores.reshape([batch_size, self.num_mc_samples]).mean(-1)
            for i in range(batch_size):
                num_blanks = torch.round((1 - torch.sigmoid(iou_scores[i])) * self.num_preds).int()
                
                # [:, :, -0:] will give entire array not empty array
                if num_blanks == 0:
                    continue
                pred_masks[:, :, -num_blanks:] = -1 * torch.inf"""
        
        self.set_click_strategy(old_strategy)

        return SamMultimaskOutput(
            loss=total_loss,
            iou_scores=torch.zeros(pred_masks.shape[0], dtype=torch.float32, device=pred_masks.device).unsqueeze(1).unsqueeze(1),
            pred_masks=pred_masks,
            input_points=input_points,
            input_labels=input_labels,
        )

    def _clustering(
        self,
        pred_masks,
        return_idxs=False,
    ):

        p_pred = (pred_masks > 0.0).squeeze(1).cpu().numpy()
        shape = (p_pred.shape[0], p_pred.shape[1], p_pred.shape[1])
        similarity_matrix = np.zeros(shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    similarity_matrix[i, j, k] = self.scorer.score(p_pred[i, j], p_pred[i, k])
        
        """
        # Deduplicate the predictions
        threshold = 0.99
        batch_new_preds = []
        batch_indices = []
        for i in range(shape[0]):
            new_preds = []
            indices = []
            for j in range(shape[1]):
                if j == 0:
                    new_preds.append(pred_masks[i, :, j])
                    indices.append(j)
                    continue
                if np.max(similarity_matrix[i, j, :j]) < threshold:
                    new_preds.append(pred_masks[i, :, j])
                    indices.append(j)
            batch_new_preds.append(torch.stack(new_preds, dim=1))
            batch_indices.append(indices)
        pred_masks = batch_new_preds

        new_similarity_matrix = []
        for i, indices in enumerate(batch_indices):
            new_similarity_matrix.append(similarity_matrix[i, indices][:, indices])
        similarity_matrix = new_similarity_matrix
        """
        
        chosen_preds = []
        all_chosen_idxs = []
        for i in range(len(pred_masks)):
            chosen_idxs = self.search_strategy.run_search(scores=similarity_matrix[i], k=self.num_preds)
            chosen_preds.append(pred_masks[i][:, chosen_idxs])
            all_chosen_idxs.append(chosen_idxs)
        chosen_preds = torch.stack(chosen_preds, dim=0)
        
        try:
            all_chosen_idxs = torch.stack(all_chosen_idxs).to(self.device)
        except:
            all_chosen_idxs = torch.tensor(all_chosen_idxs).to(self.device)

        if return_idxs:
            return chosen_preds, all_chosen_idxs, np.array([np.mean(S, axis=-1) for S in similarity_matrix])

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

        pred_masks, idxs, pred_iou = self._clustering(new_pred_masks, pred_masks, return_idxs=True)
        iou_scores = torch.gather(torch.tensor(pred_iou).to(idxs.device), 1, idxs).unsqueeze(1)

        return SamMultimaskOutput(
            loss=loss,
            iou_scores=iou_scores,
            pred_masks=pred_masks,
        )

    @staticmethod
    def loss_fn(A, B):
        return nn.BCEWithLogitsLoss()(A, B) + monai.losses.DiceLoss(sigmoid=True)(A, B)

    def compute_loss(self, pred_masks, labels):
        labels = labels.float()
        pred_masks = pred_masks[:, 0, 0, :, :]
        if self.multiple_annotations:
            loss = 0.0
            for i in range(labels.shape[1]):
                loss += self.loss_fn(pred_masks, labels[:, i])
            return loss / labels.shape[1]
        
        return self.loss_fn(pred_masks, labels)

    def forward(
        self, 
        pixel_values=None,
        image_embeddings=None,
        input_boxes=None,
        labels=None,
        label_mask=None,
        original_sizes=None,
        reshaped_input_sizes=None,
    ):
        
        
        if image_embeddings is None:
            image_embeddings = self.sam.get_image_embeddings(pixel_values)

        kwargs = {
            "image_embeddings": image_embeddings,
            "input_boxes": input_boxes,
            "original_sizes": original_sizes,
            "reshaped_input_sizes": reshaped_input_sizes,
            "labels": labels,
            "label_mask": label_mask
        }
        
        if self.training:
            return self._forward_train(**kwargs)

        return self._forward_train(**kwargs)


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

    

from functools import partial
import math



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
        output_tokens = torch.cat([self.iou_token.weight, sampled_tokens], dim=0)
        
        output_tokens = output_tokens.repeat(batch_size, point_batch_size, 1, 1)

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
            current_mlp = self.output_hypernetworks_mlps[i]
            hyper_in_list += [current_mlp(mask_tokens_out[:, :, i, :])]
        hyper_in = torch.stack(hyper_in_list, dim=2)

        """
        _mean = mean(hyper_in_list[0].squeeze(1))
        _cov = cov(hyper_in_list[0].squeeze(1)).reshape(batch_size, 32, 32)
        Sigma_k = torch.bmm(_cov, _cov.transpose(2, 1))
        eye = torch.eye(32).to(Sigma_k.device).unsqueeze(0).expand(Sigma_k.shape[0], -1, -1)
        Sigma_k = Sigma_k + eye

        samples = []
        for i in range(batch_size):
            dist = MultivariateNormal(_mean[i], Sigma_k[i])
            samples.append(dist.rsample([5]))
        hyper_in = torch.stack(samples, dim=0).unsqueeze(1)"""

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


class StochasticSam(SLIP):

    def __init__(
        self,
        config, 
        do_clustering: bool = False,
        **kwargs
    ):

        self.do_clustering = do_clustering
        super().__init__(config, **kwargs)
        self.sam = SamModel(config)

        # Create mean and covariance weights
        self.dist = nn.Linear(256, 256)
        self.mean = nn.Linear(32, 32)
        self.cov = nn.Linear(32, 32*32)

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

        # Sample from multivariate normal distribution
        Sigma_k = torch.mm(self.dist.weight, self.dist.weight.T)
        Sigma_k.add_(torch.eye(256).to(Sigma_k.device))
        
        dist = MultivariateNormal(self.dist.bias, Sigma_k)
        sampled_tokens = dist.rsample([self.num_simulations + 1])
        self.sam.mask_decoder.forward = partial(forward, self=self.sam.mask_decoder, sampled_tokens=sampled_tokens, mean=self.mean, cov=self.cov)
        self.sam.mask_decoder.num_mask_tokens = self.num_simulations + 1

        outputs = self.sam(
            image_embeddings=image_embeddings,
            input_boxes=input_boxes,
            multimask_output=True,
        )
        labels = labels.float()

        batch_size = outputs.pred_masks.shape[0]
        num_mc_samples = outputs.pred_masks.shape[2]
        logit_sample = outputs.pred_masks.reshape((batch_size * num_mc_samples, -1))
        target = labels[:, :1].expand(-1, num_mc_samples, -1, -1).reshape((batch_size * num_mc_samples, -1))
        log_prob = -F.cross_entropy(logit_sample, target, reduction='none').reshape((batch_size, num_mc_samples, -1))
        loglikelihood = torch.mean(torch.logsumexp(torch.sum(log_prob, dim=-1), dim=1) - math.log(num_mc_samples))
        loss = -loglikelihood / (labels.shape[-1] * labels.shape[-2])

        if self.do_clustering:
            # Find intersection between all masks along dim 2
            outputs.pred_masks = self._clustering(outputs.pred_masks)

        return SamMultimaskOutput(
            loss=loss,
            pred_masks=outputs.pred_masks,
            iou_scores=torch.zeros(outputs.pred_masks.shape[0], dtype=torch.float32, device=outputs.pred_masks.device).unsqueeze(1).unsqueeze(1)
        )



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


from zca import ZCA

class ZcaSam(Model):
    def __init__(self, config):
        super().__init__(config)
        self.sam = SamModel(config)

        self.trf = None # Can only be initialised after the model is loaded
        self.num_preds = 3

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

        if self.trf is None:    
            X = self.sam.mask_decoder.mask_tokens.weight.detach().cpu().numpy()
            self.trf = ZCA().fit(X)
        
        if image_embeddings is None:
            image_embeddings = self.sam.get_image_embeddings(pixel_values)

        # Sample from N(0, 1) and apply ZCA whitening to the samples
        loss = 0.0
        pred_masks = []
        for _ in range(self.num_preds):
            
            sampled_tokens = torch.tensor(self.trf.inverse_transform(np.random.normal(0, 1, (4, 256)))).to(self.device)

            self.sam.mask_decoder.forward = partial(forward, self=self.sam.mask_decoder, sampled_tokens=sampled_tokens)

            outputs = self.sam(
                image_embeddings=image_embeddings,
                input_boxes=input_boxes,
                multimask_output=False,
            )

            pred_masks.append(outputs.pred_masks)

            loss += nn.BCEWithLogitsLoss()(outputs.pred_masks.reshape(-1), labels[:, 0].reshape(-1).float())

        pred_masks = torch.cat(pred_masks, dim=2)

        return SamMultimaskOutput(
            loss=loss,
            iou_scores=torch.zeros(outputs.pred_masks.shape[0], dtype=torch.float32, device=outputs.pred_masks.device).unsqueeze(1).unsqueeze(1),
            pred_masks=pred_masks,
        )
        


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


class ProbabilisticSam(SLIP):
    """
    Probabilistic UNet model which replaces Unet with a SAM backbone.
    """

    def __init__(
        self,
        config, 
        do_clustering: bool = False,
        **kwargs
    ):

        self.do_clustering = do_clustering
        super().__init__(config, **kwargs)
        self.sam = SamModel(config)

        # Create mean and covariance weights
        self.dist = nn.Linear(256, 256)
        self.mean = nn.Linear(32, 32)
        self.cov = nn.Linear(32, 32*32)

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

        # Sample from multivariate normal distribution
        Sigma_k = torch.mm(self.dist.weight, self.dist.weight.T)
        Sigma_k.add_(torch.eye(256).to(Sigma_k.device))
        
        dist = MultivariateNormal(self.dist.bias, Sigma_k)
        sampled_tokens = dist.rsample([self.num_simulations + 1])
        self.sam.mask_decoder.forward = partial(forward, self=self.sam.mask_decoder, sampled_tokens=sampled_tokens, mean=self.mean, cov=self.cov)
        self.sam.mask_decoder.num_mask_tokens = self.num_simulations + 1

        outputs = self.sam(
            image_embeddings=image_embeddings,
            input_boxes=input_boxes,
            multimask_output=True,
        )
        labels = labels.float()

        batch_size = outputs.pred_masks.shape[0]
        num_mc_samples = outputs.pred_masks.shape[2]
        logit_sample = outputs.pred_masks.reshape((batch_size * num_mc_samples, -1))
        target = labels[:, :1].expand(-1, num_mc_samples, -1, -1).reshape((batch_size * num_mc_samples, -1))
        log_prob = -F.cross_entropy(logit_sample, target, reduction='none').reshape((batch_size, num_mc_samples, -1))
        loglikelihood = torch.mean(torch.logsumexp(torch.sum(log_prob, dim=-1), dim=1) - math.log(num_mc_samples))
        loss = -loglikelihood / (labels.shape[-1] * labels.shape[-2])

        if self.do_clustering:
            # Find intersection between all masks along dim 2
            outputs.pred_masks = self._clustering(outputs.pred_masks)

        return SamMultimaskOutput(
            loss=loss,
            pred_masks=outputs.pred_masks,
            iou_scores=torch.zeros(outputs.pred_masks.shape[0], dtype=torch.float32, device=outputs.pred_masks.device).unsqueeze(1).unsqueeze(1)
        )
