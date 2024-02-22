import json
import numpy as np

from abc import ABC, abstractmethod
from typing import Callable, Union
from scipy.stats import pearsonr, spearmanr
from surface_distance import compute_surface_dice_at_tolerance, compute_surface_distances

def sigmoid_fn(z):
    return 1/(1 + np.exp(-z))

def compute_metrics(eval_pred, write_path: str = "data/results.json"): 
    metrics = {}
    metrics["mean_dice"] = Metric.create("dice", reduce_fn="mean")
    metrics["mean_nsd"] = Metric.create("nsd", reduce_fn="mean")
    metrics["self_dice"] = Metric.create("self_dice", reduce_fn="max")
    metrics["self_nsd"] = Metric.create("self_nsd", reduce_fn="max")
    
    metrics["iou"] = Metric.create("iou")
    predictions = eval_pred.predictions[1]

    results = {
        name: metric.compute(
            predictions, eval_pred.label_ids
            ) for name, metric in metrics.items()
    }

    num_preds = eval_pred.predictions[0].shape[2]
    all_iou_scores = []
    all_iou_preds = []
    for k in range(num_preds):
        all_iou_scores += metrics["iou"].compute(predictions[:, :, k:k+1], eval_pred.label_ids)
        all_iou_preds += eval_pred.predictions[0].squeeze(1)[:, k].tolist()
    
    iou_pearson = pearsonr(all_iou_preds, all_iou_scores)[0]
    iou_spearman = spearmanr(all_iou_preds, all_iou_scores)[0]

    results["mean_dice"] = [s if not np.isnan(s) else 0.0 for s in results["mean_dice"]]
    results["mean_nsd"] = [s if not np.isnan(s) else 0.0 for s in results["mean_nsd"]]
    results["self_dice"] = [s if not np.isnan(s) else 1.0 for s in results["self_dice"]]
    results["self_nsd"] = [s if not np.isnan(s) else 1.0 for s in results["self_nsd"]]

    results["iou_pearson"] = [iou_pearson]
    results["iou_spearman"] = [iou_spearman]
    results["all_iou_scores"] = all_iou_scores
    results["all_iou_preds"] = all_iou_preds

    if write_path is not None:
        with open(write_path, "w") as f:
            json.dump(results, f)

    results = {name: np.mean(result) for name, result in results.items()}
    
    # Check for nan values
    for name, result in results.items():
        if np.isnan(result):
            results[name] = 0.0
    
    return results


class Metric(ABC):
    """Base class for metrics."""

    subclasses = {}

    def __init__(self, reduce_fn: Union[str, Callable] = np.max):
        if isinstance(reduce_fn, str):
            if reduce_fn == "mean":
                reduce_fn = np.mean
            elif reduce_fn == "max":
                reduce_fn = np.max
            else:
                raise ValueError(f"Bad reduce_fn: {reduce_fn}.")
        self.reduce_fn = reduce_fn

    @classmethod
    def register_subclass(cls, name: str):
        """Register a subclass with a given name."""

        def decorator(subclass):
            cls.subclasses[name] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, name: str, *args, **kwargs):
        if name not in cls.subclasses:
            raise ValueError(f"Bad type: {name}.")
        
        return cls.subclasses[name](*args, **kwargs)

    @abstractmethod
    def iter_compute(self, pred: np.array, label: np.array) -> float:
        """Compute the metric for a single prediction and label."""
        raise NotImplementedError

    def compute(self, preds: np.array, labels: np.array, sigmoid: bool = True) -> float:
        """Compute the metric."""
        results = []
        if sigmoid:
            preds = sigmoid_fn(preds) > 0.5
        
        preds = preds.squeeze(1)  # Remove class dimension
        for i in range(len(preds)):
            result = self.iter_compute(preds[i], labels[i])
            results.append(result)

        return results


class SelfMixin:

    def self_iter_compute(self, pred: np.array, metric_fn: Callable) -> float:
        results = []
        for i in range(len(pred)):
            results.append(
                self.reduce_fn([metric_fn(pred[i], pred[j]) for j in range(len(pred)) if j != i])
            )
        return np.mean(results)


@Metric.register_subclass("iou")
class IoUMetric(Metric):

    def iter_compute(self, pred: np.array, label: np.array) -> float:
        return self.reduce_fn([iou(pred[i], label) for i in range(len(pred))])

def iou(A: np.array, B: np.array) -> float:
    """Calculate the intersection over union of two binary masks."""
    intersection = np.logical_and(A, B)
    union = np.logical_or(A, B)
    if np.sum(union) == 0:
        return 0.0
    return np.sum(intersection) / np.sum(union)


@Metric.register_subclass("nsd")
class NormalisedSurfaceDistance(Metric):

    def iter_compute(self, pred: np.array, label: np.array) -> float:
        return self.reduce_fn([nsd(pred[i], label) for i in range(len(pred))])

def nsd(pred: np.array, label: np.array) -> float:
    """Calculate the normalised surface distance of two binary masks."""
    return compute_surface_dice_at_tolerance(
        compute_surface_distances(pred, label, spacing_mm=(1, 1)), 2.0
    )


@Metric.register_subclass("dice")
class DiceMetric(Metric):

    def iter_compute(self, pred: np.array, label: np.array) -> float:
        return self.reduce_fn([dice(pred[i], label) for i in range(len(pred))])


def dice(A: np.array, B: np.array) -> float:
    #Calculate the dice score of two binary masks.
    intersection = np.logical_and(A, B)
    return 2 * np.sum(intersection) / (np.sum(A) + np.sum(B))


@Metric.register_subclass("self_nsd")
class SelfNSDMetric(NormalisedSurfaceDistance, SelfMixin):

    def iter_compute(self, pred: np.array, label: np.array) -> float:
        try:
            return self.self_iter_compute(pred, nsd)
        except ValueError:
            return 0


@Metric.register_subclass("self_iou")
class SelfIoUMetric(IoUMetric, SelfMixin):

    def iter_compute(self, pred: np.array, label: np.array) -> float:
        try:
            return self.self_iter_compute(pred, iou)
        except ValueError:
            return 0


@Metric.register_subclass("self_dice")
class SelfDiceMetric(DiceMetric, SelfMixin):

    def iter_compute(self, pred: np.array, label: np.array) -> float:
        try:
            return self.self_iter_compute(pred, dice)
        except ValueError:
            return 0
