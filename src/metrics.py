import json
import numpy as np
from medpy.metric import jc, dc

def calc_dsc(image_0, image_1):
    if np.sum(image_0) == 0 and np.sum(image_1) == 0:
        return np.nan
    else:
        return dc(image_1, image_0)

def compute_dice(predictions, labels):
    results = []
    for pred, label in zip(predictions, labels):
        pred = np.round(np.mean((pred > 0.0).astype(np.float32), axis=0)).astype(np.int64)
        label = label.astype(np.int64)
        #results.append(
            #np.mean([np.mean([dice(target == i, pred == i) for i in range(2)]) for target in label]))
        results.append(np.mean([calc_dsc(target, pred) for target in label]))

    results = [result if not np.isnan(result) else 0.0 for result in results ]

    return results

def compute_dice_max(predictions, labels):
    results = []
    for pred, label in zip(predictions, labels):
        pred = np.round(np.mean((pred > 0.0).astype(np.float32), axis=0)).astype(np.int64)
        label = label.astype(np.int64)
        #results.append(
            #np.mean([np.mean([dice(target == i, pred == i) for i in range(2)]) for target in label]))
        scores = [calc_dsc(target, pred) for target in label]

        # 1.0 when undefined see below:
        # https://openaccess.thecvf.com/content/CVPR2023/papers/Rahman_Ambiguous_Medical_Image_Segmentation_Using_Diffusion_Models_CVPR_2023_paper.pdf
        scores = [s if not np.isnan(s) else 1.0 for s in scores] 
        results.append(np.max(scores))

    return results

def compute_dice_nod(predictions, labels):
    results = []
    for pred, label in zip(predictions, labels):
        pred = np.round(np.mean((pred > 0.0).astype(np.float32), axis=0)).astype(np.int64)
        label = label.astype(np.int64)

        if np.sum(label) == 0:
            continue

        #results.append(
         #   np.mean([np.mean([calc_dsc(target == i, pred == i) for i in range(2)]) for target in label if np.sum(target) > 0]))
        results.append(np.mean([calc_dsc(target, pred) for target in label if np.sum(target) > 0]))

    return results

def compute_ged(predictions, labels):
    _geds = []
    diversities = []
    for pred, label in zip(predictions, labels):
        pred = (pred > 0.0).astype(np.int64)
        _ged, diversity = generalised_energy_distance(pred, label, 1, range(1, 2))
        _geds.append(_ged)
        diversities.append(diversity)

    return _geds, diversities

def compute_metrics(eval_pred, write_path: str = "data/results.json"): 
    predictions = eval_pred.predictions[1]
    labels = eval_pred.label_ids.astype(bool)
    non_empty_predictions = []
    non_empty_labels = []
    for prediction, label in zip(predictions, labels):
        if np.sum(label) > 0:
            non_empty_predictions.append(prediction)
            non_empty_labels.append(label)
    non_empty_predictions = np.array(non_empty_predictions)
    non_empty_labels = np.array(non_empty_labels)

    predictions = predictions.squeeze(1)
    ged, diversity = compute_ged(predictions, labels)

    results = {
        "dice": compute_dice(predictions, labels),
        "dice_max": compute_dice_max(predictions, labels),
        "dice_nod": compute_dice_nod(predictions, labels),
        "ged": ged,
        "sample_diversity": diversity,
    }

    if write_path is not None:
        with open(write_path, "w") as f:
            json.dump(results, f)

    results = {name: np.mean(result) for name, result in results.items()}

    return results

def iou(A: np.array, B: np.array) -> float:
    """Calculate the intersection over union of two binary masks."""
    intersection = np.logical_and(A, B)
    union = np.logical_or(A, B)
    if np.sum(union) == 0:
        return 0.0
    return np.sum(intersection) / np.sum(union)


def dice(A: np.array, B: np.array) -> float:
    #Calculate the dice score of two binary masks.
    intersection = np.logical_and(A, B)
    return 2 * np.sum(intersection) / (np.sum(A) + np.sum(B))


def dist_fct(m1, m2, nlabels, label_range):
    per_label_iou = []
    for lbl in label_range:

        # assert not lbl == 0  # tmp check
        m1_bin = (m1 == lbl) * 1
        m2_bin = (m2 == lbl) * 1

        if np.sum(m1_bin) == 0 and np.sum(m2_bin) == 0:
            per_label_iou.append(1)
        elif np.sum(m1_bin) > 0 and np.sum(m2_bin) == 0 or np.sum(m1_bin) == 0 and np.sum(m2_bin) > 0:
            per_label_iou.append(0)
        else:
            per_label_iou.append(jc(m1_bin, m2_bin))

    return 1 - (sum(per_label_iou) / nlabels)


def generalised_energy_distance(sample_arr, gt_arr, nlabels, label_range):
    """
    :param sample_arr: expected shape N x X x Y 
    :param gt_arr: M x X x Y
    :return: 
    """

    N = sample_arr.shape[0]
    M = gt_arr.shape[0]

    d_sy = []
    d_ss = []
    d_yy = []

    for i in range(N):
        for j in range(M):
            d_sy.append(dist_fct(sample_arr[i, ...], gt_arr[j, ...], nlabels, label_range))
        for j in range(N):
            d_ss.append(dist_fct(sample_arr[i, ...], sample_arr[j, ...], nlabels, label_range))

    for i in range(M):
        for j in range(M):
            # print(dist_fct(gt_arr[i,...], gt_arr[j,...]))
            d_yy.append(dist_fct(gt_arr[i, ...], gt_arr[j, ...], nlabels, label_range))
    diversity = (1. / N ** 2) * sum(d_ss)
    return (2. / (N * M)) * sum(d_sy) - diversity - (1. / M ** 2) * sum(d_yy), diversity
