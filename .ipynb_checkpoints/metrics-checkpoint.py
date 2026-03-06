# ==================== metrics.py (完整替换) ====================
from sklearn import metrics
from skimage import measure

import cv2
import numpy as np
import pandas as pd


def _prepare_valid_region_mask(region_mask, shape_hw):
    if region_mask is None:
        return None

    mask = np.asarray(region_mask)
    if mask.ndim == 2:
        mask = np.expand_dims(mask, axis=0)
    if mask.ndim != 3:
        raise ValueError("region_mask must have shape [H, W] or [N, H, W].")

    if mask.shape[-2:] != shape_hw:
        raise ValueError(
            f"region_mask spatial shape {mask.shape[-2:]} "
            f"must match segmentation shape {shape_hw}."
        )

    return mask > 0.5


def _squeeze_gt(arr):
    """Squeeze singleton channel dim: (N,1,H,W) → (N,H,W)."""
    if arr.ndim == 4 and arr.shape[1] == 1:
        return arr[:, 0]
    return arr


def compute_best_pr_re(anomaly_ground_truth_labels, anomaly_prediction_weights):
    precision, recall, thresholds = metrics.precision_recall_curve(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_threshold = thresholds[np.argmax(f1_scores)]
    best_precision = precision[np.argmax(f1_scores)]
    best_recall = recall[np.argmax(f1_scores)]
    print(best_threshold, best_precision, best_recall)
    return best_threshold, best_precision, best_recall


def compute_imagewise_retrieval_metrics(
    anomaly_prediction_weights, anomaly_ground_truth_labels, path='training'
):
    auroc = metrics.roc_auc_score(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    ap = (
        0.0
        if path == 'training'
        else metrics.average_precision_score(
            anomaly_ground_truth_labels, anomaly_prediction_weights
        )
    )
    return {"auroc": auroc, "ap": ap}


def compute_pixelwise_retrieval_metrics(
    anomaly_segmentations, ground_truth_masks, path='train', region_mask=None
):
    if isinstance(anomaly_segmentations, list):
        anomaly_segmentations = np.stack(anomaly_segmentations)
    if isinstance(ground_truth_masks, list):
        ground_truth_masks = np.stack(ground_truth_masks)

    # ★ FIX: squeeze (N,1,H,W) → (N,H,W)
    anomaly_segmentations = _squeeze_gt(anomaly_segmentations)
    ground_truth_masks = _squeeze_gt(ground_truth_masks)

    valid_mask = _prepare_valid_region_mask(
        region_mask, anomaly_segmentations.shape[-2:]
    )

    if valid_mask is not None:
        if valid_mask.shape[0] == 1 and anomaly_segmentations.shape[0] > 1:
            valid_mask = np.repeat(valid_mask, anomaly_segmentations.shape[0], axis=0)
        elif valid_mask.shape[0] != anomaly_segmentations.shape[0]:
            raise ValueError(
                "region_mask batch size must be 1 or match data batch size."
            )
        flat_anomaly_segmentations = anomaly_segmentations[valid_mask]
        flat_ground_truth_masks = ground_truth_masks[valid_mask]
    else:
        flat_anomaly_segmentations = anomaly_segmentations.ravel()
        flat_ground_truth_masks = ground_truth_masks.ravel()

    unique_labels = np.unique(flat_ground_truth_masks.astype(int))
    if unique_labels.shape[0] < 2:
        auroc = 0.5
        ap = 0.0 if path == 'training' else 0.0
        return {"auroc": auroc, "ap": ap}

    auroc = metrics.roc_auc_score(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    ap = (
        0.0
        if path == 'training'
        else metrics.average_precision_score(
            flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
        )
    )
    return {"auroc": auroc, "ap": ap}


def compute_pro(masks, amaps, num_th=200, region_mask=None):
    if isinstance(masks, list):
        masks = np.stack(masks)
    if isinstance(amaps, list):
        amaps = np.stack(amaps)
    masks = _squeeze_gt(np.asarray(masks, dtype=np.float64))
    amaps = _squeeze_gt(np.asarray(amaps, dtype=np.float64))

    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=bool)

    valid_mask = _prepare_valid_region_mask(region_mask, amaps.shape[-2:])
    if valid_mask is not None:
        if valid_mask.shape[0] == 1 and amaps.shape[0] > 1:
            valid_mask = np.repeat(valid_mask, amaps.shape[0], axis=0)
        elif valid_mask.shape[0] != amaps.shape[0]:
            raise ValueError(
                "region_mask batch size must be 1 or match data batch size."
            )
        # ★ 提前剪枝：区域内无异常像素则直接返回
        if (masks * valid_mask).sum() == 0:
            return 0.0

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for idx, (binary_amap, mask) in enumerate(zip(binary_amaps, masks)):
            binary_amap = cv2.dilate(binary_amap.astype(np.uint8), k)
            if valid_mask is not None:
                valid = valid_mask[idx]
                binary_amap = np.logical_and(binary_amap > 0, valid)
                mask = np.logical_and(mask > 0, valid).astype(mask.dtype)
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        if valid_mask is not None:
            inverse_masks = np.logical_and(inverse_masks > 0, valid_mask).astype(
                np.uint8
            )
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        denom = inverse_masks.sum()
        if denom <= 0:
            continue
        fpr = fp_pixels / denom

        mean_pro = np.mean(pros) if len(pros) > 0 else 0.0
        df = pd.concat(
            [df, pd.DataFrame({"pro": mean_pro, "fpr": fpr, "threshold": th}, index=[0])]
        )

    if df.empty:
        return 0.0
    df = df[df["fpr"] < 0.3]
    if df.empty:
        return 0.0
    df["fpr"] = (df["fpr"] - df["fpr"].min()) / (
        df["fpr"].max() - df["fpr"].min() + 1e-10
    )

    pro_auc = metrics.auc(df["fpr"], df["pro"])
    return pro_auc

def compute_f1_max(labels, scores):
    """Compute maximum F1 score across all thresholds."""
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores)
    if len(np.unique(labels)) < 2:
        return 0.0
    precision, recall, _ = metrics.precision_recall_curve(labels, scores)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    return float(np.max(f1))


def compute_fpr_at_tpr(labels, scores, target_tpr=0.98):
    """FPR when TPR >= target_tpr."""
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores)
    if len(np.unique(labels)) < 2:
        return 1.0
    fpr, tpr, _ = metrics.roc_curve(labels, scores)
    idx = np.where(tpr >= target_tpr)[0]
    if len(idx) == 0:
        return 1.0
    return float(fpr[idx[0]])