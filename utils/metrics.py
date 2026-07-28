"""
评估指标计算.

包含:
- seg_acc: 16 类动作分类准确率
- fall_f1 / fallen_f1: 二分类 F1
- 完整分类报告
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)


def compute_seg_acc(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    计算 16 类动作分割准确率.

    Args:
        predictions: (N,) int — 预测的类别标签 (0-15).
        targets: (N,) int — 真实标签 (0-15).
    Returns:
        seg_acc: float — 准确率.
    """
    return float(accuracy_score(targets, predictions))


def compute_binary_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    pos_label: int = 1,
) -> dict:
    """
    计算二分类指标.

    Args:
        predictions: (N,) int — 预测标签 (0/1).
        targets: (N,) int — 真实标签 (0/1).
        pos_label: 正类标签, 默认 1.
    Returns:
        dict with keys: accuracy, precision, recall, f1.
    """
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "precision": float(precision_score(targets, predictions, pos_label=pos_label, zero_division=0)),
        "recall": float(recall_score(targets, predictions, pos_label=pos_label, zero_division=0)),
        "f1": float(f1_score(targets, predictions, pos_label=pos_label, zero_division=0)),
    }


def compute_classification_report(
    predictions: np.ndarray,
    targets: np.ndarray,
    class_names: list[str],
) -> str:
    """
    生成完整的分类报告.

    Args:
        predictions: (N,) int — 预测标签.
        targets: (N,) int — 真实标签.
        class_names: 类名列表.
    Returns:
        格式化的分类报告字符串.
    """
    return classification_report(
        targets, predictions, target_names=class_names, zero_division=0
    )


def compute_ternary_metrics(
    pred_ternary: np.ndarray,
    gt_ternary: np.ndarray,
) -> dict:
    """
    Compute per-class metrics for ternary classification (Phase 6).

    Classes: 0=fall, 1=fallen, 2=normal.

    Args:
        pred_ternary: (N,) int — predicted class (0, 1, 2).
        gt_ternary: (N,) int — ground truth class (0, 1, 2).
    Returns:
        dict with per-class F1/precision/recall, macro_f1, confusion_matrix,
        ternary_acc, and classification report.
    """
    TERNARY_CLASS_NAMES = ["fall", "fallen", "normal"]

    # Per-class binary metrics (OvR)
    def _binary_metrics(pred_mask, gt_mask):
        return {
            "precision": float(precision_score(gt_mask, pred_mask, zero_division=0)),
            "recall": float(recall_score(gt_mask, pred_mask, zero_division=0)),
            "f1": float(f1_score(gt_mask, pred_mask, zero_division=0)),
        }

    fall_m = _binary_metrics(pred_ternary == 0, gt_ternary == 0)
    fallen_m = _binary_metrics(pred_ternary == 1, gt_ternary == 1)
    normal_m = _binary_metrics(pred_ternary == 2, gt_ternary == 2)

    # 3-class accuracy
    ternary_acc = float(accuracy_score(gt_ternary, pred_ternary))

    # 3x3 confusion matrix
    cm = confusion_matrix(gt_ternary, pred_ternary, labels=[0, 1, 2])

    # Classification report
    report = classification_report(
        gt_ternary, pred_ternary,
        target_names=TERNARY_CLASS_NAMES,
        zero_division=0,
    )

    results = {
        "ternary_acc": ternary_acc,
        "random_baseline": 1.0 / 3,
        "fall_f1": fall_m["f1"],
        "fall_precision": fall_m["precision"],
        "fall_recall": fall_m["recall"],
        "fallen_f1": fallen_m["f1"],
        "fallen_precision": fallen_m["precision"],
        "fallen_recall": fallen_m["recall"],
        "normal_f1": normal_m["f1"],
        "normal_precision": normal_m["precision"],
        "normal_recall": normal_m["recall"],
        "avg_f1": (fall_m["f1"] + fallen_m["f1"] + normal_m["f1"]) / 3.0,
        "confusion_matrix": cm.tolist(),
        "cls_report": report,
    }
    return results


def compute_all_metrics(
    pred_cls: np.ndarray,
    pred_fall: np.ndarray,
    pred_fallen: np.ndarray,
    gt_cls: np.ndarray,
    gt_fall: np.ndarray,
    gt_fallen: np.ndarray,
    class_names: list[str],
) -> dict:
    """
    计算所有评估指标 (老师要求的 5 项).

    Args:
        pred_cls: (N,) int — 预测的 16 类标签.
        pred_fall: (N,) int — 预测的 fall 标签 (0/1).
        pred_fallen: (N,) int — 预测的 fallen 标签 (0/1).
        gt_cls: (N,) int — 真实 16 类标签.
        gt_fall: (N,) int — 真实 fall 标签.
        gt_fallen: (N,) int — 真实 fallen 标签.
        class_names: 16 类名称列表.
    Returns:
        dict: 包含所有指标的字典.
    """
    seg_acc = compute_seg_acc(pred_cls, gt_cls)
    fall_metrics = compute_binary_metrics(pred_fall, gt_fall)
    fallen_metrics = compute_binary_metrics(pred_fallen, gt_fallen)

    results = {
        "seg_acc": seg_acc,
        "random_baseline": 1.0 / len(class_names),  # 1/16 = 6.25%
        "fall_f1": fall_metrics["f1"],
        "fall_precision": fall_metrics["precision"],
        "fall_recall": fall_metrics["recall"],
        "fallen_f1": fallen_metrics["f1"],
        "fallen_precision": fallen_metrics["precision"],
        "fallen_recall": fallen_metrics["recall"],
        "avg_f1": (fall_metrics["f1"] + fallen_metrics["f1"]) / 2.0,
        "cls_report": compute_classification_report(pred_cls, gt_cls, class_names),
    }
    return results
