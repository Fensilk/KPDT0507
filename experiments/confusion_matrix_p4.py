"""
为 Phase 4 最优模型 (p4d_vitg) 生成混淆矩阵。

严格控制变量:
- 从 checkpoint 读取原始参数，完全复现训练时的推理配置
- 使用相同的 validate_epoch 逻辑
- 唯一区别: 额外计算和保存混淆矩阵

用法:
    python experiments/confusion_matrix_p4.py
    python experiments/confusion_matrix_p4.py --exp_dir logs/phase4/p4d_vitg
    python experiments/confusion_matrix_p4.py --exp_dir logs/phase4/p4a_dino_baseline
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.transformer_model import MultimodalFeatureTransformer
from models.dataset import create_longseq_dataloaders
from utils.visualization import (
    CLASS_NAMES,
    plot_confusion_matrix,
)

KMP_DUPLICATE_LIB_OK = "TRUE"  # 解决 Windows OpenMP 冲突


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate confusion matrix for Phase 4 best model"
    )
    parser.add_argument("--exp_dir", type=str,
                        default="logs/phase4/p4d_vitg",
                        help="实验日志目录 (包含 transformer_best.pt)")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


@torch.no_grad()
def validate_and_collect(model, loader, device, class_names):
    """
    与 train_transformer.validate_epoch 完全一致的推理逻辑,
    额外收集所有预测和真实标签用于绘制混淆矩阵。
    """
    model.eval()

    all_pred_cls, all_gt_cls = [], []
    all_pred_fall, all_gt_fall = [], []
    all_pred_fallen, all_gt_fallen = [], []

    for batch in loader:
        features = batch["features"].to(device)
        labels_16 = batch["labels_16"].to(device)
        fall_gt = batch["fall_labels"].to(device)
        fallen_gt = batch["fallen_labels"].to(device)

        logits_cls, logits_fall, logits_fallen = model(features)

        pred_cls = logits_cls.argmax(dim=-1)
        pred_fall = (torch.sigmoid(logits_fall) > 0.5).long().squeeze(-1)
        pred_fallen = (torch.sigmoid(logits_fallen) > 0.5).long().squeeze(-1)

        all_pred_cls.append(pred_cls.cpu().numpy().ravel())
        all_pred_fall.append(pred_fall.cpu().numpy().ravel())
        all_pred_fallen.append(pred_fallen.cpu().numpy().ravel())
        all_gt_cls.append(labels_16.cpu().numpy().ravel())
        all_gt_fall.append(fall_gt.cpu().numpy().ravel())
        all_gt_fallen.append(fallen_gt.cpu().numpy().ravel())

    pred_cls = np.concatenate(all_pred_cls)
    pred_fall = np.concatenate(all_pred_fall)
    pred_fallen = np.concatenate(all_pred_fallen)
    gt_cls = np.concatenate(all_gt_cls)
    gt_fall = np.concatenate(all_gt_fall)
    gt_fallen = np.concatenate(all_gt_fallen)

    return pred_cls, pred_fall, pred_fallen, gt_cls, gt_fall, gt_fallen


def main():
    args = parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(base_dir)

    device = get_device(args.device)
    print(f"[INFO] Using device: {device}")

    # ============================================================
    # 1. 从 checkpoint 读取原始参数 (保证完全复现)
    # ============================================================
    ckpt_path = os.path.join(args.exp_dir, "transformer_best.pt")
    print(f"[INFO] Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    orig_args = checkpoint["args"]
    best_epoch = checkpoint["epoch"]
    best_fall_f1 = checkpoint["best_fall_f1"]

    print(f"[INFO] Best epoch: {best_epoch}, Best val Fall F1: {best_fall_f1:.4f}")
    print(f"[INFO] Original experiment config:")
    for k in sorted(orig_args.keys()):
        print(f"       {k}: {orig_args[k]}")

    # ============================================================
    # 2. 用原始参数重建数据加载器
    # ============================================================
    print("\n[INFO] Loading test data with original parameters...")

    frame_npz_path = os.path.join(base_dir, orig_args["frame_npz"])
    splits_dir = os.path.join(base_dir, orig_args["splits_dir"])
    pose_npz_path = None
    if orig_args.get("pose_npz"):
        pose_npz_path = os.path.join(base_dir, orig_args["pose_npz"])

    _, _, test_loader = create_longseq_dataloaders(
        npz_path=frame_npz_path,
        splits_dir=splits_dir,
        window_size=orig_args["window_size"],
        stride=orig_args["stride"],
        batch_size=orig_args["batch_size"],
        num_workers=orig_args.get("num_workers", 0),
        use_weighted_sampler=False,  # 测试集不需要 weighted sampler
        use_diff=orig_args.get("use_diff", False),
        pose_npz_path=pose_npz_path,
    )

    input_dim = test_loader.dataset.feature_dim
    num_classes = orig_args.get("num_classes", 16)
    print(f"[INFO] Feature dim: {input_dim}")
    print(f"[INFO] Number of classes: {num_classes}")
    print(f"[INFO] Test samples (windows): {len(test_loader.dataset):,}")
    print(f"[INFO] Test batches: {len(test_loader)}")

    # ============================================================
    # 3. 用原始参数重建模型并加载权重
    # ============================================================
    print("\n[INFO] Building model with original architecture...")

    model = MultimodalFeatureTransformer(
        input_dim=input_dim,
        hidden_dim=orig_args["hidden_dim"],
        num_layers=orig_args["num_layers"],
        num_heads=orig_args["num_heads"],
        dropout=orig_args["dropout"],
        max_len=orig_args["window_size"] + 10,
        num_classes=num_classes,
    ).to(device)

    # 严格加载保存的权重
    model.load_state_dict(checkpoint["model_state_dict"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model params: {n_params:,}")
    print(f"[INFO] Model weights loaded successfully")

    # ============================================================
    # 4. 测试集推理 (与原始 validate_epoch 逻辑一致)
    # ============================================================
    print("\n[INFO] Running inference on test set...")

    # 选择合适的类名列表
    if num_classes == 11:
        cls_names = ['walk', 'fall', 'fallen', 'sit_down', 'sitting', 'lie_down',
                     'lying', 'stand_up', 'standing', 'other', 'rare']
    else:
        cls_names = CLASS_NAMES

    pred_cls, pred_fall, pred_fallen, gt_cls, gt_fall, gt_fallen = \
        validate_and_collect(model, test_loader, device, cls_names)

    total_frames = len(pred_cls)
    print(f"[INFO] Total predicted frames: {total_frames:,}")

    # ============================================================
    # 5. 验证结果一致性 (与已知 test_results.json 对比)
    # ============================================================
    from utils.metrics import compute_all_metrics

    metrics = compute_all_metrics(
        pred_cls, pred_fall, pred_fallen,
        gt_cls, gt_fall, gt_fallen, cls_names)

    print(f"\n[INFO] Reproduced metrics (should match original test_results.json):")
    print(f"       seg_acc:  {metrics['seg_acc']:.4f}")
    print(f"       fall_f1:  {metrics['fall_f1']:.4f}")
    print(f"       fallen_f1:{metrics['fallen_f1']:.4f}")
    print(f"       avg_f1:   {metrics['avg_f1']:.4f}")

    # 加载原始结果做对比
    orig_results_path = os.path.join(args.exp_dir, "test_results.json")
    if os.path.exists(orig_results_path):
        with open(orig_results_path) as f:
            orig_results = json.load(f)
        eps = 0.002  # 允许 fp16 带来的微小浮点误差
        for key in ["seg_acc", "fall_f1", "fallen_f1", "avg_f1"]:
            diff = abs(metrics[key] - orig_results[key])
            status = "MATCH" if diff < eps else f"DIFF ({diff:.6f})"
            print(f"       {key}: reproduced={metrics[key]:.4f}  "
                  f"original={orig_results[key]:.4f}  {status}")

    # ============================================================
    # 6. 生成混淆矩阵
    # ============================================================
    print("\n[INFO] Computing confusion matrices...")

    # 6a. 16 类混淆矩阵 (原始计数, 对数色阶 — 解决类别不平衡导致稀有类不可见)
    exp_name = os.path.basename(args.exp_dir)  # e.g. "p4d_vitg" or "p4a_dino_baseline"
    cm_raw = confusion_matrix(gt_cls, pred_cls, labels=range(num_classes))
    cm_path_raw = os.path.join(args.exp_dir, "confusion_matrix_counts.png")
    plot_confusion_matrix(
        cm_raw, cls_names, cm_path_raw,
        title=f"Confusion Matrix (Counts, Log Scale) — {exp_name}\n"
              f"seg_acc={metrics['seg_acc']:.3f}  fall_f1={metrics['fall_f1']:.3f}  "
              f"fallen_f1={metrics['fallen_f1']:.3f}",
        normalize=False,
        log_scale=True,
    )

    # 6b. 16 类混淆矩阵 (按行归一化 → recall)
    cm_norm_path = os.path.join(args.exp_dir, "confusion_matrix_normalized.png")
    plot_confusion_matrix(
        cm_raw, cls_names, cm_norm_path,
        title=f"Confusion Matrix (Row-Normalized) — {exp_name}\n"
              f"seg_acc={metrics['seg_acc']:.3f}  fall_f1={metrics['fall_f1']:.3f}  "
              f"fallen_f1={metrics['fallen_f1']:.3f}",
        normalize=True,
    )

    # 6c. 保存混淆矩阵数据为 JSON (方便后续分析)
    cm_data = {
        "class_names": cls_names,
        "confusion_matrix_counts": cm_raw.tolist(),
        "confusion_matrix_normalized": (
            cm_raw.astype(float) / np.where(cm_raw.sum(axis=1, keepdims=True) == 0,
                                             1, cm_raw.sum(axis=1, keepdims=True))
        ).round(4).tolist(),
        "total_frames": int(total_frames),
        "metrics": {k: v for k, v in metrics.items() if k != "cls_report"},
    }
    cm_json_path = os.path.join(args.exp_dir, "confusion_matrix.json")
    with open(cm_json_path, "w") as f:
        json.dump(cm_data, f, indent=2)
    print(f"[INFO] Confusion matrix data saved: {cm_json_path}")

    # ============================================================
    # 7. 打印混淆矩阵摘要
    # ============================================================
    print(f"\n{'='*60}")
    print("CONFUSION MATRIX — Row-Normalized (Recall per class)")
    print(f"{'='*60}")

    row_sums = cm_raw.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    cm_norm = cm_raw.astype(float) / row_sums

    # 打印每行 top-3 混淆目标
    for i, name in enumerate(cls_names):
        if cm_raw[i].sum() == 0:
            print(f"  {name:>12s}: NO SAMPLES")
            continue
        # 找 top-3 预测 (包括对角线)
        top_idx = np.argsort(cm_norm[i])[::-1][:3]
        parts = []
        for j in top_idx:
            pct = cm_norm[i, j]
            marker = "<-" if i == j else "  "
            parts.append(f"{cls_names[j]}:{pct:.1%}{marker}")
        print(f"  {name:>12s}: {',  '.join(parts)}")

    print(f"\n[INFO] Done. All outputs saved to: {args.exp_dir}")
    print(f"       - {cm_path_raw}")
    print(f"       - {cm_norm_path}")
    print(f"       - {cm_json_path}")


if __name__ == "__main__":
    main()
