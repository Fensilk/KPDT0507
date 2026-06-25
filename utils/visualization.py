"""
可视化工具.

包含:
- 预测时间线对比图 (老师要求第 5 项: "预测时间线能看到 standing -> fall -> fallen")
- 混淆矩阵热力图
- 训练曲线图
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 非交互式后端, 无 GUI 依赖
import matplotlib.pyplot as plt


# 16 类名称 (与预处理一致)
CLASS_NAMES = [
    "walk", "fall", "fallen",
    "sit_down", "sitting", "lie_down", "lying",
    "stand_up", "standing", "other",
    "kneel_down", "kneeling", "squat_down", "squatting",
    "crawl", "jump",
]


def plot_timeline_comparison(
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    gt_fall: np.ndarray,
    pred_fall: np.ndarray,
    gt_fallen: np.ndarray,
    pred_fallen: np.ndarray,
    save_path: str,
    video_idx: int = 0,
):
    """
    绘制单个视频的预测时间线对比图.

    三行:
    - Row 1: 16 类动作标签 (GT vs Pred)
    - Row 2: fall 概率 (GT vs Pred)
    - Row 3: fallen 概率 (GT vs Pred)

    Args:
        gt_labels: (T,) int — 真实 16 类标签.
        pred_labels: (T,) int — 预测 16 类标签.
        gt_fall: (T,) int — 真实 fall 标签.
        pred_fall: (T,) float — 预测 fall 概率.
        gt_fallen: (T,) int — 真实 fallen 标签.
        pred_fallen: (T,) float — 预测 fallen 概率.
        save_path: 保存路径.
        video_idx: 视频编号 (用于标题).
    """
    T = len(gt_labels)
    clip_positions = np.arange(T)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # ---- Row 1: 16-class action labels ----
    ax = axes[0]
    gt_names = [CLASS_NAMES[l] for l in gt_labels]
    pred_names = [CLASS_NAMES[l] for l in pred_labels]
    ax.scatter(clip_positions, [1] * T, c=gt_labels,
               cmap="tab20", vmin=0, vmax=15, s=100, marker="s", label="GT")
    ax.scatter(clip_positions, [0.5] * T, c=pred_labels,
               cmap="tab20", vmin=0, vmax=15, s=100, marker="o", label="Pred")
    for i in range(T):
        ax.annotate(gt_names[i], (clip_positions[i], 1.15), ha="center", fontsize=8, rotation=45)
        ax.annotate(pred_names[i], (clip_positions[i], 0.35), ha="center", fontsize=8, rotation=45)
    ax.set_ylim(0, 1.5)
    ax.set_ylabel("16-Class Action")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_yticks([])
    ax.set_title(f"Video {video_idx}: Prediction Timeline")
    ax.grid(axis="x", alpha=0.3)

    # ---- Row 2: Fall probability ----
    ax = axes[1]
    colors = ["green" if g == 0 else "red" for g in gt_fall]
    ax.bar(clip_positions, pred_fall, color=colors, alpha=0.6, edgecolor="black")
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="Threshold=0.5")
    ax.set_ylabel("Fall Prob")
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # ---- Row 3: Fallen probability ----
    ax = axes[2]
    colors = ["green" if g == 0 else "red" for g in gt_fallen]
    ax.bar(clip_positions, pred_fallen, color=colors, alpha=0.6, edgecolor="black")
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="Threshold=0.5")
    ax.set_xlabel("Clip Index (time ->)")
    ax.set_ylabel("Fallen Prob")
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Timeline saved: {save_path}")


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    save_path: str,
    title: str = "Confusion Matrix",
    normalize: bool = True,
):
    """
    绘制混淆矩阵热力图.

    Args:
        cm: (N, N) int — 混淆矩阵.
        class_names: 类名列表.
        save_path: 保存路径.
        title: 图标题.
        normalize: 是否按行归一化.
    """
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)  # 避免除零
        display = cm.astype(float) / row_sums
        fmt = ".2f"
    else:
        display = cm
        fmt = "d"

    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(10, n * 0.6), max(8, n * 0.5)))
    im = ax.imshow(display, cmap="Blues", aspect="auto")

    # 标注数值
    for i in range(n):
        for j in range(n):
            val = display[i, j]
            color = "white" if val > 0.5 else "black"
            if normalize:
                text = f"{val:.2f}" if val > 0.01 else ""
            else:
                text = str(int(cm[i, j])) if cm[i, j] > 0 else ""
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=7)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Confusion matrix saved: {save_path}")


def plot_training_curves(
    history: dict,
    save_path: str,
):
    """
    绘制训练曲线.
    Args:
        history: 包含历史记录的字典, keys: train_loss, val_loss, seg_acc,
                 fall_f1, fallen_f1, lr. 每个 value 是 list.
        save_path: 保存路径.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Loss
    ax = axes[0, 0]
    ax.plot(epochs, history["train_loss"], label="Train", color="blue")
    ax.plot(epochs, history["val_loss"], label="Val", color="orange")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Total Loss"); ax.legend(); ax.grid(alpha=0.3)

    # seg_acc
    ax = axes[0, 1]
    ax.plot(epochs, history["seg_acc"], color="blue")
    ax.axhline(y=1.0 / 16, color="red", linestyle="--", label="Random (6.25%)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.set_title("Segmentation Accuracy (16-class)")
    ax.legend(); ax.grid(alpha=0.3)

    # fall_f1
    ax = axes[0, 2]
    ax.plot(epochs, history["fall_f1"], color="green")
    ax.set_xlabel("Epoch"); ax.set_ylabel("F1")
    ax.set_title("Fall F1 Score"); ax.grid(alpha=0.3)

    # fallen_f1
    ax = axes[1, 0]
    ax.plot(epochs, history["fallen_f1"], color="purple")
    ax.set_xlabel("Epoch"); ax.set_ylabel("F1")
    ax.set_title("Fallen F1 Score"); ax.grid(alpha=0.3)

    # avg_f1
    ax = axes[1, 1]
    avg = [(f + fn) / 2 for f, fn in zip(history["fall_f1"], history["fallen_f1"])]
    ax.plot(epochs, avg, color="teal")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Avg F1")
    ax.set_title("Average (Fall + Fallen) F1"); ax.grid(alpha=0.3)

    # lr
    ax = axes[1, 2]
    ax.plot(epochs, history["lr"], color="red")
    ax.set_xlabel("Epoch"); ax.set_ylabel("LR")
    ax.set_title("Learning Rate"); ax.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Training curves saved: {save_path}")
