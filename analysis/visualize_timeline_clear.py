"""
改进版预测时间线可视化 — 让 standing→fall→fallen 模式一目了然.

用法: python visualize_timeline_clear.py
"""

import os, sys, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

from models.tcn_model import TCNModel
from models.dataset import create_dataloaders
from utils.visualization import CLASS_NAMES

# 颜色方案: 强调 standing(蓝), fall(红), fallen(橙)
HIGHLIGHT_CLASSES = {
    8: ("standing", "#4A90D9"),   # 蓝
    1: ("fall",     "#E74C3C"),   # 红
    2: ("fallen",   "#F39C12"),   # 橙
}


def plot_timeline_clear(
    gt_cls, pred_cls, gt_fall, pred_fall_prob,
    gt_fallen, pred_fallen_prob,
    save_path, video_idx
):
    """清晰版时间线图: 用色块代替散点, 一目了然."""
    T = len(gt_cls)
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True,
                              gridspec_kw={'height_ratios': [1.2, 1, 1]})

    # ====== Row 1: 16-class with colored blocks ======
    ax = axes[0]
    y_gt, y_pred = 1.0, 0.0
    ax.set_ylim(-0.8, 1.8)

    for t in range(T):
        # GT block
        gt_label = gt_cls[t]
        gt_name = CLASS_NAMES[gt_label]
        if gt_label in HIGHLIGHT_CLASSES:
            color = HIGHLIGHT_CLASSES[gt_label][1]
        else:
            color = "#BDC3C7"  # 灰 — 非关键类
        rect = mpatches.Rectangle((t - 0.4, y_gt - 0.3), 0.8, 0.6,
                                   linewidth=2, edgecolor='black',
                                   facecolor=color, alpha=0.85)
        ax.add_patch(rect)
        ax.text(t, y_gt, gt_name, ha='center', va='center',
                fontsize=10, fontweight='bold', color='white' if gt_label in HIGHLIGHT_CLASSES else 'black')

        # Pred block
        pred_label = pred_cls[t]
        pred_name = CLASS_NAMES[pred_label]
        if pred_label in HIGHLIGHT_CLASSES:
            color = HIGHLIGHT_CLASSES[pred_label][1]
        else:
            color = "#BDC3C7"
        correct = "✓" if gt_cls[t] == pred_cls[t] else "✗"
        rect = mpatches.Rectangle((t - 0.4, y_pred - 0.3), 0.8, 0.6,
                                   linewidth=2, edgecolor='green' if gt_cls[t] == pred_cls[t] else 'red',
                                   facecolor=color, alpha=0.6)
        ax.add_patch(rect)
        ax.text(t, y_pred, f"{pred_name}", ha='center', va='center',
                fontsize=10, fontweight='bold', color='white' if pred_label in HIGHLIGHT_CLASSES else 'black')

    ax.set_xlim(-0.6, T - 0.4)
    ax.text(-0.8, y_gt, "GT", ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(-0.8, y_pred, "Pred", ha='center', va='center', fontsize=12, fontweight='bold')
    ax.set_yticks([])
    ax.set_title(f"Video #{video_idx} — 16-Class Action Prediction", fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="#4A90D9", label="standing"),
        mpatches.Patch(facecolor="#E74C3C", label="fall"),
        mpatches.Patch(facecolor="#F39C12", label="fallen"),
        mpatches.Patch(facecolor="#BDC3C7", label="other"),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9, ncol=4)

    # ====== Row 2: Fall probability ======
    ax = axes[1]
    colors = ["#2ECC71" if g == 0 else "#E74C3C" for g in gt_fall]
    bars = ax.bar(range(T), pred_fall_prob, color=colors, alpha=0.7,
                  edgecolor='black', linewidth=1.5, width=0.7)
    ax.axhline(y=0.5, color='black', linestyle='--', linewidth=2, alpha=0.7, label='Threshold=0.5')

    # Annotate probability values
    for t, prob in enumerate(pred_fall_prob):
        ax.text(t, prob + 0.05, f'{prob:.2f}', ha='center', fontsize=10, fontweight='bold')
        # 标注真实标签
        status = "FALL!" if gt_fall[t] == 1 else ""
        if status:
            ax.text(t, 1.05, status, ha='center', fontsize=11, fontweight='bold', color='#E74C3C')

    ax.set_ylabel("Fall Probability", fontsize=12)
    ax.set_ylim(0, 1.25)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ====== Row 3: Fallen probability ======
    ax = axes[2]
    colors = ["#2ECC71" if g == 0 else "#F39C12" for g in gt_fallen]
    bars = ax.bar(range(T), pred_fallen_prob, color=colors, alpha=0.7,
                  edgecolor='black', linewidth=1.5, width=0.7)
    ax.axhline(y=0.5, color='black', linestyle='--', linewidth=2, alpha=0.7, label='Threshold=0.5')

    for t, prob in enumerate(pred_fallen_prob):
        ax.text(t, prob + 0.05, f'{prob:.2f}', ha='center', fontsize=10, fontweight='bold')
        status = "FALLEN!" if gt_fallen[t] == 1 else ""
        if status:
            ax.text(t, 1.05, status, ha='center', fontsize=11, fontweight='bold', color='#F39C12')

    ax.set_xlabel("Clip Index  →  (time flow)", fontsize=12)
    ax.set_ylabel("Fallen Probability", fontsize=12)
    ax.set_xticks(range(T))
    ax.set_xticklabels([f"clip {i}" for i in range(T)])
    ax.set_ylim(0, 1.25)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 底部标注期望模式
    fig.text(0.5, 0.01,
             "Expected pattern:  standing → fall → fallen   |   "
             "Row2: fall prob drops after fall moment   |   "
             "Row3: fallen prob rises after falling",
             ha='center', fontsize=10, style='italic', color='#7F8C8D',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#F8F9FA', alpha=0.8))

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] {save_path}")


def main():
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 加载数据和模型
    loaders = create_dataloaders(
        os.path.join(BASE, 'data', 'omnifall_preprocessed.npz'),
        os.path.join(BASE, 'data', 'ofsyn_clips.csv'),
        os.path.join(BASE, 'DATASET-omnifall', 'splits', 'syn', 'random'),
        batch_size=64, num_workers=0,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TCNModel().to(device)
    ckpt = torch.load(os.path.join(BASE, 'logs', 'tcn_full', 'tcn_best.pt'),
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    test_ds = loaders[2].dataset

    # 选最好的例子
    # Video 126: fall->fallen->fallen->fallen->fallen, 满分15/15
    # Video 76:  standing->standing->fall->fall->fallen, GT有完整standing→fall→fallen
    selected = [126, 136, 76, 49]

    out_dir = os.path.join(BASE, 'logs', 'tcn_full', 'timelines_v2')
    os.makedirs(out_dir, exist_ok=True)

    for vid in selected:
        sample = test_ds[vid]
        features = sample['features'].unsqueeze(0).to(device)
        with torch.no_grad():
            logits_cls, logits_fall, logits_fallen = model(features)

        gt_cls = sample['labels_16'].numpy()
        pred_cls = logits_cls.argmax(-1).squeeze(0).cpu().numpy()
        gt_fall = sample['fall_labels'].numpy()
        pred_fall_prob = torch.sigmoid(logits_fall).squeeze().cpu().numpy()
        gt_fallen = sample['fallen_labels'].numpy()
        pred_fallen_prob = torch.sigmoid(logits_fallen).squeeze().cpu().numpy()

        # 计算正确率
        cls_ok = (pred_cls == gt_cls).sum()
        fall_ok = ((pred_fall_prob > 0.5).astype(int) == gt_fall).sum()
        fallen_ok = ((pred_fallen_prob > 0.5).astype(int) == gt_fallen).sum()

        plot_timeline_clear(
            gt_cls, pred_cls, gt_fall, pred_fall_prob,
            gt_fallen, pred_fallen_prob,
            os.path.join(out_dir, f"video_{vid:03d}.png"),
            vid,
        )

    print(f"\nGenerated {len(selected)} clear timelines in {out_dir}/")


if __name__ == "__main__":
    main()
