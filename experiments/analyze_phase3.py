"""
Phase 3 汇总分析脚本.

扫描 logs/phase3/ 目录, 汇总所有实验的 test_results.json,
生成 CSV 表格 + 比较柱状图 + Markdown 表格.
"""

import os
import sys
import json
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np

# 项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE_DIR)
sys.path.insert(0, BASE_DIR)

LOG_ROOT = "logs/phase3"


def load_experiments(log_root: str) -> list[dict]:
    """加载所有实验的 test_results.json."""
    experiments = []
    root = Path(log_root)
    if not root.exists():
        print(f"[WARN] Log directory '{log_root}' not found. No experiments to analyze.")
        return experiments

    for exp_dir in sorted(root.iterdir()):
        if not exp_dir.is_dir():
            continue
        results_path = exp_dir / "test_results.json"
        if not results_path.exists():
            print(f"  [SKIP] {exp_dir.name}: no test_results.json")
            continue

        with open(results_path, "r") as f:
            results = json.load(f)

        config = results.get("config", {})

        # 推断特征组合
        feature_combo = _infer_feature(config)

        row = {
            "exp_tag": exp_dir.name,
            "feature_combo": feature_combo,
            "w_cls": config.get("w_cls", 1.0),
            "w_fall": config.get("w_fall", 0.5),
            "w_fallen": config.get("w_fallen", 0.5),
            "input_dim": config.get("input_dim", "?"),
            "use_diff": config.get("use_diff", False),
            "pose_npz": config.get("pose_npz", False),
            "seg_acc": results.get("seg_acc", 0),
            "fall_f1": results.get("fall_f1", 0),
            "fall_precision": results.get("fall_precision", 0),
            "fall_recall": results.get("fall_recall", 0),
            "fallen_f1": results.get("fallen_f1", 0),
            "fallen_precision": results.get("fallen_precision", 0),
            "fallen_recall": results.get("fallen_recall", 0),
            "avg_f1": results.get("avg_f1", 0),
            "test_loss": results.get("loss", 0),
            "best_epoch": config.get("best_epoch", "?"),
            "epochs_run": config.get("epochs_run", "?"),
        }
        # 跷跷板差距
        row["gap"] = abs(row["fall_f1"] - row["fallen_f1"])
        row["phase"] = _infer_phase(exp_dir.name)

        experiments.append(row)

    return experiments


def _infer_feature(config: dict) -> str:
    """从配置推断特征组合名称."""
    use_diff = config.get("use_diff", False)
    pose = config.get("pose_npz", False)
    if use_diff and pose:
        return "RGB+Δclip+Pose"
    elif use_diff:
        return "RGB+Δclip"
    elif pose:
        return "RGB+Pose"
    else:
        return "RGB"


def _infer_phase(tag: str) -> str:
    """从 tag 推断实验阶段."""
    if tag.startswith("p3a"):
        return "3a_ablation"
    elif tag.startswith("p3b"):
        return "3b_weight_search"
    elif tag.startswith("p3c"):
        return "3c_seesaw_fix"
    return "unknown"


def print_summary_table(experiments: list[dict]):
    """打印汇总表格."""
    if not experiments:
        print("No experiments found.")
        return

    # 按 phase, avg_f1 排序
    experiments.sort(key=lambda r: (r["phase"], -r["avg_f1"]))

    header = (
        f"{'Experiment':<28} {'Feature':<18} {'w(C/F/L)':<18} "
        f"{'seg_acc':>7} {'fall_f1':>7} {'fallen_f1':>7} {'avg_f1':>7} {'gap':>7} {'#Ep':>4}"
    )
    sep = "-" * len(header)

    print("\n" + "=" * len(header))
    print("PHASE 3 EXPERIMENT SUMMARY")
    print("=" * len(header))
    print(header)
    print(sep)

    for r in experiments:
        weights = f"{r['w_cls']:.1f}/{r['w_fall']:.1f}/{r['w_fallen']:.1f}"
        print(
            f"{r['exp_tag']:<28} {r['feature_combo']:<18} {weights:<18} "
            f"{r['seg_acc']:>7.4f} {r['fall_f1']:>7.4f} {r['fallen_f1']:>7.4f} "
            f"{r['avg_f1']:>7.4f} {r['gap']:>7.4f} {r['epochs_run']:>4}"
        )
    print(sep)
    print(f"Total: {len(experiments)} experiments\n")


def find_best(experiments: list[dict]):
    """找出各维度最佳."""
    if not experiments:
        return

    best_avg = max(experiments, key=lambda r: r["avg_f1"])
    best_fall = max(experiments, key=lambda r: r["fall_f1"])
    best_fallen = max(experiments, key=lambda r: r["fallen_f1"])
    most_balanced = min(experiments, key=lambda r: r["gap"])

    print("=" * 60)
    print("BEST RESULTS")
    print("=" * 60)
    print(f"  Best avg_f1:     {best_avg['exp_tag']:<28} avg_f1={best_avg['avg_f1']:.4f}  "
          f"(fall={best_avg['fall_f1']:.4f}, fallen={best_avg['fallen_f1']:.4f})")
    print(f"  Best fall_f1:    {best_fall['exp_tag']:<28} fall_f1={best_fall['fall_f1']:.4f}  "
          f"(fallen={best_fall['fallen_f1']:.4f}, avg={best_fall['avg_f1']:.4f})")
    print(f"  Best fallen_f1:  {best_fallen['exp_tag']:<28} fallen_f1={best_fallen['fallen_f1']:.4f}  "
          f"(fall={best_fallen['fall_f1']:.4f}, avg={best_fallen['avg_f1']:.4f})")
    print(f"  Most balanced:   {most_balanced['exp_tag']:<28} gap={most_balanced['gap']:.4f}  "
          f"(avg_f1={most_balanced['avg_f1']:.4f})")
    print()


def print_phase_analysis(experiments: list[dict]):
    """分阶段专题分析."""
    # Phase 3a: 消融对比
    exps_3a = [r for r in experiments if r["phase"] == "3a_ablation"]
    if exps_3a:
        print("=" * 60)
        print("Phase 3a: Ablation Study (Feature Comparison)")
        print("=" * 60)
        exps_3a.sort(key=lambda r: -r["avg_f1"])
        header = f"{'Feature':<20} {'input_dim':>9} {'fall_f1':>7} {'fallen_f1':>7} {'avg_f1':>7}"
        print(header)
        print("-" * len(header))
        for r in exps_3a:
            print(f"{r['feature_combo']:<20} {r['input_dim']:>9} "
                  f"{r['fall_f1']:>7.4f} {r['fallen_f1']:>7.4f} {r['avg_f1']:>7.4f}")
        print()

    # Phase 3b: 权重搜索
    exps_3b = [r for r in experiments if r["phase"] == "3b_weight_search"]
    if exps_3b:
        print("=" * 60)
        print("Phase 3b: Loss Weight Search (RGB+Δclip+Pose)")
        print("=" * 60)
        exps_3b.sort(key=lambda r: -r["avg_f1"])
        header = f"{'Experiment':<25} {'w(C/F/L)':<18} {'fall_f1':>7} {'fallen_f1':>7} {'avg_f1':>7} {'gap':>7}"
        print(header)
        print("-" * len(header))
        for r in exps_3b:
            weights = f"{r['w_cls']:.1f}/{r['w_fall']:.1f}/{r['w_fallen']:.1f}"
            print(f"{r['exp_tag']:<25} {weights:<18} "
                  f"{r['fall_f1']:>7.4f} {r['fallen_f1']:>7.4f} "
                  f"{r['avg_f1']:>7.4f} {r['gap']:>7.4f}")
        print()

    # Phase 3c: 跷跷板修复
    exps_3c = [r for r in experiments if r["phase"] == "3c_seesaw_fix"]
    if exps_3c:
        print("=" * 60)
        print("Phase 3c: Seesaw Fix (RGB+Pose)")
        print("=" * 60)
        exps_3c.sort(key=lambda r: -r["avg_f1"])
        header = f"{'Experiment':<28} {'w(C/F/L)':<18} {'fall_f1':>7} {'fallen_f1':>7} {'avg_f1':>7} {'gap':>7}"
        print(header)
        print("-" * len(header))
        for r in exps_3c:
            weights = f"{r['w_cls']:.1f}/{r['w_fall']:.1f}/{r['w_fallen']:.1f}"
            print(f"{r['exp_tag']:<28} {weights:<18} "
                  f"{r['fall_f1']:>7.4f} {r['fallen_f1']:>7.4f} "
                  f"{r['avg_f1']:>7.4f} {r['gap']:>7.4f}")
        print()


def save_csv(experiments: list[dict], save_path: str):
    """保存 CSV 汇总表."""
    if not experiments:
        return
    fieldnames = [
        "exp_tag", "phase", "feature_combo", "w_cls", "w_fall", "w_fallen",
        "input_dim", "use_diff", "pose_npz",
        "seg_acc", "fall_f1", "fall_precision", "fall_recall",
        "fallen_f1", "fallen_precision", "fallen_recall",
        "avg_f1", "gap", "test_loss", "best_epoch", "epochs_run",
    ]
    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(experiments, key=lambda r: (r["phase"], -r["avg_f1"])))
    print(f"[INFO] CSV saved to: {save_path}")


def plot_comparison(experiments: list[dict], save_path: str):
    """生成比较柱状图 (fall_f1 + fallen_f1 分组, avg_f1 标注)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available, skipping plot.")
        return

    if not experiments:
        return

    # 分组
    phases_order = ["3a_ablation", "3b_weight_search", "3c_seesaw_fix"]
    phase_labels = {
        "3a_ablation": "Ablation\n(feature combos)",
        "3b_weight_search": "Weight Search\n(RGB+Δclip+Pose)",
        "3c_seesaw_fix": "Seesaw Fix\n(RGB+Pose)",
    }

    grouped = defaultdict(list)
    for r in experiments:
        grouped[r["phase"]].append(r)

    n_groups = sum(1 for p in phases_order if p in grouped)
    if n_groups == 0:
        return

    fig, axes = plt.subplots(1, n_groups, figsize=(5 * n_groups, 6), squeeze=False)
    axes = axes[0]

    for ax, phase in zip(axes, [p for p in phases_order if p in grouped]):
        exps = grouped[phase]
        exps.sort(key=lambda r: -r["avg_f1"])

        labels = [r["exp_tag"].replace("phase3/", "") for r in exps]
        x = np.arange(len(labels))
        width = 0.35

        fall_vals = [r["fall_f1"] for r in exps]
        fallen_vals = [r["fallen_f1"] for r in exps]
        avg_vals = [r["avg_f1"] for r in exps]

        bars1 = ax.bar(x - width/2, fall_vals, width, label="fall_f1",
                       color="#e74c3c", alpha=0.85)
        bars2 = ax.bar(x + width/2, fallen_vals, width, label="fallen_f1",
                       color="#3498db", alpha=0.85)

        # avg_f1 标注
        for i, avg in enumerate(avg_vals):
            ax.annotate(f"{avg:.3f}", (x[i], max(fall_vals[i], fallen_vals[i]) + 0.02),
                        ha="center", fontsize=7, fontweight="bold", color="#2c3e50")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_title(phase_labels.get(phase, phase), fontsize=10, fontweight="bold")
        ax.set_ylim(0, 0.75)
        ax.set_ylabel("F1 Score")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Phase 3: Ablation + Loss Weight Optimization Results",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Comparison plot saved to: {save_path}")


def main():
    print("=" * 60)
    print("Phase 3 Analysis")
    print(f"Scanning: {LOG_ROOT}")
    print("=" * 60)

    experiments = load_experiments(LOG_ROOT)

    if not experiments:
        print("\n[INFO] No experiment results found yet.")
        print("Run experiments first: bash experiments/run_phase3.sh")
        return 0

    experiments.sort(key=lambda r: (r["phase"], -r["avg_f1"]))

    # 1. 汇总表
    print_summary_table(experiments)

    # 2. 各维度最佳
    find_best(experiments)

    # 3. 分阶段分析
    print_phase_analysis(experiments)

    # 4. 保存 CSV
    csv_path = os.path.join(LOG_ROOT, "phase3_summary.csv")
    save_csv(experiments, csv_path)

    # 5. 生成柱状图
    png_path = os.path.join(LOG_ROOT, "phase3_comparison.png")
    plot_comparison(experiments, png_path)

    # 6. Markdown 表格 (便于复制到报告)
    print("\n" + "=" * 60)
    print("MARKDOWN TABLE (for report)")
    print("=" * 60)
    print_markdown_table(experiments)

    return 0


def print_markdown_table(experiments: list[dict]):
    """打印 Markdown 格式表格."""
    print()
    print("| Experiment | Feature | w_cls | w_fall | w_fallen | seg_acc | fall_f1 | fallen_f1 | avg_f1 | gap |")
    print("|-----------|---------|-------|--------|----------|---------|---------|-----------|--------|-----|")
    for r in experiments:
        print(f"| {r['exp_tag']} | {r['feature_combo']} | {r['w_cls']:.1f} | {r['w_fall']:.1f} | "
              f"{r['w_fallen']:.1f} | {r['seg_acc']:.4f} | {r['fall_f1']:.4f} | "
              f"{r['fallen_f1']:.4f} | {r['avg_f1']:.4f} | {r['gap']:.4f} |")
    print()


if __name__ == "__main__":
    sys.exit(main())
