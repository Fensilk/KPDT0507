"""
时间线质量自动化评估.

不依赖人工看图, 将"预测时间线好不好"变成可统计的数字指标.

指标设计:
    1. 时序正确率 — fall 预测是否在 fallen 之前出现
    2. 状态稳定性 — 预测序列中 0/1 翻转次数
    3. 检测延迟   — GT 事件首次出现后, 模型滞后几个 clip 才检测到
    4. Fall/Fallen 混淆 — 两个阶段间的交叉误报
    5. 逐视频 F1 分布 — 而非全局 flatten 后的单一 F1
    6. 覆盖率     — 有多少视频的事件被至少检测到一次

用法:
    python utils/timeline_metrics.py \
        --log_dir logs/phase3/p3a_diff_pose \
        --npz data/omnifall_preprocessed.npz \
        --clips_csv data/ofsyn_clips.csv \
        --splits_dir DATASET-omnifall/splits/syn/random \
        --use_diff --pose_npz data/omnifall_pose.npz
"""

import os
import sys
import json
import argparse
from collections import defaultdict

import numpy as np
import torch

# 项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.tcn_model import TCNModel
from models.dataset import create_dataloaders


# ============================================================
# 逐视频指标计算
# ============================================================

def compute_video_timeline_metrics(
    pred_fall: np.ndarray,      # (T,) int
    pred_fallen: np.ndarray,    # (T,) int
    gt_fall: np.ndarray,        # (T,) int
    gt_fallen: np.ndarray,      # (T,) int
) -> dict:
    """
    对单个视频 (T 个 clips) 计算时间线质量指标.

    Returns:
        dict with per-video metrics.
    """
    T = len(pred_fall)
    result = {"T": T}

    # ---- 1. 时序正确性 ----
    # 在 GT 同时有 fall 和 fallen 的视频中, 检查预测的 fall 首发 < fallen 首发
    gt_has_fall = gt_fall.sum() > 0
    gt_has_fallen = gt_fallen.sum() > 0
    result["gt_has_fall"] = bool(gt_has_fall)
    result["gt_has_fallen"] = bool(gt_has_fallen)

    pred_first_fall = int(np.argmax(pred_fall > 0)) if (pred_fall > 0).any() else T
    pred_first_fallen = int(np.argmax(pred_fallen > 0)) if (pred_fallen > 0).any() else T
    gt_first_fall = int(np.argmax(gt_fall > 0)) if gt_has_fall else T
    gt_first_fallen = int(np.argmax(gt_fallen > 0)) if gt_has_fallen else T

    result["pred_first_fall"] = pred_first_fall
    result["pred_first_fallen"] = pred_first_fallen
    result["gt_first_fall"] = gt_first_fall
    result["gt_first_fallen"] = gt_first_fallen

    # 时序是否正确: predicted fall 首次出现 < predicted fallen 首次出现
    if pred_first_fall < T and pred_first_fallen < T:
        result["order_correct"] = pred_first_fall < pred_first_fallen
    elif pred_first_fall < T:
        result["order_correct"] = True   # 只检测到 fall, 勉强算对
    elif pred_first_fallen < T:
        result["order_correct"] = gt_first_fall >= T  # GT 无 fall 则对, 有则错
    else:
        result["order_correct"] = True   # 都没检测到, 不算错 (归为漏检)

    # ---- 2. 状态稳定性 (翻转次数) ----
    def count_flips(seq: np.ndarray) -> int:
        """统计 0/1 序列中相邻元素变化的次数."""
        if len(seq) <= 1:
            return 0
        return int((seq[1:] != seq[:-1]).sum())

    result["fall_flips"] = count_flips(pred_fall)
    result["fallen_flips"] = count_flips(pred_fallen)
    result["total_flips"] = result["fall_flips"] + result["fallen_flips"]

    # ---- 3. 检测延迟 ----
    if gt_has_fall and pred_first_fall < T:
        result["fall_delay"] = max(0, pred_first_fall - gt_first_fall)
    else:
        result["fall_delay"] = None  # GT 无 fall 或模型未检测到

    if gt_has_fallen and pred_first_fallen < T:
        result["fallen_delay"] = max(0, pred_first_fallen - gt_first_fallen)
    else:
        result["fallen_delay"] = None

    # ---- 4. Fall/Fallen 混淆 ----
    # 在 GT_fall=1 的 clip 上, 模型错误预测 fallen=1 的比例
    fall_clips = gt_fall == 1
    if fall_clips.sum() > 0:
        result["fall_as_fallen_rate"] = float(pred_fallen[fall_clips].mean())
    else:
        result["fall_as_fallen_rate"] = None

    # 在 GT_fallen=1 的 clip 上, 模型错误预测 fall=1 的比例
    fallen_clips = gt_fallen == 1
    if fallen_clips.sum() > 0:
        result["fallen_as_fall_rate"] = float(pred_fall[fallen_clips].mean())
    else:
        result["fallen_as_fall_rate"] = None

    # ---- 5. 逐视频 F1 (仅在含事件的视频上计算) ----
    def binary_counts(pred: np.ndarray, gt: np.ndarray):
        tp = int(((pred == 1) & (gt == 1)).sum())
        fp = int(((pred == 1) & (gt == 0)).sum())
        fn = int(((pred == 0) & (gt == 1)).sum())
        tn = int(((pred == 0) & (gt == 0)).sum())
        return tp, fp, fn, tn

    def safe_f1(tp, fp, fn):
        return 2 * tp / max(2 * tp + fp + fn, 1)

    fall_tp, fall_fp, fall_fn, fall_tn = binary_counts(pred_fall, gt_fall)
    fallen_tp, fallen_fp, fallen_fn, fallen_tn = binary_counts(pred_fallen, gt_fallen)

    # 只在 GT 有该事件时 F1 才有意义
    if gt_has_fall:
        result["fall_f1"] = safe_f1(fall_tp, fall_fp, fall_fn)
    else:
        result["fall_f1"] = None  # 无事件, F1 不适用
        result["fall_false_alarm"] = fall_fp > 0  # 无事件却预测了 = 误报

    if gt_has_fallen:
        result["fallen_f1"] = safe_f1(fallen_tp, fallen_fp, fallen_fn)
    else:
        result["fallen_f1"] = None
        result["fallen_false_alarm"] = fallen_fp > 0

    # avg_f1: 只在两个事件都存在的视频上计算
    if gt_has_fall and gt_has_fallen:
        result["avg_f1"] = (result["fall_f1"] + result["fallen_f1"]) / 2.0
    elif gt_has_fall:
        result["avg_f1"] = result["fall_f1"]
    elif gt_has_fallen:
        result["avg_f1"] = result["fallen_f1"]
    else:
        result["avg_f1"] = None

    # ---- 6. 检测覆盖率 + 误报率 ----
    result["fall_detected"] = bool((pred_fall > 0).any())
    result["fallen_detected"] = bool((pred_fallen > 0).any())

    return result


# ============================================================
# 汇总统计
# ============================================================

def aggregate_timeline_metrics(per_video: list[dict]) -> dict:
    """将逐视频指标汇总为整体统计量."""
    n = len(per_video)
    if n == 0:
        return {}

    # 辅助函数
    def mean_of(vals):
        valid = [v for v in vals if v is not None]
        return float(np.mean(valid)) if valid else None

    def median_of(vals):
        valid = [v for v in vals if v is not None]
        return float(np.median(valid)) if valid else None

    def rate_of(vals):
        valid = [v for v in vals if v is not None]
        return float(np.mean(valid)) if valid else None

    agg = {}

    # --- 基本统计 ---
    agg["n_videos"] = n
    agg["n_has_fall"] = sum(1 for v in per_video if v["gt_has_fall"])
    agg["n_has_fallen"] = sum(1 for v in per_video if v["gt_has_fallen"])
    agg["n_has_both"] = sum(1 for v in per_video if v["gt_has_fall"] and v["gt_has_fallen"])

    # --- 时序正确率 ---
    both_videos = [v for v in per_video if v["gt_has_fall"] and v["gt_has_fallen"]]
    agg["order_correct_rate"] = mean_of([v["order_correct"] for v in both_videos])

    # --- 状态稳定性 ---
    agg["mean_fall_flips"] = mean_of([v["fall_flips"] for v in per_video])
    agg["mean_fallen_flips"] = mean_of([v["fallen_flips"] for v in per_video])
    agg["mean_total_flips"] = mean_of([v["total_flips"] for v in per_video])
    # 翻转次数分布
    agg["flip_distribution"] = {
        "0_flips": sum(1 for v in per_video if v["total_flips"] == 0),
        "1_flips": sum(1 for v in per_video if v["total_flips"] == 1),
        "2_flips": sum(1 for v in per_video if v["total_flips"] == 2),
        "3_flips": sum(1 for v in per_video if v["total_flips"] == 3),
        "4+_flips": sum(1 for v in per_video if v["total_flips"] >= 4),
    }
    # 高频翻转视频比例 (total_flips >= 4 表示平均每个 clip 都在变)
    agg["unstable_rate"] = sum(1 for v in per_video if v["total_flips"] >= 4) / n

    # --- 检测延迟 ---
    agg["mean_fall_delay"] = mean_of([v["fall_delay"] for v in per_video])
    agg["median_fall_delay"] = median_of([v["fall_delay"] for v in per_video])
    agg["mean_fallen_delay"] = mean_of([v["fallen_delay"] for v in per_video])
    agg["median_fallen_delay"] = median_of([v["fallen_delay"] for v in per_video])

    # --- Fall/Fallen 混淆 ---
    agg["mean_fall_as_fallen_rate"] = rate_of([v["fall_as_fallen_rate"] for v in per_video])
    agg["mean_fallen_as_fall_rate"] = rate_of([v["fallen_as_fall_rate"] for v in per_video])

    # --- 逐视频 F1 分布 (仅含事件的视频) ---
    fall_f1s = [v["fall_f1"] for v in per_video if v["fall_f1"] is not None]
    fallen_f1s = [v["fallen_f1"] for v in per_video if v["fallen_f1"] is not None]
    avg_f1s = [v["avg_f1"] for v in per_video if v["avg_f1"] is not None]

    def f1_dist(f1_list, event_count):
        if not f1_list:
            return {"n": 0, "mean": None, "median": None, "std": None}
        return {
            "n": len(f1_list),
            "mean": float(np.mean(f1_list)),
            "median": float(np.median(f1_list)),
            "std": float(np.std(f1_list)),
            "0.0": sum(1 for v in f1_list if v == 0.0),
            "0.0-0.5": sum(1 for v in f1_list if 0.0 < v < 0.5),
            "0.5-0.8": sum(1 for v in f1_list if 0.5 <= v < 0.8),
            "0.8-1.0": sum(1 for v in f1_list if v >= 0.8),
            "1.0": sum(1 for v in f1_list if v == 1.0),
        }

    agg["per_video_fall_f1"] = f1_dist(fall_f1s, agg["n_has_fall"])
    agg["per_video_fallen_f1"] = f1_dist(fallen_f1s, agg["n_has_fallen"])
    agg["per_video_avg_f1"] = {
        "n": len(avg_f1s),
        "mean": float(np.mean(avg_f1s)) if avg_f1s else None,
        "median": float(np.median(avg_f1s)) if avg_f1s else None,
        "std": float(np.std(avg_f1s)) if avg_f1s else None,
    }

    # --- 误报统计 (无事件视频中错误预测) ---
    no_fall_videos = [v for v in per_video if not v["gt_has_fall"]]
    no_fallen_videos = [v for v in per_video if not v["gt_has_fallen"]]
    agg["fall_false_alarm_rate"] = (
        sum(1 for v in no_fall_videos if v.get("fall_false_alarm", False)) / max(len(no_fall_videos), 1)
    )
    agg["fallen_false_alarm_rate"] = (
        sum(1 for v in no_fallen_videos if v.get("fallen_false_alarm", False)) / max(len(no_fallen_videos), 1)
    )

    # --- 检测覆盖率 ---
    fall_videos = [v for v in per_video if v["gt_has_fall"]]
    fallen_videos = [v for v in per_video if v["gt_has_fallen"]]
    agg["fall_detection_rate"] = mean_of([v["fall_detected"] for v in fall_videos]) if fall_videos else None
    agg["fallen_detection_rate"] = mean_of([v["fallen_detected"] for v in fallen_videos]) if fallen_videos else None

    # --- 综合通过率: 在有事件的视频上, fall_f1>=0.5 且 fallen_f1>=0.5, 且 order_correct ---
    eligible = [v for v in per_video if v["gt_has_fall"] and v["gt_has_fallen"]]
    agg["pass_rate"] = sum(
        1 for v in eligible
        if (v["fall_f1"] is not None and v["fall_f1"] >= 0.5)
        and (v["fallen_f1"] is not None and v["fallen_f1"] >= 0.5)
        and v["order_correct"]
    ) / max(len(eligible), 1) if eligible else None

    return agg


# ============================================================
# 主流程
# ============================================================

def evaluate_timelines(
    model: torch.nn.Module,
    test_loader,
    device: torch.device,
) -> tuple[list[dict], dict]:
    """对所有测试视频计算逐视频时间线指标并汇总."""
    model.eval()
    per_video = []

    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device)            # (B, T, D)
            labels_16 = batch["labels_16"]                      # (B, T)
            fall_gt = batch["fall_labels"]                      # (B, T)
            fallen_gt = batch["fallen_labels"]                  # (B, T)

            logits_cls, logits_fall, logits_fallen = model(features)

            pred_cls = logits_cls.argmax(-1).cpu().numpy()                         # (B, T)
            pred_fall = (torch.sigmoid(logits_fall) > 0.5).long().squeeze(-1).cpu().numpy()
            pred_fallen = (torch.sigmoid(logits_fallen) > 0.5).long().squeeze(-1).cpu().numpy()

            B = features.size(0)
            for i in range(B):
                metrics = compute_video_timeline_metrics(
                    pred_fall=pred_fall[i],
                    pred_fallen=pred_fallen[i],
                    gt_fall=fall_gt[i].numpy(),
                    gt_fallen=fallen_gt[i].numpy(),
                )
                per_video.append(metrics)

    agg = aggregate_timeline_metrics(per_video)
    return per_video, agg


def parse_args():
    p = argparse.ArgumentParser(description="Timeline quality evaluation")
    p.add_argument("--log_dir", type=str, required=True,
                   help="实验日志目录 (含 tcn_best.pt)")
    p.add_argument("--npz", type=str, default="data/omnifall_preprocessed.npz")
    p.add_argument("--clips_csv", type=str, default="data/ofsyn_clips.csv")
    p.add_argument("--splits_dir", type=str, default="DATASET-omnifall/splits/syn/random")
    p.add_argument("--use_diff", action="store_true")
    p.add_argument("--pose_npz", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def main():
    args = parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(base_dir)

    # 设备
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[INFO] Device: {device}")

    # 加载模型
    ckpt_path = os.path.join(args.log_dir, "tcn_best.pt")
    if not os.path.exists(ckpt_path):
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # 从 checkpoint 的 args 读取特征配置 (Phase 3 格式)
    chkpt_args = ckpt.get("args", {})
    use_diff = args.use_diff or chkpt_args.get("use_diff", False)
    pose_npz = args.pose_npz or chkpt_args.get("pose_npz", None)

    # 推断 input_dim (避免为取一个数字加载整个 NPZ)
    input_dim = 512
    if use_diff:
        input_dim += 512
    if pose_npz:
        # 从 NPZ 读取 pose 维度 (只读 metadata, 不加载全部数据)
        pose_path = os.path.join(base_dir, pose_npz)
        if os.path.exists(pose_path):
            pose_data = np.load(pose_path, allow_pickle=True)
            input_dim += pose_data["pose_features"].shape[1]
            pose_data.close()

    print(f"[INFO] Input dim: {input_dim}, use_diff={use_diff}, pose_npz={pose_npz}")

    model = TCNModel(
        input_dim=input_dim,
        hidden_dim=256,
        num_blocks=2,
        kernel_size=3,
        dilations=[1, 2],
        tcn_dropout=0.2,
        num_classes=16,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[INFO] Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    # 数据 (使用 checkpoint 中的特征配置, 而非命令行参数)
    _, _, test_loader = create_dataloaders(
        npz_path=args.npz,
        clips_csv_path=args.clips_csv,
        splits_dir=args.splits_dir,
        batch_size=args.batch_size,
        num_workers=0,
        use_weighted_sampler=False,
        use_diff=use_diff,
        pose_npz_path=pose_npz,
    )

    # 评估
    per_video, agg = evaluate_timelines(model, test_loader, device)

    # ---- 输出 ----
    print("\n" + "=" * 60)
    print("TIMELINE QUALITY REPORT")
    print("=" * 60)

    print(f"\n--- 基本数据 ---")
    print(f"  测试视频总数:        {agg['n_videos']}")
    print(f"  含 fall 事件的视频:  {agg['n_has_fall']}")
    print(f"  含 fallen 事件视频:  {agg['n_has_fallen']}")
    print(f"  同时含两者的视频:    {agg['n_has_both']}")

    print(f"\n--- 1. 时序正确性 ---")
    print(f"  fall→fallen 顺序正确率: {agg['order_correct_rate']:.1%}  "
          f"({agg['n_has_both']} 个含两事件的视频)")

    print(f"\n--- 2. 状态稳定性 (翻转次数, 越低越好) ---")
    print(f"  平均 fall 翻转:   {agg['mean_fall_flips']:.2f}")
    print(f"  平均 fallen 翻转: {agg['mean_fallen_flips']:.2f}")
    print(f"  不稳定视频占比:   {agg['unstable_rate']:.1%}  (>=4次翻转)")
    print(f"  翻转分布: 0次={agg['flip_distribution']['0_flips']}, "
          f"1次={agg['flip_distribution']['1_flips']}, "
          f"2次={agg['flip_distribution']['2_flips']}, "
          f"3次={agg['flip_distribution']['3_flips']}, "
          f"4+次={agg['flip_distribution']['4+_flips']}")

    print(f"\n--- 3. 检测延迟 (clips, 越低越好) ---")
    print(f"  fall 延迟:   均值={agg['mean_fall_delay']:.2f}, 中位={agg['median_fall_delay']:.2f}")
    print(f"  fallen 延迟: 均值={agg['mean_fallen_delay']:.2f}, 中位={agg['median_fallen_delay']:.2f}")

    print(f"\n--- 4. Fall/Fallen 混淆 ---")
    print(f"  GT fall → 错判为 fallen: {agg['mean_fall_as_fallen_rate']:.1%}")
    print(f"  GT fallen → 错判为 fall: {agg['mean_fallen_as_fall_rate']:.1%}")

    print(f"\n--- 5. 逐视频 F1 分布 (仅含事件的视频) ---")
    for name in ["per_video_fall_f1", "per_video_fallen_f1"]:
        d = agg[name]
        label = "Fall" if "fall" in name else "Fallen"
        if d["n"] == 0:
            print(f"  {label}: (无含事件视频)")
            continue
        print(f"  {label} (n={d['n']}): mean={d['mean']:.3f}, median={d['median']:.3f}, std={d['std']:.3f}")
        print(f"        F1=0: {d['0.0']}, (0,0.5): {d['0.0-0.5']}, "
              f"[0.5,0.8): {d['0.5-0.8']}, [0.8,1.0): {d['0.8-1.0']}, F1=1.0: {d['1.0']}")

    print(f"\n--- 6. 检测覆盖率 ---")
    print(f"  Fall 检测率:     {agg['fall_detection_rate']:.1%}  ({agg['n_has_fall']} 个含 fall 的测试视频中)")
    print(f"  Fallen 检测率:   {agg['fallen_detection_rate']:.1%}  ({agg['n_has_fallen']} 个含 fallen 的测试视频中)")
    print(f"  Fall 误报率:     {agg['fall_false_alarm_rate']:.1%}  ({agg['n_videos'] - agg['n_has_fall']} 个无 fall 视频中)")
    print(f"  Fallen 误报率:   {agg['fallen_false_alarm_rate']:.1%}  ({agg['n_videos'] - agg['n_has_fallen']} 个无 fallen 视频中)")

    print(f"\n--- 7. 综合通过率 ---")
    print(f"  (fall_f1>=0.5 & fallen_f1>=0.5 & 顺序正确): "
          f"{agg['pass_rate']:.1%}  ({agg['n_has_both']} 个含双事件视频中)" if agg['pass_rate'] is not None else "  N/A")

    # 保存结果
    out_path = os.path.join(args.log_dir, "timeline_quality.json")
    # 只保存汇总, 不保存所有逐视频细节 (避免文件过大)
    with open(out_path, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"\n[INFO] Summary saved to {out_path}")

    # 也保存逐视频结果 (用于后续分析最差视频)
    per_video_path = os.path.join(args.log_dir, "timeline_per_video.json")
    with open(per_video_path, "w") as f:
        json.dump(per_video, f, indent=2)
    print(f"[INFO] Per-video details saved to {per_video_path}")


if __name__ == "__main__":
    main()
