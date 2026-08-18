"""
真加速度诊断：髋部关键点位置的二阶差分在 fall vs lie_down 过渡段是否可分。

背景：方向 1 的"加速度"是对 ViT-g 语义向量做二阶差分（已证伪）。
本脚本改对"物理量"做二阶差分——MediaPipe 关键点的 2D 位置。
髋部中点 y 坐标（hip_y）的垂直位移，其二阶差分就是髋部的真实（图像空间）加速度。

关键：pose npz 与标签 CSV 的**视频顺序不同**（同一集合，不同排序），
因此不能按平坦位置对齐，必须按 path 对齐。

输出三类对比：
  1. 逐帧 |accel| 分布（fall vs lie_down）
  2. 过渡段峰值 |accel| 分布（每个连续 fall/lie_down 段取峰值）
  3. 单特征 AUC（0.5=不可分, 1.0=完美可分）

用法：
    conda run -n py310 python experiments/probe_pose_accel.py
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_POSE = os.path.join(ROOT, "data", "omnifall_pose_frame.npz")
LABELS_CSV = os.path.join(ROOT, "data", "ofsyn_frame_labels.csv")

# MediaPipe 关键点索引（与 step9 一致）
NOSE = 0
L_SH, R_SH = 11, 12
L_HIP, R_HIP = 23, 24

# 16 类标签 id（label2id.csv）
ID = {"fall": 1, "fallen": 2, "lie_down": 5, "lying": 6}


def clean_path(p):
    s = str(p)
    return s[:-4] if s.endswith(".mp4") else s


def manual_auc(pos, neg):
    """两个一维分布的 ROC AUC（对方向不敏感，取 max(auc, 1-auc)）。"""
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    all_vals = np.concatenate([pos, neg])
    ranks = np.argsort(np.argsort(all_vals, kind="mergesort"), kind="mergesort") + 1.0
    pos_ranks = ranks[: len(pos)]
    auc = (pos_ranks.sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))
    return max(auc, 1.0 - auc)


def per_video_second_diff(x, starts, counts):
    """逐视频中心二阶差分。首尾帧=0（跨视频边界不差分）。"""
    acc = np.zeros_like(x, dtype=np.float32)
    for s, c in zip(starts, counts):
        if c < 3:
            continue
        seg = x[s:s + c]
        acc[s + 1:s + c - 1] = seg[2:] - 2.0 * seg[1:-1] + seg[:-2]
    return acc


def stats(name, vals):
    vals = np.asarray(vals, dtype=float)
    if len(vals) == 0:
        return f"{name:<22s} n=0"
    return (f"{name:<22s} n={len(vals):>7d}  mean={vals.mean():.5f}  "
            f"median={np.median(vals):.5f}  std={vals.std():.5f}  "
            f"p90={np.percentile(vals, 90):.5f}  max={vals.max():.5f}")


def main():
    print(f"RAW_POSE: {RAW_POSE}")
    print(f"LABELS:   {LABELS_CSV}\n")

    # --- 加载标签，建立 path -> label 数组（按 frame_idx 排序） ---
    df = pd.read_csv(LABELS_CSV)
    label_by_video = {}
    for path, grp in df.groupby("path", sort=False):
        grp = grp.sort_values("frame_idx")
        label_by_video[path] = grp["label"].values.astype(np.int64)
    n_csv_frames = int(df["frame_idx"].count())
    print(f"标签 CSV: {len(label_by_video):,} 个视频, {n_csv_frames:,} 帧")

    # --- 加载原始姿态 (mmap) ---
    raw = np.load(RAW_POSE, allow_pickle=True, mmap_mode="r")
    pose = raw["pose_features"]                       # (960000, 99) = 33 x (x,y,vis)
    video_paths = raw["video_paths"]
    starts = raw["video_start_indices"].astype(np.int64)
    counts = raw["video_clip_counts"].astype(np.int64)
    n_pose_frames = pose.shape[0]
    print(f"原始姿态: {pose.shape}, {len(video_paths):,} 个视频\n")

    # --- 按 path 对齐标签到 pose 帧顺序 ---
    labels_aligned = np.zeros(n_pose_frames, dtype=np.int64)
    n_missing = 0
    for vi in range(len(video_paths)):
        p = clean_path(video_paths[vi])
        lbl = label_by_video.get(p)
        if lbl is None:
            n_missing += 1
            continue
        s = starts[vi]
        c = counts[vi]
        n = min(len(lbl), c)
        labels_aligned[s:s + n] = lbl[:n]
    print(f"对齐: 缺失标签的视频 {n_missing} 个（应为 0）")

    # --- 髋部中点 y（垂直位置） ---
    hip_y = (pose[:, L_HIP * 3 + 1].astype(np.float32)
             + pose[:, R_HIP * 3 + 1].astype(np.float32)) / 2.0

    # --- 逐视频一阶（速度）与二阶（加速度） ---
    acc = per_video_second_diff(hip_y, starts, counts)
    abs_acc = np.abs(acc)

    # --- 躯干倾角（次级信号） ---
    sh_mid_y = (pose[:, L_SH * 3 + 1].astype(np.float32)
                + pose[:, R_SH * 3 + 1].astype(np.float32)) / 2.0
    sh_mid_x = (pose[:, L_SH * 3].astype(np.float32)
                + pose[:, R_SH * 3].astype(np.float32)) / 2.0
    hip_mid_x = (pose[:, L_HIP * 3].astype(np.float32)
                 + pose[:, R_HIP * 3].astype(np.float32)) / 2.0
    torso_angle = np.arctan2(sh_mid_x - hip_mid_x, -(sh_mid_y - hip_y))
    acc_torso = per_video_second_diff(torso_angle, starts, counts)

    # --- 【1】逐帧 |accel| 分布：fall vs lie_down ---
    print("\n" + "=" * 80)
    print("【1】逐帧 |髋部加速度| 分布（过渡帧）")
    print("=" * 80)
    for cls in ["fall", "lie_down"]:
        m = labels_aligned == ID[cls]
        print(stats(cls, abs_acc[m]))
    m_fall = labels_aligned == ID["fall"]
    m_lie = labels_aligned == ID["lie_down"]
    if m_fall.sum() and m_lie.sum():
        print(f"\n逐帧 |accel| 作为 fall/lie_down 二分类器 AUC = "
              f"{manual_auc(abs_acc[m_fall], abs_acc[m_lie]):.4f}")

    # --- 【2】过渡段峰值对比 ---
    def segment_peaks(mask):
        peaks = []
        d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
        s_seg = np.where(d == 1)[0]
        e_seg = np.where(d == -1)[0]
        for a, b in zip(s_seg, e_seg):
            if b - a >= 2:
                peaks.append(abs_acc[a:b].max())
        return np.array(peaks)

    print("\n" + "=" * 80)
    print("【2】过渡段峰值 |髋部加速度| 分布（每个连续段取 max）")
    print("=" * 80)
    peaks_fall = segment_peaks(m_fall)
    peaks_lie = segment_peaks(m_lie)
    print(stats("fall 段峰值", peaks_fall))
    print(stats("lie_down 段峰值", peaks_lie))
    if len(peaks_fall) and len(peaks_lie):
        print(f"\n过渡段峰值 |accel| 二分类 AUC = "
              f"{manual_auc(peaks_fall, peaks_lie):.4f}")

    # --- 【对照】fallen vs lying（静态，加速度应都≈0） ---
    print("\n" + "=" * 80)
    print("【对照】fallen vs lying（静态类，加速度应都≈0）")
    print("=" * 80)
    for cls in ["fallen", "lying"]:
        m = labels_aligned == ID[cls]
        print(stats(cls, abs_acc[m]))
    m_fallen = labels_aligned == ID["fallen"]
    m_lying = labels_aligned == ID["lying"]
    if m_fallen.sum() and m_lying.sum():
        print(f"\nfallen/lying 逐帧 |accel| AUC = "
              f"{manual_auc(abs_acc[m_fallen], abs_acc[m_lying]):.4f}")

    # --- 【附加】躯干倾角加速度 ---
    print("\n" + "=" * 80)
    print("【附加】躯干倾角加速度（fall vs lie_down）")
    print("=" * 80)
    for cls in ["fall", "lie_down"]:
        m = labels_aligned == ID[cls]
        print(stats(cls + " |躯干角accel|", np.abs(acc_torso[m])))
    if m_fall.sum() and m_lie.sum():
        print(f"\n躯干角 |accel| fall/lie_down AUC = "
              f"{manual_auc(np.abs(acc_torso[m_fall]), np.abs(acc_torso[m_lie])):.4f}")

    print("\n完成。")


if __name__ == "__main__":
    main()
