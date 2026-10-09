# -*- coding: utf-8 -*-
"""Phase 11 性能诊断（第 1 部分）：**纯数据侧**的每 batch 耗时。

**为什么要它**：E1 训练在 3090 上实测 **5.28 s/步**（batch=2, window=64），
E1-full 的 42.26 h 就是它推出来的。要判断"**换更强的卡是否有用**"，
必须先知道这 5.28 s 里**数据侧占多少**：

  - 数据侧 ≪ 5.28 s  → GPU 是串行瓶颈，换更快的卡**有**用
  - 数据侧 ≈ 5.28 s  → 瓶颈在解码/搬运，换卡**无用**（该做的是缓存帧 or 加 workers）

**做法**：只构造 dataset + DataLoader，**不建模型、不做前向**，跑 N 个 batch 计时。
⚠ 与训练侧**严格同参**：同一 csv / window / stride / batch / workers / collate_fn。
⚠ 数据是 mp4 现解码（AV1，经 `preprocessing/av_decode.py`）——这正是训练侧的真实路径。

用法（远端项目根）：
    python -u analysis/p11_profile_dataloader.py
    python -u analysis/p11_profile_dataloader.py --pose_npz data/omnifall_pose_semantic_accel.npz
"""

import os
import sys
import time
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from experiments.train_p9e1 import Mp4WindowCache, collate_windows  # noqa: E402

DEFAULT_CSV = os.path.join(BASE, "DATASET-omnifall", "splits", "syn", "random", "train.csv")
DEFAULT_NPZ = os.path.join(BASE, "data", "omnifall_dinov2_giant_frame.npz")
TRAIN_STEP_S = 5.28          # E1-full 实测（3090）/ 5.28 s/步；用于算占比


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--npz", default=DEFAULT_NPZ)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit_videos", type=int, default=300,
                    help="只取 N 个视频计时。⚠ 这是**前 N 个**（有偏），"
                         "但本脚本只测吞吐、不算指标，偏差不影响结论。")
    ap.add_argument("--steps", type=int, default=40, help="计时 batch 数")
    ap.add_argument("--warmup", type=int, default=8, help="预热 batch 数（worker 启动开销）")
    ap.add_argument("--pose_npz", default=None,
                    help="测 arm 4 的数据侧时给出（含姿态读取的开销）")
    ap.add_argument("--train_step_s", type=float, default=TRAIN_STEP_S,
                    help="训练实测的 s/步，用于算数据侧占比")
    args = ap.parse_args()

    print("=" * 78)
    print("Phase 11 性能诊断 · 第 1 部分：纯数据侧")
    print("=" * 78)
    print(f"  csv      = {os.path.basename(args.csv)}")
    print(f"  window={args.window} stride={args.stride} batch={args.batch} workers={args.workers}")
    print(f"  limit_videos={args.limit_videos}  pose={'ON' if args.pose_npz else 'OFF'}")
    print()

    t0 = time.time()
    ds = Mp4WindowCache(args.csv, args.npz, args.window, args.stride,
                        limit_videos=args.limit_videos, pose_npz=args.pose_npz)
    print(f"  数据集构造 {time.time()-t0:.1f}s | 视频 {len(ds.videos)} | 窗口 {len(ds)}")
    print()

    loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, collate_fn=collate_windows,
                        persistent_workers=args.workers > 0)

    ts = []
    for i, batch in enumerate(loader):
        if i < args.warmup:
            t = time.time()
            continue
        ts.append(time.time() - t)
        t = time.time()
        if len(ts) >= args.steps:
            break

    ts = np.array(ts)
    mean, med = float(ts.mean()), float(np.median(ts))
    print("  数据侧（每 batch 耗时）：")
    print(f"    batch 数 {len(ts)} | 均值 {mean:.3f} s/batch | 中位 {med:.3f} | "
          f"P90 {np.percentile(ts,90):.3f} | 最大 {ts.max():.3f}")
    print()
    print(f"  对照训练实测 {args.train_step_s:.2f} s/步：")
    print(f"    数据侧占 **{100*mean/args.train_step_s:.1f}%**（按均值）"
          f" / {100*med/args.train_step_s:.1f}%（按中位）")
    print()
    print("  判读：")
    print("    ≪50%  → 数据不是瓶颈，GPU 是 → **换更快的卡有用**")
    print("    ≈100% → 数据就是瓶颈 → **换卡无用**，应缓存帧 or 加 workers")
    print()
    print("⚠ 注意：DataLoader 在训练中是**与模型并发**跑的（16 workers 预取），")
    print("   所以这里测的是数据管线的**吞吐上限**，用来排除「数据拖后腿」这一可能。")


if __name__ == "__main__":
    main()
