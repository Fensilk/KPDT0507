"""
E3 Stage1 帧级数据准备：从训练集视频抽取采样帧，存 per-video .pt 缓存。

设计：
  - 视频按类别打散后随机取 N 个（保证类均衡，避免前 N 个全是同类别）
  - 每视频均匀抽 K 帧（跨 80 帧），三元标签从 NPZ labels_16 派生
  - 帧存 224x224 uint8 RGB（DINOv2 输入尺寸），归一化在训练时做

用法（项目根目录，py310）：
    python experiments/prep_p9_frames.py --n_videos 2000 --frames_per_video 10
产出：
    data/p9_frames/{i:05d}_{path}.pt  每个含 frames(K,224,224,3) uint8 + labels(K,) int64 + 元信息
"""
import os
import sys
import argparse
import cv2
import numpy as np
import pandas as pd
import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ = os.path.join(BASE, "data", "omnifall_dinov2_giant_frame.npz")
VIDEO_ROOT = os.path.join(BASE, "DATASET-omnifall", "data_files", "extracted")
TRAIN_CSV = os.path.join(BASE, "DATASET-omnifall", "splits", "syn", "random", "train.csv")
OUT = os.path.join(BASE, "data", "p9_frames")
SIZE = (224, 224)
N_FRAMES = 80


def decode_frames(mp4, n=N_FRAMES):
    cap = cv2.VideoCapture(mp4)
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    if len(frames) < n:
        while len(frames) < n:
            frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))
    return frames[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_videos", type=int, default=2000)
    ap.add_argument("--frames_per_video", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # 视频索引
    d = np.load(NPZ, allow_pickle=True, mmap_mode="r")
    vp = [str(p) for p in d["video_paths"]]
    vsi = d["video_start_indices"]
    labels16 = d["labels_16"]
    path2idx = {p: i for i, p in enumerate(vp)}

    train_paths = list(pd.read_csv(TRAIN_CSV)["path"].str.strip())
    rng = np.random.default_rng(args.seed)
    rng.shuffle(train_paths)
    selected = train_paths[: args.n_videos]

    os.makedirs(OUT, exist_ok=True)
    n_ok = 0
    class_count = np.zeros(3, dtype=int)
    for i, rel in enumerate(selected):
        mp4 = os.path.join(VIDEO_ROOT, rel + ".mp4")
        if rel not in path2idx or not os.path.exists(mp4):
            continue
        vi = path2idx[rel]
        st = int(vsi[vi])
        frames = decode_frames(mp4)
        idx = np.linspace(0, N_FRAMES - 1, args.frames_per_video, dtype=int)
        imgs = np.stack([cv2.resize(frames[j], SIZE, interpolation=cv2.INTER_LINEAR)
                         for j in idx]).astype(np.uint8)  # (K,H,W,3)
        lab16 = labels16[st + idx]
        ternary = np.where(lab16 == 1, 0, np.where(lab16 == 2, 1, 2)).astype(np.int64)
        class_count += np.bincount(ternary, minlength=3)
        torch.save({"frames": imgs, "labels": ternary, "frame_idx": idx, "video": rel},
                   os.path.join(OUT, f"{n_ok:05d}_{rel.replace('/', '_')}.pt"))
        n_ok += 1
        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{len(selected)}] 已存 {n_ok} 视频")

    print(f"\n完成：{n_ok} 视频 × {args.frames_per_video} 帧 = {n_ok*args.frames_per_video} 帧")
    print(f"三元标签分布 fall/fallen/normal: {class_count.tolist()}")
    print(f"缓存目录: {OUT}")


if __name__ == "__main__":
    main()
