"""
E1 连续帧缓存：与 E3 同批 2,000 训练视频，抽全部 80 连续帧（PyAV 解码）。

设计：
  - 训练视频用与 E3 完全相同的 seed 打散后取前 N 个（受控对比 → 视频逐一对齐 E3）
  - 每视频解码全部 80 连续帧（不抽帧），resize 224x224 uint8 RGB（DINOv2 输入尺寸）
  - 缓存不含标签：训练时按 video path 从 NPZ 对齐标签，80 帧即视频前 80 帧、位置即帧序 0..79
  - 另缓存一个 val 子集（--n_val）供训练监控；用 fresh rng(seed) 独立打散，写独立目录、计数器从 0 重启

用法（项目根目录，py310）：
    python experiments/prep_p9e1_frames.py --n_videos 2000 --n_val 150
产出：
    data/p9e1_frames/{i:05d}_{path}.pt       每视频 {"frames": (80,224,224,3) uint8, "video": path}
    data/p9e1_frames_val/{i:05d}_{path}.pt   同上（val 子集 --n_val 个）
"""
import os
import sys
import argparse

import cv2
import numpy as np
import pandas as pd
import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "preprocessing"))  # 供 av_decode 导入
NPZ = os.path.join(BASE, "data", "omnifall_dinov2_giant_frame.npz")
VIDEO_ROOT = os.path.join(BASE, "DATASET-omnifall", "data_files", "extracted")
TRAIN_CSV = os.path.join(BASE, "DATASET-omnifall", "splits", "syn", "random", "train.csv")
VAL_CSV = os.path.join(BASE, "DATASET-omnifall", "splits", "syn", "random", "val.csv")
OUT = os.path.join(BASE, "data", "p9e1_frames")
OUT_VAL = os.path.join(BASE, "data", "p9e1_frames_val")
SIZE = (224, 224)
N_FRAMES = 80


def decode_frames(mp4, n=N_FRAMES):
    """返回 list of (H,W,3) uint8；PyAV 优先（AV1 软件解码），长度补/截到 n。"""
    from av_decode import read_rgb_frames
    return read_rgb_frames(mp4, n)


def cache_videos(rel_paths, out_dir, path2idx):
    """
    解码并缓存一组视频的 N_FRAMES 连续帧到 out_dir。

    rel_paths: 视频相对路径（如 fall/fall_ch_001）
    path2idx:  video path -> NPZ 索引（用于校验视频属于数据集）
    返回 (n_ok, first)：成功写入数，以及首个成功视频 (rel, frames均值)。
    不在 path2idx 或磁盘缺失的视频打印警告并跳过（预期为 0）。
    """
    os.makedirs(out_dir, exist_ok=True)
    n_ok = 0
    first = None  # (rel, mean)
    n_total = len(rel_paths)
    for i, rel in enumerate(rel_paths):
        mp4 = os.path.join(VIDEO_ROOT, rel + ".mp4")
        if rel not in path2idx or not os.path.exists(mp4):
            print(f"  [警告] 跳过 {rel}: 不在 NPZ 索引或视频缺失 ({mp4})")
            continue
        frames = decode_frames(mp4)
        imgs = np.stack([cv2.resize(f, SIZE, interpolation=cv2.INTER_LINEAR)
                         for f in frames]).astype(np.uint8)  # (80,224,224,3)
        torch.save({"frames": imgs, "video": rel},
                   os.path.join(out_dir, f"{n_ok:05d}_{rel.replace('/', '_')}.pt"))
        if n_ok == 0:
            first = (rel, float(imgs.mean()))
        n_ok += 1
        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{n_total}] 已存 {n_ok} 视频")
    return n_ok, first


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_videos", type=int, default=2000)
    ap.add_argument("--n_val", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # 视频索引（仅用 video_paths 做归属校验；标签由训练侧按 path 从 NPZ 对齐）
    d = np.load(NPZ, allow_pickle=True, mmap_mode="r")
    vp = [str(p) for p in d["video_paths"]]
    path2idx = {p: i for i, p in enumerate(vp)}
    print(f"NPZ 视频索引: {len(path2idx)} 条")

    # ---- train：与 E3 同 seed 打散 → 同批同序视频 ----
    train_paths = list(pd.read_csv(TRAIN_CSV)["path"].str.strip())
    rng = np.random.default_rng(args.seed)   # 与 E3 同 seed → 同批视频
    rng.shuffle(train_paths)
    selected = train_paths[: args.n_videos]
    print(f"训练候选: {len(selected)}（train.csv {len(train_paths)} 条, seed={args.seed}）")

    n_ok, first = cache_videos(selected, OUT, path2idx)
    if first:
        print(f"首训练视频 {first[0]}: 帧均值 {first[1]:.1f}（>20 为真实解码帧）")
    print(f"完成：训练 {n_ok} 视频 × {N_FRAMES} 帧 → {OUT}")

    # ---- val：fresh rng(seed) 独立打散 → 固定 val 子集供训练监控 ----
    val_paths = list(pd.read_csv(VAL_CSV)["path"].str.strip())
    rng_val = np.random.default_rng(args.seed)   # fresh，独立于上方训练打散
    rng_val.shuffle(val_paths)
    val_selected = val_paths[: args.n_val]
    print(f"验证候选: {len(val_selected)}（val.csv {len(val_paths)} 条, fresh rng seed={args.seed}）")

    n_val_ok, first_val = cache_videos(val_selected, OUT_VAL, path2idx)
    if first_val:
        print(f"首验证视频 {first_val[0]}: 帧均值 {first_val[1]:.1f}（>20 为真实解码帧）")
    print(f"完成：验证 {n_val_ok} 视频 × {N_FRAMES} 帧 → {OUT_VAL}")


if __name__ == "__main__":
    main()
