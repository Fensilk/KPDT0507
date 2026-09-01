"""
E3 前置一致性检查：本地加载 facebook/dinov2-giant，用与 step7 完全相同的预处理
抽 CLS 特征，与现有 omnifall_dinov2_giant_frame.npz 逐帧对比（余弦相似度）。

若余弦 ≈ 1.0，说明本地复现 == 当年抽特征的版本，Stage 2 的"只有特征不同"
对比才成立；若不匹配（预处理/模型版本差异），需先解决再微调。

用法（项目根目录，py310）：
    python experiments/check_p9_consistency.py [--n 3]
"""
import os
import sys
import argparse
import numpy as np
import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "preprocessing"))

from step7_extract_dinov2 import (  # noqa: E402
    extract_dinov2_features, DINOV2_MEAN, DINOV2_STD, DINOV2_SIZE, N_FRAMES,
)

NPZ_PATH = os.path.join(BASE, "data", "omnifall_dinov2_giant_frame.npz")
VIDEO_ROOT = os.path.join(BASE, "DATASET-omnifall", "data_files", "extracted")

# 跨类别抽几个样例视频（存在性由脚本检查跳过）
SAMPLE_VIDEOS = [
    "fall/fall_ch_001",
    "fallen/fallen_ch_116",
    "lying/lying_ch_012",
    "lie_down/lie_down_ch_019",
    "stand_up/stand_up_ch_042",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3, help="检查视频数")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from transformers import AutoModel
    print(f"[1] 加载 facebook/dinov2-giant → {args.device} ...")
    model = AutoModel.from_pretrained("facebook/dinov2-giant").to(args.device)
    model = model.half().eval()

    print(f"[2] 加载 NPZ 特征 ...")
    d = np.load(NPZ_PATH, allow_pickle=True, mmap_mode="r")
    vp = d["video_paths"]; vsi = d["video_start_indices"]; feats = d["features"]
    path2idx = {str(p): i for i, p in enumerate(vp)}

    print(f"[3] 逐视频对比（前 {args.n} 个可用的样例）...")
    checked = 0
    for rel in SAMPLE_VIDEOS:
        if checked >= args.n:
            break
        mp4 = os.path.join(VIDEO_ROOT, rel + ".mp4")
        if not os.path.exists(mp4) or rel not in path2idx:
            print(f"  [skip] {rel}")
            continue
        vi = path2idx[rel]
        start = int(vsi[vi])
        npz_feat = feats[start:start + N_FRAMES]  # (80, 1536) fp16

        ext_feat = extract_dinov2_features(mp4, model, args.device)  # (80, 1536) fp16

        # 逐帧余弦相似度
        a = ext_feat.astype(np.float32); b = npz_feat.astype(np.float32)
        cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)
        mae = np.abs(a - b).mean()
        print(f"  {rel:28s} 余弦 min={cos.min():.4f} mean={cos.mean():.4f}  MAE={mae:.4f}")
        checked += 1

    print("\n判定：mean 余弦 > 0.99 → 复现一致，可进入微调。")
    if checked == 0:
        print("  ⚠ 没有可用的样例视频，检查 VIDEO_ROOT 与 SAMPLE_VIDEOS。")


if __name__ == "__main__":
    main()
