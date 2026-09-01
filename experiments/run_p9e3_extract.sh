#!/bin/bash
# E3 特征重抽：用微调后 ViT-g 重抽全量 12,000 视频特征 → 新 NPZ
#（全量请在 autodl 3090 上跑；本地跑通用 --limit 100）
set -e
cd "$(dirname "$0")/.."
echo "[E3 重抽] 用微调 ViT-g 重抽特征 | $(pwd)"

python experiments/extract_p9_features.py

echo ""
echo "[E3 重抽] Done. NPZ: data/omnifall_dinov2_finetuned_frame.npz"
echo "下一步: python experiments/train_phase6.py --frame_npz data/omnifall_dinov2_finetuned_frame.npz ..."
