#!/bin/bash
# E3 Stage1：帧级 LoRA 微调 ViT-g（依赖 data/p9_frames/ 帧缓存，先跑 prep_p9_frames.py）
set -e
cd "$(dirname "$0")/.."
echo "[E3 Stage1] 帧级 LoRA 微调 | $(pwd)"

python experiments/train_p9_e3_stage1.py \
    --lr 1e-4 --epochs 5 --batch_size 64 --rank 16

echo ""
echo "[E3 Stage1] Done. Checkpoint: logs/phase9/e3_stage1/finetuned_lora.pt"
echo "下一步: python experiments/extract_p9_features.py"
