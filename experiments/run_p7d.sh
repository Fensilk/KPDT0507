#!/bin/bash
# ============================================================
# P7d: Δframe — ViT-g + frame-to-frame diff features
#
# Same as p6_bridge, only difference: input = ViT-g + Δframe (3072d)
# ============================================================
set -e
cd "$(dirname "$0")/.."
echo "[P7d] Δframe | Working directory: $(pwd)"

python experiments/train_phase6.py \
    --bridge \
    --use_diff \
    --window_size 64 --stride 8 \
    --exp_tag phase7/p7d_delta \
    --epochs 100 --patience 15 \
    --batch_size 32 --lr 1e-3 \
    --seed 42

echo ""
echo "[P7d] Done. Results: logs/phase7/p7d_delta/"
