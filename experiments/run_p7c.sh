#!/bin/bash
# ============================================================
# P7c: Binary Event Detection — Ternary vs Binary capacity ablation
#
# Same as p6_bridge, only difference: 2-class output instead of 3.
# ============================================================
set -e
cd "$(dirname "$0")/.."
echo "[P7c] Working directory: $(pwd)"
echo "[P7c] Binary Event Detection"

python experiments/train_p7c_binary.py \
    --exp_tag phase7/p7c_binary \
    --window_size 64 --stride 8 \
    --epochs 100 --patience 15 \
    --batch_size 32 --lr 1e-3 \
    --seed 42

echo ""
echo "[P7c] Done. Results: logs/phase7/p7c_binary/"
