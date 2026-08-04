#!/bin/bash
# ============================================================
# P7b: Event-gated ternary detection
#
# Architecture: Bridge backbone + binary event gate head (385 params)
# Training: L = CE(ternary_mod) + lambda × BCE(event_score, event_gt)
#
# Usage:
#   bash experiments/run_p7b.sh                          # default λ=0.5
#   bash experiments/run_p7b.sh --event_lambda 0.3       # weaker event loss
#   bash experiments/run_p7b.sh --event_lambda 1.0       # stronger event loss
# ============================================================

set -e
cd "$(dirname "$0")/.."
echo "[P7b] Working directory: $(pwd)"

EVENT_LAMBDA=0.5
EXTRA=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --event_lambda) EVENT_LAMBDA="$2"; shift 2 ;;
        *) EXTRA="$EXTRA $1"; shift ;;
    esac
done

echo "[P7b] Event lambda: ${EVENT_LAMBDA}"

python experiments/train_p7b.py \
    --exp_tag phase7/p7b_event_gate \
    --event_lambda ${EVENT_LAMBDA} \
    --window_size 64 \
    --stride 8 \
    --epochs 100 \
    --patience 15 \
    --batch_size 32 \
    --lr 1e-3 \
    --seed 42 \
    ${EXTRA}

echo ""
echo "[P7b] Done. Results: logs/phase7/p7b_event_gate/"
