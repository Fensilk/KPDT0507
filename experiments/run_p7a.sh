#!/bin/bash
# ============================================================
# P7a: Event-level causality — Cross-window voting
# Run on AutoDL server after uploading the packaged files.
#
# Prerequisites:
#   - Python 3.10+, PyTorch with CUDA
#   - p6_bridge checkpoint at logs/phase6/p6_bridge/best_model.pt
#   - DINOv2-giant NPZ at data/omnifall_dinov2_giant_frame.npz
#   - Splits at DATASET-omnifall/splits/syn/random/
#
# Usage:
#   bash experiments/run_p7a.sh
#   bash experiments/run_p7a.sh --vote_threshold 3
# ============================================================

set -e

cd "$(dirname "$0")/.."
echo "[P7a] Working directory: $(pwd)"

# ── Default args ──
VOTE_THRESHOLD=2
DEVICE="auto"

# ── Parse extra args ──
while [[ $# -gt 0 ]]; do
    case $1 in
        --vote_threshold) VOTE_THRESHOLD="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "[P7a] Vote threshold: ${VOTE_THRESHOLD}/3"
echo "[P7a] Device: ${DEVICE}"

# ── Run ──
python experiments/run_p7a_event_vote.py \
    --exp_dir logs/phase6/p6_bridge \
    --npz_path data/omnifall_dinov2_giant_frame.npz \
    --splits_dir DATASET-omnifall/splits/syn/random \
    --output_dir logs/phase7/p7a_event_vote \
    --vote_threshold ${VOTE_THRESHOLD} \
    --window_size 64 \
    --stride 8 \
    --batch_size 32 \
    --device ${DEVICE}

echo ""
echo "[P7a] Done. Results saved to logs/phase7/p7a_event_vote/"
echo "[P7a] Key file: logs/phase7/p7a_event_vote/test_results.json"
