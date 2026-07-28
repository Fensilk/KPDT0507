#!/bin/bash
# ================================================================
# Phase 6: Ternary Classification + Temporal Decoder Comparison
#
# Bridge experiment (1) + 3 decoders x 3 gamma values (9) = 10 experiments
#
# Config:
#   Feature: DINOv2 ViT-g (1536d)
#   Window:  T=64, stride=8
#   Model:   hidden_dim=384, dropout=0.3, PE=sinusoidal
#   Train:   AdamW lr=1e-3, batch_size=32, epochs=100, patience=15
#
# Usage:
#   bash experiments/run_phase6.sh bridge   # Bridge only
#   bash experiments/run_phase6.sh ct       # Causal Transformer only
#   bash experiments/run_phase6.sh lstm     # LSTM only
#   bash experiments/run_phase6.sh mamba    # Mamba only
#   bash experiments/run_phase6.sh all      # All 10 experiments
# ================================================================

set -e
cd "$(dirname "$0")/.."

SHARED_ARGS="--window_size 64 --stride 8 --epochs 100 --patience 15 \
--frame_npz data/omnifall_dinov2_giant_frame.npz \
--hidden_dim 384 --dropout 0.3 --batch_size 32 \
--num_workers 4 --seed 42 --fast"

echo "=============================================="
echo "Phase 6: Ternary Classification"
echo "Decoder: Causal Transformer | LSTM | Mamba"
echo "Loss: Focal Loss (gamma=2,3,5)"
echo "=============================================="
echo ""
echo "Shared config:"
echo "  Feature: DINOv2 ViT-g (1536d)"
echo "  Window:  T=64, stride=8"
echo "  Model:   hidden_dim=384, dropout=0.3"
echo "  Batch:   32, lr=1e-3, epochs=100"
echo ""

# Check mamba_ssm availability
python -c "from mamba_ssm import Mamba; print('mamba-ssm: available')" 2>/dev/null || \
    echo "[WARN] mamba_ssm not installed. Mamba experiments will use SimplifiedMambaBlock (pure PyTorch fallback)."
echo ""

# ============================================================
# Bridge Experiment
# Bidirectional Transformer + CE (no Focal Loss)
# ============================================================
run_bridge() {
    echo "=============================================="
    echo ">>> [bridge] p6_bridge: Bidirectional Transformer + CE"
    echo "=============================================="
    python experiments/train_phase6.py \
        --exp_tag phase6/p6_bridge \
        --decoder_type transformer --bridge \
        $SHARED_ARGS
}

# ============================================================
# A. Causal Transformer x gamma in {2, 3, 5}
# ============================================================
run_ct() {
    for gamma in 2 3 5; do
        echo ""
        echo "=============================================="
        echo ">>> [ct/g${gamma}] p6a_ct_g${gamma}: Causal Transformer + Focal(gamma=${gamma})"
        echo "=============================================="
        python experiments/train_phase6.py \
            --exp_tag "phase6/p6a_ct_g${gamma}" \
            --decoder_type transformer --num_layers 1 \
            --focal_gamma ${gamma} \
            $SHARED_ARGS
    done
}

# ============================================================
# B. LSTM x gamma in {2, 3, 5}
# ============================================================
run_lstm() {
    for gamma in 2 3 5; do
        echo ""
        echo "=============================================="
        echo ">>> [lstm/g${gamma}] p6b_lstm_g${gamma}: LSTM 2L + Focal(gamma=${gamma})"
        echo "=============================================="
        python experiments/train_phase6.py \
            --exp_tag "phase6/p6b_lstm_g${gamma}" \
            --decoder_type lstm --num_layers 2 \
            --focal_gamma ${gamma} \
            $SHARED_ARGS
    done
}

# ============================================================
# C. Mamba x gamma in {2, 3, 5}
# ============================================================
run_mamba() {
    for gamma in 2 3 5; do
        echo ""
        echo "=============================================="
        echo ">>> [mamba/g${gamma}] p6c_mamba_g${gamma}: Mamba 2L + Focal(gamma=${gamma})"
        echo "=============================================="
        python experiments/train_phase6.py \
            --exp_tag "phase6/p6c_mamba_g${gamma}" \
            --decoder_type mamba --num_layers 2 \
            --focal_gamma ${gamma} \
            $SHARED_ARGS
    done
}

# ============================================================
# Dispatch
# ============================================================
case "${1:-all}" in
    bridge)
        run_bridge
        ;;
    ct|transformer)
        run_ct
        ;;
    lstm)
        run_lstm
        ;;
    mamba)
        run_mamba
        ;;
    all)
        echo "Running all 10 experiments in priority order:"
        echo "  [1/10] Bridge"
        echo "  [2-4/10] Causal Transformer (gamma=2,3,5)"
        echo "  [5-7/10] Mamba (gamma=2,3,5)"
        echo "  [8-10/10] LSTM (gamma=2,3,5)"
        echo ""
        run_bridge
        run_ct
        run_mamba
        run_lstm
        ;;
    *)
        echo "Usage: bash experiments/run_phase6.sh {bridge|ct|lstm|mamba|all}"
        echo ""
        echo "  bridge  — Bidirectional Transformer + CE (1 experiment)"
        echo "  ct      — Causal Transformer x gamma=2,3,5 (3 experiments)"
        echo "  lstm    — LSTM 2L x gamma=2,3,5 (3 experiments)"
        echo "  mamba   — Mamba 2L x gamma=2,3,5 (3 experiments)"
        echo "  all     — All 10 experiments"
        exit 1
        ;;
esac

echo ""
echo "=============================================="
echo "Phase 6 complete!"
echo "Logs: logs/phase6/"
echo ""
echo "Experiment matrix:"
echo "  p6_bridge       Bidirectional Transformer + CE"
echo "  p6a_ct_g{2,3,5}  Causal Transformer + Focal"
echo "  p6b_lstm_g{2,3,5} LSTM 2L + Focal"
echo "  p6c_mamba_g{2,3,5} Mamba 2L + Focal"
echo "=============================================="
