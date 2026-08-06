#!/bin/bash
# ================================================================
# Phase 6: Ternary Classification + Temporal Decoder Comparison
#
# Original causal experiments (10): bridge + 3 decoders x 3 gamma
# Bidirectional rerun (6): Bi-Transformer x 3 gamma + Bi-LSTM x 3 gamma
#
# Config:
#   Feature: DINOv2 ViT-g (1536d)
#   Window:  T=64, stride=8
#   Model:   hidden_dim=384, dropout=0.3, PE=sinusoidal
#   Train:   AdamW lr=1e-3, batch_size=32, epochs=100, patience=15
#
# Usage:
#   bash experiments/run_phase6.sh bridge       # Bridge only
#   bash experiments/run_phase6.sh ct            # Causal Transformer only
#   bash experiments/run_phase6.sh lstm          # Causal LSTM only
#   bash experiments/run_phase6.sh mamba         # Causal Mamba only
#   bash experiments/run_phase6.sh bidir_trans   # Bidirectional Transformer
#   bash experiments/run_phase6.sh bilstm        # Bi-LSTM
#   bash experiments/run_phase6.sh bidir_all     # Bidirectional: trans + lstm (6)
#   bash experiments/run_phase6.sh causal_all    # Original causal: all 10
#   bash experiments/run_phase6.sh all           # Everything (bridge + 6 bidir + 9 causal)
# ================================================================

set -e
cd "$(dirname "$0")/.."

SHARED_ARGS="--window_size 64 --stride 8 --epochs 100 --patience 15 \
--frame_npz data/omnifall_dinov2_giant_frame.npz \
--hidden_dim 384 --dropout 0.3 --batch_size 32 \
--num_workers 4 --seed 42 --fast"

echo "=============================================="
echo "Phase 6: Ternary Classification"
echo "Decoder: Transformer | Bi-LSTM (bidirectional)"
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
# Bridge Experiment (Bidirectional Transformer + CE, already run)
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
# B. Causal LSTM x gamma in {2, 3, 5}
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
# C. Causal Mamba x gamma in {2, 3, 5}
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
# D. Bidirectional Transformer x gamma in {2, 3, 5}  [NEW]
#    --no_causal = bidirectional attention (no future mask)
# ============================================================
run_bidir_trans() {
    for gamma in 2 3 5; do
        echo ""
        echo "=============================================="
        echo ">>> [bidir_trans/g${gamma}] p6d_bt_g${gamma}: Bidirectional Transformer + Focal(gamma=${gamma})"
        echo "=============================================="
        python experiments/train_phase6.py \
            --exp_tag "phase6/p6d_bt_g${gamma}" \
            --decoder_type transformer --num_layers 1 \
            --no_causal --focal_gamma ${gamma} \
            $SHARED_ARGS
    done
}

# ============================================================
# E. Bi-LSTM x gamma in {2, 3, 5}  [NEW]
#    --no_causal + --bidirectional = Bi-LSTM
# ============================================================
run_bilstm() {
    for gamma in 2 3 5; do
        echo ""
        echo "=============================================="
        echo ">>> [bilstm/g${gamma}] p6e_bilstm_g${gamma}: Bi-LSTM 2L + Focal(gamma=${gamma})"
        echo "=============================================="
        python experiments/train_phase6.py \
            --exp_tag "phase6/p6e_bilstm_g${gamma}" \
            --decoder_type lstm --num_layers 2 \
            --no_causal --bidirectional --focal_gamma ${gamma} \
            $SHARED_ARGS
    done
}

# ============================================================
# Dispatch
# ============================================================
case "${1:-bidir_all}" in
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
    bidir_trans)
        run_bidir_trans
        ;;
    bilstm)
        run_bilstm
        ;;
    causal_all)
        echo "Running all 10 causal experiments in priority order:"
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
    bidir_all)
        echo "Running all 6 bidirectional experiments:"
        echo "  [1-3/6] Bidirectional Transformer (gamma=2,3,5)"
        echo "  [4-6/6] Bi-LSTM (gamma=2,3,5)"
        echo ""
        run_bidir_trans
        run_bilstm
        ;;
    all)
        echo "Running everything (bridge + 6 bidir + 9 causal = 16 experiments):"
        echo "  [1/16] Bridge"
        echo "  [2-4/16] Bidirectional Transformer (gamma=2,3,5)"
        echo "  [5-7/16] Bi-LSTM (gamma=2,3,5)"
        echo "  [8-10/16] Causal Transformer (gamma=2,3,5)"
        echo "  [11-13/16] Mamba (gamma=2,3,5)"
        echo "  [14-16/16] Causal LSTM (gamma=2,3,5)"
        echo ""
        run_bridge
        run_bidir_trans
        run_bilstm
        run_ct
        run_mamba
        run_lstm
        ;;
    *)
        echo "Usage: bash experiments/run_phase6.sh {bridge|ct|lstm|mamba|bidir_trans|bilstm|bidir_all|causal_all|all}"
        echo ""
        echo "  bridge       — Bidirectional Transformer + CE (1 experiment)"
        echo "  ct           — Causal Transformer x gamma=2,3,5 (3 experiments)"
        echo "  lstm         — Causal LSTM 2L x gamma=2,3,5 (3 experiments)"
        echo "  mamba        — Causal Mamba 2L x gamma=2,3,5 (3 experiments)"
        echo "  bidir_trans  — Bidirectional Transformer x gamma=2,3,5 (3 experiments)  [NEW]"
        echo "  bilstm       — Bi-LSTM 2L x gamma=2,3,5 (3 experiments)  [NEW]"
        echo "  bidir_all    — bidir_trans + bilstm (6 experiments, default)  [NEW]"
        echo "  causal_all   — Original causal: bridge + ct + mamba + lstm (10 experiments)"
        echo "  all          — Everything (16 experiments)"
        exit 1
        ;;
esac

echo ""
echo "=============================================="
echo "Phase 6 complete!"
echo "Logs: logs/phase6/"
echo ""
echo "Experiment matrix:"
echo "  p6_bridge          Bidirectional Transformer + CE"
echo "  p6d_bt_g{2,3,5}    Bidirectional Transformer + Focal  [NEW]"
echo "  p6e_bilstm_g{2,3,5} Bi-LSTM 2L + Focal  [NEW]"
echo "  p6a_ct_g{2,3,5}    Causal Transformer + Focal"
echo "  p6b_lstm_g{2,3,5}  Causal LSTM 2L + Focal"
echo "=============================================="
