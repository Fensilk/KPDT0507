#!/bin/bash
# ================================================================
# Phase 4 Experiment Runner
# DINOv2 + Transformer 跌倒检测实验
#
# 共享配置: T=32, stride=8, 1L/384d/drop0.3
# DINOv2 NPZ: data/omnifall_dinov2_frame.npz (需先运行 step7)
# Pose NPZ:   data/omnifall_pose_frame.npz   (需先运行 step8)
#
# 用法: bash experiments/run_phase4.sh [phase]
#   bash experiments/run_phase4.sh 4a    # 只运行 Phase 4a
#   bash experiments/run_phase4.sh 4b    # 只运行 Phase 4b
#   bash experiments/run_phase4.sh all   # 全部 (4a → 4b → 4c)
# ================================================================

set -e
cd "$(dirname "$0")/.."

SHARED_ARGS="--window_size 32 --stride 8 --epochs 100 --patience 15"

# ============================================================
# Phase 4a: 特征提取器验证 (2 experiments)
# 核心 A/B: ResNet18 vs DINOv2, 相同 Transformer 架构
# ============================================================
run_phase_4a() {
    echo "=============================================="
    echo "Phase 4a: Feature Extractor Comparison"
    echo "=============================================="

    # 4a.1 - ResNet18 基线
    echo ""
    echo ">>> [4a/1] p4a_rn18_baseline: ResNet18 + Transformer"
    python experiments/train_transformer.py \
        --exp_tag phase4/p4a_rn18_baseline \
        --frame_npz data/omnifall_frame_preprocessed.npz \
        --hidden_dim 384 --num_layers 1 --num_heads 6 --dropout 0.3 \
        $SHARED_ARGS

    # 4a.2 - DINOv2 基线
    echo ""
    echo ">>> [4a/2] p4a_dino_baseline: DINOv2 + Transformer"
    python experiments/train_transformer.py \
        --exp_tag phase4/p4a_dino_baseline \
        --frame_npz data/omnifall_dinov2_frame.npz \
        --hidden_dim 384 --num_layers 1 --num_heads 6 --dropout 0.3 \
        $SHARED_ARGS

    echo ""
    echo "[Phase 4a DONE] Compare: p4a_rn18_baseline vs p4a_dino_baseline"
    echo "Success criterion: p4a_dino_baseline fall_f1 >= 0.65"
}

# ============================================================
# Phase 4b: 辅助特征消融 (4 experiments)
# DINOv2 + Δframe / Pose 的增量价值
# ============================================================
run_phase_4b() {
    echo "=============================================="
    echo "Phase 4b: Auxiliary Feature Ablation"
    echo "=============================================="

    # 4b.1 - DINOv2 only (same as 4a.2)
    echo ""
    echo ">>> [4b/1] p4b_dino: DINOv2 only"
    python experiments/train_transformer.py \
        --exp_tag phase4/p4b_dino \
        --frame_npz data/omnifall_dinov2_frame.npz \
        --hidden_dim 384 --num_layers 1 --num_heads 6 --dropout 0.3 \
        $SHARED_ARGS

    # 4b.2 - DINOv2 + Δframe
    echo ""
    echo ">>> [4b/2] p4b_dino_diff: DINOv2 + Δframe diff"
    python experiments/train_transformer.py \
        --exp_tag phase4/p4b_dino_diff \
        --frame_npz data/omnifall_dinov2_frame.npz \
        --use_diff \
        --hidden_dim 384 --num_layers 1 --num_heads 6 --dropout 0.3 \
        $SHARED_ARGS

    # 4b.3 - DINOv2 + Pose
    echo ""
    echo ">>> [4b/3] p4b_dino_pose: DINOv2 + Pose"
    python experiments/train_transformer.py \
        --exp_tag phase4/p4b_dino_pose \
        --frame_npz data/omnifall_dinov2_frame.npz \
        --pose_npz data/omnifall_pose_frame.npz \
        --hidden_dim 384 --num_layers 1 --num_heads 6 --dropout 0.3 \
        $SHARED_ARGS

    # 4b.4 - DINOv2 + Δframe + Pose
    echo ""
    echo ">>> [4b/4] p4b_dino_diff_pose: DINOv2 + Δframe + Pose"
    python experiments/train_transformer.py \
        --exp_tag phase4/p4b_dino_diff_pose \
        --frame_npz data/omnifall_dinov2_frame.npz \
        --use_diff --pose_npz data/omnifall_pose_frame.npz \
        --hidden_dim 384 --num_layers 1 --num_heads 6 --dropout 0.3 \
        $SHARED_ARGS

    echo ""
    echo "[Phase 4b DONE] Decision thresholds (Δfall_f1 vs p4b_dino):"
    echo "  Δ < 0.010 → drop Pose+Diff, 4c uses pure DINOv2"
    echo "  Δ 0.010~0.015 → run 4c with both pure and full configs"
    echo "  Δ > 0.015 → keep Pose+Diff for 4c"
}

# ============================================================
# Phase 4c: Transformer 架构微调 (5 experiments)
# 使用 4b 最优特征配置
# ============================================================
run_phase_4c() {
    echo "=============================================="
    echo "Phase 4c: Transformer Architecture Tuning"
    echo "=============================================="
    echo "NOTE: Replace FEATURE_ARGS below with best config from 4b"
    echo ""

    # Default: assume pure DINOv2 (no diff, no pose) from 4b
    FEATURE_ARGS="--frame_npz data/omnifall_dinov2_frame.npz"

    # 4c.1 - baseline (same as 4b winner)
    echo ">>> [4c/1] p4c_base: 1L/384d/drop0.3"
    python experiments/train_transformer.py \
        --exp_tag phase4/p4c_base \
        $FEATURE_ARGS \
        --hidden_dim 384 --num_layers 1 --num_heads 6 --dropout 0.3 \
        $SHARED_ARGS

    # 4c.2 - wider hidden_dim
    echo ""
    echo ">>> [4c/2] p4c_h768: 1L/768d/drop0.3"
    python experiments/train_transformer.py \
        --exp_tag phase4/p4c_h768 \
        $FEATURE_ARGS \
        --hidden_dim 768 --num_layers 1 --num_heads 8 --dropout 0.3 \
        $SHARED_ARGS

    # 4c.3 - deeper (2 layers)
    echo ""
    echo ">>> [4c/3] p4c_L2: 2L/384d/drop0.3"
    python experiments/train_transformer.py \
        --exp_tag phase4/p4c_L2 \
        $FEATURE_ARGS \
        --hidden_dim 384 --num_layers 2 --num_heads 6 --dropout 0.3 \
        $SHARED_ARGS

    # 4c.4 - wider + deeper
    echo ""
    echo ">>> [4c/4] p4c_L2h768: 2L/768d/drop0.2"
    python experiments/train_transformer.py \
        --exp_tag phase4/p4c_L2h768 \
        $FEATURE_ARGS \
        --hidden_dim 768 --num_layers 2 --num_heads 8 --dropout 0.2 \
        $SHARED_ARGS

    # 4c.5 - reduced regularization
    echo ""
    echo ">>> [4c/5] p4c_drop01: 1L/384d/drop0.1"
    python experiments/train_transformer.py \
        --exp_tag phase4/p4c_drop01 \
        $FEATURE_ARGS \
        --hidden_dim 384 --num_layers 1 --num_heads 6 --dropout 0.1 \
        $SHARED_ARGS

    echo ""
    echo "[Phase 4c DONE] Compare architectures; watch train-val gap"
}

# ============================================================
# Main
# ============================================================
PHASE="${1:-all}"

case "$PHASE" in
    4a) run_phase_4a ;;
    4b) run_phase_4b ;;
    4c) run_phase_4c ;;
    all)
        echo "Running Phase 4a → 4b → 4c"
        echo "Check results between each phase before proceeding."
        echo ""
        run_phase_4a
        echo ""
        read -p "Review 4a results. Continue to 4b? [y/N] " yn
        if [ "$yn" != "y" ] && [ "$yn" != "Y" ]; then
            echo "Stopping after 4a."
            exit 0
        fi
        run_phase_4b
        echo ""
        read -p "Review 4b results. Continue to 4c? [y/N] " yn
        if [ "$yn" != "y" ] && [ "$yn" != "Y" ]; then
            echo "Stopping after 4b."
            exit 0
        fi
        # Update FEATURE_ARGS in 4c based on 4b results
        run_phase_4c
        ;;
    *)
        echo "Usage: bash experiments/run_phase4.sh {4a|4b|4c|all}"
        exit 1
        ;;
esac

echo ""
echo "=============================================="
echo "Phase 4 experiments complete!"
echo "Logs: logs/phase4/"
echo "=============================================="
