#!/bin/bash
# ================================================================
# Phase 5b: 模块消融 — 特征提取器 × 时序模型 贡献拆分
# DINOv2+TCN 补全 Phase 4 缺失的消融实验
#
# 2×3 矩阵:
#               TCN (T=32,h=256)    Transformer (T=32,h=384)
#   ResNet18    [5b/1] p5b_rn18     p4a_rn18_baseline (已跑)
#   DINOv2 L    [5b/2] p5b_dino     p4a_dino_baseline (已跑)
#   DINOv2 g    [5b/3] p5b_dinog    p4d_vitg          (已跑)
#
# 用法: bash experiments/run_phase5_ablation.sh
# ================================================================

set -e
cd "$(dirname "$0")/.."

SHARED_ARGS="--window_size 32 --stride 8 --epochs 100 --patience 15 \
--hidden_dim 256 --batch_size 32 --num_workers 4 --seed 42 --fast"

echo "=============================================="
echo "Phase 5b: Module Ablation"
echo "Feature Extractor x Temporal Model"
echo "=============================================="

# --------------------------------------------------
# Experiment 1: ResNet18 + TCN (T=32)
# Baseline cell: weakest feature + convolutional temporal model
# --------------------------------------------------
echo ""
echo ">>> [5b/1] p5b_rn18_tcn: ResNet18 (512) + TCN"
python experiments/train_tcn.py \
    --exp_tag phase5/ablation/p5b_rn18_tcn \
    --frame_npz data/omnifall_frame_preprocessed.npz \
    $SHARED_ARGS

# --------------------------------------------------
# Experiment 2: DINOv2 ViT-L + TCN (T=32)
# Core ablation: strong features with convolutional temporal model
# Answers: how much does Transformer self-attention matter?
# --------------------------------------------------
echo ""
echo ">>> [5b/2] p5b_dino_tcn: DINOv2 ViT-L (1024) + TCN"
python experiments/train_tcn.py \
    --exp_tag phase5/ablation/p5b_dino_tcn \
    --frame_npz data/omnifall_dinov2_frame.npz \
    $SHARED_ARGS

# --------------------------------------------------
# Experiment 3: DINOv2 ViT-g + TCN (T=32)
# Optional cell: best features with convolutional temporal model
# Answers: does ViT-g's fallen boost require Transformer attention?
# --------------------------------------------------
echo ""
echo ">>> [5b/3] p5b_dinog_tcn: DINOv2 ViT-g (1536) + TCN"
python experiments/train_tcn.py \
    --exp_tag phase5/ablation/p5b_dinog_tcn \
    --frame_npz data/omnifall_dinov2_giant_frame.npz \
    $SHARED_ARGS

echo ""
echo "=============================================="
echo "Phase 5b complete!"
echo "Logs: logs/phase5/ablation/"
echo ""
echo "Full 2x3 matrix:"
echo "                     TCN                  Transformer"
echo "  ResNet18 (512)     p5b_rn18_tcn         p4a_rn18 (0.545)"
echo "  DINOv2 L (1024)    p5b_dino_tcn         p4a_dino (0.660)"
echo "  DINOv2 g (1536)    p5b_dinog_tcn        p4d_vitg (0.657)"
echo "=============================================="
