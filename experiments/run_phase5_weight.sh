#!/bin/bash
# ================================================================
# Phase 5a: 损失函数权重调优
# DINOv2 + Transformer, 6 组单变量对照实验
#
# 基线: p4a_dino_baseline (w_cls=1.0, w_fall=0.5, w_fallen=0.5)
#        fall_f1=0.660, fallen_f1=0.591, avg_f1=0.626
#
# 用法: bash experiments/run_phase5_weight.sh
# ================================================================

set -e
cd "$(dirname "$0")/.."

SHARED_ARGS="--window_size 32 --stride 8 --epochs 100 --patience 15 \
--frame_npz data/omnifall_dinov2_frame.npz \
--hidden_dim 384 --num_layers 1 --num_heads 6 --dropout 0.3 \
--num_workers 4 --seed 42 --fast"

echo "=============================================="
echo "Phase 5a: Loss Weight Tuning"
echo "6 single-variable experiments"
echo "=============================================="

# --------------------------------------------------
# Experiment 1: cls_down (w_cls=0.3)
# Hypothesis: reduce cls dominance → release fall/fallen gradient
# --------------------------------------------------
echo ""
echo ">>> [5a/1] cls_down: w_cls=0.3, w_fall=0.5, w_fallen=0.5"
python experiments/train_transformer.py \
    --exp_tag phase5/weight/cls_down \
    --w_cls 0.3 --w_fall 0.5 --w_fallen 0.5 \
    $SHARED_ARGS

# --------------------------------------------------
# Experiment 2: cls_up (w_cls=2.0)
# Hypothesis: cls signal underweighted → boost classification
# --------------------------------------------------
echo ""
echo ">>> [5a/2] cls_up: w_cls=2.0, w_fall=0.5, w_fallen=0.5"
python experiments/train_transformer.py \
    --exp_tag phase5/weight/cls_up \
    --w_cls 2.0 --w_fall 0.5 --w_fallen 0.5 \
    $SHARED_ARGS

# --------------------------------------------------
# Experiment 3: fall_down (w_fall=0.2)
# Hypothesis: suppress fall false positives → Precision↑
# --------------------------------------------------
echo ""
echo ">>> [5a/3] fall_down: w_cls=1.0, w_fall=0.2, w_fallen=0.5"
python experiments/train_transformer.py \
    --exp_tag phase5/weight/fall_down \
    --w_cls 1.0 --w_fall 0.2 --w_fallen 0.5 \
    $SHARED_ARGS

# --------------------------------------------------
# Experiment 4: fall_up (w_fall=1.0)
# Hypothesis: boost fall recall → Recall↑
# --------------------------------------------------
echo ""
echo ">>> [5a/4] fall_up: w_cls=1.0, w_fall=1.0, w_fallen=0.5"
python experiments/train_transformer.py \
    --exp_tag phase5/weight/fall_up \
    --w_cls 1.0 --w_fall 1.0 --w_fallen 0.5 \
    $SHARED_ARGS

# --------------------------------------------------
# Experiment 5: fallen_down (w_fallen=0.2)
# Hypothesis: suppress fallen false positives → Precision↑
# --------------------------------------------------
echo ""
echo ">>> [5a/5] fallen_down: w_cls=1.0, w_fall=0.5, w_fallen=0.2"
python experiments/train_transformer.py \
    --exp_tag phase5/weight/fallen_down \
    --w_cls 1.0 --w_fall 0.5 --w_fallen 0.2 \
    $SHARED_ARGS

# --------------------------------------------------
# Experiment 6: fallen_up (w_fallen=1.0)
# Hypothesis: boost fallen recall → Recall↑
# --------------------------------------------------
echo ""
echo ">>> [5a/6] fallen_up: w_cls=1.0, w_fall=0.5, w_fallen=1.0"
python experiments/train_transformer.py \
    --exp_tag phase5/weight/fallen_up \
    --w_cls 1.0 --w_fall 0.5 --w_fallen 1.0 \
    $SHARED_ARGS

echo ""
echo "=============================================="
echo "Phase 5a complete!"
echo "Logs: logs/phase5/weight/"
echo ""
echo "Baseline (p4a_dino_baseline): fall_f1=0.660"
echo "Success threshold:           fall_f1 >= 0.680 (+3%)"
echo "=============================================="
