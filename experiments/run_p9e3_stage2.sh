#!/bin/bash
# E3 Stage2：用微调特征 NPZ 训练时序模型（与 baseline p7d_delta 同配置）
# 前置：extract_p9_features.py 已产出 data/omnifall_dinov2_finetuned_frame.npz
set -e
cd "$(dirname "$0")/.."
echo "[E3 Stage2] 时序训练（微调特征）| $(pwd)"

python experiments/train_phase6.py \
    --bridge \
    --use_diff \
    --window_size 64 --stride 8 \
    --frame_npz data/omnifall_dinov2_finetuned_frame.npz \
    --exp_tag phase9/e3_stage2 \
    --epochs 100 --patience 15 \
    --batch_size 32 --lr 1e-3 \
    --seed 42

echo ""
echo "[E3 Stage2] Done. Results: logs/phase9/e3_stage2/test_results.json"
echo "对比 baseline: logs/phase7/p7d_delta/test_results.json"
