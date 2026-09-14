#!/bin/bash
# ============================================================
# P9 数据侧实验：剔除人工确认的"全程保持躺着"无增益视频
#
# 与 baseline p7d_delta 完全同配置，仅 --splits_dir 换成过滤集：
#   train 9600 -> 8518 | val 1200 -> 1079 | test 1200 -> 1046
# 过滤逻辑见 analysis/build_filtered_splits.py（人工审查结论）
# ============================================================
set -e
cd "$(dirname "$0")/.."
echo "[P9 no_keeplying] | Working directory: $(pwd)"

python experiments/train_phase6.py \
    --bridge \
    --use_diff \
    --window_size 64 --stride 8 \
    --splits_dir DATASET-omnifall/splits/syn/no_keeplying \
    --exp_tag phase9/no_keeplying \
    --epochs 100 --patience 15 \
    --batch_size 32 --lr 1e-3 \
    --seed 42

echo ""
echo "[P9] Done. Results: logs/phase9/no_keeplying/"
echo "对比 baseline: logs/phase7/p7d_delta/ (fall_rec 0.792 / fallen_rec 0.698 / fallen_prec 0.627)"
