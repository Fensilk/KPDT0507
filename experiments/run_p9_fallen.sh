#!/bin/bash
# ============================================================
# P9b 变体：只剔除"全程保持躺卧 AND 原标注含 fallen"的视频（141 个）
#
# 修正 P9 的反面结果：P9 剔了全部 keep_lying（连"地面=normal"先验也删了），
# 模型更激进判 fallen。本变体只剔"地面=fallen"矛盾侧（98% 是 fallen 帧），
# 保留 causeless_lying 的 normal 先验。
#
# 与 baseline p7d_delta 完全同配置，仅换 --splits_dir：
#   train 9600 -> 9483 | val 1200 -> 1187 | test 1200 -> 1189
# ============================================================
set -e
cd "$(dirname "$0")/.."
echo "[P9b no_keeplying_fallen] | Working directory: $(pwd)"

python experiments/train_phase6.py \
    --bridge \
    --use_diff \
    --window_size 64 --stride 8 \
    --splits_dir DATASET-omnifall/splits/syn/no_keeplying_fallen \
    --exp_tag phase9/no_keeplying_fallen \
    --epochs 100 --patience 15 \
    --batch_size 32 --lr 1e-3 \
    --seed 42

echo ""
echo "[P9b] Done. Results: logs/phase9/no_keeplying_fallen/"
echo "对比: baseline logs/phase7/p7d_delta/  |  上一变体 logs/phase9/no_keeplying/"
