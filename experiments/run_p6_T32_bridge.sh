#!/bin/bash
# ============================================================
# P6 T=32 Bridge — 消融窗口大小对过拟合的影响
#
# 与 p6_bridge (T=64) 唯一差异: T=32
# 其余完全相同: bidirectional Transformer 1L/384d, CE loss
#
# 数据变化:
#   T=64: (80-64)//8+1 = 3 窗口/视频, train ~29K 窗口
#   T=32: (80-32)//8+1 = 7 窗口/视频, train ~67K 窗口 (2.3x)
# ============================================================

set -e
cd "$(dirname "$0")/.."
echo "[P6-T32] Working directory: $(pwd)"

python experiments/train_phase6.py \
    --bridge \
    --window_size 32 \
    --stride 8 \
    --exp_tag phase6/p6_T32_bridge \
    --epochs 100 \
    --patience 15 \
    --batch_size 32 \
    --lr 1e-3 \
    --seed 42

echo ""
echo "[P6-T32] Done. Results: logs/phase6/p6_T32_bridge/"
