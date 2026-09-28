#!/bin/bash
# Phase 11 无人值守链：① 补 A0 的 epoch 子集评测 → ② 启动 W1 第一个臂 Exp-1a（λ=0.3）。
#
# 为什么要串：GPU 只有一块，两件事不能并行；且 ① 是 ② 之后做轨迹比较的前提
# （A0 的 epoch_* 原跑的是全量 1200，而各臂将跑 300 随机子集，不可直接比）。
#
# 预估：① 3 × ~13min ≈ 40min；② 训练 ~8.8h + 评测(best 全量 53min + 3 子集 ~39min) ≈ 10.5h
#      合计 ≈ 11.2h
#
# 终止预案：本链只是"顺序驱动"，各步失败不阻塞后续（② 有自己的 watchdog 与 ALL_DONE）。
#   若 ① 某步失败 → 只影响轨迹基准对齐，② 照常跑；事后单独补该步即可。
#   若 ② 崩溃 → 按其自身 run.log 的 verdict 处理（resume 用法见 run_p11_w1.sh 头注释）。
set -u
cd /root/autodl-tmp || exit 1
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1
PY=/root/miniconda3/bin/python
A0=logs/phase11/a0_2000
CHAIN=$A0/chain_0929.log

echo "===== 链启动 $(date) =====" >> "$CHAIN"

echo "--- ① A0 epoch 子集评测开始 $(date) ---" >> "$CHAIN"
for f in epoch_1 epoch_2 epoch_3; do
  echo "== A0 $f subset-300 start $(date) ==" >> "$CHAIN"
  $PY -u experiments/eval_p9e1_dual.py \
      --ckpt "$A0/${f}_model.pt" \
      --out "$A0/eval_${f}_model.subset.json" \
      --limit_videos 300 --subset_seed 42 >> "$CHAIN" 2>&1 < /dev/null
  echo "== A0 $f subset-300 done rc=$? $(date) ==" >> "$CHAIN"
done
echo "--- ① 完成 $(date) ---" >> "$CHAIN"
echo "A0_SUBSET_DONE $(date)" >> "$CHAIN"

echo "--- ② 启动 W1 臂 1a $(date) ---" >> "$CHAIN"
bash experiments/run_p11_w1.sh 1a >> "$CHAIN" 2>&1
echo "CHAIN_LAUNCHED_1a $(date)" >> "$CHAIN"
