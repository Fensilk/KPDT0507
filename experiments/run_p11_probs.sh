#!/bin/bash
# 用**新版 eval**（带逐帧概率 dump）重跑已有 ckpt，供 prec-rec 前沿曲线比较。
#
# 为什么必须重跑：旧 pv 只存 argmax 标签，**概率无法事后补**。而前沿曲线比较需要连续分数。
#
# 覆盖三个已有臂的 best ckpt（全量 1200，每臂约 53 min，合计约 2.65 h）：
#   A0     恒定 lr 锚点
#   1a     boundary head λ=0.3（被硬否决的那一臂）
#   a0cos  A0 + 余弦退火（诊断臂）
#
# 产物：各臂目录下 pv_best_probs.pt（含 probs 字段）。**不覆盖** pv_best.pt，
#       便于与旧结果对照。
#
# 用法（autodl，/root/autodl-tmp 内）：
#   bash experiments/run_p11_probs.sh
set -u
cd /root/autodl-tmp || exit 1
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1
PY=/root/miniconda3/bin/python
LOG=logs/phase11/prec_rec_reeval.log

echo "===== 前沿曲线重评启动 $(date) =====" >> "$LOG"
for arm in a0_2000 w1_1a w1_a0cos; do
  CK=logs/phase11/$arm/best_model.pt
  [ -f "$CK" ] || { echo "== $arm: ckpt 缺失 $CK，跳过 ==" >> "$LOG"; continue; }
  echo "== $arm start $(date) ==" >> "$LOG"
  $PY -u experiments/eval_p9e1_dual.py \
      --ckpt "$CK" \
      --out "logs/phase11/$arm/eval_best_probs.json" \
      --dump_pv "logs/phase11/$arm/pv_best_probs.pt" >> "$LOG" 2>&1 < /dev/null
  echo "== $arm done rc=$? $(date) ==" >> "$LOG"
done
echo "ALL_DONE_PROBS $(date)" >> "$LOG"
