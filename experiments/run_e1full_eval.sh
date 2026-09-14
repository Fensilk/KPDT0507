#!/bin/bash
# E1-full 全量 test 双口径评估（顺序跑多候选）。
# 为什么多候选：val 排序不可靠（E1@6000 的 val-best ep3 在 test 上被 ep4 反超），
# 故 best(val-best 0.9125) + epoch_3(段末 0.850) + epoch_2(0.8885) 三个 ckpt 全评。
# 每个 = eval_p9e1_dual 单次推理双口径（instance 对官方线 + merged/RuleV2 对 0.7735）。
# 用法（autodl，/root/autodl-tmp 内）: bash experiments/run_e1full_eval.sh
set -u
cd /root/autodl-tmp || exit 1
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1
OUT=logs/phase9/e1_full
PY=/root/miniconda3/bin/python
LOG=$OUT/eval_all.log
: > "$LOG"
echo "== eval start $(date) ==" >> "$LOG"
for pair in "best_model.pt:eval_best.json" "epoch_3_model.pt:eval_ep3.json" "epoch_2_model.pt:eval_ep2.json"; do
  ckpt=${pair%%:*}; out=${pair##*:}
  echo "== eval $ckpt -> $out @ $(date) ==" >> "$LOG"
  "$PY" -u experiments/eval_p9e1_dual.py --ckpt "$OUT/$ckpt" --out "$OUT/$out" >> "$LOG" 2>&1
  echo "== done $out rc=$? @ $(date) ==" >> "$LOG"
done
echo "== ALL_DONE $(date) ==" >> "$LOG"
