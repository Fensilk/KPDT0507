#!/bin/bash
# E1-full：为"真·事件级指标"补跑逐视频预测 dump —— best + ep3 两个 ckpt 并发。
# 为什么：E1-full 的 eval 当初没存逐视频预测，无法事后补算 timeline 指标（§7.1）。
#   ep2 经确认不测（无任何轴显著占优，仅 fallen_prec +0.01 在噪声内）。
# 产出：logs/phase9/e1_full/pv_best.pt、pv_ep3.pt
#       + eval_best_dump.json / eval_ep3_dump.json（**新名字，不覆盖已归档的 eval_best/ep3.json**，
#         两者应逐位一致 → 顺便当一次可复现性校验）
# 用法（autodl，/root/autodl-tmp 内）: bash experiments/run_e1full_eval_dump.sh
set -u
cd /root/autodl-tmp || exit 1
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1
OUT=logs/phase9/e1_full
PY=/root/miniconda3/bin/python
LOG=$OUT/eval_dump.log

echo "== dump-eval start $(date) ==" >> "$LOG"

# 两个并发；本脚本作为父进程 wait，退出码可辨
"$PY" -u experiments/eval_p9e1_dual.py --ckpt "$OUT/best_model.pt" \
    --out "$OUT/eval_best_dump.json" --dump_pv "$OUT/pv_best.pt" >> "$LOG" 2>&1 &
P1=$!
"$PY" -u experiments/eval_p9e1_dual.py --ckpt "$OUT/epoch_3_model.pt" \
    --out "$OUT/eval_ep3_dump.json" --dump_pv "$OUT/pv_ep3.pt" >> "$LOG" 2>&1 &
P2=$!
echo "PIDS: best=$P1 ep3=$P2" >> "$LOG"

# 8h 上限保护（正常 ~1.5h）
( sleep 28800; kill -TERM $P1 $P2 2>/dev/null ) &
GUARD=$!

wait $P1; r1=$?
wait $P2; r2=$?
kill -TERM $GUARD 2>/dev/null

echo "== dump-eval done best rc=$r1 ep3 rc=$r2 @ $(date) ==" >> "$LOG"
ls -la "$OUT"/pv_*.pt >> "$LOG" 2>&1
echo "== DUMP_ALL_DONE $(date) ==" >> "$LOG"
