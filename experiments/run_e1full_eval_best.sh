#!/bin/bash
# E1-full：仅补跑 best_model.pt 的双口径 eval（首次被游离 SIGTERM rc=143 打断）。
# 关键：每个 python 用 setsid 放进独立会话，避免被 ssh 会话清理信号误伤。
set -u
cd /root/autodl-tmp || exit 1
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1
OUT=logs/phase9/e1_full
PY=/root/miniconda3/bin/python
echo "== eval-best start $(date) ==" >> "$OUT/eval_all.log"
setsid "$PY" -u experiments/eval_p9e1_dual.py --ckpt "$OUT/best_model.pt" --out "$OUT/eval_best.json" \
    >> "$OUT/eval_all.log" 2>&1
echo "== done eval_best.json rc=$? @ $(date) ==" >> "$OUT/eval_all.log"
echo "== BEST_DONE $(date) ==" >> "$OUT/eval_all.log"
