#!/usr/bin/env bash
# Phase 10 / X3D-S finalisation.
#
# Waiting for the T_max=15 run to finish its evaluation, this script:
#   1. archives that run as logs/phase10/x3d_s_cos15  (evidence for the LR-schedule finding)
#   2. promotes the T_max=5 run (the adopted protocol) to logs/phase10/x3d_s
#   3. evaluates the adopted checkpoint: dense per-frame dump -> timeline -> comparison vs E1-full
#
# Usage: setsid bash experiments/p10_x3d_finalize.sh > logs/phase10/x3d_finalize.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
PY=${PY:-/root/miniconda3/bin/python}
LOGROOT=logs/phase10

banner () { echo; echo "=============== [$(date '+%F %T')] $* ==============="; echo; }

banner "WAIT for the T_max=15 run to finish"
while pgrep -f "run_p10_video_baseline.sh x3d_s [x]3d_s" > /dev/null; do sleep 30; done
for p in $(pgrep -f "train_p10_video.py --model x3d_s --tag [x]3d_s"); do kill $p 2>/dev/null; done
sleep 3

banner "ARCHIVE T_max=15 run -> ${LOGROOT}/x3d_s_cos15"
rm -rf "$LOGROOT/x3d_s_cos15"
mv "$LOGROOT/x3d_s" "$LOGROOT/x3d_s_cos15"

banner "PROMOTE T_max=5 run (adopted protocol) -> ${LOGROOT}/x3d_s"
mkdir -p "$LOGROOT/x3d_s"
cp "$LOGROOT/x3d_s_5ep/best_model.pt" "$LOGROOT/x3d_s_5ep/training_history.json" \
   "$LOGROOT/x3d_s_5ep/args.json" "$LOGROOT/x3d_s_5ep/run.log" "$LOGROOT/x3d_s/"

banner "EVAL: dense per-frame dump (stride 1)"
"$PY" -u experiments/train_p10_video.py --model x3d_s --tag x3d_s \
  --eval_only_ckpt "$LOGROOT/x3d_s/best_model.pt" --test_dense \
  --batch_size 32 --workers 8

banner "EVAL: timeline metrics"
"$PY" -u experiments/eval_p10_timeline.py --dump "$LOGROOT/x3d_s/test_dense_preds.npz"

banner "EVAL: same-frame comparison vs E1-full"
"$PY" -u experiments/eval_p10_compare_frames.py \
  --ref "$LOGROOT/x3d_s/test_dense_preds.npz" --ref-name X3D-S \
  --out "$LOGROOT/x3d_s/compare_e1_vs_x3d_s.json"

touch "$LOGROOT/x3d_s/P10_x3d_s_DONE"
banner "ALL DONE — $LOGROOT/x3d_s/P10_x3d_s_DONE"
