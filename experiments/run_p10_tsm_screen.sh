#!/usr/bin/env bash
# Phase 10 — F3 "equal tuning effort": screen a few lr / weight-decay / label-smoothing configs
# for TSM on a 2,000-video subset, then report which one wins on the full validation split.
#
# Rationale: the main line (E1) was tuned across many Phase-9 iterations, so giving the reference
# model a single official recipe would be an unfair comparison of *effort*. This screen is
# deliberately cheap (1 epoch, 2k videos, dense stride 1 ~ 146k clips) and pre-declared.
#
# Usage: bash experiments/run_p10_tsm_screen.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
PY=${PY:-/root/miniconda3/bin/python}

run_cfg () {  # name lr wd ls
  local name=$1 lr=$2 wd=$3 ls=$4
  echo "=== screening $name (lr=$lr wd=$wd ls=$ls) ==="
  "$PY" -u experiments/train_p10_tsm.py \
    --tag "tsm_screen_$name" \
    --limit_videos 2000 --val_limit_videos 0 \
    --train_stride 1 --val_stride 8 --test_stride 8 \
    --epochs 1 --batch_size 32 --workers 8 \
    --lr "$lr" --weight_decay "$wd" --label_smoothing "$ls" \
    --tune_from data/tsm_k400_r50_8f.pth --fast_shift --skip_test \
    2>&1 | tail -3
}

run_cfg A 0.005 5e-4 0.1     # fair default
run_cfg B 0.002 1e-3 0.1     # more conservative / stronger regularisation
run_cfg C 0.010 5e-4 0.0     # closer to TSM's official Kinetics setting

echo "=== summary (val avg_f1 from each run) ==="
for t in A B C; do
  f="logs/phase10/tsm_screen_$t/training_history.json"
  [ -f "$f" ] && "$PY" -c "
import json,sys
h=json.load(open('$f'))
print('$t', 'val_avg_f1=%.4f' % h[-1]['val_avg_f1'], 'val_acc=%.4f' % h[-1]['val_acc'])
"
done
