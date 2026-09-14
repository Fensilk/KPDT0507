#!/usr/bin/env bash
# Phase 10 — ViViT-B/16x2 **T1** baseline end-to-end: frozen Kinetics-400 backbone + linear probe.
#
# T1 (not a fine-tune) per docs/0911实验第十阶段规划.md §6.6: the ViViT release is JAX/Scenic +
# DMVR/TFRecord, and a full fine-tune (~7 h on one 3090) would mostly repeat TimeSformer's result.
# This runner therefore reports ViViT as a "video-Transformer feature quality" reference point and
# the thesis must label it as a linear probe (T1).
#
#   stage 1  features : train 67,200 + val 8,400 + dense test 58,800 clips through the frozen
#                       backbone, cached as [CLS] tokens (~1 h at ~40 clips/s)
#   stage 2  probe    : Linear(768->3), class-weighted CE, AdamW, early stop on val avg_f1
#   stage 3  timeline : the same per-frame merge / event metrics as the other baselines
#   stage 4  compare  : same-frame comparison vs E1-full
#
# Usage: setsid bash experiments/run_p10_vivit_probe.sh > logs/phase10/vivit/run.log 2>&1 &
#
# Env overrides: WORKERS
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1

PY=${PY:-/root/miniconda3/bin/python}
OUT=logs/phase10/vivit
mkdir -p "$OUT"

banner () { echo; echo "=============== [$(date '+%F %T')] $* ==============="; echo; }

banner "VIVIT T1 START model=vivit (linear probe)"

banner "STAGE 1/4: frozen-backbone [CLS] feature extraction"
"$PY" -u experiments/probe_p10_vivit.py --stage features --workers "${WORKERS:-16}"

banner "STAGE 2/4: linear probe (AdamW, class-weighted CE)"
"$PY" -u experiments/probe_p10_vivit.py --stage probe

banner "STAGE 3/4: timeline metrics"
"$PY" -u experiments/eval_p10_timeline.py --dump "$OUT/test_dense_preds.npz"

banner "STAGE 4/4: same-frame comparison vs E1-full"
"$PY" -u experiments/eval_p10_compare_frames.py \
  --ref "$OUT/test_dense_preds.npz" --ref-name vivit \
  --out "$OUT/compare_e1_vs_vivit.json"

touch "$OUT/P10_vivit_DONE"
banner "ALL DONE — $OUT/P10_vivit_DONE"
