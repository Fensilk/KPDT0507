#!/usr/bin/env bash
# Phase 10 addendum — VideoMAE-K400 (HF videomae-base-finetuned-kinetics, 87M, 16 frames).
#
# Same protocol as the other Phase 10 baselines (see docs/1102实验第十阶段增补-VideoMAE-K400对标.md):
# 3-class head, centre-frame ternary label, dense stride-1 test on the same 96,000 frames as E1-full.
#
# Usage: setsid bash experiments/run_p10_videomae.sh videomae_k400 \
#            > logs/phase10/videomae_k400/run.log 2>&1 &
#
# Env overrides: EPOCHS BATCH_SIZE EVAL_BS WORKERS LR WD PATIENCE
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

PY=${PY:-/root/miniconda3/bin/python}
TAG=${1:-videomae_k400}
OUT=logs/phase10/$TAG
mkdir -p "$OUT"

banner () { echo; echo "=============== [$(date '+%F %T')] $* ==============="; echo; }

banner "VIDEOMAE-K400 START tag=$TAG"
"$PY" -u experiments/train_p10_video.py --model videomae_k400 --tag "$TAG" \
  --epochs "${EPOCHS:-5}" --batch_size "${BATCH_SIZE:-16}" --workers "${WORKERS:-8}" \
  --lr "${LR:-1e-4}" --weight_decay "${WD:-0.05}" --optimizer adamw \
  --early_stop_patience "${PATIENCE:-2}" --num_frames 16 --test_stride 8
banner "TRAINING DONE"

banner "EVAL: dense per-frame dump"
"$PY" -u experiments/train_p10_video.py --model videomae_k400 --tag "$TAG" \
  --eval_only_ckpt "$OUT/best_model.pt" --test_dense --num_frames 16 \
  --batch_size "${EVAL_BS:-16}" --workers "${WORKERS:-8}"

banner "EVAL: timeline metrics"
"$PY" -u experiments/eval_p10_timeline.py --dump "$OUT/test_dense_preds.npz"

banner "EVAL: same-frame comparison vs E1-full"
"$PY" -u experiments/eval_p10_compare_frames.py \
  --ref "$OUT/test_dense_preds.npz" --ref-name "$TAG" \
  --out "$OUT/compare_e1_vs_${TAG}.json"

touch "$OUT/P10_${TAG}_DONE"
banner "ALL DONE — $OUT/P10_${TAG}_DONE"
