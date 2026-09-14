#!/usr/bin/env bash
# Phase 10 — run ONE pytorchvideo baseline end-to-end (X3D / I3D-R50 / SlowFast-R50).
#
# Each model is an independent experiment: own tag, own log, own result files.
# Unified recipe (documented in docs/0911实验第十阶段规划.md §4.4): Kinetics-400 init,
# 5 epochs + early stop patience 2, SGD lr 0.01 / wd 5e-4, batch 32, class-weighted CE,
# evaluation on the same 96,000 frames as E1-full.
#
# Usage: setsid bash experiments/run_p10_video_baseline.sh x3d_s x3d_s \
#            > logs/phase10/x3d_s/run.log 2>&1 &
#
# Env overrides: EPOCHS BATCH_SIZE WORKERS LR WD OPTIMIZER PATIENCE EVAL_BS NUM_FRAMES
#   EVAL_BS exists because the memory-hungry models (TimeSformer / ViViT) need a
#   smaller batch at inference than the 32 that X3D / C3D / SlowFast use.
#   NUM_FRAMES is for models whose K400 checkpoint expects a non-default window
#   (e.g. I3D-R50's official 8x8 setting = 8 frames). It is passed to BOTH the training
#   and the dense-eval invocation, because the eval path reads the window from the CLI
#   args (not from the checkpoint) and would otherwise fall back to the 16-frame default.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1

PY=${PY:-/root/miniconda3/bin/python}
MODEL=${1:?usage: run_p10_video_baseline.sh <model> [tag]}
TAG=${2:-$MODEL}
OUT=logs/phase10/$TAG
mkdir -p "$OUT"

EXTRA_ARGS=()
if [ -n "${NUM_FRAMES:-}" ]; then
  EXTRA_ARGS+=(--num_frames "$NUM_FRAMES")
fi

banner () { echo; echo "=============== [$(date '+%F %T')] $* ==============="; echo; }

banner "BASELINE START model=$MODEL tag=$TAG"
"$PY" -u experiments/train_p10_video.py --model "$MODEL" --tag "$TAG" \
  --epochs "${EPOCHS:-5}" --batch_size "${BATCH_SIZE:-32}" --workers "${WORKERS:-8}" \
  --lr "${LR:-0.01}" --weight_decay "${WD:-5e-4}" --optimizer "${OPTIMIZER:-sgd}" \
  --early_stop_patience "${PATIENCE:-2}" --test_stride 8 ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
banner "TRAINING DONE"

banner "EVAL: dense per-frame dump"
"$PY" -u experiments/train_p10_video.py --model "$MODEL" --tag "$TAG" \
  --eval_only_ckpt "$OUT/best_model.pt" --test_dense \
  --batch_size "${EVAL_BS:-32}" --workers "${WORKERS:-8}" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

banner "EVAL: timeline metrics"
"$PY" -u experiments/eval_p10_timeline.py --dump "$OUT/test_dense_preds.npz"

banner "EVAL: same-frame comparison vs E1-full"
"$PY" -u experiments/eval_p10_compare_frames.py \
  --ref "$OUT/test_dense_preds.npz" --ref-name "$TAG" \
  --out "$OUT/compare_e1_vs_${TAG}.json"

touch "$OUT/P10_${TAG}_DONE"
banner "ALL DONE — $OUT/P10_${TAG}_DONE"
