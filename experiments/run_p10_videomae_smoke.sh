#!/usr/bin/env bash
# Smoke test for the VideoMAE-K400 Phase 10 addendum — pipeline correctness only, not a result.
# 200 train videos / 20 steps / 1 epoch, then a 30-video dense evaluation.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
PY=${PY:-/root/miniconda3/bin/python}
TAG=videomae_k400_smoke
OUT=logs/phase10/$TAG
mkdir -p "$OUT"

echo "=============== [$(date '+%F %T')] SMOKE TRAIN ==============="
"$PY" -u experiments/train_p10_video.py --model videomae_k400 --tag "$TAG" \
  --limit_videos 200 --val_limit_videos 50 --epochs 1 --max_steps 20 \
  --batch_size 16 --workers 8 --lr 1e-4 --weight_decay 0.05 --optimizer adamw \
  --num_frames 16 --skip_test

echo "=============== [$(date '+%F %T')] SMOKE DENSE EVAL ==============="
"$PY" -u experiments/train_p10_video.py --model videomae_k400 --tag "$TAG" \
  --eval_only_ckpt "$OUT/best_model.pt" --test_dense --test_limit_videos 30 \
  --batch_size 16 --workers 8 --num_frames 16

echo "=============== [$(date '+%F %T')] SMOKE DONE ==============="
