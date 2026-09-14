#!/usr/bin/env bash
# Phase 10 — FAIR TSM run (F1 full-frame data + F2 dense sampling + F3 regularisation).
#
# Fairness protocol (docs/0911实验第十阶段规划.md §4.4):
#   * data  : every model sees the FULL frame (no centre crop) — squash resize, same FOV as the main line
#   * budget: label-instance matched. E1-full = 3 epochs x 128 labels/video = 3.69 M labels;
#             TSM stride 1 = 73 clips/video x 9,600 = 700,800 labels/epoch -> 5 epochs ~= 3.5 M labels
#   * recipe: same effort as the main line got (regularisation + a short lr screening on a 2 k subset)
#   * model : untouched (8-frame sliding-window TSM, K400 init)
#
# Usage: bash experiments/run_p10_tsm_fair.sh [tag]
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1

TAG=${1:-tsm_fair}
PY=${PY:-/root/miniconda3/bin/python}
FRAMES=data/ref_frames
OUT=logs/phase10/$TAG

mkdir -p "$OUT"

# ---------- F1: full-field-of-view frames ----------
if [ ! -f data/.frames_geometry_squash ]; then
  echo "[fair] (re)extracting FULL-FRAME frames (squash $((0)) )..."
  rm -rf "$FRAMES"
  "$PY" -u experiments/prep_p10_frames.py --workers "${PREP_WORKERS:-32}" --geometry squash
  touch data/.frames_geometry_squash
fi

# ---------- F2 + F3: dense sampling + regularisation ----------
"$PY" -u experiments/train_p10_tsm.py --tag "$TAG" \
  --train_stride 1 --val_stride 8 --test_stride 8 \
  --epochs "${EPOCHS:-5}" \
  --batch_size "${BATCH_SIZE:-32}" \
  --workers "${WORKERS:-8}" \
  --lr "${LR:-0.005}" \
  --weight_decay "${WD:-5e-4}" \
  --label_smoothing "${LS:-0.1}" \
  --early_stop_patience "${PATIENCE:-2}" \
  --tune_from "${TSM_CKPT:-data/tsm_k400_r50_8f.pth}" \
  --fast_shift

# ---------- evaluation: sparse test (already done inside training) + dense/timeline + comparison ----------
"$PY" -u experiments/train_p10_tsm.py --tag "$TAG" \
  --eval_only_ckpt "$OUT/best_model.pt" --test_dense \
  --batch_size 32 --workers 8 --fast_shift

"$PY" -u experiments/eval_p10_timeline.py --dump "$OUT/test_dense_preds.npz"

"$PY" -u experiments/eval_p10_compare_frames.py \
  --ref "$OUT/test_dense_preds.npz" --ref-name TSM-fair \
  --out "$OUT/compare_e1_vs_tsm_fair.json"

touch "$OUT/P10_TSM_FAIR_DONE"
echo "[fair] all done -> $OUT"
