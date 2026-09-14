#!/usr/bin/env bash
# Phase 10 — full FAIR-comparison pipeline for TSM (run on the Autodl 3090).
#
#   F1 data alignment   : re-extract frames with the FULL field of view (squash, no centre crop)
#   F3 training effort  : screen 3 lr/weight-decay/label-smoothing configs on a 2,000-video subset
#   F2 training budget  : final run with dense sampling (stride 1 = 73 clips/video) and the winning
#                         config; 5 epochs ~= 3.5 M labels ~= E1-full's 3.69 M
#   evaluation          : sparse test + dense per-frame dump + timeline + same-frame comparison vs E1
#
# Everything is unbuffered and written to logs/phase10/tsm_fair/run.log so it can be watched live.
#
# Usage: setsid bash experiments/p10_fair_pipeline.sh > logs/phase10/tsm_fair/run.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1

PY=${PY:-/root/miniconda3/bin/python}
TAG=${TAG:-tsm_fair}
OUT=logs/phase10/$TAG
mkdir -p "$OUT"

banner () { echo; echo "=============== [$(date '+%F %T')] $* ==============="; echo; }

banner "PIPELINE START (tag=$TAG)"

# ---------------- F1: full-field-of-view frames ----------------
if [ ! -f data/.frames_geometry_squash ]; then
  banner "F1: deleting old cropped frames and re-extracting FULL-FRAME frames (12,000 videos, ~25 GB)"
  rm -rf data/ref_frames
  "$PY" -u experiments/prep_p10_frames.py --workers "${PREP_WORKERS:-32}" --geometry squash
  touch data/.frames_geometry_squash
  banner "F1 DONE"
else
  banner "F1 already done (marker present)"
fi

# ---------------- F3: screening ----------------
banner "F3: screening lr / weight-decay / label-smoothing on a 2,000-video subset (1 epoch each)"
bash experiments/run_p10_tsm_screen.sh

BEST=$(/root/miniconda3/bin/python - <<'PYEOF'
import json, os
cfgs = {"A": ("0.005", "5e-4", "0.1"), "B": ("0.002", "1e-3", "0.1"), "C": ("0.010", "5e-4", "0.0")}
best = None
for name, (lr, wd, ls) in cfgs.items():
    f = "logs/phase10/tsm_screen_%s/training_history.json" % name
    if os.path.exists(f):
        v = json.load(open(f))[-1]["val_avg_f1"]
        print("[screen] %s val_avg_f1=%.4f" % (name, v), file=__import__("sys").stderr)
        if best is None or v > best[0]:
            best = (v, lr, wd, ls, name)
print("%s %s %s %s" % (best[1], best[2], best[3], best[4]) if best else "0.005 5e-4 0.1 A")
PYEOF
)
read -r LR WD LS WINNER <<< "$BEST"
banner "F3 DONE — winner=$WINNER  lr=$LR wd=$WD label_smoothing=$LS"

# ---------------- F2: final fair run ----------------
banner "F2: FINAL RUN — 9,600 videos, stride 1 (700,800 clips/epoch), 5 epochs, early stop patience 2"
"$PY" -u experiments/train_p10_tsm.py --tag "$TAG" \
  --train_stride 1 --val_stride 8 --test_stride 8 \
  --epochs "${EPOCHS:-5}" \
  --batch_size "${BATCH_SIZE:-32}" \
  --workers "${WORKERS:-8}" \
  --lr "$LR" --weight_decay "$WD" --label_smoothing "$LS" \
  --early_stop_patience "${PATIENCE:-2}" \
  --tune_from "${TSM_CKPT:-data/tsm_k400_r50_8f.pth}" \
  --fast_shift
banner "F2 DONE"

# ---------------- evaluation ----------------
banner "EVAL: dense per-frame dump (stride 1)"
"$PY" -u experiments/train_p10_tsm.py --tag "$TAG" \
  --eval_only_ckpt "$OUT/best_model.pt" --test_dense \
  --batch_size 32 --workers 8 --fast_shift

banner "EVAL: timeline metrics"
"$PY" -u experiments/eval_p10_timeline.py --dump "$OUT/test_dense_preds.npz"

banner "EVAL: same-frame comparison vs E1-full"
"$PY" -u experiments/eval_p10_compare_frames.py \
  --ref "$OUT/test_dense_preds.npz" --ref-name TSM-fair \
  --out "$OUT/compare_e1_vs_tsm_fair.json"

touch "$OUT/P10_TSM_FAIR_DONE"
banner "ALL DONE — marker written: $OUT/P10_TSM_FAIR_DONE"
