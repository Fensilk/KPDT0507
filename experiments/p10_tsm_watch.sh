#!/usr/bin/env bash
# Phase 10 TSM watcher: wait for the full training run to finish, then run the dense
# per-frame test dump and the timeline (event-level) metrics, and finally drop a marker file.
#
# Shutdown rule for the cloud instance: logs/phase10/<TAG>/P10_TSM_DONE exists => safe to stop.
set -u
cd /root/autodl-tmp

TAG=${1:-tsm_r50_8f_9600}
OUT=logs/phase10/$TAG
PY=/root/miniconda3/bin/python

rm -f "$OUT/P10_TSM_DONE"
echo "[watch] started at $(date), waiting for training (tag=$TAG)" >> "$OUT/chain.log"

while pgrep -f "train_p10_tsm.py --tag $TAG" > /dev/null; do
  sleep 60
done
echo "[watch] training process gone at $(date)" >> "$OUT/chain.log"

# dense (stride 1, per-frame merged) test dump -> used by the timeline metrics
"$PY" experiments/train_p10_tsm.py --tag "$TAG" \
  --eval_only_ckpt "$OUT/best_model.pt" --test_dense \
  --batch_size 32 --workers 8 --fast_shift >> "$OUT/chain.log" 2>&1

"$PY" experiments/eval_p10_timeline.py --dump "$OUT/test_dense_preds.npz" >> "$OUT/chain.log" 2>&1

echo "[watch] all done at $(date)" >> "$OUT/chain.log"
touch "$OUT/P10_TSM_DONE"
