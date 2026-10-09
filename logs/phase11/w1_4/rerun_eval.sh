#!/bin/bash
# Exp-4 评测补跑（训练已完成，只是评测侧漏传 --pose_npz）
cd /root/autodl-tmp || exit 1
export PYTHONIOENCODING=utf-8
PY=/root/miniconda3/bin/python
O=logs/phase11/w1_4
P="--pose_npz data/omnifall_pose_semantic_accel.npz"
echo "== RERUN EVAL best (full 1200, dump_pv) start @ $(date) ==" >> $O/eval_all.log
$PY -u experiments/eval_p9e1_dual.py --ckpt $O/best_model.pt --out $O/eval_best_model.json --dump_pv $O/pv_best.pt $P >> $O/eval_all.log 2>&1 < /dev/null
echo "== RERUN EVAL best done rc=$? @ $(date) ==" >> $O/eval_all.log
for ck in $O/epoch_*_model.pt; do
  b=$(basename $ck .pt)
  echo "== RERUN EVAL $b (subset 300) start @ $(date) ==" >> $O/eval_all.log
  $PY -u experiments/eval_p9e1_dual.py --ckpt $ck --out $O/eval_$b.subset.json --limit_videos 300 --subset_seed 42 $P >> $O/eval_all.log 2>&1 < /dev/null
  echo "== RERUN EVAL $b done rc=$? @ $(date) ==" >> $O/eval_all.log
done
echo "ALL_EVAL_DONE @ $(date)" >> $O/eval_all.log
