#!/bin/bash
set -u
cd /root/autodl-tmp || exit 1
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1
echo "== W2/4 cmd: /root/miniconda3/bin/python -u experiments/train_p9e1.py --data_mp4 DATASET-omnifall/splits/syn/random/train.csv --val_data data/p9e1_frames_val --npz data/omnifall_dinov2_giant_frame.npz --model_name facebook/dinov2-giant --epochs 3 --early_stop_patience 0 --val_every 600 --val_n 150 --window 64 --stride 16 --batch 2 --lr 1e-4 --seed 42 --workers 16 --out logs/phase11/w2_4 --pose_npz data/omnifall_pose_semantic_accel.npz" >> "logs/phase11/w2_4/run.log"
/root/miniconda3/bin/python -u experiments/train_p9e1.py --data_mp4 DATASET-omnifall/splits/syn/random/train.csv --val_data data/p9e1_frames_val --npz data/omnifall_dinov2_giant_frame.npz --model_name facebook/dinov2-giant --epochs 3 --early_stop_patience 0 --val_every 600 --val_n 150 --window 64 --stride 16 --batch 2 --lr 1e-4 --seed 42 --workers 16 --out logs/phase11/w2_4 --pose_npz data/omnifall_pose_semantic_accel.npz >> "logs/phase11/w2_4/run.log" 2>&1 < /dev/null &
PID=$!
echo "TRAIN_PID=$PID" >> "logs/phase11/w2_4/run.log"
echo $PID > "logs/phase11/w2_4/train.pid"

alive() { [ -d "/proc/$PID" ] && [ "$(cut -d' ' -f3 /proc/$PID/stat 2>/dev/null)" != "Z" ]; }
last=0; stall=0
while alive; do
  sleep 600
  n=$(wc -l < "logs/phase11/w2_4/e1_train.log" 2>/dev/null || echo 0)
  if [ "$n" -le "$last" ]; then stall=$((stall+1)); else stall=0; fi
  last=$n
  if [ "$stall" -ge 8 ]; then
    echo "WATCHDOG: 疑似挂起 pid=$PID @ $(date) 尾部: $(tail -n 3 "logs/phase11/w2_4/e1_train.log" 2>/dev/null | tr '\n' '|')" >> "logs/phase11/w2_4/run.log"
    stall=0
  fi
done
wait $PID; rc=$?

if grep -q "DIVERGED" "logs/phase11/w2_4/e1_train.log" 2>/dev/null; then      V=diverged
elif grep -q "\[DONE\]" "logs/phase11/w2_4/e1_train.log" 2>/dev/null; then    V=done
else                                                             V=crash; fi
echo "WATCHDOG: pid=$PID exited rc=$rc verdict=$V @ $(date)" >> "logs/phase11/w2_4/run.log"
rm -f "logs/phase11/w2_4/train.pid"
echo "ALL_DONE arm=4 verdict=$V rc=$rc @ $(date)" >> "logs/phase11/w2_4/run.log"

if [ "$V" = done ]; then
  # 评测预算策略（2026-09-29 实测后定）：
  #   全量评测 = 53 min/ckpt，且**是真实 GPU 前向成本**（解码仅占 4%，批量前向无收益——
  #   ViT-g 在 64 帧上 batch=1 已吃满 GPU）。4 个 ckpt × 6 臂 = 21 h，规划未计。
  #   → best 跑**全量 1200**（最终数字用这个）；
  #   → epoch_* 跑固定 **300 视频随机子集**（seed=42，跨臂同子集可比），供 §7.4 轨迹判定。
  #     抽样是**确定性随机**而非前 N 个：test.csv 按路径排序、fall/ 打头，前缀有偏。
  # best 额外 --dump_pv：逐视频预测是后续算 timeline 指标（§7.2 否决条件之一）
  # 与边界指标（规划 §5）的必要输入，不存就只能重跑 53 min。
  echo "== EVAL best (full 1200, dump_pv) start @ $(date) ==" >> "logs/phase11/w2_4/eval_all.log"
  /root/miniconda3/bin/python -u experiments/eval_p9e1_dual.py --ckpt "logs/phase11/w2_4/best_model.pt"       --out "logs/phase11/w2_4/eval_best_model.json" --dump_pv "logs/phase11/w2_4/pv_best.pt" --pose_npz data/omnifall_pose_semantic_accel.npz       >> "logs/phase11/w2_4/eval_all.log" 2>&1 < /dev/null
  echo "== EVAL best done rc=$? @ $(date) ==" >> "logs/phase11/w2_4/eval_all.log"

  for ck in "logs/phase11/w2_4"/epoch_*_model.pt; do
    [ -f "$ck" ] || continue
    b=$(basename "$ck" .pt)
    echo "== EVAL $b (subset 300) start @ $(date) ==" >> "logs/phase11/w2_4/eval_all.log"
    /root/miniconda3/bin/python -u experiments/eval_p9e1_dual.py --ckpt "$ck" --out "logs/phase11/w2_4/eval_$b.subset.json"         --limit_videos 300 --subset_seed 42 --pose_npz data/omnifall_pose_semantic_accel.npz >> "logs/phase11/w2_4/eval_all.log" 2>&1 < /dev/null
    echo "== EVAL $b done rc=$? @ $(date) ==" >> "logs/phase11/w2_4/eval_all.log"
  done
  echo "ALL_EVAL_DONE @ $(date)" >> "logs/phase11/w2_4/eval_all.log"
fi
