#!/bin/bash
set -u
cd /root/autodl-tmp || exit 1
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1
echo "== A0 cmd: /root/miniconda3/bin/python -u experiments/train_p9e1.py --data_mp4 data/e1_2000_manifest.csv --val_data data/p9e1_frames_val --npz data/omnifall_dinov2_giant_frame.npz --model_name facebook/dinov2-giant --epochs 3 --early_stop_patience 0 --val_every 600 --val_n 150 --window 64 --stride 16 --batch 2 --lr 1e-4 --seed 42 --workers 16 --out logs/phase11/a0_2000" >> "logs/phase11/a0_2000/run.log"
/root/miniconda3/bin/python -u experiments/train_p9e1.py --data_mp4 data/e1_2000_manifest.csv --val_data data/p9e1_frames_val --npz data/omnifall_dinov2_giant_frame.npz --model_name facebook/dinov2-giant --epochs 3 --early_stop_patience 0 --val_every 600 --val_n 150 --window 64 --stride 16 --batch 2 --lr 1e-4 --seed 42 --workers 16 --out logs/phase11/a0_2000 >> "logs/phase11/a0_2000/run.log" 2>&1 < /dev/null &
PID=$!
echo "TRAIN_PID=$PID" >> "logs/phase11/a0_2000/run.log"
echo $PID > "logs/phase11/a0_2000/train.pid"

# 判活：/proc/<pid>/stat 第3字段 != Z（僵尸）
alive() { [ -d "/proc/$PID" ] && [ "$(cut -d' ' -f3 /proc/$PID/stat 2>/dev/null)" != "Z" ]; }
last=0; stall=0
while alive; do
  sleep 600
  n=$(wc -l < "logs/phase11/a0_2000/e1_train.log" 2>/dev/null || echo 0)
  if [ "$n" -le "$last" ]; then stall=$((stall+1)); else stall=0; fi
  last=$n
  if [ "$stall" -ge 8 ]; then
    echo "WATCHDOG: 疑似挂起 pid=$PID @ $(date) 尾部: $(tail -n 3 "logs/phase11/a0_2000/e1_train.log" 2>/dev/null | tr '\n' '|')" >> "logs/phase11/a0_2000/run.log"
    stall=0
  fi
done
wait $PID; rc=$?

if grep -q "DIVERGED" "logs/phase11/a0_2000/e1_train.log" 2>/dev/null; then      V=diverged
elif grep -q "\[DONE\]" "logs/phase11/a0_2000/e1_train.log" 2>/dev/null; then    V=done
else                                                             V=crash; fi
echo "WATCHDOG: pid=$PID exited rc=$rc verdict=$V @ $(date)" >> "logs/phase11/a0_2000/run.log"
rm -f "logs/phase11/a0_2000/train.pid"

# runner 自己写结束标记（不依赖 watchdog 判型）
echo "ALL_DONE verdict=$V rc=$rc @ $(date)" >> "logs/phase11/a0_2000/run.log"

if [ "$V" = done ]; then
  # 自动评测：best 优先，再逐 epoch —— epoch 轨迹供规划 §7.4「比整条轨迹」判定
  for ck in "logs/phase11/a0_2000/best_model.pt" "logs/phase11/a0_2000"/epoch_*_model.pt; do
    [ -f "$ck" ] || continue
    b=$(basename "$ck" .pt)
    echo "== EVAL $b start @ $(date) ==" >> "logs/phase11/a0_2000/eval_all.log"
    /root/miniconda3/bin/python -u experiments/eval_p9e1_dual.py --ckpt "$ck" --out "logs/phase11/a0_2000/eval_$b.json" >> "logs/phase11/a0_2000/eval_all.log" 2>&1 < /dev/null
    echo "== EVAL $b done rc=$? @ $(date) ==" >> "logs/phase11/a0_2000/eval_all.log"
  done
  echo "ALL_EVAL_DONE @ $(date)" >> "logs/phase11/a0_2000/eval_all.log"
  echo "WATCHDOG: 评测全部完成 → logs/phase11/a0_2000/eval_all.log" >> "logs/phase11/a0_2000/run.log"
fi
