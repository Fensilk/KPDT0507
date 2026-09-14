#!/bin/bash
set -u
cd /root/autodl-tmp || exit 1
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1
echo "== compile OFF (闸门失败重启) @ $(date) pid=95829 ==" >> "logs/phase9/e1_full/run.log"
echo "== cmd: python experiments/train_p9e1.py --data_mp4 DATASET-omnifall/splits/syn/random/train.csv --val_data data/p9e1_frames_val --npz data/omnifall_dinov2_giant_frame.npz --model_name facebook/dinov2-giant --epochs 3 --early_stop_patience 0 --val_every 600 --val_n 150 --window 64 --stride 16 --batch 2 --lr 1e-4 --seed 42 --workers 16 --out logs/phase9/e1_full " >> "logs/phase9/e1_full/run.log"
/root/miniconda3/bin/python -u experiments/train_p9e1.py --data_mp4 DATASET-omnifall/splits/syn/random/train.csv --val_data data/p9e1_frames_val --npz data/omnifall_dinov2_giant_frame.npz --model_name facebook/dinov2-giant --epochs 3 --early_stop_patience 0 --val_every 600 --val_n 150 --window 64 --stride 16 --batch 2 --lr 1e-4 --seed 42 --workers 16 --out logs/phase9/e1_full  >> "logs/phase9/e1_full/run.log" 2>&1 < /dev/null &
PID=$!
echo "TRAIN_PID=$PID" >> "logs/phase9/e1_full/run.log"
echo $PID > "logs/phase9/e1_full/train.pid"

# 挂起探测：10min 一查，run.log 连续 8 次(≈80min)无增长且进程仍在 → 记疑似挂起
last_len=0; stall=0
while kill -0 $PID 2>/dev/null; do
  sleep 600
  n=$(wc -l < "logs/phase9/e1_full/run.log" 2>/dev/null || echo 0)
  if [ "$n" -le "$last_len" ]; then stall=$((stall+1)); else stall=0; fi
  last_len=$n
  if [ "$stall" -ge 8 ]; then
    echo "WATCHDOG: 疑似挂起(≈${stall}0min 无日志增长) pid=$PID 仍在 @ $(date)" >> "logs/phase9/e1_full/run.log"
    echo "  尾部: $(tail -n 3 "logs/phase9/e1_full/run.log" | tr '\n' '|')" >> "logs/phase9/e1_full/run.log"
    stall=0
  fi
done
wait $PID; rc=$?

# 进程退出 → 判型
if grep -q "DIVERGED" "logs/phase9/e1_full/e1_train.log" 2>/dev/null; then        V=diverged
elif grep -q "\[DONE\]" "logs/phase9/e1_full/e1_train.log" 2>/dev/null; then      V=done
else V=crash; fi
echo "WATCHDOG: pid=$PID exited rc=$rc verdict=$V @ $(date)" >> "logs/phase9/e1_full/run.log"
rm -f "logs/phase9/e1_full/train.pid"

# 正常完成 → 自动跑全量 test 双口径（best）
if [ "$V" = done ]; then
  sleep 5
  /root/miniconda3/bin/python -u experiments/eval_p9e1_dual.py --ckpt "logs/phase9/e1_full/best_model.pt"       --out "logs/phase9/e1_full/eval_best.json" > "logs/phase9/e1_full/eval_best.log" 2>&1 < /dev/null &
  EVALPID=$!
  echo "WATCHDOG: [DONE] → 自动启动 eval_best pid=$EVALPID (eval_best.log/eval_best.json)" >> "logs/phase9/e1_full/run.log"
fi
