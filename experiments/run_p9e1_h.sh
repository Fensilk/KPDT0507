#!/bin/bash
# 方案 H：E1@6000 端到端训练（decode-on-fly，无 96GB 缓存）。
# 配方 = E1@2000 原样复刻 + 数据量 6000（嵌套超集，见 data/e1_6000_manifest.csv）。
#   epochs=4 / patience=0(不开早停) / window64 / stride16 / batch2 / lr1e-4 / seed42
#   val = 同 E1@2000 的 150 视频集（data/p9e1_frames_val），val_every 600，best 按 val_avg_f1
# 用法（在 autodl 上，/root/autodl-tmp 内）:
#    bash run_h.sh          # 首次启动
#    bash run_h.sh resume <epoch_N_model.pt> <已完成的epoch数>   # 崩溃后续跑
set -u
cd /root/autodl-tmp || exit 1
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1
OUT=logs/phase9/e1_6000
RUNLOG=$OUT/run.log
mkdir -p "$OUT"

MODEL="facebook/dinov2-giant"
DATA_MP4=data/e1_6000_manifest.csv
VAL=data/p9e1_frames_val
NPZ=data/omnifall_dinov2_giant_frame.npz
COMMON="--data_mp4 $DATA_MP4 --val_data $VAL --npz $NPZ --model_name $MODEL
        --epochs 4 --early_stop_patience 0 --val_every 600
        --window 64 --stride 16 --batch 2 --lr 1e-4 --seed 42 --workers 16 --out $OUT"

if [ "${1:-}" = "resume" ]; then
    CKPT=${2:?need ckpt path}; INIT_EP=${3:?need completed epochs}
    echo "== resume $(date) from $CKPT init_epoch=$INIT_EP ==" >> "$RUNLOG"
    setsid nohup /root/miniconda3/bin/python -u experiments/train_p9e1.py \
        $COMMON --init_ckpt "$CKPT" --init_epoch "$INIT_EP" >> "$RUNLOG" 2>&1 < /dev/null &
else
    echo "== launch $(date) E1@6000 ep4 ==" >> "$RUNLOG"
    setsid nohup /root/miniconda3/bin/python -u experiments/train_p9e1.py \
        $COMMON >> "$RUNLOG" 2>&1 < /dev/null &
fi
PID=$!
echo "PID=$PID" >> "$RUNLOG"
# 进程级 watchdog：独立于会话，训练进程一退立即在 run.log 记退出码（正常/DIVERGED/崩溃均可辨）
setsid bash -c "wait $PID; rc=\$?; echo \"WATCHDOG: pid $PID exited rc=\$rc at \$(date)\" >> $RUNLOG" &
echo "launched pid=$PID -> $RUNLOG"
