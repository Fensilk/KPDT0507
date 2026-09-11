#!/bin/bash
# E1-full：全量 9,600 端到端冷启动（decode-on-fly，mp4 现解码）——Phase 9 收官实验。
# 配方 = E1@6000(H) 原样复刻 + 唯一变量：训练数据扩到全量 train.csv(~9,600)；
# 提速 = torch.compile ON（§6.2 闸门：头 30-60min 比每步耗时 vs 5.3s，无 ≥15% 提速/
# 报错/OOM → nocompile 模式 kill 重启；compile OFF 是 H 已验证配方）。
#   epochs=3 / patience=0 / window64 / stride16 / batch2 / lr1e-4 / seed42 / workers16
#   val = data/p9e1_frames_val(150)，val_every 600，best 按 val_avg_f1
# ep3 尾若 val 仍爬 → resume 到绝对目标 epochs=4（init_ep=3）续第 4 ep。
# watchdog：watch.sh 作为训练进程的父进程，进程一退即分类(正常[DONE]/DIVERGED/崩溃)写入
# run.log；[DONE] 后自动启动 eval_p9e1_dual(best) → eval_best.json；>~80min 无日志增长
# 且进程仍在 → 记疑似挂起。全部服务器端、独立于会话。
#
# --epochs 语义 = 绝对目标轮数（train_p9e1: range(init_epoch, epochs)）。
# 用法（autodl，/root/autodl-tmp 内；首次前先 scp 本地权威代码上来）:
#   bash experiments/run_p9e1_full.sh compile            # 首次启动（compile ON，目标 3 ep）
#   bash experiments/run_p9e1_full.sh nocompile          # compile 闸门失败后的重启（3 ep）
#   bash experiments/run_p9e1_full.sh resume <ckpt> <init_ep> <target_epochs> [compile|nocompile]
#                                                        # 崩溃续跑/续第4ep：ckpt=epoch_<K>_model.pt,
#                                                        # init_ep=已完成轮数K, target_epochs=绝对目标
#                                                        # (崩溃续跑 target=3；ep3后想续第4轮 target=4)
set -u
cd /root/autodl-tmp || exit 1
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1
OUT=logs/phase9/e1_full
RUNLOG=$OUT/run.log
mkdir -p "$OUT"

PY=/root/miniconda3/bin/python
MODEL="facebook/dinov2-giant"
DATA_MP4=DATASET-omnifall/splits/syn/random/train.csv   # 全量 ~9,600（seed42 打散）
VAL=data/p9e1_frames_val
NPZ=data/omnifall_dinov2_giant_frame.npz

# ---- 解析模式 ----
MODE="${1:-compile}"
RESUME_CKPT=""; RESUME_EP=""; EPOCHS=3; COMPILE_FLAG=""
case "$MODE" in
  compile)   COMPILE_FLAG="--compile"; LABEL="compile ON (首启/闸门观察)";;
  nocompile) COMPILE_FLAG="";          LABEL="compile OFF (闸门失败重启)";;
  resume)    RESUME_CKPT="${2:?resume 需 ckpt 路径}"; RESUME_EP="${3:?resume 需 init_epoch}"
             EPOCHS="${4:?resume 需绝对目标轮数 target_epochs}"
             case "${5:-compile}" in compile) COMPILE_FLAG="--compile";; *) COMPILE_FLAG="";; esac
             LABEL="resume init_epoch=$RESUME_EP target_epochs=$EPOCHS from $RESUME_CKPT ${COMPILE_FLAG:+compile ON}";;
  *) echo "usage: run_p9e1_full.sh [compile|nocompile|resume <ckpt> <init_ep> <target_epochs> [compile|nocompile]]"; exit 2;;
esac

# 注意：必须单行！COMMON 会插值进 heredoc 生成的 watch.sh；多行会把 python 命令截断成多条。
COMMON="--data_mp4 $DATA_MP4 --val_data $VAL --npz $NPZ --model_name $MODEL --epochs $EPOCHS --early_stop_patience 0 --val_every 600 --val_n 150 --window 64 --stride 16 --batch 2 --lr 1e-4 --seed 42 --workers 16 --out $OUT $COMPILE_FLAG"
[ -n "$RESUME_CKPT" ] && COMMON="$COMMON --init_ckpt $RESUME_CKPT --init_epoch $RESUME_EP"

# ---- 生成 watch.sh（训练进程的真正父进程：wait 得 rc、进程级退出分类、挂起探测、DONE→eval）----
cat > "$OUT/watch.sh" <<EOF
#!/bin/bash
set -u
cd /root/autodl-tmp || exit 1
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1
echo "== $LABEL @ \$(date) pid=$$ ==" >> "$RUNLOG"
echo "== cmd: python experiments/train_p9e1.py $COMMON" >> "$RUNLOG"
$PY -u experiments/train_p9e1.py $COMMON >> "$RUNLOG" 2>&1 < /dev/null &
PID=\$!
echo "TRAIN_PID=\$PID" >> "$RUNLOG"
echo \$PID > "$OUT/train.pid"

# 挂起探测：10min 一查，run.log 连续 8 次(≈80min)无增长且进程仍在 → 记疑似挂起
# 判活必须看 /proc/<pid>/stat 状态（Z=僵尸）——python 是本脚本子进程，退出后变僵尸，
# 而 `kill -0 僵尸` 仍返回成功 → 会死循环永不判型（E1-full 首跑即踩此坑）。
alive() { [ -d "/proc/\$PID" ] && [ "\$(cut -d' ' -f3 /proc/\$PID/stat 2>/dev/null)" != "Z" ]; }
last_len=0; stall=0
while alive; do
  sleep 600
  n=\$(wc -l < "$RUNLOG" 2>/dev/null || echo 0)
  if [ "\$n" -le "\$last_len" ]; then stall=\$((stall+1)); else stall=0; fi
  last_len=\$n
  if [ "\$stall" -ge 8 ]; then
    echo "WATCHDOG: 疑似挂起(≈\${stall}0min 无日志增长) pid=\$PID 仍在 @ \$(date)" >> "$RUNLOG"
    echo "  尾部: \$(tail -n 3 "$RUNLOG" | tr '\n' '|')" >> "$RUNLOG"
    stall=0
  fi
done
wait \$PID; rc=\$?

# 进程退出 → 判型
if grep -q "DIVERGED" "$OUT/e1_train.log" 2>/dev/null; then        V=diverged
elif grep -q "\[DONE\]" "$OUT/e1_train.log" 2>/dev/null; then      V=done
else V=crash; fi
echo "WATCHDOG: pid=\$PID exited rc=\$rc verdict=\$V @ \$(date)" >> "$RUNLOG"
rm -f "$OUT/train.pid"

# 正常完成 → 自动跑全量 test 双口径（best）
if [ "\$V" = done ]; then
  sleep 5
  $PY -u experiments/eval_p9e1_dual.py --ckpt "$OUT/best_model.pt" \
      --out "$OUT/eval_best.json" > "$OUT/eval_best.log" 2>&1 < /dev/null &
  EVALPID=\$!
  echo "WATCHDOG: [DONE] → 自动启动 eval_best pid=\$EVALPID (eval_best.log/eval_best.json)" >> "$RUNLOG"
fi
EOF
chmod +x "$OUT/watch.sh"

# ---- 脱离会话启动 watch.sh ----
setsid nohup bash "$OUT/watch.sh" >/dev/null 2>&1 < /dev/null &
echo "launched watch pid=$! -> OUT=$OUT"
echo "watch script: $OUT/watch.sh | run log: $RUNLOG | progress: $OUT/e1_train.log"
