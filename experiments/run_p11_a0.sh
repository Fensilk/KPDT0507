#!/bin/bash
# Phase 11 Exp-0b：**配方锚点 A0** —— E1@2000 复刻，3 epoch。
#
# 为什么必须跑：W1 全部实验在 2000 数据上做，需要一个"同配方 + 同数据 + 零改动"的锚点。
# 现有 logs/phase9/e1_5ep（E1@2000）**配方不同**（5 ep / 早停史 / best=step6000），
# 直接对比会重犯 Phase 9 的"配方不同 → 不是单变量对照"错误。故重跑一个干净锚点。
#
# 数据：data/e1_2000_manifest.csv = default_rng(42).permutation(train.csv)[:2000]
#       （已验证 == E1@2000 的子集，且 2000 ⊂ 6000 嵌套；与 W1 各臂共用同一清单）
#       帧源用 mp4 现解码（--data_mp4）：帧缓存 data/p9e1_frames 已被清空（磁盘仅剩 4.2G），
#       而 --data_mp4 与缓存逐字节一致（Phase 9 已验证）。
# 配方：window64 / stride16 / batch2 / lr1e-4 / seed42 / epochs3 / patience 0（不开早停）
#       compile OFF —— E1@6000/Full 已验证配方；compile 与 transformers DINOv2 +
#       梯度 checkpointing 不兼容（首前向即 TypeError），冒烟 10 分钟即可判死，不必试。
# 预估：9600 视频 × 3ep = 42.3h → 2000 视频 × 3ep ≈ **8.8 h**（+ 末尾自动评测）
#
# ---- 终止预案（启动前固化，不依赖人工发现）----
#   早停      不会发生（--early_stop_patience 0，跑满 3 轮）
#   DIVERGED  e1_train.log 出现 DIVERGED → 停、诊断（降 lr / 查数据）后重启，绝不带病跑完
#   OOM       降 --batch 到 1 重跑
#   崩溃      resume：bash experiments/run_p11_a0.sh resume <epoch_N_model.pt> <N>
#   疑似挂起  连续 ~80min e1_train.log 无增长且进程仍在 → 写入 run.log 告警
#
# ---- 进程级监视 ----
#   watch.sh 是训练进程的真正父进程：wait 得 rc，进程一退立即判型写 run.log。
#   判活必须查 /proc/<pid>/stat 第 3 字段是否为 Z —— 训练进程退出后变僵尸，
#   而 `kill -0 僵尸` 仍返回成功 → 会死循环永不判型（E1-full 首跑即踩此坑）。
#   [DONE] → 自动评测 best + 每个 epoch（epoch 轨迹是规划 §7.4 判定所需）。
#   结束标记 ALL_DONE 由 runner 自己写（不依赖 watchdog 判型：实例关机时会吞掉判型）。
#
# 用法（autodl，/root/autodl-tmp 内）：
#   bash experiments/run_p11_a0.sh                    # 首次启动
#   bash experiments/run_p11_a0.sh resume <ckpt> <N>  # 崩溃续跑，N=已完成轮数
set -u
cd /root/autodl-tmp || exit 1
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1

OUT=logs/phase11/a0_2000
RUNLOG=$OUT/run.log
mkdir -p "$OUT"

PY=/root/miniconda3/bin/python
# 注意：必须单行！COMMON 会插值进下面 heredoc 生成的 watch.sh，多行会把 python 命令截断。
COMMON="--data_mp4 data/e1_2000_manifest.csv --val_data data/p9e1_frames_val --npz data/omnifall_dinov2_giant_frame.npz --model_name facebook/dinov2-giant --epochs 3 --early_stop_patience 0 --val_every 600 --val_n 150 --window 64 --stride 16 --batch 2 --lr 1e-4 --seed 42 --workers 16 --out $OUT"

RESUME_EXTRA=""
if [ "${1:-}" = "resume" ]; then
  CKPT=${2:?resume 需要 ckpt 路径}
  INIT_EP=${3:?resume 需要已完成的 epoch 数}
  RESUME_EXTRA=" --init_ckpt $CKPT --init_epoch $INIT_EP"
  echo "== A0 resume $(date) from $CKPT init_epoch=$INIT_EP ==" >> "$RUNLOG"
else
  echo "== A0 launch $(date) 3ep on 2000 ==" >> "$RUNLOG"
fi
COMMON="$COMMON$RESUME_EXTRA"

# ---- 生成 watch.sh（训练进程的真正父进程）----
cat > "$OUT/watch.sh" <<EOF
#!/bin/bash
set -u
cd /root/autodl-tmp || exit 1
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1
echo "== A0 cmd: $PY -u experiments/train_p9e1.py $COMMON" >> "$RUNLOG"
$PY -u experiments/train_p9e1.py $COMMON >> "$RUNLOG" 2>&1 < /dev/null &
PID=\$!
echo "TRAIN_PID=\$PID" >> "$RUNLOG"
echo \$PID > "$OUT/train.pid"

# 判活：/proc/<pid>/stat 第3字段 != Z（僵尸）
alive() { [ -d "/proc/\$PID" ] && [ "\$(cut -d' ' -f3 /proc/\$PID/stat 2>/dev/null)" != "Z" ]; }
last=0; stall=0
while alive; do
  sleep 600
  n=\$(wc -l < "$OUT/e1_train.log" 2>/dev/null || echo 0)
  if [ "\$n" -le "\$last" ]; then stall=\$((stall+1)); else stall=0; fi
  last=\$n
  if [ "\$stall" -ge 8 ]; then
    echo "WATCHDOG: 疑似挂起 pid=\$PID @ \$(date) 尾部: \$(tail -n 3 "$OUT/e1_train.log" 2>/dev/null | tr '\n' '|')" >> "$RUNLOG"
    stall=0
  fi
done
wait \$PID; rc=\$?

if grep -q "DIVERGED" "$OUT/e1_train.log" 2>/dev/null; then      V=diverged
elif grep -q "\[DONE\]" "$OUT/e1_train.log" 2>/dev/null; then    V=done
else                                                             V=crash; fi
echo "WATCHDOG: pid=\$PID exited rc=\$rc verdict=\$V @ \$(date)" >> "$RUNLOG"
rm -f "$OUT/train.pid"

# runner 自己写结束标记（不依赖 watchdog 判型）
echo "ALL_DONE verdict=\$V rc=\$rc @ \$(date)" >> "$RUNLOG"

if [ "\$V" = done ]; then
  # 自动评测：best 优先，再逐 epoch —— epoch 轨迹供规划 §7.4「比整条轨迹」判定
  for ck in "$OUT/best_model.pt" "$OUT"/epoch_*_model.pt; do
    [ -f "\$ck" ] || continue
    b=\$(basename "\$ck" .pt)
    echo "== EVAL \$b start @ \$(date) ==" >> "$OUT/eval_all.log"
    $PY -u experiments/eval_p9e1_dual.py --ckpt "\$ck" --out "$OUT/eval_\$b.json" >> "$OUT/eval_all.log" 2>&1 < /dev/null
    echo "== EVAL \$b done rc=\$? @ \$(date) ==" >> "$OUT/eval_all.log"
  done
  echo "ALL_EVAL_DONE @ \$(date)" >> "$OUT/eval_all.log"
  echo "WATCHDOG: 评测全部完成 → $OUT/eval_all.log" >> "$RUNLOG"
fi
EOF
chmod +x "$OUT/watch.sh"

# ---- 脱离会话启动 ----
setsid nohup bash "$OUT/watch.sh" >/dev/null 2>&1 < /dev/null &
echo "launched watch pid=$! -> OUT=$OUT"
echo "run log:   $RUNLOG"
echo "progress:  $OUT/e1_train.log"
echo "watchdog:  $OUT/watch.sh"
