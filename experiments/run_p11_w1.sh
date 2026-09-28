#!/bin/bash
# Phase 11 W1 统一 runner —— 2000 视频端到端筛查（train_p9e1.py，非冻结特征路径）。
#
# ⚠ 目标脚本是 train_p9e1.py 而非 train_phase6.py：W1 的锚点 A0 是 **E1 端到端**，
#   各臂必须同范式同配方才可单变量对照（规划 §3.1 / §7.2）。
#
# 公共配方（唯一变量 = 各 arm 自己那一个开关）：
#   data = data/e1_2000_manifest.csv（= default_rng(42).permutation(train)[:2000]，
#          已核验与 E1@2000 同子集；类权重 [1.336,1.532,0.132] 逐位一致）
#   epochs 3 / patience 0（跑满）/ window64 / stride16 / batch2 / lr1e-4 / seed42 / workers16
#   compile OFF（与 transformers DINOv2 + 梯度 checkpointing 不兼容）
#   预估每臂 ≈ 8.6-9 h（冒烟实测 5.15-5.6 s/步 × 6000 步）
#
# ---- 终止预案（启动前固化，不依赖人工发现）----
#   早停      不会发生（--early_stop_patience 0）
#   DIVERGED  e1_train.log 出现 DIVERGED → 停、诊断（降 lr / 查数据）后重启，绝不带病跑完
#   OOM       降 --batch 到 1 重跑
#   崩溃      bash experiments/run_p11_w1.sh <arm> resume <epoch_N_model.pt> <N>
#   疑似挂起  连续 ~80min e1_train.log 无增长且进程仍在 → 写 run.log 告警
#
# ---- 进程级监视 ----
#   watch.sh 是训练进程的真正父进程，wait 得 rc、进程一退立即判型。
#   判活查 /proc/<pid>/stat 第 3 字段是否为 Z（kill -0 对僵尸仍返回成功，
#   E1-full 首跑即踩此坑 → 死循环永不判型）。
#   结束标记 ALL_DONE 由 runner 自己写（不依赖 watchdog 判型——实例关机会吞掉判型）。
#   [DONE] → 自动评测 best + 每个 epoch（epoch 轨迹供规划 §7.4「比整条轨迹」判定）。
#
# 用法（autodl，/root/autodl-tmp 内）：
#   bash experiments/run_p11_w1.sh <arm>
#   bash experiments/run_p11_w1.sh <arm> resume <ckpt> <已完成epoch数>
#
# arm 一览（对应规划 §3 的 W1 表）：
#   a0    配方锚点 A0（无改动）                    —— W1 全部判据的对照基准
#   1a    Exp-1a boundary head, λ=0.3（序贯首档）
#   1a01  Exp-1a λ=0.1（仅当 1a 完全无反应时）
#   1a10  Exp-1a λ=1.0（仅当 1a 有效但偏弱时）
#   1b1   Exp-1b 难例池 {lie_down}
#   1b2   Exp-1b 难例池 {lie_down, other}
#   2a    Exp-2a 多尺度（深度可分离卷积 k=8/16）
#   4     Exp-4 运动特征（姿态 12d 含加速度 + concat，忠实复现 Phase 8 E2c）
set -u
cd /root/autodl-tmp || exit 1
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1

ARM="${1:?用法: run_p11_w1.sh <arm> [resume <ckpt> <N>]}"
shift

case "$ARM" in
  a0)    EXTRA="";                                             DESC="A0 配方锚点（无改动）";;
  1a)    EXTRA="--use_boundary_head --boundary_lambda 0.3";    DESC="Exp-1a boundary λ=0.3";;
  1a01)  EXTRA="--use_boundary_head --boundary_lambda 0.1";    DESC="Exp-1a boundary λ=0.1";;
  1a10)  EXTRA="--use_boundary_head --boundary_lambda 1.0";    DESC="Exp-1a boundary λ=1.0";;
  1b1)   EXTRA="--hard_neg_classes lie_down --hard_neg_alpha 2.0";            DESC="Exp-1b 难例 {lie_down}";;
  1b2)   EXTRA="--hard_neg_classes lie_down,other --hard_neg_alpha 2.0";      DESC="Exp-1b 难例 {lie_down,other}";;
  2a)    EXTRA="--multiscale";                                 DESC="Exp-2a 多尺度 k=8/16";;
  4)     EXTRA="--pose_npz data/omnifall_pose_semantic_accel.npz"
                                                               DESC="Exp-4 姿态12d+concat";;
  *) echo "[ERROR] 未知 arm: $ARM"; exit 2;;
esac

OUT=logs/phase11/w1_$ARM
RUNLOG=$OUT/run.log
mkdir -p "$OUT"

PY=/root/miniconda3/bin/python
# 必须单行！会插值进下方 heredoc 生成的 watch.sh，多行会把 python 命令截断。
COMMON="--data_mp4 data/e1_2000_manifest.csv --val_data data/p9e1_frames_val --npz data/omnifall_dinov2_giant_frame.npz --model_name facebook/dinov2-giant --epochs 3 --early_stop_patience 0 --val_every 600 --val_n 150 --window 64 --stride 16 --batch 2 --lr 1e-4 --seed 42 --workers 16 --out $OUT $EXTRA"

RESUME_EXTRA=""
if [ "${1:-}" = "resume" ]; then
  CKPT=${2:?resume 需要 ckpt 路径}; INIT_EP=${3:?resume 需要已完成 epoch 数}
  RESUME_EXTRA=" --init_ckpt $CKPT --init_epoch $INIT_EP"
  echo "== W1/$ARM resume $(date) from $CKPT init_epoch=$INIT_EP ==" >> "$RUNLOG"
else
  echo "== W1/$ARM launch $(date) | $DESC ==" >> "$RUNLOG"
fi
COMMON="$COMMON$RESUME_EXTRA"

cat > "$OUT/watch.sh" <<EOF
#!/bin/bash
set -u
cd /root/autodl-tmp || exit 1
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1
echo "== W1/$ARM cmd: $PY -u experiments/train_p9e1.py $COMMON" >> "$RUNLOG"
$PY -u experiments/train_p9e1.py $COMMON >> "$RUNLOG" 2>&1 < /dev/null &
PID=\$!
echo "TRAIN_PID=\$PID" >> "$RUNLOG"
echo \$PID > "$OUT/train.pid"

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
echo "ALL_DONE arm=$ARM verdict=\$V rc=\$rc @ \$(date)" >> "$RUNLOG"

if [ "\$V" = done ]; then
  # 评测预算策略（2026-09-29 实测后定）：
  #   全量评测 = 53 min/ckpt，且**是真实 GPU 前向成本**（解码仅占 4%，批量前向无收益——
  #   ViT-g 在 64 帧上 batch=1 已吃满 GPU）。4 个 ckpt × 6 臂 = 21 h，规划未计。
  #   → best 跑**全量 1200**（最终数字用这个）；
  #   → epoch_* 跑固定 **300 视频随机子集**（seed=42，跨臂同子集可比），供 §7.4 轨迹判定。
  #     抽样是**确定性随机**而非前 N 个：test.csv 按路径排序、fall/ 打头，前缀有偏。
  # best 额外 --dump_pv：逐视频预测是后续算 timeline 指标（§7.2 否决条件之一）
  # 与边界指标（规划 §5）的必要输入，不存就只能重跑 53 min。
  echo "== EVAL best (full 1200, dump_pv) start @ \$(date) ==" >> "$OUT/eval_all.log"
  $PY -u experiments/eval_p9e1_dual.py --ckpt "$OUT/best_model.pt" \
      --out "$OUT/eval_best_model.json" --dump_pv "$OUT/pv_best.pt" \
      >> "$OUT/eval_all.log" 2>&1 < /dev/null
  echo "== EVAL best done rc=\$? @ \$(date) ==" >> "$OUT/eval_all.log"

  for ck in "$OUT"/epoch_*_model.pt; do
    [ -f "\$ck" ] || continue
    b=\$(basename "\$ck" .pt)
    echo "== EVAL \$b (subset 300) start @ \$(date) ==" >> "$OUT/eval_all.log"
    $PY -u experiments/eval_p9e1_dual.py --ckpt "\$ck" --out "$OUT/eval_\$b.subset.json" \
        --limit_videos 300 --subset_seed 42 >> "$OUT/eval_all.log" 2>&1 < /dev/null
    echo "== EVAL \$b done rc=\$? @ \$(date) ==" >> "$OUT/eval_all.log"
  done
  echo "ALL_EVAL_DONE @ \$(date)" >> "$OUT/eval_all.log"
fi
EOF
chmod +x "$OUT/watch.sh"

setsid nohup bash "$OUT/watch.sh" >/dev/null 2>&1 < /dev/null &
echo "launched arm=$ARM watch pid=$! -> OUT=$OUT"
echo "  $DESC"
echo "  run log:  $RUNLOG"
echo "  progress: $OUT/e1_train.log"
