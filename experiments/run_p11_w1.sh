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
#   a0cos 诊断：A0 + 余弦退火                    —— 查"val 游走"是否由恒定 lr 引起
#   1a    Exp-1a boundary head, λ=0.3（序贯首档）
#   1a01  Exp-1a λ=0.1（仅当 1a 完全无反应时）
#   1a10  Exp-1a λ=1.0（仅当 1a 有效但偏弱时）
#   1b1   Exp-1b 难例池 {lie_down}
#   1b2   Exp-1b 难例池 {lie_down, other}
#   1c    Exp-1c 显式静态轴辅助损失，λ=0.3（**替代 1b 路线**；1b 已证杠杆偏弱，见进度文档 §2）
#   1c01  Exp-1c λ=0.1（仅当 1c 把 fallen 召回打崩时）
#   1c10  Exp-1c λ=1.0（仅当 1c 有效但偏弱时）
#   2a    Exp-2a 多尺度（深度可分离卷积 k=8/16）
#   4     Exp-4 运动特征（姿态 12d 含加速度 + concat，忠实复现 Phase 8 E2c）
#
# ⚠ a0cos 不是 W1 竞争臂，是**配方诊断**：判据不是 §7.2，而是"val 带宽度是否显著收窄"。
set -u
cd /root/autodl-tmp || exit 1
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1

ARM="${1:?用法: run_p11_w1.sh <arm> [resume <ckpt> <N>]}"
shift

case "$ARM" in
  a0)    EXTRA="";                                             DESC="A0 配方锚点（无改动）";;
  a0cos) EXTRA="--lr_schedule cosine";                         DESC="诊断：A0 + 余弦退火";;
  1a)    EXTRA="--use_boundary_head --boundary_lambda 0.3";    DESC="Exp-1a boundary λ=0.3";;
  1a01)  EXTRA="--use_boundary_head --boundary_lambda 0.1";    DESC="Exp-1a boundary λ=0.1";;
  1a10)  EXTRA="--use_boundary_head --boundary_lambda 1.0";    DESC="Exp-1a boundary λ=1.0";;
  1b1)   EXTRA="--hard_neg_classes lie_down --hard_neg_alpha 2.0";            DESC="Exp-1b 难例 {lie_down}";;
  1b2)   EXTRA="--hard_neg_classes lie_down,other --hard_neg_alpha 2.0";      DESC="Exp-1b 难例 {lie_down,other}";;
  # Exp-1c：显式静态轴辅助损失（替代 1b 的难例采样路线）。
  # 只罚真值 normal 帧的 P(fallen)，不罚 P(fall)（跌倒前那几帧的 fall 概率本该升）。
  # λ=0.3 起步，与 1a 同档；序贯试（有效但偏弱→1.0；把 fallen 召回打崩→0.1），不铺网格。
  1c)    EXTRA="--aux_static_lambda 0.3";                      DESC="Exp-1c 静态轴辅助损失 λ=0.3";;
  1c10)  EXTRA="--aux_static_lambda 1.0";                      DESC="Exp-1c 静态轴辅助损失 λ=1.0";;
  1c01)  EXTRA="--aux_static_lambda 0.1";                      DESC="Exp-1c 静态轴辅助损失 λ=0.1";;
  2a)    EXTRA="--multiscale";                                 DESC="Exp-2a 多尺度 k=8/16";;
  4)     EXTRA="--pose_npz data/omnifall_pose_semantic_accel.npz"
         EVAL_EXTRA="$EXTRA"
                                                               DESC="Exp-4 姿态12d+concat";;
  *) echo "[ERROR] 未知 arm: $ARM"; exit 2;;
esac
# ⚠⚠ 评测专用参数，**不能**直接用 $EXTRA 代替。
#   2026-10-02 Exp-4 就栽在这：训练侧传了 --pose_npz、**评测侧没传**，
#   于是评测按 ckpt 里的 input_dim=3084 建模型却喂 3072 维特征 →
#   `RuntimeError: mat1 and mat2 shapes cannot be multiplied (192x3072 and 3084x384)`。
#   训练 8.8h 全部跑完（rc=0），**四次评测各 10 秒全挂**，白等一轮。
#   为什么不能透传 $EXTRA：eval_p9e1_dual.py 只认 --pose_npz，
#   不认 --use_boundary_head / --multiscale / --hard_neg_* / --aux_static_lambda，
#   整个透传会在别的 arm 上直接报未知参数。
#   **凡"新开一路输入/输出"的功能，都要同时检查：训练侧、val 侧、评测侧——三处都接上了才算接上。**
EVAL_EXTRA="${EVAL_EXTRA:-}"

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

# ⚠⚠ 2026-10-05：与 run_p11_w2.sh 同一处修正（那里有完整说明）。
#   原实现用**一个 sleep 600 轮询循环**同时扛「等训练结束」与「查是否停滞」，
#   于是 wait 最快也要等下一次醒来才返回 → 训练结束后最多 10 分钟 GPU 空转才接评测，
#   而那段时间容器里没有任何进程在跑（外部监控看到 CPU/GPU/显存全 0，像挂起）。
#   修法：解耦 —— `wait` 负责零延迟切换，后台子 shell 只管周期性查停滞。
( last=0; stall=0
  while [ -d "/proc/\$PID" ]; do
    sleep 600
    n=\$(wc -l < "$OUT/e1_train.log" 2>/dev/null || echo 0)
    if [ "\$n" -le "\$last" ]; then stall=\$((stall+1)); else stall=0; fi
    last=\$n
    if [ "\$stall" -ge 8 ]; then
      echo "WATCHDOG: 疑似挂起 pid=\$PID @ \$(date) 尾部: \$(tail -n 3 "$OUT/e1_train.log" 2>/dev/null | tr '\n' '|')" >> "$RUNLOG"
      stall=0
    fi
  done ) &
STALLER=\$!
wait \$PID; rc=\$?
kill \$STALLER 2>/dev/null; wait \$STALLER 2>/dev/null

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
      --out "$OUT/eval_best_model.json" --dump_pv "$OUT/pv_best.pt" $EVAL_EXTRA \
      >> "$OUT/eval_all.log" 2>&1 < /dev/null
  echo "== EVAL best done rc=\$? @ \$(date) ==" >> "$OUT/eval_all.log"

  for ck in "$OUT"/epoch_*_model.pt; do
    [ -f "\$ck" ] || continue
    b=\$(basename "\$ck" .pt)
    echo "== EVAL \$b (subset 300) start @ \$(date) ==" >> "$OUT/eval_all.log"
    $PY -u experiments/eval_p9e1_dual.py --ckpt "\$ck" --out "$OUT/eval_\$b.subset.json" \
        --limit_videos 300 --subset_seed 42 $EVAL_EXTRA >> "$OUT/eval_all.log" 2>&1 < /dev/null
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
