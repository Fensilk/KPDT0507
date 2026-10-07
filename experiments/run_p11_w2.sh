#!/bin/bash
# Phase 11 **W2** runner —— 两个 W1 过闸臂在 **9600 全量**上确认（对照 E1-full 锚点）。
#
# 为什么是这两个臂（2026-10-03 定案）：
#   W1 判出 `1a`（boundary head）与 `4`（姿态 12d）双双"强通过"，且**判据上不可分辨**。
#   而规划 §3.1 已预登记 **@2000 → @9600 不可靠**（有跨尺度秩反转的实测先例）。
#   **只跑一个 = 赌"我挑的那个恰好能转移"**；跑两个则无论结果如何都有干净归因。
#   84 h 仍在 §3 声明的 W2 预算（42–84 h）内。
#
# 配方 = **E1-full 原样复刻**（这就是对照锚点的配方，唯一变量 = 各 arm 自己那一个开关）：
#   data = DATASET-omnifall/splits/syn/random/train.csv（全量 ~9600，seed42 打散）
#   epochs 3 / patience 0 / window64 / stride16 / batch2 / lr1e-4 / seed42 / workers16
#   val = data/p9e1_frames_val(150)，val_every 600（全程 48 次），best 按 val_avg_f1
#   compile **OFF**（E1-full 当年也是闸门失败后 nocompile 跑的；本 runner 不再试 compile）
#   实测成本参考：E1-full = 28800 步 / 152144.5 s = **42.26 h**，5.28 s/步 → 每臂 ≈42.3 h
#
# ---- 终止预案（启动前固化，不依赖人工发现）----
#   早停      不会发生（--early_stop_patience 0）
#   DIVERGED  e1_train.log 出现 DIVERGED → 停、诊断（降 lr / 查数据）后重启，绝不带病跑完
#   OOM       降 --batch 到 1 重跑
#   崩溃      bash experiments/run_p11_w2.sh <arm> resume <epoch_N_model.pt> <N>
#   疑似挂起  连续 ~80min e1_train.log 无增长且进程仍在 → 写 run.log 告警
#
# ---- 进程级监视 ----
#   watch.sh 是训练进程的真正父进程，wait 得 rc、进程一退立即判型。
#   判活查 /proc/<pid>/stat 第 3 字段是否为 Z（kill -0 对僵尸仍返回成功，E1-full 首跑即踩此坑）。
#   结束标记 ALL_DONE 由 runner 自己写。  [DONE] → 自动评测 best（全量）+ 每个 epoch（300 子集）。
#
# ⚠⚠ **EVAL_EXTRA**：`4` 的评测侧必须一并传 --pose_npz，否则评测按 ckpt 里的 input_dim=3084
#   建模型却喂 3072 维特征 → 2026-10-02 Exp-4 就这么白等了一轮 8.8h（详见进度文档 §3.1）。
#   **凡"新开一路输入/输出"的功能，训练侧 / val 侧 / 评测侧三处都要接上。**
#
# 用法（autodl，/root/autodl-tmp 内）：
#   bash experiments/run_p11_w2.sh <arm>            # arm ∈ {1a, 4}
#   bash experiments/run_p11_w2.sh <arm> resume <ckpt> <已完成epoch数>
set -u
set -u
cd /root/autodl-tmp || exit 1
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1

ARM="${1:?用法: run_p11_w1.sh <arm> [resume <ckpt> <N>]}"
shift

case "$ARM" in
  # ── W2：两个 W1 过闸臂，在 9600 上按 E1-full 同配方跑（对照 E1-full 锚点，判 §7.3）──
  # 1a：boundary head λ=0.3（实现最简：无额外输入、只 +385 参数）
  1a)    EXTRA="--use_boundary_head --boundary_lambda 0.3";    DESC="W2-Exp-1a boundary λ=0.3 @9600";;
  # 4：姿态 12d + concat（⚠ 推理需外部姿态特征 → EVAL_EXTRA 必须同步给评测侧）
  4)     EXTRA="--pose_npz data/omnifall_pose_semantic_accel.npz"
         EVAL_EXTRA="$EXTRA"
                                                               DESC="W2-Exp-4 姿态12d+concat @9600";;
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

OUT=logs/phase11/w2_$ARM
RUNLOG=$OUT/run.log
mkdir -p "$OUT"

# ⚠ 解释器路径可覆盖：AutoDL 的 **PyTorch 2.8.0/CUDA 12.8 镜像**（5090 用的那个）
#   未必把 python 放在 /root/miniconda3/bin。跑前用 `which python` 确认，不同就传 PY=。
PY="${PY:-/root/miniconda3/bin/python}"

# ⚠ --workers 必须按**实例的 vCPU 配额**配，不能照抄 16：
#   · 3090 实例：16 够用
#   · **A100 实例只有 10 vCPU → 用 WORKERS=8**（16 个解码进程挤 10 核会互相拖）
#   · 5090 实例：25 vCPU → 16 原样即可
#   调它**不破坏可比性**：worker 数只影响取数速度，**不改变 batch 顺序与内容**
#   （采样顺序由主进程的 --seed 42 决定，解码是确定性的）。
#   实测依据：数据侧只占每步 3.5%（3090 上），A100 把 GPU 段压快 ~2.7× 后约 9.5%，
#   在 10 核上仍应有余量；但**跑起来后要看 util 确认**（掉到 90% 以下就是不够）。
WORKERS="${WORKERS:-16}"
# 必须单行！会插值进下方 heredoc 生成的 watch.sh，多行会把 python 命令截断。
COMMON="--data_mp4 DATASET-omnifall/splits/syn/random/train.csv --val_data data/p9e1_frames_val --npz data/omnifall_dinov2_giant_frame.npz --model_name facebook/dinov2-giant --epochs 3 --early_stop_patience 0 --val_every 600 --val_n 150 --window 64 --stride 16 --batch 2 --lr 1e-4 --seed 42 --workers $WORKERS --out $OUT $EXTRA"

RESUME_EXTRA=""
if [ "${1:-}" = "resume" ]; then
  CKPT=${2:?resume 需要 ckpt 路径}; INIT_EP=${3:?resume 需要已完成 epoch 数}
  RESUME_EXTRA=" --init_ckpt $CKPT --init_epoch $INIT_EP"
  echo "== W2/$ARM resume $(date) from $CKPT init_epoch=$INIT_EP ==" >> "$RUNLOG"
else
  echo "== W2/$ARM launch $(date) | $DESC ==" >> "$RUNLOG"
fi
COMMON="$COMMON$RESUME_EXTRA"

cat > "$OUT/watch.sh" <<EOF
#!/bin/bash
set -u
cd /root/autodl-tmp || exit 1
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1
echo "== W2/$ARM cmd: $PY -u experiments/train_p9e1.py $COMMON" >> "$RUNLOG"
$PY -u experiments/train_p9e1.py $COMMON >> "$RUNLOG" 2>&1 < /dev/null &
PID=\$!
echo "TRAIN_PID=\$PID" >> "$RUNLOG"
echo \$PID > "$OUT/train.pid"

# ⚠⚠ 2026-10-05 修正：原实现把「等训练结束」与「查是否停滞」**放进同一个 sleep 600 轮询循环**，
#   于是 `wait` 最快也要等下一次醒来才返回 → **训练结束后最多 10 分钟 GPU 空转**才接上评测，
#   而 CLI 侧的判型时刻也随之滞后。实测：两台的判型时刻**精确等于**各自 watch.sh 启动后
#   第 98 个 600 秒 tick（15:32:25 / 15:32:29），而训练真实完成是 15:23:30 / 15:27:25
#   —— 白等 8分55秒 / 5分04秒，且"0 占用"窗口看起来像挂起（用户据此提问）。
#   本项目 CLAUDE.md 要求"进程一退立即判型"，原实现只对到 10 分钟粒度。
#   修法：**把两件事解耦** —— `wait` 负责"一退就接评测"（零延迟），
#   后台子 shell 只负责"周期性查停滞"（它慢没关系，10 分钟粒度对②足够）。
#   **判型门槛不变**：仍然只有 verdict=done 才评测（不能拿废 ckpt 去评）。
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
wait \$PID; rc=\$?          # ← 子进程一退立刻返回，不再等轮询
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
