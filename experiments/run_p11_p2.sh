#!/usr/bin/env bash
# Phase 11 增补 P2：三臂同环境重跑（预登记见 docs/1007实验第十一阶段增补-P2三臂重跑冻结.md）
#
# 参照系 = **基线 E1-full**（不是 W2）。p2_anchor = E1-full 命令的逐字复刻，
# 唯一差异 = 环境（3090/torch2.5.1/py3.10 → 5090/torch2.8.0/py3.12）。
# W2 只作为 1a/4 的「旧对照」，用于 R2（且两臂同在 5090 上跑过 → R2 是**纯 run 间 band**）。
#
# 三臂唯一差异 = 各自那一个开关；其余 flag（含 --val_data/--val_n/--val_every/--seed/--workers）
# **一律与 E1-full 相同**。训练期 val 原样保留：它不影响训练轨迹（patience 0、无 BN、
# eval 下不抽 dropout），留着是为了让「与基线的差异只剩环境」成为**可核查的事实**而非假设。
#
# 新增 --save_every_check：每个 val 检查点存一份 step_{N}_model.pt（48 份/臂 ≈2 GB），
# 供**离线两段式选点**用（① 48 个在小 val 上评分取 top-8 ② top-8 在全量 1200 上重评）。
# 训练计算量不变。
#
# 用法（项目根目录）：
#   bash experiments/run_p11_p2.sh anchor          # 跑单臂（前台带看门狗）
#   bash experiments/run_p11_p2.sh all             # 三臂**串行**
#   # 三臂并行（每卡一臂）——⚠ 并行时把 WORKERS 调小（默认 16→建议 8），
#   #   因为解码是 CPU 密集，3×16 个 worker 会互相抢核：
#   CUDA_VISIBLE_DEVICES=0 WORKERS=8 bash experiments/run_p11_p2.sh anchor &
#   CUDA_VISIBLE_DEVICES=1 WORKERS=8 bash experiments/run_p11_p2.sh 1a &
#   CUDA_VISIBLE_DEVICES=2 WORKERS=8 bash experiments/run_p11_p2.sh 4 &
# 脱离式启动（ssh 立刻返回）：
#   ssh ... "cd /root/autodl-tmp && setsid nohup bash experiments/run_p11_p2.sh all \
#            >/dev/null 2>&1 </dev/null & echo LAUNCHED"
set -u

cd "$(dirname "$0")/.." || exit 1

PY=${PY:-/root/miniconda3/bin/python}
WORKERS=${WORKERS:-16}
# 复跑用（默认=基线值，不改任何东西）：SEED 只影响 --seed；OUTDIR 只影响输出目录。
# ⚠ 复跑必须**只**改 SEED，OUTDIR 只用来避免覆盖 —— 其余 flag 逐字不动。
SEED=${SEED:-42}
OUTDIR=${OUTDIR:-}
ARM="${1:?用法: run_p11_p2.sh <anchor|1a|4|all>}"

export HF_HUB_OFFLINE=1          # 模型已缓存；联网会卡
export PYTHONIOENCODING=utf-8    # GBK 控制台编不了 ✅（附录 A.2 #3）

# ── 与 E1-full 逐字相同的基础命令（见 logs/phase9/e1_full/run.log）──
BASE_ARGS="--data_mp4 DATASET-omnifall/splits/syn/random/train.csv \
--val_data data/p9e1_frames_val --npz data/omnifall_dinov2_giant_frame.npz \
--model_name facebook/dinov2-giant --epochs 3 --early_stop_patience 0 \
--val_every 600 --val_n 150 --window 64 --stride 16 --batch 2 --lr 1e-4 \
--seed $SEED --workers $WORKERS --save_every_check"

arm_args() {
  case "$1" in
    anchor) echo "";;
    1a)     echo "--use_boundary_head --boundary_lambda 0.3";;
    4)      echo "--pose_npz data/omnifall_pose_semantic_accel.npz";;
    *) echo "[ERROR] 未知 arm: $1" >&2; exit 2;;
  esac
}

run_one() {
  local A="$1"
  local OUT="${OUTDIR:-logs/phase11/p2_$A}"
  local EXTRA; EXTRA="$(arm_args "$A")"
  mkdir -p "$OUT"
  local RUNLOG="$OUT/run.log"

  echo "== P2/$A launch $(date) ==" >> "$RUNLOG"
  echo "   cmd extra: ${EXTRA:-（无，逐字复刻 E1-full）}" >> "$RUNLOG"
  echo "   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<未设=全部可见>}  workers=$WORKERS" >> "$RUNLOG"

  "$PY" -u experiments/train_p9e1.py $BASE_ARGS $EXTRA --out "$OUT" \
      >> "$OUT/train.log" 2>&1 < /dev/null &
  local PID=$!
  echo "PID=$PID" >> "$RUNLOG"
  echo "[$A] launched pid=$PID  log=$OUT/train.log"

  # 进程级判型：以**进程消失**为信号。
  # ⚠ 用 /proc/<pid>/stat 第 3 字段是否 Z 判活，**不用 kill -0**（僵尸时 kill -0 仍成功）。
  while kill -0 "$PID" 2>/dev/null; do
    [ "$(awk '{print $3}' "/proc/$PID/stat" 2>/dev/null)" = "Z" ] && break
    sleep 30
  done
  wait "$PID"; local rc=$?
  echo "ARM_DONE arm=$A rc=$rc @ $(date)" >> "$RUNLOG"

  # 终止预案：非 0 退出即停（下一步；不要带着未完成的臂往下跑）
  if [ "$rc" != "0" ]; then
    echo "ARM_FAILED arm=$A rc=$rc —— 见 $OUT/train.log；已停，不继续下一臂" >> "$RUNLOG"
    return 1
  fi
  return 0
}

if [ "$ARM" = "all" ]; then
  for a in anchor 1a 4; do
    run_one "$a" || exit 1
  done
  echo "ALL_DONE_P2 @ $(date)" >> "logs/phase11/p2_anchor/run.log"
  echo "三臂全部完成"
else
  run_one "$ARM"
fi
