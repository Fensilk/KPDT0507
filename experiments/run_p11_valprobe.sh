#!/usr/bin/env bash
# Phase 11 增补 step 1：**新 val 重评**（12 个 ckpt）—— 选点规则补执行的前置试算
#
# 冻结口径见 docs/1006实验第十一阶段增补-选点规则冻结.md（F1–F8）：
#   val = val.csv 全量 1200 ｜ stride 16（2 窗，覆盖帧 0–79）
#   聚合 = 每帧只计一次，重叠区归第一个窗口 ｜ 指标 = 三类 macro F1
#
# 实测成本：单 ckpt 1200 视频 ≈12.2 min（val 帧一次性解码进 RAM，12 个 ckpt 共用）
#           + 一次性解码 ≈4 min  →  单卡串行 ≈2.5 h
#
# 终止预案（启动前已定死）：
#   * 脚本**每完成一个 ckpt 就落盘** → 中途崩掉不丢已算的，重跑会自动从头覆盖
#   * 解码失败（RuntimeError）→ 直接停，**不重试**；先查是哪个视频、为何解不出
#   * OOM 极不可能（val 帧 14.5 GB vs 本机 625 GB available），若发生 → 降 --batch
#
# 用法（项目根目录）：
#   bash experiments/run_p11_valprobe.sh                 # setsid 后台启动，立刻返回
#   tail -n +1 -f logs/phase11/val_probe/reeval.log      # 观看
set -u

cd "$(dirname "$0")/.." || exit 1

PY=${PY:-/root/miniconda3/bin/python}
OUT=logs/phase11/val_probe
LOG=$OUT/reeval.log
RUNLOG=$OUT/run.log
mkdir -p "$OUT"

export HF_HUB_OFFLINE=1          # 模型已缓存；联网会卡
export PYTHONIOENCODING=utf-8    # Windows/GBK 控制台编不了 ✅（附录 A.2 #3）

CKPTS=""
for run in logs/phase9/e1_full logs/phase11/w2_1a logs/phase11/w2_4; do
  for c in best epoch_1 epoch_2 epoch_3; do
    f="$run/${c}_model.pt"
    [ -f "$f" ] || { echo "[ERROR] 缺 ckpt: $f" >> "$RUNLOG"; exit 2; }
    CKPTS="$CKPTS $f"
  done
done

echo "== valprobe launch $(date) ==" >> "$RUNLOG"
echo "   ckpts:$CKPTS" >> "$RUNLOG"

# setsid + </dev/null：让 ssh 立刻返回，不被后台子进程拖住
setsid nohup "$PY" -u analysis/reeval_val_ckpts.py --out "$OUT/reval.json" $CKPTS \
    >> "$LOG" 2>&1 < /dev/null &
PID=$!
echo "PID=$PID" >> "$RUNLOG"
echo "launched pid=$PID  log=$LOG"

# 进程级判型：以**进程消失**为信号。
# ⚠ 用 /proc/<pid>/stat 第 3 字段是否为 Z 判活，**不用 kill -0**：
#   子进程退出后变僵尸时 `kill -0` 仍返回成功 → 循环永不退出（E1-full 首跑即卡死于此）。
for _ in $(seq 1 2000); do
  kill -0 "$PID" 2>/dev/null || break
  st=$(awk '{print $3}' "/proc/$PID/stat" 2>/dev/null)
  [ "$st" = "Z" ] && break
  sleep 15
done
wait "$PID"; rc=$?
echo "ALL_DONE rc=$rc @ $(date)" >> "$RUNLOG"
