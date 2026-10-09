#!/usr/bin/env bash
# 服务器端「停滞」检测 —— 补 CLAUDE.md 强制项的缺口。
#
# 为什么需要它：会话内的监视器（Claude Monitor）**随会话结束被拆**（2026-10-07 实测发生过一次），
# 而 CLAUDE.md 明写「定时进度检查是强制项，且**不得只依赖"会话内 + 仅 REPL 空闲才触发"的机制**，
# 须由服务器端 cron / 启动脚本自带 watchdog 承担」。
#
# 分工：
#   p2_extwatch.sh  —— 管**进程退出**（往 run.log 写 EXT_WATCHDOG_EXIT）
#   本脚本          —— 管**进程活着但不产出**（往 run.log 写 STALL），这才是"空转"的形态
#
# 判据：train.log 的 mtime 超过 STALL_MIN 分钟没更新 → 判停滞。
#   正常节奏：每 50 步一行（≈1.7 min）、每 600 步一次 [VAL]（≈20 min）。
#   取 45 min 作阈值：足够宽（跨过 val 检查），又能在 1 小时内发现问题。
#
# 用法：
#   setsid nohup bash experiments/p2_stallwatch.sh <train.log> <run.log> [STALL_MIN] [PID] \
#       >/dev/null 2>&1 < /dev/null &
#
# 可选第 4 参 PID = **被监视的训练进程**。给了它就多一条退出条件：
#   进程消失（含变僵尸）→ 正常退出，不再判停滞。
#   为什么必须给：本脚本只看 train.log 的 mtime，**完全不认识训练进程**。训练正常结束后
#   train.log 永久冻结（ARM_DONE 写的是 run.log），45 min 后**必然**误报一次 STALL
#   （2026-10-08 实测确认）。那时它既无监控价值——离线阶段的产出在 offline.log——
#   又污染 run.log，还会让"STALL"这个唯一管"活着但不产出"的信号贬值。
set -u

TRAIN="${1:?用法: p2_stallwatch.sh <train.log> <run.log> [STALL_MIN] [PID]}"
RUN="${2:?用法: p2_stallwatch.sh <train.log> <run.log> [STALL_MIN] [PID]}"
STALL_MIN="${3:-45}"
PID="${4:-}"

while true; do
  sleep 600                       # 每 10 分钟看一次
  if [ -n "$PID" ]; then
    # 判活沿用 p2_extwatch.sh：查 /proc/<pid>/stat 第 3 字段是否 Z。
    # 不用 kill -0 —— 子进程成僵尸时 kill -0 仍返回成功（E1-full 首跑即卡死于此）。
    if [ ! -d "/proc/$PID" ]; then
      echo "STALLWATCH_RETIRED 训练进程 $PID 已消失，正常退出（非停滞）@ $(date)" >> "$RUN"
      exit 0
    fi
    if [ "$(awk '{print $3}' "/proc/$PID/stat" 2>/dev/null)" = "Z" ]; then
      echo "STALLWATCH_RETIRED 训练进程 $PID 已变僵尸，正常退出（非停滞）@ $(date)" >> "$RUN"
      exit 0
    fi
  fi
  [ -f "$TRAIN" ] || continue
  now=$(date +%s)
  mt=$(stat -c %Y "$TRAIN" 2>/dev/null) || continue
  age=$(( (now - mt) / 60 ))
  if [ "$age" -ge "$STALL_MIN" ]; then
    # 只在**跨过阈值的那一刻**报一次，避免刷屏
    if [ ! -f "${RUN}.stalled" ]; then
      echo "STALL train.log 已 ${age} 分钟未更新（阈值 ${STALL_MIN}）@ $(date)" >> "$RUN"
      touch "${RUN}.stalled"
    fi
  else
    [ -f "${RUN}.stalled" ] && rm -f "${RUN}.stalled"    # 恢复产出则复位
  fi
done
