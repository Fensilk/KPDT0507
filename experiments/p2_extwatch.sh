#!/usr/bin/env bash
# 外部看门狗：只做一件事 —— **等 PID 消失，然后往 run.log 追加一行**。
#
# 为什么不依赖 runner 自带的看门狗：runner 的看门狗活在与 ssh **同一个会话**里，
# ssh 一断它就没（本阶段刚踩过：setsid nohup 仍会让 ssh 挂住，靠 timeout 兜住）。
# 而这个由调用方用 setsid 脱离启动，能兜住"**进程已死、但 ARM_DONE 没写出来**"的情况——
# 而 ARM_DONE 正是监视器判定结束的依据。
#
# ⚠ 判活用 /proc/<pid>/stat 第 3 字段是否为 Z，**不用 kill -0**：
#   子进程退出后变僵尸时 kill -0 仍返回成功 → 循环永不退出（E1-full 首跑即卡死于此）。
#
# 用法：
#   setsid nohup bash experiments/p2_extwatch.sh <pid> <run.log> >/dev/null 2>&1 < /dev/null &
set -u

P="${1:?用法: p2_extwatch.sh <pid> <run.log>}"
LOG="${2:?用法: p2_extwatch.sh <pid> <run.log>}"

while kill -0 "$P" 2>/dev/null; do
  [ "$(awk '{print $3}' "/proc/$P/stat" 2>/dev/null)" = "Z" ] && break
  sleep 60
done
echo "EXT_WATCHDOG_EXIT pid=$P @ $(date)" >> "$LOG"
