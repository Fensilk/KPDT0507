#!/bin/bash
# 进程级 watchdog（独立会话，session 无关）：pid 一消失就在 logfile 追加标记。
# 用法: bash watch_pid.sh <pid> <logfile>
PID=$1; LOG=$2
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
echo "WATCHDOG: pid $PID gone at $(date)" >> "$LOG"
