#!/usr/bin/env bash
# Server-side watchdog for a long Phase 10 addendum run.
# Checks process liveness by PID (NOT `pgrep -f <name>`, which self-matches), GPU state,
# log growth and NaN markers every CHECK_SEC seconds; kills the run on NaN.
#
# Usage: setsid bash experiments/watch_p10_videomae.sh <log> <pidfile> <status> [check_sec]
set -uo pipefail
LOG=${1:?log}
PIDFILE=${2:?pidfile}
STATUS=${3:?status}
CHECK=${4:-300}

while true; do
  if [ ! -f "$PIDFILE" ]; then
    echo "[watch] $(date '+%F %T') pidfile missing -> exit" >> "$STATUS"
    break
  fi
  PID=$(cat "$PIDFILE" 2>/dev/null || echo "")
  if [ -z "$PID" ] || ! kill -0 "$PID" 2>/dev/null; then
    echo "[watch] $(date '+%F %T') process $PID exited" >> "$STATUS"
    tail -30 "$LOG" >> "$STATUS" 2>/dev/null
    break
  fi
  GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader 2>/dev/null | head -1)
  NAN=$(grep -Eic "loss=(nan|inf)|loss: (nan|inf)|NaN detected" "$LOG" 2>/dev/null || true)
  echo "[watch] $(date '+%F %T') alive pid=$PID gpu=[$GPU] nan_hits=$NAN log_bytes=$(stat -c%s "$LOG" 2>/dev/null || echo 0)" >> "$STATUS"
  if [ "${NAN:-0}" -gt 0 ]; then
    echo "[watch] $(date '+%F %T') NaN/Inf detected -> terminating process group $PID" >> "$STATUS"
    kill -TERM -- "-$PID" 2>/dev/null || kill -TERM "$PID" 2>/dev/null || true
    sleep 10
    kill -KILL -- "-$PID" 2>/dev/null || true
    tail -30 "$LOG" >> "$STATUS" 2>/dev/null
    break
  fi
  sleep "$CHECK"
done
