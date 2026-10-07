#!/bin/bash
# 服务器端定时进度快照（每 30 min）——不依赖任何会话脚本/Claude cron。
cd /root/autodl-tmp || exit 1
O=logs/phase11/w1_2a
L=$O/progress_watch.log
prev=""
while true; do
  echo "=== $(date '+%F %T') ===" >> "$L"
  cur=$(tail -1 "$O/e1_train.log" 2>/dev/null)
  echo "$cur" >> "$L"
  if [ "$cur" = "$prev" ]; then echo "  [WARN] 与上次快照相同，日志可能停滞" >> "$L"; fi
  prev="$cur"
  if grep -q "^ALL_DONE" "$O/run.log" 2>/dev/null; then
    grep "^ALL_DONE" "$O/run.log" | tail -1 >> "$L"
    break
  fi
  sleep 1800
done
