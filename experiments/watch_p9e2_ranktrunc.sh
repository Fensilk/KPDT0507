#!/bin/bash
# rank 截断探针 watchdog（服务器端，autodl）。模式复用 logs/phase9/e1_full/watch.sh，
# 但两处按项目教训修正：
#   1. 判活用 /proc/<pid>/stat 状态（kill -0 对僵尸子进程仍返回成功 → 会死循环永不判型）；
#   2. 失败签名纳入判型（Traceback / CUDA out of memory / FAILED），不只认正常完成。
# 预估 ~2.7h（3 × ~55min）。产物：eval_k{16,8,4}.json，脚本可断点续跑。
set -u
cd /root/autodl-tmp || exit 1
OUT=logs/phase9/e2_rank_trunc
LOG=$OUT/eval_all.log
RUN=$OUT/watch.log
mkdir -p "$OUT"
: > "$RUN"

alive() {   # 真活 = 进程存在且非僵尸
  local p=$1
  [ -d "/proc/$p" ] || return 1
  local st
  st=$(awk '{print $3}' "/proc/$p/stat" 2>/dev/null) || return 1
  [ "$st" != "Z" ]
}

# runner 自带日志（进度 tee 到 $LOG、python 输出显式 >> $LOG），故 stdout 不可再指回 $LOG，
# 否则每行写两遍。这里单独收 console 留档。
PY310=/root/miniconda3/bin/python nohup bash experiments/run_p9e2_ranktrunc.sh >> "$OUT/runner.console.log" 2>&1 < /dev/null &
PID=$!
echo "WATCHDOG: launched runner pid=$PID @ $(date)" >> "$RUN"
echo $PID > "$OUT/runner.pid"

last=0; stall=0
while alive $PID; do
  sleep 300
  n=$(wc -l < "$LOG" 2>/dev/null || echo 0)
  if [ "$n" -le "$last" ]; then stall=$((stall+1)); else stall=0; fi
  last=$n
  if [ "$stall" -ge 6 ]; then
    echo "WATCHDOG: 疑似挂起(≈$((stall*5))min 无日志增长) pid=$PID 仍活 @ $(date)" >> "$RUN"
    echo "  尾部: $(tail -n 3 "$LOG" | tr '\n' '|')" >> "$RUN"
    stall=0
  fi
done
wait $PID; rc=$?

if   grep -q "ALL_DONE" "$LOG" 2>/dev/null;                       then V=all_done
elif grep -qE "Traceback|CUDA out of memory|FAILED|Killed" "$LOG" 2>/dev/null; then V=crash
else V=unknown_exit; fi
echo "WATCHDOG: pid=$PID exited rc=$rc verdict=$V @ $(date)" >> "$RUN"
rm -f "$OUT/runner.pid"

# 每个 k 的结果落位一览（便于一眼看进度）
for k in 16 8 4; do
  f="$OUT/eval_k${k}.json"
  [ -f "$f" ] && echo "  k=$k: $(ls -l "$f" | awk '{print $5}') bytes" >> "$RUN" \
              || echo "  k=$k: (缺)" >> "$RUN"
done
echo "WATCHDOG: END @ $(date)" >> "$RUN"
