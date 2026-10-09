#!/bin/bash
# Phase 11 **手机友好**状态检查 —— 极短、极简、只输出两行。
#
# **为什么单独做一个**：手机浏览器里在 JupyterLab 终端敲长命令很痛苦，
# 而 AutoDL 的「实例监控」曲线又不能直接回答"实验还在正常跑吗"（尤其 GPU 利用率那条不可信）。
# 本脚本把两臂的状态压成**一人一行**，输入只需一个字符（已配 alias `h`）。
#
# 判决只看**五个互不重叠的信号**：
#   ① ALL_DONE  已收工
#   ② DIVERGED / loss nan  发散
#   ③ 训练进程不在      进程死了
#   ④ 日志 >10 分钟没更新  疑似挂起/空转（**这条最关键**：进程还在 ≠ 在推进）
#   ⑤ 其余        正常
#
# 用法（JupyterLab 终端里）：h        （alias，见 /root/.bashrc）
#                       或：bash /root/h.sh
set -u
TOTAL=28800
NOW=$(date +%s)

for pair in "1a:/root/autodl-tmp/logs/phase11/w2_1a" "4 :/root/autodl-tmp/logs/phase11/w2_4"; do
  ARM="${pair%%:*}"; OUT="${pair##*:}"
  LOG="$OUT/e1_train.log"; RUN="$OUT/run.log"

  # ⚠ 每台机器只跑一个臂 —— 另一臂的目录不存在是**正常的**，
  #   必须显式标出，否则会显示成 "step 0 ✅正常" 误导人（初版就这么错了）。
  [ -d "$OUT" ] || { printf '%-3s —（本机未跑此臂）\n' "$ARM"; continue; }

  step=$(grep -oE '^step [0-9]+' "$LOG" 2>/dev/null | tail -1 | cut -d' ' -f2); step=${step:-0}
  alive=no; pgrep -f 'train_p9e1.p[y]' >/dev/null 2>&1 && alive=yes
  alldone=0;      grep -q 'ALL_DONE'      "$RUN"              2>/dev/null && alldone=1
  allevdone=0;    grep -q 'ALL_EVAL_DONE' "$OUT/eval_all.log" 2>/dev/null && allevdone=1
  bad=$(grep -ciE 'DIVERGED|loss[ =]nan' "$LOG" 2>/dev/null); bad=${bad:-0}
  # 评测阶段的进度与"新鲜度"都看 eval_all.log —— 训练日志此时已经不再更新了
  # ⚠ 必须允许**前导空格**：实际行是 "  [400/1200] videos done, skip 0"，
  #   用 '^\[' 匹配不到（初版就这么静默显示成"启动中"）。
  evprog=$(grep -oE '\[[0-9]+/[0-9]+\] videos done' "$OUT/eval_all.log" 2>/dev/null | tail -1 | grep -oE '[0-9]+/[0-9]+')
  evage=$(( NOW - $(stat -c %Y "$OUT/eval_all.log" 2>/dev/null || echo 0) ))
  trage=$(( NOW - $(stat -c %Y "$LOG" 2>/dev/null || echo "$NOW") ))

  # ⚠ 判决顺序有讲究：`ALL_DONE` 由 runner 在**评测开始前**写（早了 ≈35 min），
  #   所以**不能**拿它当"全部结束"。真正结束的判据是 eval_all.log 里的 `ALL_EVAL_DONE`。
  #   （2026-10-05 初版把 ALL_DONE 当收工，在手机上误报"✅ 已收工"，而当时评测才跑到 400/1200。）
  if   [ "$allevdone" = 1 ]; then
    verdict="✅ 全部完成（可归档）"
  elif [ "$bad" -gt 0 ]; then
    verdict="❌ 发散/NaN → 要人工处置"
  elif [ "$alldone" = 1 ]; then
    # 训练已结束，评测阶段：看评测进度，而不是训练日志的新鲜度
    if [ "$evage" -gt 900 ]; then verdict="⚠ 评测无进展"
    else                           verdict="🔄 训练完，评测中"; fi
  elif [ "$alive" = no ]; then
    verdict="❌ 进程已死 → 要人工处置"
  elif [ "$trage" -gt 600 ]; then
    verdict="⚠ 日志 $((trage/60)) 分钟没更新 → 疑似挂起"
  else
    verdict="✅ 正常"
  fi
  age="$trage"; [ "$alldone" = 1 ] && age="$evage"

  # ETA：从 run.log 首行的启动时刻 + 累计平均外推（首次查也能给）
  eta="—"
  launch=$(sed -n '1s/.*launch \(.*\) |.*/\1/p' "$RUN" 2>/dev/null)
  t0=$( [ -n "$launch" ] && date -d "$launch" +%s 2>/dev/null || echo 0 )
  if [ "${t0:-0}" -gt 0 ] && [ "$step" -gt 50 ] && [ "$step" -lt "$TOTAL" ]; then
    el=$(( NOW - t0 ))
    rem=$(awk -v s="$step" -v T="$TOTAL" -v t="$el" 'BEGIN{printf "%d", (T-s)*t/s}')
    eta=$(date -d "@$(( NOW + rem ))" '+%m-%d %H:%M')
  fi

  if [ "$alldone" = 1 ] && [ "$allevdone" = 0 ]; then
    printf '%-3s %-24s 训练 28800/28800 ✅   评测 %-9s 日志 %4ss 前\n' \
           "$ARM" "$verdict" "${evprog:-启动中}" "$evage"
  else
    pct=$(awk -v s="$step" 'BEGIN{printf "%.1f", 100*s/28800}')
    printf '%-3s %-24s step %5s/28800 (%s%%)  日志 %4ss 前  训练完 ~%s\n' \
           "$ARM" "$verdict" "$step" "$pct" "$age" "$eta"
  fi
done
echo "  （现在 $(date '+%m-%d %H:%M')；训练完自动接评测：best 全量 1200 + 3 个 epoch 各 300 子集）"
