#!/bin/bash
# Phase 11 长任务**健康检查** —— 一条命令给出"正常 / 异常"的判决。
#
# **为什么需要它**：会话和本地机器随时会关，而"实验还在正常跑吗"必须能**从任何地方**回答
# （AutoDL 网页 JupyterLab 终端、另一台机器、手机 ssh 都行）。
# 原始日志当然能看，但**判读成本高**：要自己算"这次和上次差了几步"、比时间、辨 NaN、
# 辨"进程还活着但已经不产出"（空转）——而这些正是最容易被忽略的失败模式。
# 本脚本把这一整套压成一行判决 + 关键数字，**不依赖任何会话内机制**。
#
# 用法（实例上）：
#   bash /root/autodl-tmp/experiments/p11_health.sh logs/phase11/w2_1a
# 可选：TOTAL_STEPS=28800（默认即此）
#
# ⚠ 它**只读不写**业务产物，唯一的写入是自己的一份状态文件（用于算"距上次检查的增量"）。
set -u
OUT="${1:?用法: p11_health.sh <out_dir>（如 logs/phase11/w2_1a）}"
[ -d "$OUT" ] || { echo "❌ 目录不存在: $OUT"; exit 2; }
TOTAL="${TOTAL_STEPS:-28800}"
LOG="$OUT/e1_train.log"; RUN="$OUT/run.log"
# ⚠ 状态文件必须放在**产物目录外**（2026-10-05 修正）：
#   它是本工具的临时状态，不是实验产物；放在 $OUT 里会进 tar 却因隐藏文件不进 `md5sum *`
#   清单 → 归档核对时永远报一条"清单外多出"，是**永久性假阳性**。
#   本项目核对纪律要求"必须区分文件缺失与哈希不符"，假阳性会让人对这条检查脱敏。
#   （同型先例见进度文档 §3.3 的 `.part.*` 残留坑。）
STATE="/root/autodl-tmp/.p11_health_$(basename "$OUT").state"
now=$(date '+%s')

step=$(grep -oE '^step [0-9]+' "$LOG" 2>/dev/null | tail -1 | cut -d' ' -f2); step=${step:-0}
last=$(tail -n 1 "$LOG" 2>/dev/null | cut -c1-110)
alive=no; pgrep -f 'train_p9e1.p[y]' >/dev/null 2>&1 && alive=yes
diverged=$(grep -c 'DIVERGED' "$LOG" 2>/dev/null); diverged=${diverged:-0}
nanhit=$(grep -ciE 'loss[ =]nan' "$LOG" 2>/dev/null); nanhit=${nanhit:-0}
alldone=0; grep -q 'ALL_DONE' "$RUN" 2>/dev/null && alldone=1
# run.log 里 watchdog 判型 / 疑似挂起告警
verdict=$(grep -oE 'verdict=[a-z]+' "$RUN" 2>/dev/null | tail -1 | cut -d= -f2)
stall=$(grep -c 'WATCHDOG: 疑似挂起' "$RUN" 2>/dev/null); stall=${stall:-0}
disk=$(df -h /root/autodl-tmp 2>/dev/null | tail -1 | awk '{print $4}')

# ── 增量与速率（与**上一次**运行本脚本时比）─────────────────────────────
delta="—"; rate="—"; dsec=0; dstep=0
if [ -f "$STATE" ]; then
  read -r p_t p_s < "$STATE" 2>/dev/null || true
  if [ -n "${p_t:-}" ] && [ "${p_s:-}" != "" ] && [ "$now" -gt "$p_t" ]; then
    dsec=$(( now - p_t )); dstep=$(( step - p_s ))
    delta="$dstep 步 / $((dsec/60)) 分钟"
    [ "$dstep" -gt 0 ] && rate=$(awk -v d="$dstep" -v t="$dsec" 'BEGIN{printf "%.3f", t/d}')
  fi
fi
printf '%s %s\n' "$now" "$step" > "$STATE" 2>/dev/null

# ── 判决 ────────────────────────────────────────────────────────────────
if [ "$alldone" = 1 ]; then
  st="✅ 已收工（ALL_DONE 已写，verdict=${verdict:-?}）"
elif [ "$diverged" -gt 0 ] || [ "$nanhit" -gt 0 ]; then
  st="❌ **发散 / NaN —— 立即停机诊断**（降 lr / 查数据），绝不带病跑完污染下游"
elif [ "$alive" = no ]; then
  st="❌ **进程已死**（verdict=${verdict:-未知}）→ 看 run.log 尾部判型，按终止预案处置"
elif [ "$stall" -gt 0 ]; then
  st="⚠ **watchdog 报过疑似挂起**（$stall 次）→ 查 run.log 与 e1_train.log 尾部"
elif [ "$dstep" -eq 0 ] && [ "$dsec" -ge 900 ]; then
  st="⚠ **无进展**（$((dsec/60)) 分钟 0 步）→ 疑似挂起/死循环"
elif [ "$rate" != "—" ] && [ "$(awk -v r="$rate" 'BEGIN{print (r>3.0)?1:0}')" = 1 ]; then
  st="⚠ 速率异常慢（$rate s/步，正常 ≈2.0）"
elif [ "$rate" != "—" ]; then
  st="✅ 训练中，正常（$rate s/步）"
elif [ "$step" -gt 0 ]; then
  st="✅ 训练中（首次检查，无历史可比）"
else
  st="⏳ 尚未产生 step 行（还在加载模型/首批解码？）"
fi

echo "══════════ Phase 11 健康检查 $(date '+%F %T') ══════════"
echo "  目录        : $OUT"
echo "  判决        : $st"
echo "  当前步      : $step / $TOTAL$([ "$TOTAL" -gt 0 ] && awk -v s="$step" -v t="$TOTAL" 'BEGIN{printf "  (%.1f%%)", 100*s/t}')"
echo "  距上次检查  : $delta$([ "$rate" != "—" ] && echo "   →  $rate s/步")"
echo "  训练进程    : $alive$([ "$alive" = yes ] && echo "  (pid $(pgrep -f 'train_p9e1.p[y]' | head -1))")"
echo "  最近日志    : $last"
echo "  磁盘可用    : $disk"

# ── ETA ─────────────────────────────────────────────────────────────────
if [ "$step" -gt 0 ] && [ "$step" -lt "$TOTAL" ]; then
  # 用累计平均（启动至今）外推：不依赖本次增量，**首次检查也能给 ETA**。
  # ⚠ 启动时刻必须从 run.log **首行的文本**里取，不能用它的 mtime ——
  #   训练每秒都在写 run.log，mtime≈now，el 会算成 0（初版就这么踩了，
  #   ETA 块被 `el > 120` 静默跳过，什么都不打印，还不报错）。
  launch=$(sed -n '1s/.*launch \(.*\) |.*/\1/p' "$RUN" 2>/dev/null)
  t0=$( [ -n "$launch" ] && date -d "$launch" +%s 2>/dev/null || echo 0 )
  el=$(( now - t0 ))
  if [ "${t0:-0}" -gt 0 ] && [ "$el" -gt 120 ] && [ "$step" -gt 50 ]; then
    cum=$(awk -v s="$step" -v t="$el" 'BEGIN{printf "%.3f", t/s}')
    rem=$(awk -v s="$step" -v T="$TOTAL" -v r="$cum" 'BEGIN{printf "%d", (T-s)*r}')
    echo "  累计均速    : $cum s/步（含启动开销）"
    echo "  预计训练完成: $(date -d "@$(( now + rem ))" '+%F %H:%M')"
    echo "  预计全部完成: $(date -d "@$(( now + rem + 2100 ))" '+%F %H:%M')  (≈+35 min 评测)"
  fi
fi

# ── 最近 3 次 VAL（趋势） ─────────────────────────────────────────────────
vals=$(grep -E '^\[VAL\]' "$LOG" 2>/dev/null | tail -3)
[ -n "$vals" ] && { echo "  最近 VAL    :"; echo "$vals" | sed 's/^/                /'; }

# ── 若已进入评测 ─────────────────────────────────────────────────────────
if [ -s "$OUT/eval_all.log" ]; then
  echo "  评测        :"; tail -n 3 "$OUT/eval_all.log" | sed 's/^/                /'
fi
echo "═══════════════════════════════════════════════════════════"
