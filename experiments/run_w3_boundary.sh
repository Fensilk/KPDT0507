#!/bin/bash
# W3：边界指标（本阶段自建指标 §5）—— 对 W2 全部 ckpt + 锚点补算。零 GPU，纯本地。
#
# ⚠ 2026-10-05 修：**必须 export PYTHONIOENCODING=utf-8**。
#   这些工具会 printEmoji（✅/❌），而 Windows 控制台默认 GBK 编不了 → UnicodeEncodeError，
#   **整个脚本 rc=1 却没有产出**。首次运行 9/9 全静默失败，而我当时看到 DONE 标记就以为成功了
#   —— 教训与项目一贯纪律一致：**"脚本跑完"≠"产物正确"，必须校验产物本身**。
#   （本会话同一个 GBK 坑已踩三次：judge_w1_arm.py ×2、verify_boundary_metrics.py ×1。）
set -u
export PYTHONIOENCODING=utf-8
# 按仓库约定从项目根运行（脚本已从仓库根移入 experiments/，2026-10-09）
cd "$(dirname "$0")/.."
PY=/c/Users/Lizhe/anaconda3/envs/py310/python.exe
SUMMARY=logs/phase11/w2_boundary_all.log
: > "$SUMMARY"
ok=0; bad=0
run() {  # $1=tag  $2=pv  $3=out_json  $4=out_stdout
  rm -f "$3"
  "$PY" -u analysis/verify_boundary_metrics.py --pv "$2" --out "$3" --tag "$1" > "$4" 2>&1
  rc=$?
  if [ "$rc" -eq 0 ] && [ -s "$3" ]; then
    ok=$((ok+1)); status="✅"
  else
    bad=$((bad+1)); status="❌ rc=$rc"
  fi
  { echo "######## $1 ########"; echo "  状态: $status";
    sed -n '/的边界指标/,$p' "$4"; echo; } >> "$SUMMARY"
}
[ -f logs/phase9/e1_full/pv_ep3.pt ] && \
  run "E1-full ep3" logs/phase9/e1_full/pv_ep3.pt \
      logs/phase9/e1_full/boundary_metrics_e1full_ep3.json \
      logs/phase9/e1_full/boundary_e1full_ep3.stdout.log
for arm in w2_1a w2_4; do
  for ck in best 1 2 3; do
    if [ "$ck" = best ]; then pv="logs/phase11/$arm/pv_best.pt"; else pv="logs/phase11/$arm/pv_epoch_$ck.pt"; fi
    [ -f "$pv" ] || continue
    run "$arm $ck" "$pv" "logs/phase11/$arm/boundary_metrics_$ck.json" "logs/phase11/$arm/boundary_$ck.stdout.log"
  done
done
echo "W3_BOUNDARY_DONE  成功 $ok / 失败 $bad" >> "$SUMMARY"
