#!/bin/bash
# Phase 11 启动前预检：**本地与远端的 Phase 11 代码是否逐字节一致**。
#
# 为什么存在（2026-09-29 事故）：
#   Exp-4 时改了 models/phase9_e1_model.py（forward 加 aux 参数），但启动 a0cos 前
#   只同步了 train_p9e1.py 与 runner，**漏了模型文件** → 远端仍是 2 参数 forward，
#   而新训练脚本用 model(x, aux) 调用 → TypeError 崩溃（rc=1 verdict=crash）。
#   更糟的是我当时的"启动确认"只查了"进程是否存在"（启动后 8 秒），
#   而崩溃发生在模型加载后的首个前向（约 5 分钟）——**检查点恰好差一点没抓到**。
#
# 两条规则由此固化：
#   ① 启动任何远端实验前，先跑本脚本；不一致就**不要启动**。
#   ② 启动后的验证必须等到"训练真的在推进"的证据（出现 step 行），
#      而不是"进程存在"——进程存在只说明它没在启动瞬间死掉。
#
# 用法（本地 Git Bash，项目根目录）：
#   bash experiments/p11_preflight.sh          # 只检查
#   bash experiments/p11_preflight.sh --sync   # 先同步再检查
set -u
cd "$(dirname "$0")/.." || exit 1

REMOTE=/root/autodl-tmp
FILES="
models/phase6_model.py
models/dataset.py
models/phase9_e1_model.py
experiments/train_p9e1.py
experiments/eval_p9e1_dual.py
experiments/run_p11_w1.sh
experiments/run_p11_chain_0929.sh
analysis/verify_boundary_metrics.py
experiments/judge_w1_arm.py
analysis/prec_rec_curves.py
experiments/run_p11_probs.sh
"

if [ "${1:-}" = "--sync" ]; then
  echo "== 同步 =="
  scp -q models/phase6_model.py models/dataset.py models/phase9_e1_model.py "autodl:$REMOTE/models/" || exit 1
  scp -q experiments/train_p9e1.py experiments/eval_p9e1_dual.py \
         experiments/run_p11_w1.sh experiments/run_p11_chain_0929.sh \
         experiments/judge_w1_arm.py "autodl:$REMOTE/experiments/" || exit 1
  scp -q analysis/verify_boundary_metrics.py "autodl:$REMOTE/analysis/" || exit 1
  # .sh 必须去 CR，否则远端 bash 会因为 \r 报错
  ssh autodl "cd $REMOTE && for f in experiments/run_p11_w1.sh experiments/run_p11_chain_0929.sh experiments/p11_preflight.sh; do [ -f \$f ] && sed -i 's/\r\$//' \$f; done; echo synced"
fi

echo "== 预检：逐文件哈希（忽略行尾符）=="
fail=0
for f in $FILES; do
  [ -f "$f" ] || { printf "  ⚠ %-40s 本地不存在，跳过\n" "$f"; continue; }
  lh=$(tr -d '\r' < "$f" | md5sum | cut -d' ' -f1)
  rh=$(ssh autodl "tr -d '\r' < $REMOTE/$f 2>/dev/null | md5sum | cut -d' ' -f1" 2>/dev/null)
  # d41d8cd9... = 空串的 md5。远端文件不存在时 `tr < missing` 无输出，
  # 也会得到这个值 —— 必须与"内容不同"区分开，否则提示会误导。
  EMPTY=d41d8cd98f00b204e9800998ecf8427e
  if [ "$rh" = "$EMPTY" ]; then
    printf "  ❌ %-40s 远端缺失（从未同步）\n" "$f"
    fail=$((fail+1))
  elif [ -n "$rh" ] && [ "$lh" = "$rh" ]; then
    printf "  ✅ %-40s\n" "$f"
  else
    printf "  ❌ %-40s 内容不同 本地=%s 远端=%s\n" "$f" "${lh:0:8}" "${rh:0:8}"
    fail=$((fail+1))
  fi
done

echo
if [ "$fail" -eq 0 ]; then
  echo "✅ 预检通过（$(( $(echo "$FILES" | wc -w) )) 个文件一致）——可以启动"
  echo "   启动后请等到出现 'step ' 行再宣布'已启动'（进程存在 ≠ 训练在推进）"
else
  echo "❌ 预检失败：$fail 个文件不一致 —— **不要启动**，先跑 --sync"
  exit 1
fi
