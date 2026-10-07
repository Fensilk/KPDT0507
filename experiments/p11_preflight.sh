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

# 目标主机：默认 3090（autodl），换 A100 时用 HOST=a100 bash experiments/p11_preflight.sh --sync
# —— 两台机的远端路径都是 /root/autodl-tmp，故只有主机名需要参数化。
HOST="${HOST:-autodl}"
REMOTE=/root/autodl-tmp
FILES="
models/phase6_model.py
models/dataset.py
models/phase9_e1_model.py
experiments/train_p9e1.py
experiments/eval_p9e1_dual.py
experiments/run_p11_w1.sh
experiments/run_p11_w2.sh
experiments/run_p11_chain_0929.sh
analysis/verify_boundary_metrics.py
experiments/judge_w1_arm.py
analysis/prec_rec_curves.py
analysis/p11_hardneg_dose.py
analysis/p11_profile_dataloader.py
experiments/run_p11_profile.sh
experiments/run_p11_batchprobe.sh
experiments/setup_a100_w2.sh
experiments/setup_5090_w2.sh
experiments/p11_env_check.sh
experiments/make_wheelhouse.sh
experiments/run_p11_probs.sh
"
# ── 目录级清单（2026-10-04 补）────────────────────────────────────────
# FILES 是 per-file 清单，**已经漏过三次**：
#   models/phase9_e1_model.py（Phase 11 更早的 a0cos 事故）
#   → preprocessing/（train_p9e1 依赖 step7_extract_dinov2，其内又用 av_decode 解 AV1）
#   → experiments/lora.py（models/phase9_e1_model 反过来 import 它）
# 每次都要"真的跑到那一步"才报错，而那是**起跑之后** —— 代价是整臂白跑。
# → 结构性修法：**整目录同步 + 目录级联合哈希**，不再打地鼠。
#   analysis/ 不整目录同步（内含 472M 可视化产物 `diagnose_lying_event/`），
#   仍在 FILES 里逐文件列。
DIRS="models experiments preprocessing utils"

if [ "${1:-}" = "--sync" ]; then
  echo "== 同步 =="
  # ⚠ 必须先建远端目录：本脚本原先只在"代码已就位"的实例上用过，
  #   而**全新实例**上 $REMOTE/{models,experiments,analysis} 都不存在，
  #   直接 scp 会因为"目标目录不存在"整条失败（2026-10-04 换 5090 冷启动时暴露）。
  ssh $HOST "mkdir -p $REMOTE/models $REMOTE/experiments $REMOTE/analysis" || exit 1
  scp -q models/phase6_model.py models/dataset.py models/phase9_e1_model.py "$HOST:$REMOTE/models/" || exit 1
  scp -q experiments/train_p9e1.py experiments/eval_p9e1_dual.py \
         experiments/run_p11_w1.sh experiments/run_p11_w2.sh \
         experiments/run_p11_chain_0929.sh \
         experiments/run_p11_probs.sh \
         experiments/run_p11_profile.sh \
         experiments/run_p11_batchprobe.sh \
         experiments/setup_a100_w2.sh \
         experiments/setup_5090_w2.sh \
         experiments/p11_env_check.sh \
         experiments/make_wheelhouse.sh \
         experiments/judge_w1_arm.py "$HOST:$REMOTE/experiments/" || exit 1
  scp -q analysis/verify_boundary_metrics.py analysis/prec_rec_curves.py \
         analysis/p11_hardneg_dose.py analysis/p11_profile_dataloader.py \
         "$HOST:$REMOTE/analysis/" || exit 1
  # 整目录同步（$DIRS）。用 tar 而非 scp -r：可以顺手排除 __pycache__/*.pyc，
  # 免得把本地的陈旧字节码带到远端。
  tar cf - --exclude=__pycache__ --exclude='*.pyc' $DIRS | ssh $HOST "cd $REMOTE && tar xf -" || exit 1
  # ⚠ .sh 必须去 CR，否则远端 bash 会因行尾的 \r 报错。
  #   同步块与上面的检查清单**必须保持一致**——只加检查不加同步，
  #   会让预检永远报"远端缺失"（本次就发生了一次）。
  ssh $HOST "cd $REMOTE && for f in experiments/run_p11_w1.sh experiments/run_p11_w2.sh experiments/run_p11_chain_0929.sh experiments/run_p11_probs.sh experiments/run_p11_profile.sh experiments/run_p11_batchprobe.sh experiments/setup_a100_w2.sh experiments/setup_5090_w2.sh experiments/p11_env_check.sh experiments/make_wheelhouse.sh experiments/p11_preflight.sh; do [ -f \$f ] && sed -i 's/\r\$//' \$f; done; echo synced"
fi

echo "== 预检：逐文件哈希（忽略行尾符）=="
fail=0
for f in $FILES; do
  [ -f "$f" ] || { printf "  ⚠ %-40s 本地不存在，跳过\n" "$f"; continue; }
  lh=$(tr -d '\r' < "$f" | md5sum | cut -d' ' -f1)
  rh=$(ssh $HOST "tr -d '\r' < $REMOTE/$f 2>/dev/null | md5sum | cut -d' ' -f1" 2>/dev/null)
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

# 目录级：目录内每个 .py 取 md5 → 排序 → 再取一次**联合 md5**。
# 这样"少了一个文件"和"某个文件内容不同"都会被同一个标量暴露出来。
echo "== 预检：目录级联合哈希（$DIRS）=="
for d in $DIRS; do
  lh=$(cd "$d" 2>/dev/null && find . -type f -name '*.py' -printf '%P\n' | sort | \
       while read -r f; do printf '%s  %s\n' "$(tr -d '\r' < "$f" | md5sum | cut -d' ' -f1)" "$f"; done | md5sum | cut -d' ' -f1)
  rh=$(ssh $HOST "cd $REMOTE/$d 2>/dev/null && find . -type f -name '*.py' -printf '%P\n' | sort | while read -r f; do printf '%s  %s\n' \"\$(tr -d '\r' < \"\$f\" | md5sum | cut -d' ' -f1)\" \"\$f\"; done | md5sum | cut -d' ' -f1" 2>/dev/null)
  n=$(find "$d" -name '*.py' 2>/dev/null | wc -l)
  if [ -z "$rh" ]; then
    printf "  ❌ %-40s 远端目录缺失\n" "$d/"
    fail=$((fail+1))
  elif [ "$lh" = "$rh" ]; then
    printf "  ✅ %-40s (%s 个 .py)\n" "$d/" "$n"
  else
    printf "  ❌ %-40s 内容不同 本地=%s 远端=%s\n" "$d/" "${lh:0:8}" "${rh:0:8}"
    fail=$((fail+1))
  fi
done

echo
if [ "$fail" -eq 0 ]; then
  echo "✅ 预检通过（$(( $(echo "$FILES" | wc -w) )) 个文件 + $(( $(echo "$DIRS" | wc -w) )) 个目录一致）——可以启动"
  echo "   启动后请等到出现 'step ' 行再宣布'已启动'（进程存在 ≠ 训练在推进）"
else
  echo "❌ 预检失败：$fail 个文件不一致 —— **不要启动**，先跑 --sync"
  exit 1
fi
