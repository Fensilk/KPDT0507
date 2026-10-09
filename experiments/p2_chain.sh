#!/usr/bin/env bash
# 训练 → 离线的**链式触发**（服务器端脱离运行）。
#
# 为什么需要：训练约在明早 11:30–11:50 结束，那时未必有人在会话里；
# 而离线阶段（① 48 ckpt × 全量 val ≈10.1 h + ② 5 ckpt × 全量 test ≈2 h）必须紧接着跑。
# CLAUDE.md 明写「定时检查不得只依赖会话内机制」——本脚本就是那条的落地。
#
# 触发条件：run.log 里出现 `ARM_DONE arm=<A> rc=0`。
#   若 rc≠0 → **不起离线**，写 CHAIN_ABORT 后退出（训练失败时跑离线毫无意义）。
#
# 用法（项目根目录）：
#   setsid nohup bash experiments/p2_chain.sh <anchor|1a|4> >/dev/null 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.." || exit 1

ARM="${1:?用法: p2_chain.sh <anchor|1a|4>}"
# RUN 可用 OUTDIR 覆盖（**复跑用**：复跑的输出目录是 p2_anchor_seed43，
# 而 $ARM 只能拼出 p2_anchor —— 硬编码会让本脚本在复跑机上**静默空转 66 h**，
# 因为 `[ -f "$RUN" ] || continue` 永远不会命中一个不存在的文件）。
# 默认值与原来逐字相同 → 主臂行为不变。
# OUTDIR / PROBE_DIR / RERUN 由环境继承传给下游 run_p11_p2_offline.sh。
RUN="${OUTDIR:-logs/phase11/p2_$ARM}/run.log"

for _ in $(seq 1 2000); do            # 上限 ≈ 2000×2min = 66 h
  sleep 120
  [ -f "$RUN" ] || continue
  if grep -q "ARM_DONE arm=$ARM rc=0" "$RUN" 2>/dev/null; then
    break
  fi
  if grep -qE "ARM_DONE arm=$ARM rc=[^0]" "$RUN" 2>/dev/null; then
    echo "CHAIN_ABORT 训练非 0 退出，**不起离线** @ $(date)" >> "$RUN"
    exit 1
  fi
done

echo "CHAIN_TRIGGER 训练完成，起离线阶段 @ $(date)" >> "$RUN"
bash experiments/run_p11_p2_offline.sh "$ARM"
rc=$?
echo "CHAIN_OFFLINE_DONE rc=$rc @ $(date)" >> "$RUN"
