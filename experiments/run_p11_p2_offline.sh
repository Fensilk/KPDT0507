#!/usr/bin/env bash
# P2 离线阶段：① 48 个密集 ckpt 全评全量 val → 选点  ② 5 个 ckpt 全量 test 评测
#
# 预登记：docs/1007实验第十一阶段增补-P2三臂重跑冻结.md（§2.2 曲线为主 / §2.4 多 ckpt 平均 / §3 全评）
#
# 用法（在**该臂所在的那台机器**上跑，项目根目录）：
#   bash experiments/run_p11_p2_offline.sh <anchor|1a|4>
#
# 成本（实测单价）：① ≈10.1 h（一次性解码 + 48×12.5 min）  ② ≈2 h（5×24 min）
#
# ⚠ 命名约定：**被选中 ckpt 的 pv 必须叫 `pv_best.pt`** —— 因为 R3 的曲线判定要调用冻结的
#   `experiments/judge_w1_arm.py`，而它的 find_pv() 只认 `pv_best_probs.pt` / `pv_best.pt`。
#   这里"best"= 按冻结规则选出的代表，不是 val 单点极大（那是旧做法）。
set -u

cd "$(dirname "$0")/.." || exit 1
PY=${PY:-/root/miniconda3/bin/python}
ARM="${1:?用法: run_p11_p2_offline.sh <anchor|1a|4>}"
# 目录可由环境变量覆盖（**复跑用**：输出目录与 probe 目录都得另置，
# 否则会写进主臂的目录、并覆盖 selection.json 里同名的选点记录）。
# 默认值与原来**逐字相同** → 主臂路径行为不变。
OUT="${OUTDIR:-logs/phase11/p2_$ARM}"
PROBE="${PROBE_DIR:-logs/phase11/p2_val_probe}"
# RERUN=1：**复跑／固定网格模式**（预登记 docs/1007…冻结.md §4b）
#   · 候选固定为 epoch_1/2/3，不做 48 ckpt 选点（复跑比的是跨 run 可比的固定网格）
#   · val 也只评这三份 —— 正是 §4b 成本栏里的「val 评测 0.6 h」
#   · **每个候选都 dump pv + timeline**：协同性检验要逐 epoch 的曲线值与逐视频口径，
#     而主臂只给"选中者" dump pv，照抄会让复跑静默产出空结果
#   · **不写 selection.json**——复跑没有"选中者"，写进去会覆盖主臂记录
RERUN="${RERUN:-0}"
LOG="$OUT/offline.log"
export HF_HUB_OFFLINE=1
export PYTHONIOENCODING=utf-8

[ -d "$OUT" ] || { echo "[ERROR] 缺 $OUT"; exit 2; }
mkdir -p "$PROBE"

# 评测侧开关：eval_p9e1_dual.py **只认 --pose_npz**（不认 use_boundary_head 等训练开关）
case "$ARM" in
  anchor) EVAL_EXTRA="";;
  1a)     EVAL_EXTRA="";;      # boundary head 不进评测路径（评测只用三元 logits）
  4)      EVAL_EXTRA="--pose_npz data/omnifall_pose_semantic_accel.npz";;
  *) echo "[ERROR] 未知 arm: $ARM"; exit 2;;
esac

say(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# ── 终止预案：任一步 rc≠0 立即停，不带着半成品往下走 ──
run(){ say "RUN: $*"; "$@" >>"$LOG" 2>&1; rc=$?; [ $rc -eq 0 ] || { say "FAILED rc=$rc: $*"; exit 1; }; }

# ════════ 阶段 ①：候选 × 全量 val ════════
# 候选集：主臂 = 48 个密集 ckpt（供选点）；复跑 = 固定网格 epoch_1/2/3。
if [ "$RERUN" = "1" ]; then
  CKPTS=()
  for c in epoch_1 epoch_2 epoch_3; do
    [ -f "$OUT/${c}_model.pt" ] && CKPTS+=("$OUT/${c}_model.pt")
  done
  say "=== 阶段①（RERUN 固定网格）：$ARM 的 val 评测，候选 ${#CKPTS[@]} 个 ==="
  [ "${#CKPTS[@]}" -ge 1 ] || { say "[ERROR] RERUN 模式找不到 epoch_1/2/3 的 ckpt"; exit 3; }
else
  mapfile -t CKPTS < <(ls -v "$OUT"/step_*_model.pt 2>/dev/null)
  say "=== 阶段① 开始：$ARM 的密集 ckpt 全评全量 val ==="
  say "step ckpt 数 = ${#CKPTS[@]}"
  [ "${#CKPTS[@]}" -ge 10 ] || { say "[ERROR] step ckpt 太少（${#CKPTS[@]}）——--save_every_check 没生效？"; exit 3; }
fi

run "$PY" -u analysis/reeval_val_ckpts.py --out "$PROBE/$ARM.json" "${CKPTS[@]}"

if [ "$RERUN" = "1" ]; then
  say "阶段① 完成（RERUN：跳过选点，**不写 selection.json**）"
else
# ── 选点：按冻结规则（读法 A：新 val 取最大），并登记到 selection.json ──
run "$PY" -u - "$PROBE/$ARM.json" "$PROBE/selection.json" "$ARM" <<'PY'
import json, sys, os
src, dst, arm = sys.argv[1], sys.argv[2], sys.argv[3]
d = json.load(open(src, encoding="utf-8"))
res = sorted(d["results"], key=lambda r: r["val_avg_f1"], reverse=True)
r = res[0]
sel = json.load(open(dst, encoding="utf-8")) if os.path.exists(dst) else {}
sel[arm] = os.path.basename(r["ckpt"]).replace("_model.pt", "")
sel[arm + "_val_avg_f1"] = round(r["val_avg_f1"], 4)
sel[arm + "_runner_up"] = os.path.basename(res[1]["ckpt"]).replace("_model.pt", "")
sel[arm + "_runner_up_val"] = round(res[1]["val_avg_f1"], 4)
sel[arm + "_n_candidates"] = len(res)
sel[arm + "_val_spread_top1_top2"] = round(r["val_avg_f1"] - res[1]["val_avg_f1"], 4)
json.dump(sel, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"[select] {arm}: {sel[arm]} (val {sel[arm+'_val_avg_f1']}), "
      f"次优 {sel[arm+'_runner_up']} (差 {sel[arm+'_val_spread_top1_top2']})")
PY
say "阶段① 完成，selection.json 已更新"
fi

# ════════ 阶段 ②：候选 × 全量 test ════════
# 主臂：选中者 + epoch_1/2/3 + 次优（去重）—— epoch 末尾 3 个是 R1/R3 的固定网格，必须都在。
# 复跑（RERUN）：只有固定网格 epoch_1/2/3，**没有"选中者"**（复跑不做选点）。
if [ "$RERUN" = "1" ]; then
  SEL="__none__"
  CANDS=$(printf '%s\n' epoch_1 epoch_2 epoch_3)
else
  SEL=$("$PY" -c "import json;print(json.load(open('$PROBE/selection.json'))['$ARM'])")
  RUNNER=$("$PY" -c "import json;print(json.load(open('$PROBE/selection.json'))['${ARM}_runner_up'])")
  CANDS=$(printf '%s\n' "$SEL" "epoch_1" "epoch_2" "epoch_3" "$RUNNER" | awk '!seen[$0]++')
fi
say "阶段② 候选：$(echo $CANDS | tr '\n' ' ')"

for c in $CANDS; do
  CK="$OUT/${c}_model.pt"
  [ -f "$CK" ] || { say "跳过（缺 ckpt）：$CK"; continue; }
  # **每个候选都 dump pv**，统一命名 `pv_<ckpt>.pt`。
  # 为什么不能只 dump 选中者（原先的做法）：预登记 §4b 的**协同性检验**要拿 anchor
  # **两次 run** 的固定网格（epoch_1/2/3）逐个对减，§7.3 的模式稳定性也要逐 epoch 的
  # 逐视频口径 —— 二者**都从 pv 派生**（timeline 也是喂 pv 算的）。只 dump 一个，
  # 这些量会**全算不出来**，而且是静默出空结果（`[ -f "$P" ] || continue`）。
  run "$PY" -u experiments/eval_p9e1_dual.py --ckpt "$CK" --out "$OUT/eval_${c}.json" \
      --dump_pv "$OUT/pv_${c}.pt" $EVAL_EXTRA
done

# 主臂额外把"选中者"的 pv **复制**一份成 pv_best.pt —— judge_w1_arm 的 find_pv 只认这个名，
# R3 的曲线判定靠它。复制而非重算：同源同 ckpt，零额外 GPU 时间。
if [ "$RERUN" != "1" ] && [ -f "$OUT/pv_${SEL}.pt" ]; then
  cp -f "$OUT/pv_${SEL}.pt" "$OUT/pv_best.pt"
  say "已复制 pv_${SEL}.pt → pv_best.pt（供 judge_w1_arm 的 find_pv）"
fi

# ── 逐视频口径（§7.3 的 ③a/③c 与事件轴读数）──
# ⚠ eval_timeline_metrics.py 收**位置参数** `name=path[:wrap]`，不是 --pv
for c in $CANDS; do
  P="$OUT/pv_${c}.pt"
  [ -f "$P" ] || continue
  run "$PY" -u experiments/eval_timeline_metrics.py "${c}=${P}" --out "$OUT/timeline_${c}.json"
done

say "=== 阶段② 完成 rc=0 ==="
say "产物：$OUT/eval_*.json, pv_best.pt, timeline_*.json；选点：$PROBE/selection.json"
say "下一步（本地）：python -u analysis/p2_judge.py"
