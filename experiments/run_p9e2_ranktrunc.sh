#!/bin/bash
# LoRA 秩截断探针：k=16 / 8 / 4 顺序评估（全量 test，官方口径 230,400 帧）。
#
# 目的：零训练地回答"r=16 是否过剩"。k=16 是本地对照——它同时验证本地流水线与
# autodl 的 eval_best.json 是否一致；有它，k=8/k=4 的差异才能归因到截断本身而非机器。
#
# 终止预案（预估 ~9h = 3 × ~3h）：
#   - 每个 k 独立产出 json，**已存在的 json 会被跳过** → 中断后重跑本脚本即自动续跑。
#   - OOM：不自动重试（batch 是脚本内定的），记入日志并停在原位，由人决定是否降窗口重试。
#   - 机器休眠会杀死进程：跑之前请把电源计划设为"不睡眠"。
#   - 失败判据：日志出现 Traceback / CUDA out of memory / 无 DONE 标记。
#
# 用法（项目根）: bash experiments/run_p9e2_ranktrunc.sh
set -u
cd "$(dirname "$0")/.." || exit 1

PY="${PY310:-/c/Users/Lizhe/anaconda3/envs/py310/python.exe}"
CKPT_SRC=logs/phase9/e1_full/best_model.pt
OUT=logs/phase9/e2_rank_trunc
LOG=$OUT/eval_all.log
export KMP_DUPLICATE_LIB_OK=TRUE PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1

mkdir -p "$OUT"
echo "== rank-trunc eval start $(date) (py=$PY) ==" | tee -a "$LOG"

for k in 16 8 4; do
  ckpt=$OUT/trunc_k${k}.pt
  out=$OUT/eval_k${k}.json
  pv=$OUT/pv_k${k}.pt
  # 跳过条件必须同时看 json 与 pv：只看 json 会在"补 dump_pv"的重跑里误跳过。
  if [ -f "$out" ] && [ -f "$pv" ]; then
    echo "== skip k=$k (json+pv 均在位) @ $(date) ==" | tee -a "$LOG"
    continue
  fi
  if [ ! -f "$ckpt" ]; then
    echo "== build trunc_k${k}.pt @ $(date) ==" | tee -a "$LOG"
    "$PY" -u experiments/lora_rank_truncate.py --ckpt "$CKPT_SRC" --out "$ckpt" --k "$k" \
      >> "$LOG" 2>&1 || { echo "!! truncate k=$k FAILED rc=$? @ $(date)" | tee -a "$LOG"; exit 1; }
  fi
  # --dump_pv：产出逐视频预测缓存，供 eval_timeline_metrics.py 算真·事件级指标
  # （覆盖率/误报率/逐视频F1/翻转/延迟）。不传它就只有 instance 口径 + 已废除的逐帧二值 ev_f1。
  echo "== eval k=$k -> $out (+$pv) @ $(date) ==" | tee -a "$LOG"
  "$PY" -u experiments/eval_p9e1_dual.py --ckpt "$ckpt" --out "$out" --dump_pv "$pv" >> "$LOG" 2>&1
  rc=$?
  echo "== done k=$k rc=$rc @ $(date) ==" | tee -a "$LOG"
  [ $rc -ne 0 ] && { echo "!! k=$k FAILED @ $(date)" | tee -a "$LOG"; exit $rc; }
done

echo "== ALL_DONE $(date) ==" | tee -a "$LOG"
