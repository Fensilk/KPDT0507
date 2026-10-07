#!/bin/bash
# Phase 11 性能诊断（第 2 部分）：**真实训练步的 GPU 忙碌率**。
#
# **要回答的问题**：E1 训练在 3090 上实测 5.28 s/步、42.26 h/轮。若换成
# A100/H100（单价 3–5 倍），值不值？
#
# **判据**（不依赖任何估算，只看实测）：
#     GPU 在步内 busy ≈100%  → GPU 是串行瓶颈 → **换更快的卡有用**（按吞吐比近似）
#     GPU busy 明显偏低      → 步被 kernel 之间的**空隙**主导（Python/launch/数据）
#                              → 换卡**基本无用**，该动的是 batch / compile / 数据管线
#
# **做法**：跑一段**真实**的 train_p9e1.py（W2 同配方，仅 --max_steps 截断），
#   同时用 nvidia-smi 以 1 Hz 采样，算 busy 的均值/中位/P90。
#   **不改任何训练代码**；采样从**首个 `step ` 行出现后**才开始（避开模型加载期，
#   否则那段近零的 util 会把均值拖垮）。
#
# 用法（autodl，/root/autodl-tmp 内）：
#   bash experiments/run_p11_profile.sh                 # 默认：arm 1a 的配置（无额外输入）
#   bash experiments/run_p11_profile.sh pose            # arm 4 的配置（带 --pose_npz）
set -u
cd /root/autodl-tmp || exit 1
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1

PY=/root/miniconda3/bin/python
STEPS="${STEPS:-60}"
OUT=logs/phase11/_profile
mkdir -p "$OUT"
TRACE=$OUT/gpu_trace.csv
LOG=$OUT/profile.log

EXTRA=""
[ "${1:-}" = "pose" ] && EXTRA="--pose_npz data/omnifall_pose_semantic_accel.npz"

: > "$TRACE"; : > "$LOG"
echo "== profile start $(date) | steps=$STEPS extra='$EXTRA' ==" | tee -a "$LOG"

# 训练命令行**与 W2 完全同配方**（仅 --max_steps 截断 + 独立 out 目录）
CMD="--data_mp4 DATASET-omnifall/splits/syn/random/train.csv --val_data data/p9e1_frames_val --npz data/omnifall_dinov2_giant_frame.npz --model_name facebook/dinov2-giant --epochs 3 --early_stop_patience 0 --val_every 999999 --val_n 150 --window 64 --stride 16 --batch 2 --lr 1e-4 --seed 42 --workers 16 --max_steps $STEPS --out $OUT $EXTRA"

$PY -u experiments/train_p9e1.py $CMD >> "$OUT/train.log" 2>&1 &
TRAIN_PID=$!
echo "train pid=$TRAIN_PID，等首个 'step ' 行（模型加载约 5 min）…" | tee -a "$LOG"

# 等训练真正开始（首个 step 行）再开采样 —— 这是本脚本的关键，别省
for _ in $(seq 1 240); do
  grep -q '^step ' "$OUT/train.log" 2>/dev/null && break
  kill -0 $TRAIN_PID 2>/dev/null || { echo "❌ 训练进程已退，见 $OUT/train.log" | tee -a "$LOG"; tail -20 "$OUT/train.log" | tee -a "$LOG"; exit 1; }
  sleep 5
done
echo "首个 step 行已出现，开始 1 Hz 采样 GPU…" | tee -a "$LOG"

nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw \
           --format=csv,noheader,nounits -l 1 > "$TRACE" 2>/dev/null &
SMI_PID=$!

wait $TRAIN_PID; RC=$?
sleep 2; kill $SMI_PID 2>/dev/null; wait $SMI_PID 2>/dev/null

echo "== train rc=$RC @ $(date) ==" | tee -a "$LOG"
echo | tee -a "$LOG"
echo "===== 训练尾部 =====" | tee -a "$LOG"
tail -6 "$OUT/train.log" | tee -a "$LOG"
echo | tee -a "$LOG"
echo "===== GPU 忙碌率（采样 $(wc -l < "$TRACE") 点）=====" | tee -a "$LOG"
$PY - "$TRACE" <<'PYEOF' | tee -a "$LOG"
import sys, csv
rows=[r for r in csv.reader(open(sys.argv[1])) if len(r)>=3 and r[0].strip().isdigit()]
if not rows:
    print("  ❌ 无采样数据"); raise SystemExit
u=[int(r[0]) for r in rows]; m=[int(r[1]) for r in rows]; mem=[int(r[2]) for r in rows]
s=sorted(u); n=len(s)
q=lambda p: s[min(n-1,int(p*n))]
print(f"  GPU util : 均值 {sum(u)/n:6.1f}%  中位 {q(.5):3d}%  P10 {q(.1):3d}%  P90 {q(.9):3d}%  最大 {max(u)}%")
print(f"  mem  util: 均值 {sum(m)/n:6.1f}%  显存峰值 {max(mem)} MiB")
print(f"  低忙占比 : util<50% 的点 {100*sum(1 for x in u if x<50)/n:.1f}%  |  util<20% 的点 {100*sum(1 for x in u if x<20)/n:.1f}%")
print()
med=q(.5)
if med >= 85:
    print("  判读：GPU **几乎全程忙碌** → 它是串行瓶颈 → **换更快的卡有用**（按吞吐比近似）")
elif med >= 50:
    print("  判读：GPU 一半左右时间在忙 → 换卡**部分有用**，但有一半时间花在空隙上")
else:
    print("  判读：GPU **大部分时间空闲** → 瓶颈在 kernel 之间的空隙（Python/launch/数据）")
    print("        → **换卡基本无用**，该动的是 batch / compile / 数据管线")
PYEOF
echo | tee -a "$LOG"
echo "ⓘ 如需 kernel 级归因（哪些算子最耗时），再上 torch.profiler（需给 train_p9e1.py 加开关）。" | tee -a "$LOG"
