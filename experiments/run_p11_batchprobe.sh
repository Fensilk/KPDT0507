#!/bin/bash
# Phase 11 性能诊断（第 3 部分）：**batch 扫描** —— 判"瓶颈是延迟还是吞吐"。
#
# **要回答的问题**（比 GPU 利用率 % 更决定性）：
#   E1 训练 5.28 s/步（3090, batch=2, window=64）。这 5.28 s 是**被延迟主导**
#   还是**被吞吐主导**？
#
#   s/step 随 batch 几乎不变 → **延迟/启动开销主导**（GPU 没吃满）
#        → 换更快的卡**基本无用**；真正的杠杆是**加 batch**（若显存允许，等于免费加速）
#   s/step 随 batch 近似线性 → **吞吐主导**（GPU 吃满）
#        → 换更快的卡**有用**（按吞吐比近似）
#
# **为什么值得单独测**：用户观察到"显存没占满"。显存占用与算力利用率是**两种独立资源**，
#   前者满不满推不出后者——但"显存有余量"确实意味着**batch 可以往上加**，
#   而 batch 加大正好能验证上面这条判据。**这一步把那个观察变成可判读的数。**
#
# **做法**：同一配方、同一数据，只改 --batch，各跑 N 步，比较 s/step。
#   ⚠ 只改 batch 会改变训练动力学 → **本脚本仅供诊断，产物在 `_profile` 目录，不参与任何实验判定**。
#   ⚠ 某个 batch 若 OOM，会被捕获并报告——那本身也是有用信息（显存天花板在哪）。
#
# 用法（autodl，/root/autodl-tmp 内）：
#   bash experiments/run_p11_batchprobe.sh              # 默认扫 2 4 8
#   BATCHES="2 4 8 16" STEPS=40 bash experiments/run_p11_batchprobe.sh
set -u
cd /root/autodl-tmp || exit 1
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1

PY=/root/miniconda3/bin/python
BATCHES="${BATCHES:-2 4 8}"
STEPS="${STEPS:-40}"
OUT=logs/phase11/_profile
mkdir -p "$OUT"
RES=$OUT/batchprobe.txt
: > "$RES"

echo "== batch 扫描 start $(date) | batches='$BATCHES' steps=$STEPS ==" | tee -a "$RES"

for B in $BATCHES; do
  LOG=$OUT/bp_b$B.log
  : > "$LOG"
  # 与 W2 完全同配方，仅 --batch 与 --max_steps 不同
  $PY -u experiments/train_p9e1.py \
      --data_mp4 DATASET-omnifall/splits/syn/random/train.csv \
      --val_data data/p9e1_frames_val --npz data/omnifall_dinov2_giant_frame.npz \
      --model_name facebook/dinov2-giant --epochs 3 --early_stop_patience 0 \
      --val_every 999999 --val_n 150 --window 64 --stride 16 \
      --batch $B --lr 1e-4 --seed 42 --workers 16 \
      --max_steps $STEPS --log_every 10 --out "$OUT/bp_b$B" >> "$LOG" 2>&1
  RC=$?

  if [ $RC -ne 0 ] || grep -qiE "out of memory|CUDA out of memory" "$LOG"; then
    echo "  batch=$B  ❌ 失败(OOM 或报错) rc=$RC —— 显存天花板在此之前" | tee -a "$RES"
    grep -iE "out of memory|Error" "$LOG" | tail -2 | sed 's/^/       /' | tee -a "$RES"
    continue
  fi

  # 从 [TOTAL] 行取总耗时；用**尾部**几个 step 区间更接近稳态（避开前几步的 cudnn 预热）
  total=$(grep -oE '^\[TOTAL\] [0-9]+ 步, [0-9.]+s' "$LOG" | grep -oE '[0-9.]+s' | tr -d 's')
  peak=$(grep -oE '显存[^0-9]*[0-9]+' "$LOG" | tail -1 || true)
  if [ -n "$total" ]; then
    echo "  batch=$B  ✅ ${total}s / $STEPS 步 = $(awk "BEGIN{printf \"%.3f\", $total/$STEPS}") s/步   ${peak:+| $peak}" | tee -a "$RES"
  else
    echo "  batch=$B  ⚠ 未取到 [TOTAL] 行，见 $LOG" | tee -a "$RES"
  fi
done

echo | tee -a "$RES"
echo "===== 判读 =====" | tee -a "$RES"
$PY - "$RES" <<'PYEOF' | tee -a "$RES"
import re, sys
rows=[]
for l in open(sys.argv[1], encoding='utf-8', errors='replace'):
    m=re.search(r'batch=(\d+)\s+✅\s+[\d.]+s / (\d+) 步 = ([\d.]+) s/步', l)
    if m: rows.append((int(m.group(1)), float(m.group(3))))
if len(rows) < 2:
    print("  可用数据点不足（≥2 个 batch 才算得出斜率），见上面逐条结果。"); raise SystemExit
b0,s0 = rows[0]
print(f"  基准：batch={b0} → {s0:.3f} s/步")
for b,s in rows[1:]:
    ratio_b, ratio_s = b/b0, s/s0
    shape = "近似线性 → **吞吐主导**（GPU 吃满）→ 换更快的卡**有用**" if ratio_s > 0.6*ratio_b \
            else "几乎不变 → **延迟/启动开销主导**（GPU 没吃满）→ 换卡**基本无用**，杠杆是加 batch"
    print(f"  batch={b:<3} 倍数 {ratio_b:.1f}× → s/步 {s:.3f}（{ratio_s:.2f}×）  {shape}")
print()
print("  ⚠ 补充：若 batch 加上去 s/步几乎不变，说明**同一张 3090 上就能免费加速**——")
print("     但那会改变训练动力学，**不能用于 W2**（W2 必须与 E1-full 的 batch=2 可比）。")
PYEOF
