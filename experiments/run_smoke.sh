#!/bin/bash
# 换卡冒烟 —— 一次回答三个问题：① sm_120 兼容性 ② 真实 s/step ③ 管线跟不跟得上
#
# ⚠⚠ **这不是一个可直接运行的仓库 runner** —— 它是**换卡冒烟包的入口脚本**。
#    它 `cd "$(dirname "$0")"`，然后按**同目录相对路径**找下列载荷；
#    载荷已于 2026-10-09 随 `_smoke_5090_pkg/`（170 MB 打包副本）删除，此处**只留流程与参数**。
#    要真跑，须重建一个包并把本脚本放进包根：见文末「重建冒烟包」。
#
# 与 3090 的 **5.380 s/步** 比 → 得真实加速比（不靠规格表猜）。
# ⚠ 本脚本**只做诊断**，产物在 logs/smoke/，不参与任何实验判定；
#    用的是**裁剪后的冒烟数据**（200 视频 / labels-only NPZ），**不能用来出任何指标**。
set -u
cd "$(dirname "$0")" || exit 1
export PYTHONIOENCODING=utf-8

PY=${PY:-python}
STEPS="${STEPS:-40}"
# ⚠ --workers 按实例 **vCPU 配额** 配，别照抄：
#   · 5090 实例 25 核 → 16 够用
#   · **A100 实例 10 核 → WORKERS=8**（16 个解码进程挤 10 核会互相拖）
WORKERS="${WORKERS:-16}"
OUT=logs/smoke
mkdir -p "$OUT"
: > "$OUT/smoke.log"

echo "========== 0) 环境 ==========" | tee -a "$OUT/smoke.log"
$PY - <<'PYEOF' 2>&1 | tee -a "$OUT/smoke.log"
import torch, sys
print(f"  python      : {sys.version.split()[0]}")
print(f"  torch       : {torch.__version__}")
print(f"  torch CUDA  : {torch.version.cuda}")
print(f"  支持算力 sm : {torch.cuda.get_arch_list()}")
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability()
    print(f"  设备        : {torch.cuda.get_device_name(0)}  算力 sm_{cap[0]}{cap[1]}")
    print(f"  显存        : {torch.cuda.get_device_properties(0).total_memory/1073741824:.1f} GB")
    need = f"sm_{cap[0]}{cap[1]}"
    ok = any(need in a for a in torch.cuda.get_arch_list())
    print(f"  ⚠ 本 torch 构建{'**包含**' if ok else '**不包含**'} {need} 的 kernel")
    if not ok:
        print("     → 若为 sm_120 缺失，需要 CUDA 12.8+ 的 PyTorch（如 torch>=2.7 的 cu128 轮子）")
else:
    print("  ❌ torch.cuda 不可用")
PYEOF

echo | tee -a "$OUT/smoke.log"
echo "========== 1) 前向连通性（sm_120 真正的判据）==========" | tee -a "$OUT/smoke.log"
$PY - <<'PYEOF' 2>&1 | tee -a "$OUT/smoke.log"
# 只做一次真实的 DINOv2 前向 —— 若 sm_120 无 kernel，会在这里明确报错
import torch
try:
    from transformers import AutoModel
    m = AutoModel.from_pretrained("facebook/dinov2-giant").eval()
    if torch.cuda.is_available():
        m = m.half().cuda()
        x = torch.randn(1, 3, 224, 224, dtype=torch.half, device="cuda")
    else:
        x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        y = m(pixel_values=x)
    print(f"  ✅ DINOv2-giant 前向通过，输出 {tuple(y.last_hidden_state.shape)}")
    print("  → **sm_120 兼容性 = 通过**")
except Exception as e:
    print(f"  ❌ 前向失败：{type(e).__name__}: {str(e)[:300]}")
    print("  → 若为 'no kernel image' 类错误，说明当前 torch 不支持 sm_120")
PYEOF

echo | tee -a "$OUT/smoke.log"
echo "========== 2) 真实训练步（测 s/step，同时采 GPU）==========" | tee -a "$OUT/smoke.log"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -l 1 > "$OUT/gpu_trace.csv" 2>/dev/null &
SMI=$!

$PY -u experiments/train_p9e1.py \
    --data_mp4 smoke_5090_manifest.csv \
    --npz data/smoke_labels_only.npz \
    --model_name facebook/dinov2-giant \
    --window 64 --stride 16 --batch 2 --lr 1e-4 --seed 42 --workers "$WORKERS" \
    --epochs 3 --early_stop_patience 0 --val_every 999999 \
    --max_steps "$STEPS" --log_every 10 --out "$OUT/train" >> "$OUT/smoke.log" 2>&1
RC=$?

sleep 2; kill $SMI 2>/dev/null; wait $SMI 2>/dev/null
echo "  train rc=$RC" | tee -a "$OUT/smoke.log"
grep -E '^\[TOTAL\]|^step |^\[DONE\]|Error|error' "$OUT/smoke.log" | tail -8 | sed 's/^/    /' | tee -a "$OUT/smoke.log"

echo | tee -a "$OUT/smoke.log"
echo "========== 3) 结果 ==========" | tee -a "$OUT/smoke.log"
$PY - "$OUT" <<'PYEOF' | tee -a "$OUT/smoke.log"
import sys, os, re, csv
out = sys.argv[1]
log = open(os.path.join(out, "smoke.log"), encoding="utf-8", errors="replace").read()
m = re.search(r'^\[TOTAL\] (\d+) 步, ([\d.]+)s', log, re.M)
if m:
    n, t = int(m.group(1)), float(m.group(2))
    sp = t / n
    print(f"  s/step = {sp:.3f} s   （{n} 步 / {t:.1f}s）")
    print(f"  对照 3090 的 5.380 s/步 → **加速比 {5.380/sp:.2f}×**")
    print(f"  （参考：A100-40G 的估算加速比是 2.70×）")
else:
    print("  ❌ 未取到 [TOTAL] 行 —— 训练没跑完，看上面的报错")
rows = [r for r in csv.reader(open(os.path.join(out, "gpu_trace.csv"))) if len(r) >= 2 and r[0].strip().isdigit()]
if rows:
    u = sorted(int(r[0]) for r in rows)
    print(f"  GPU util 中位 {u[len(u)//2]}%  P10 {u[len(u)//10]}%  显存峰值 {max(int(r[1]) for r in rows)} MiB")
    print("  → util ≥90%：管线跟得上，**GPU 是瓶颈**，换卡的加速比按算力比兑现" if u[len(u)//2] >= 90
          else "  → ⚠ util 偏低：**CPU/数据侧顶住了**（单核慢或 workers 过多）→ 换更强的卡也拿不到满额加速")
print()
print("  ⚠ 本冒烟只验兼容性与速度，**不出任何指标**（数据是裁剪过的 200 视频）。")
PYEOF

# ─────────────────────────────────────────────────────────────────────────────
# 重建冒烟包（原 _smoke_5090_pkg/，解包后 ≈170 MB；2026-10-09 删除，配方留档）
#
#   代码   models/ experiments/ utils/ analysis/(仅 .py) preprocessing/   ~1.5 MB
#   清单   smoke_5090_manifest.csv   —— 200 行 `--data_mp4`（每类 20 个，seed=42 分层随机，
#          **不是有偏的"前 N 个"**；路径相对 DATASET-omnifall/data_files/extracted/）
#   视频   DATASET-omnifall/data_files/extracted/  200 个 mp4，157 MB
#   标签   data/smoke_labels_only.npz   8.6 MB —— 裁剪版 NPZ，只留 labels_16 /
#          video_start_indices / video_paths（Mp4WindowCache **完全不读** features，
#          所以 2.5 GB 的大头可整块砍掉）
#   不带   完整 NPZ（2.5 GB）／p9e1_frames_val（2.4 GB，--val_every 999999 下永不触发）／
#          dinov2-giant 权重（4.3 GB，让新实例自己下）／analysis/diagnose_lying_event/（472 MB 可视化产物）
#
#   用法：tar 打到新实例 /root/autodl-tmp/ 解开 → 装 HF 权重 → `WORKERS=8 bash run_smoke.sh`
