#!/bin/bash
# Phase 11 上机环境自检（**任意卡型通用**）：一次回答四问 ——
#   ① **这张卡能不能跑**：torch 构建里有没有本机的 sm_XX —— **唯一的一票否决项**
#      （2026-10-03 教训：5090 是 sm_120，而镜像自带 torch 2.5.1+cu124 只编到 sm_90，
#       SASS 不跨代前向兼容 → 必须在**选镜像**时解决，不能事后 pip 升级）
#   ② **依赖齐不齐**：缺哪个就打印**可直接粘的 pip 命令**（含 AutoDL 内网镜像）
#   ③ **配额**：vCPU / 磁盘 —— 决定 `--workers` 与 HF 缓存要不要挪到数据盘
#   ④ **数据三件套**在不在
#
# 用法（实例上，项目根目录内）：
#   bash experiments/p11_env_check.sh
#
# 覆盖项：
#   PY=/root/miniconda3/bin/python   # AutoDL 的 PyTorch 2.8 镜像里解释器路径可能不同
#   WORKERS=16                        # 只影响打印出的建议值
set -u
cd "$(dirname "$0")/.." || exit 1

PY="${PY:-/root/miniconda3/bin/python}"
if ! command -v "$PY" >/dev/null 2>&1 && [ ! -x "$PY" ]; then
  echo "⚠ 找不到 PY=$PY —— 退回到 python"
  PY=python
fi

echo "==================== 0) 机器与配额 ===================="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "  ❌ nvidia-smi 不可用"
NPROC=$(nproc 2>/dev/null || echo 0)
echo "  vCPU(nproc) : $NPROC"
# worker 建议：实测三点——3090(16 核)→16、A100(10 核)→8、5090(25 核)→16。
# 调 --workers 不破坏可比性（只影响取数速度，不改变 batch 顺序与内容）。
if [ -z "${WORKERS:-}" ]; then
  if [ "$NPROC" -ge 16 ]; then WORKERS=16; else WORKERS=8; fi
fi
echo "  → 建议 --workers $WORKERS"
echo -n "  系统盘: "; df -h / | tail -1
echo -n "  数据盘: "; df -h /root/autodl-tmp 2>/dev/null | tail -1 || echo "(无 /root/autodl-tmp)"
[ -f /sys/fs/cgroup/cpu.max ] && echo "  cgroup CPU 上限: $(awk '{print $1/$2" 核"}' /sys/fs/cgroup/cpu.max)"

echo
echo "==================== 1) ★ 算力代际（一票否决项）===================="
"$PY" - <<'PYEOF'
import sys
try:
    import torch
except Exception as e:
    print(f"  ❌ import torch 失败：{type(e).__name__}: {str(e)[:200]}")
    sys.exit(0)
print(f"  python      : {sys.version.split()[0]}")
print(f"  torch       : {torch.__version__}   CUDA {torch.version.cuda}")
arch = torch.cuda.get_arch_list() if hasattr(torch.cuda, "get_arch_list") else []
print(f"  支持算力 sm : {arch}")
if not torch.cuda.is_available():
    print("  ❌ torch.cuda 不可用 —— 驱动/容器没把 GPU 透进来")
    sys.exit(0)
cap = torch.cuda.get_device_capability()
need = f"sm_{cap[0]}{cap[1]}"
print(f"  设备        : {torch.cuda.get_device_name(0)}  {need}")
ok = any(need in a for a in arch)
if ok:
    print(f"  ✅ 本 torch 构建**包含** {need} —— 可以跑")
else:
    print(f"  ❌ 本 torch 构建**不包含** {need} —— 跑起来必报 ")
    print(f"     'no kernel image is available for execution on the device'")
    print(f"     → **不要在这里 pip 升级**（cu128 的 nvidia-* 依赖链在公网很慢，")
    print(f"       2026-10-03 就是这么耗掉一整轮的）。正确做法是**换镜像重建实例**：")
    print(f"       AutoDL 基础镜像选 PyTorch 2.8.0 / Python 3.12 / CUDA 12.8（cu128 含 sm_120）")
PYEOF

echo
echo "==================== 2) 依赖自检 ===================="
"$PY" - <<'PYEOF'
import importlib
# 全项目扫出来的第三方依赖（stdlib 不列）；huggingface_hub/safetensors 随 transformers 装
mods = ["torch", "transformers", "av", "cv2", "numpy", "pandas", "sklearn", "matplotlib"]
missing = []
for m in mods:
    try:
        mod = importlib.import_module(m)
        v = getattr(mod, "__version__", "?")
        print(f"  ✅ {m:15s} {v}")
    except Exception as e:
        print(f"  ❌ {m:15s} {type(e).__name__}")
        missing.append(m)
if missing:
    # AutoDL 内网 pip 镜像很快（公网 download.pytorch.org 才是慢的那条）
    pkgs = " ".join({"cv2": "opencv-python-headless", "sklearn": "scikit-learn"}.get(m, m) for m in missing)
    print()
    print("  → 缺依赖，粘这条装（走内网镜像，通常 1 分钟内）：")
    print(f"    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple {pkgs}")
else:
    print("  ✅ 依赖齐全")
PYEOF

echo
echo "==================== 3) 数据三件套 ===================="
for f in data/omnifall_dinov2_giant_frame.npz data/p9e1_frames_val data/omnifall_pose_semantic_accel.npz; do
  if [ -e "$f" ]; then
    if [ -d "$f" ]; then printf "  ✅ %-45s (%s 个文件)\n" "$f" "$(ls "$f" | wc -l)"
    else printf "  ✅ %-45s (%s)\n" "$f" "$(du -h "$f" | cut -f1)"; fi
  else
    printf "  ❌ %-45s ← 需从本地上传\n" "$f"
  fi
done
V=$(find DATASET-omnifall/data_files/extracted -name '*.mp4' 2>/dev/null | wc -l)
printf "  %s 视频 %-38s (%s 个 mp4)\n" "$([ "$V" -ge 12000 ] && echo ✅ || echo ⚠)" "DATASET-omnifall/data_files/extracted/" "$V"
[ "$V" -lt 12000 ] && echo "     → 少于 12000：跑 experiments/setup_5090_w2.sh 从镜像预置 tar 解出（11.6 秒）"

echo
echo "==================== 4) HF 缓存 ===================="
if [ -L /root/.cache/huggingface ]; then
  echo "  ✅ 已软链到数据盘：$(readlink /root/.cache/huggingface)"
elif [ -d /root/.cache/huggingface ]; then
  echo "  ⚠ 在系统盘上（$(du -sh /root/.cache/huggingface 2>/dev/null | cut -f1)），系统盘 30G 会爆 → 跑 setup_5090_w2.sh 迁移"
else
  echo "  ⚠ 没有 /root/.cache/huggingface（若模型要在线下则无妨）"
fi
du -sh /root/.cache/huggingface/hub/models--facebook--dinov2-giant 2>/dev/null | sed 's/^/     dinov2-giant: /' \
  || echo "     ⚠ 未见 dinov2-giant 权重 → 见 run_smoke.sh 第 2 步（HF_ENDPOINT=https://hf-mirror.com）"

echo
echo "==================== 完成 ===================="
