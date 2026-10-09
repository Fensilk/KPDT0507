#!/bin/bash
# 5090（Blackwell / sm_120）实例的**冷启动**环境准备（W2 用）。
#
# **与 setup_a100_w2.sh 的差别（两处，都是实测才知道的）**：
#   ① A100 那台的镜像**预置了** OmniFall tar + dinov2 权重（省 13 GB 上传）；
#      **5090 用的 PyTorch 2.8.0/cu128 镜像没有**（系统盘只占 939 M，`.cache/` 是空的）
#      → 一切都要自己下。本脚本走 hf-mirror（官方 huggingface.co 在此网络下被墙）。
#   ② 镜像自带 torch 2.8.0+cu128（含 sm_120），**零环境改动**——但**代际必须先把关**：
#      torch 构建里没有 sm_120 就必然 `no kernel image`，而"从 cu124 现场升级"这条路
#      2026-10-03 已经耗掉一整轮。→ 第 0 步就判死。
#
# 实测（2026-10-04，AutoDL 5090 32G / 25 vCPU / py3.12.3）：
#   hf-mirror 下载 dinov2-giant 4.3G ≈ 10 分钟；omnifall tar 9.05G ≈ 15 分钟（可达 10 MB/s）
#   解包 12000 个 mp4 ≈ 9 秒；pip 装 5 个包 ≈ 33 秒
#
# 用法（实例上，脚本本身先 scp 上去）：
#   bash /root/autodl-tmp/setup_5090_w2.sh              # 前台
#   setsid nohup bash /root/autodl-tmp/setup_5090_w2.sh >/root/autodl-tmp/setup.log 2>&1 &
#   幂等：已完成的步骤会自动跳过（判据都是"产物存在"，不是"跑过没"）
set -u
ROOT=/root/autodl-tmp
PY=/root/miniconda3/bin/python
PIP=/root/miniconda3/bin/pip
HF=https://hf-mirror.com
TAR=$ROOT/omnifall-synthetic_av1.tar
TAR_SIZE=9716480000          # data_files/omnifall-synthetic_av1.tar 的字节数，用于判完整
EXTRACTED=$ROOT/DATASET-omnifall/data_files/extracted

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "==================== 0) ★ 算力代际硬闸门 ===================="
"$PY" - <<'PYEOF' || exit 1
import sys, torch
arch = torch.cuda.get_arch_list()
cap = torch.cuda.get_device_capability()
need = f"sm_{cap[0]}{cap[1]}"
print(f"  torch {torch.__version__} | CUDA {torch.version.cuda} | {torch.cuda.get_device_name(0)} {need}")
print(f"  arch: {arch}")
if not any(need in a for a in arch):
    print(f"  ❌ 不含 {need} —— 不要 pip 升级（cu128 依赖链走公网极慢），换镜像重建实例")
    sys.exit(1)
print(f"  ✅ 含 {need}")
PYEOF
[ $? -ne 0 ] && exit 1

log "==================== 1) 装依赖（幂等）===================="
# ⚠ transformers **必须钉 5.13.0**：那是 E1/E3 已验证过的栈。
#   装到 5.18.0 会让 W2 与 E1-full 锚点多一个版本变量——本项目的硬纪律是"唯一变量"。
"$PIP" install -q --no-input -i https://pypi.tuna.tsinghua.edu.cn/simple \
    "transformers==5.13.0" av opencv-python-headless pandas scikit-learn 2>&1 | tail -3
"$PY" -c "import transformers,av,cv2,pandas,sklearn; print('  deps ok: transformers', transformers.__version__, '| av', av.__version__)"

log "==================== 2) HF 缓存挪到数据盘 ===================="
# 系统盘只有 30G，权重 + 缓存会撑爆；用软链接保持路径不变（代码/脚本零改动）。
mkdir -p "$ROOT/huggingface"
if [ -L /root/.cache/huggingface ]; then
  log "  已是软链接：$(readlink /root/.cache/huggingface)"
else
  [ -d /root/.cache/huggingface ] && mv /root/.cache/huggingface/* "$ROOT/huggingface"/ 2>/dev/null
  rm -rf /root/.cache/huggingface
  ln -sfn "$ROOT/huggingface" /root/.cache/huggingface
  log "  已建软链接 /root/.cache/huggingface -> $ROOT/huggingface"
fi

log "==================== 3) dinov2-giant 权重（4.3 G）===================="
if [ -d "$ROOT/huggingface/hub/models--facebook--dinov2-giant" ] && \
   [ "$(du -sm "$ROOT/huggingface/hub/models--facebook--dinov2-giant" | cut -f1)" -gt 4000 ]; then
  log "  已存在，跳过"
else
  log "  下载中（hf-mirror）…"
  HF_ENDPOINT=$HF "$PY" -u -c \
    "from transformers import AutoModel; m=AutoModel.from_pretrained('facebook/dinov2-giant'); print('DINOV2_OK params=', sum(p.numel() for p in m.parameters()))" \
    || { log "  ❌ 权重下载失败"; exit 1; }
fi

log "==================== 4) OmniFall OF-Syn 视频（tar 9.05 G）===================="
# 本地那份 `DATASET-omnifall/data_files/extracted/` 就是这个 tar 解出来的（12000 个 mp4 / 10 类各 1200）。
if [ "$(find "$EXTRACTED" -name '*.mp4' 2>/dev/null | wc -l)" -eq 12000 ]; then
  log "  视频已就位（12000），跳过"
else
  if [ "$(stat -c %s "$TAR" 2>/dev/null || echo 0)" != "$TAR_SIZE" ]; then
    log "  下载 tar（支持断点续传，可重复跑）…"
    curl -sSL -C - -o "$TAR" \
      "$HF/datasets/simplexsigil2/omnifall/resolve/main/data_files/omnifall-synthetic_av1.tar" || true
  fi
  if [ "$(stat -c %s "$TAR" 2>/dev/null || echo 0)" != "$TAR_SIZE" ]; then
    log "  ❌ tar 不完整（$(stat -c %s "$TAR" 2>/dev/null || echo 0) / $TAR_SIZE）——再跑一次本脚本续传"
    exit 1
  fi
  log "  tar 完整，解包中…"
  mkdir -p "$EXTRACTED"
  tar xf "$TAR" -C "$EXTRACTED" || { log "  ❌ 解包失败"; exit 1; }
fi
N=$(find "$EXTRACTED" -name '*.mp4' 2>/dev/null | wc -l)
log "  视频总数 $N"
for d in "$EXTRACTED"/*/; do printf "    %-12s %s\n" "$(basename "$d")" "$(ls "$d" 2>/dev/null | wc -l)"; done

log "==================== 5) 总校验 ===================="
bash "$(dirname "$0")/p11_env_check.sh" 2>/dev/null || \
  bash /root/autodl-tmp/experiments/p11_env_check.sh 2>/dev/null || \
  log "  （p11_env_check.sh 未就位，跳过——代码同步后手动跑一次）"
# ⚠ 收尾提示必须**按实际状态**给：此前是两行无条件 log，在已经铺好的实例上会谎报
#   "还差 data/ 三件套"（2026-10-04 第二台铺完时真误报了一次，差点以为上传没成功）。
MISS=""
[ -f "$ROOT/data/omnifall_dinov2_giant_frame.npz" ]  || MISS="$MISS data/omnifall_dinov2_giant_frame.npz"
[ -d "$ROOT/data/p9e1_frames_val" ]                  || MISS="$MISS data/p9e1_frames_val"
[ -f "$ROOT/data/omnifall_pose_semantic_accel.npz" ] || MISS="$MISS data/omnifall_pose_semantic_accel.npz"
[ -f "$ROOT/experiments/train_p9e1.py" ]             || MISS="$MISS 代码"
[ -f "$ROOT/DATASET-omnifall/splits/syn/random/train.csv" ] || MISS="$MISS DATASET-omnifall/splits"
if [ -n "$MISS" ]; then
  log "  ⚠ 本实例还差：$MISS"
  log "    · data/ 三件套 → 从本地上传（npz 2.6G + p9e1_frames_val 2.4G + pose 40M）"
  log "    · 代码与 splits → 本地跑：HOST=<别名> bash experiments/p11_preflight.sh --sync"
else
  log "  ✅ 本实例已具备 W2 全部输入（代码 / splits / data 三件套 / 视频 / 权重 / 依赖）"
fi
log "==================== 完成 ===================="
