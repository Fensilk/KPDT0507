#!/bin/bash
# A100 实例的一次性环境准备（W2 用）。
#
# **为什么需要它**：A100 实例是**空白实例**（只能靠系统镜像复原），但——
#   · 镜像里**预置了 OmniFall 数据集**（HF 缓存里一个 9GB 的 tar）→ **视频不必上传**
#   · 系统盘只有 **30GB**，而 HF 缓存有 ~14GB → **必须挪到数据盘**，否则装不下/撑爆
#
# 本脚本做三件事：① HF 缓存迁到数据盘（软链接保持路径不变）
#                ② 从预置 tar 解出视频到代码期望的位置
#                ③ 校验（视频数 / torch / sm_80 / 磁盘）
#
# 用法（A100 上，/root/autodl-tmp 内，项目代码已就位后）：
#   bash experiments/setup_a100_w2.sh
set -u
cd /root/autodl-tmp || exit 1

echo "==================== 0) 环境概览 ===================="
echo -n "  系统盘: "; df -h / | tail -1
echo -n "  数据盘: "; df -h /root/autodl-tmp | tail -1
echo -n "  可见 GPU: "; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo -n "  cgroup CPU 上限: "; awk '{print $1/$2" 核"}' /sys/fs/cgroup/cpu.max 2>/dev/null || nproc

echo
echo "==================== 1) HF 缓存迁到数据盘 ===================="
# 系统盘 30GB 装不下 ~14GB 缓存 + pip 解包，必须挪。
# 用**移动 + 软链接**而不是改 HF_HOME：所有路径照旧解析，代码与脚本零改动。
if [ -L /root/.cache/huggingface ]; then
  echo "  已是软链接，跳过：$(readlink /root/.cache/huggingface)"
elif [ -d /root/.cache/huggingface ]; then
  sz=$(du -sh /root/.cache/huggingface | cut -f1)
  echo "  迁移中（$sz）…"
  mv /root/.cache/huggingface /root/autodl-tmp/huggingface
  ln -s /root/autodl-tmp/huggingface /root/.cache/huggingface
  echo "  ✅ 已迁到 /root/autodl-tmp/huggingface 并建软链接"
else
  echo "  ⚠ 没有 /root/.cache/huggingface（镜像可能不同）——若模型要在线下则无妨"
fi
echo -n "  迁移后系统盘: "; df -h / | tail -1 | awk '{print $4" 可用 ("$5")"}'

echo
echo "==================== 2) 从镜像预置的 tar 解出视频 ===================="
# 代码期望：DATASET-omnifall/data_files/extracted/<class>/<name>.mp4
# 镜像的 tar 里正是 ./<class>/<name>.mp4 —— 结构一致，直接解到 extracted/ 即可。
EXTRACTED=/root/autodl-tmp/DATASET-omnifall/data_files/extracted
if [ -d "$EXTRACTED/fall" ] && [ "$(ls "$EXTRACTED/fall" 2>/dev/null | wc -l)" -gt 100 ]; then
  echo "  视频已就位（fall/ 下 $(ls "$EXTRACTED/fall" | wc -l) 个），跳过"
else
  TAR=$(find /root/.cache/huggingface /root/autodl-tmp/huggingface -name '*' -type f -size +5G \
         -path '*omnifall*' 2>/dev/null | head -1)
  if [ -z "$TAR" ]; then
    echo "  ❌ 没找到预置的 OmniFall tar —— 需要改为上传视频（见 README 的备选路径）"
  else
    echo "  找到 tar: $TAR ($(du -h "$TAR" | cut -f1))"
    mkdir -p "$EXTRACTED"
    echo "  解包中（约 9GB，放到数据盘）…"
    tar xf "$TAR" -C "$EXTRACTED"
    echo "  ✅ 解包完成"
  fi
fi
echo -n "  视频总数: "; find "$EXTRACTED" -name '*.mp4' 2>/dev/null | wc -l
echo "  各类计数:"; for d in "$EXTRACTED"/*/; do printf "    %-12s %s\n" "$(basename "$d")" "$(ls "$d" 2>/dev/null | wc -l)"; done 2>/dev/null | head -12

echo
echo "==================== 3) 校验 ===================="
echo -n "  torch: "; /root/miniconda3/bin/python -c "import torch;print(torch.__version__,'| CUDA',torch.version.cuda)" 2>&1 | tail -1
echo -n "  支持的 sm: "; /root/miniconda3/bin/python -c "import torch;print(torch.cuda.get_arch_list())" 2>&1 | tail -1
echo -n "  本机算力: "; /root/miniconda3/bin/python -c "import torch;c=torch.cuda.get_device_capability();print('sm_%d%d'%c)" 2>&1 | tail -1
echo
echo "  需要上传的（镜像里没有）："
for f in data/omnifall_dinov2_giant_frame.npz data/p9e1_frames_val data/omnifall_pose_semantic_accel.npz; do
  [ -e "$f" ] && echo "    ✅ $f" || echo "    ❌ 缺 $f  ← 需从本地上传"
done
echo
echo -n "  最终磁盘: "; df -h / /root/autodl-tmp | tail -2 | tr '\n' ' '; echo
echo
echo "==================== 完成。下一步：跑 W2 ===================="
echo "  bash experiments/run_p11_w2.sh 1a     # 单卡串行；--workers 已按 10 vCPU 设为 8"
