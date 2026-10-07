#!/bin/bash
# 把当前实例上**已经跑通**的 Python 环境，打包成可离线复用的 wheelhouse。
#
# **为什么需要**（2026-10-03/04 实测教训）：在 AutoDL 实例上装 cu128 那套依赖极慢——
# 不是内网 pip 镜像慢，而是 `download.pytorch.org` 的 cu128 轮子（torch + 一整套
# `nvidia-*` CUDA 运行时）公网拉取慢，从 cu124 **现场升级**这条路走过一次，直接耗掉一整轮。
#
# **正确姿势**（本脚本固化的顺序）：
#   ① 先把**某一张实例**的环境弄通（优先靠"选对镜像"，而不是现场装）；
#   ② 环境一通，立刻在本脚本里 `pip download` 出**版本锁定的整包轮子**；
#   ③ 拉回本地存着 —— 此后任何新实例都是"上传即用"，不再依赖公网。
#
# ⚠ 关键在 ② 的**版本锁定**：`pip freeze` 出来的是 `torch==2.8.0+cu128` 这种精确 pin，
#   pip 只会去找这个版本，因此混用 PyPI 镜像与 pytorch 索引也**不会串版本**。
#
# 用法（**在已跑通的实例上**，项目根目录内）：
#   bash experiments/make_wheelhouse.sh
# 覆盖项：
#   PY=/root/miniconda3/bin/python
#   OUT=/root/autodl-tmp/wheelhouse
#   SKIP_TORCH=1     # 只打包小依赖（torch 那套若镜像自带就不必再存 3–5 GB）
set -u
cd /root/autodl-tmp 2>/dev/null || cd "$(dirname "$0")/.." || exit 1

PY="${PY:-/root/miniconda3/bin/python}"
[ -x "$PY" ] || PY=python
OUT="${OUT:-/root/autodl-tmp/wheelhouse}"
SKIP_TORCH="${SKIP_TORCH:-0}"

mkdir -p "$OUT"
echo "==================== 0) 记录环境指纹 ===================="
"$PY" - "$OUT" <<'PYEOF' | tee "$OUT/ENV_INFO.txt"
import sys, platform, datetime
out = sys.argv[1]
print(f"采集时间 : {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
print(f"python   : {sys.version.split()[0]}  ({platform.machine()})")
try:
    import torch
    print(f"torch    : {torch.__version__}  | CUDA {torch.version.cuda}")
    print(f"arch list: {torch.cuda.get_arch_list()}")
    if torch.cuda.is_available():
        print(f"GPU      : {torch.cuda.get_device_name(0)}  sm_{''.join(map(str, torch.cuda.get_device_capability()))}")
except Exception as e:
    print(f"torch    : ❌ {e}")
PYEOF
sed 's/^/  /' "$OUT/ENV_INFO.txt"
echo "  （已写入 $OUT/ENV_INFO.txt）"

echo
echo "==================== 1) 锁定依赖清单 ===================="
# pip freeze 里可能有 `-e .` / `@ file://` 这类本地包，pip download 读不了，先剔除。
"$PY" -m pip freeze | grep -vE '^(-e |.*@ file://)' > "$OUT/requirements.lock.txt"
N=$(wc -l < "$OUT/requirements.lock.txt")
echo "  $N 个包 → $OUT/requirements.lock.txt"

if [ "$SKIP_TORCH" = "1" ]; then
  # 只留小依赖：torch/nvidia-* 那套动辄 3–5 GB，镜像已自带就不必重复存
  grep -vE '^(torch|torchvision|torchaudio|triton|nvidia-|nvidia_)' "$OUT/requirements.lock.txt" \
    > "$OUT/requirements.lock.small.txt"
  REQ="$OUT/requirements.lock.small.txt"
  echo "  SKIP_TORCH=1 → 只下小依赖（$(wc -l < "$REQ") 个）"
else
  REQ="$OUT/requirements.lock.txt"
fi

echo
echo "==================== 2) 下载轮子（离线可复用）===================="
# 双索引：内网 PyPI 镜像拿小包（快），pytorch 索引补 cu128 的 torch/nvidia-*。
# 因为清单是**精确 pin**，不存在"从错索引拿了错版本"的风险。
"$PY" -m pip download -r "$REQ" -d "$OUT/wheels" \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --extra-index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -15

echo
echo "==================== 3) 离线安装脚本 ===================="
cat > "$OUT/install_offline.sh" <<'EOS'
#!/bin/bash
# 新实例上"上传即用"：完全不联网。
# 用法：tar xzf wheelhouse.tar.gz && bash wheelhouse/install_offline.sh
set -u
cd "$(dirname "$0")" || exit 1
PY="${PY:-/root/miniconda3/bin/python}"
[ -x "$PY" ] || PY=python
$PY -m pip install --no-index --find-links "$PWD/wheels" -r "$PWD/requirements.lock.txt"
EOS
chmod +x "$OUT/install_offline.sh"
cat >> "$OUT/README.txt" <<EOS
本目录 = 一张**已跑通**的实例环境的离线段本（$(date +%F) 采集）。
  ENV_INFO.txt           环境指纹（torch/CUDA/arch list/GPU）——换卡前先看它含不含目标 sm_XX
  requirements.lock.txt  pip freeze 的精确 pin
  wheels/                对应轮子
  install_offline.sh     pip install --no-index --find-links wheels
⚠ 轮子是 **linux/x86_64 + 本机 python 版本**专用的。换 python 版本（如 3.12→3.10）需重新采集。
EOS

echo
echo "==================== 4) 打包 ===================="
du -sh "$OUT"/wheels
tar czf /root/autodl-tmp/wheelhouse.tar.gz -C "$(dirname "$OUT")" "$(basename "$OUT")"
echo -n "  包: "; du -h /root/autodl-tmp/wheelhouse.tar.gz | cut -f1
md5sum /root/autodl-tmp/wheelhouse.tar.gz
echo
echo "  → 用「浏览器下载」把它拉回本地（scp 在 200MB+ 上会掉到 ~60 KB/s，见进度文档 §3.3），"
echo "    本地存到 _env_pkgs/ 下，以后新实例直接上传展开即可。"
