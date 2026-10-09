#!/usr/bin/env bash
# Phase 11 P2 上机自检 —— **只读**，不传任何东西；缺什么就把要传的命令打出来。
#
# 为什么要有它：新实例可能是全新的（旧实例的 /root/autodl-tmp 不跟着走）。
# 本项目已几次栽在"以为环境里有、结果跑到那一步才发现没有"（A.4：预检清单漏过三次）。
#
# 用法（项目根目录）：
#   bash experiments/p2_onboard.sh
set -u

cd "$(dirname "$0")/.." || exit 1
PY=${PY:-/root/miniconda3/bin/python}
ROOT=$(pwd)
LOCAL_HINT="本地（Windows）执行"

RED=$'\033[31m'; GRN=$'\033[32m'; YEL=$'\033[33m'; NC=$'\033[0m'
ok(){ printf "  ${GRN}✅${NC} %-52s %s\n" "$1" "$2"; }
bad(){ printf "  ${RED}❌${NC} %-52s %s\n" "$1" "$2"; MISSING="$MISSING\n    $1"; }
warn(){ printf "  ${YEL}⚠${NC}  %-52s %s\n" "$1" "$2"; }
MISSING=""

echo "==================== Phase 11 P2 上机自检 ===================="
echo "  host=$(hostname)  time=$(date)  root=$ROOT"
echo

echo "── 1. 硬件与环境 ──"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | sed 's/^/     GPU /'
NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
[ "$NGPU" -ge 3 ] && ok "GPU 数 = $NGPU（够三臂并行）" || warn "GPU 数 = $NGPU" "三臂并行需要 3 张；否则串行，墙钟 ×3"
"$PY" - <<'PY' 2>/dev/null || bad "python/torch" "无法导入 torch"
import torch, transformers, sys
print(f"     py{sys.version_info.major}.{sys.version_info.minor}  torch {torch.__version__}  "
      f"cuda {torch.version.cuda}  transformers {transformers.__version__}  cuda可用={torch.cuda.is_available()}")
PY
for d in /root/autodl-tmp/huggingface/hub/models--facebook--dinov2-giant "$HOME/.cache/huggingface/hub/models--facebook--dinov2-giant"; do
  [ -d "$d" ] && ok "HF 缓存 dinov2-giant" "$d" && FOUND_HF=1 && break
done
[ "${FOUND_HF:-0}" = 1 ] || bad "HF 缓存 dinov2-giant" "缺 → 训练要联网下 1.1B 模型，务必先备好"
echo

echo "── 2. 代码（必需）──"
for f in experiments/train_p9e1.py experiments/run_p11_p2.sh \
         analysis/reeval_val_ckpts.py analysis/val_probe_checks.py; do
  [ -f "$f" ] && ok "$f" "$(wc -l < "$f") 行" || bad "$f" "缺 → 整目录同步（见文末）"
done
grep -q "save_every_check" experiments/train_p9e1.py 2>/dev/null \
  && ok "train_p9e1.py 含 --save_every_check" "（P2 唯一训练侧改动）" \
  || bad "train_p9e1.py 缺 --save_every_check" "旧版代码，必须同步"
grep -q -- "--val_csv" analysis/reeval_val_ckpts.py 2>/dev/null \
  && ok "reeval_val_ckpts.py 含 --val_csv" "" \
  || warn "reeval_val_ckpts.py 缺 --val_csv" "仍可跑（默认全量 val.csv），但版本偏旧"
echo

echo "── 3. 数据（必需；缺的要从本地传）──"
chk(){ # path desc need_size
  if [ -e "$1" ]; then ok "$1" "$2"; else bad "$1" "$3"; fi
}
chk "data/omnifall_dinov2_giant_frame.npz"   "$(du -sh data/omnifall_dinov2_giant_frame.npz 2>/dev/null|cut -f1)  期望≈2.6G" "缺（本地 2.6G）"
chk "data/omnifall_pose_semantic_accel.npz"  "$(du -sh data/omnifall_pose_semantic_accel.npz 2>/dev/null|cut -f1)  期望≈39M（arm 4 必需）" "缺（本地 39M）"
chk "DATASET-omnifall/splits/syn/random"     "train/val/test.csv" "缺（本地 276K）"
chk "DATASET-omnifall/data_files/extracted"  "$(du -sh DATASET-omnifall/data_files/extracted 2>/dev/null|cut -f1)  期望≈9.1G" "缺（本地 9.1G）"
NV=$(ls DATASET-omnifall/data_files/extracted/*/*.mp4 2>/dev/null | wc -l)
[ "$NV" -eq 12000 ] && ok "视频数" "12000" || bad "视频数 = $NV" "应为 12000"
NVL=$(ls data/p9e1_frames_val/*.pt 2>/dev/null | wc -l)
if [ "$NVL" -eq 150 ]; then ok "data/p9e1_frames_val" "150 个（E1-full 当年用的同一份）"
elif [ "$NVL" -gt 0 ]; then warn "val 缓存 = $NVL 个" "应为 150；数目不符会改变与 E1-full 的可比性"
else bad "data/p9e1_frames_val" "缺（本地 2.4G）"; fi
echo

echo "── 4. 旧 ckpt（**不阻塞训练**；只需在**一台**上齐，用于 R1/R2 的补评）──"
NOLD=0
for d in logs/phase9/e1_full logs/phase11/w2_1a logs/phase11/w2_4; do
  for c in best epoch_1 epoch_2 epoch_3; do [ -f "$d/${c}_model.pt" ] && NOLD=$((NOLD+1)); done
done
if [ "$NOLD" -eq 12 ]; then
  ok "旧 ckpt" "12 个（3 run × 4）—— 本机可承担补评"
else
  warn "旧 ckpt 只有 $NOLD/12" "本机不承担补评即可（训练不受影响；补评只需一台齐）"
fi
echo

echo "── 5. 磁盘 ──"
DF=$(df -h /root/autodl-tmp 2>/dev/null | tail -1)
echo "     $DF"
AVAIL_G=$(df -BG --output=avail /root/autodl-tmp 2>/dev/null | tail -1 | tr -dc '0-9')
# 需要：训练 3 臂各 48 份 step ckpt ≈2G/臂 = 6G + 日志/评测产物 ≈2G → 8G 起步
[ "${AVAIL_G:-0}" -ge 15 ] && ok "可用磁盘 ${AVAIL_G}G" "≥15G 充足" || bad "可用磁盘 ${AVAIL_G}G" "需 ≥15G（step ckpt 6G + 评测产物）"
echo

echo "==================== 结论 ===================="
if [ -n "$MISSING" ]; then
  echo "${RED}缺以下项${NC}：$(echo -e "$MISSING")"
  echo
  echo "在**本地**项目根目录执行（按需删掉已存在的行）："
  echo "  H='-P <端口> -i ~/.ssh/<你的私钥> root@<新实例host>'"
  echo "  rsync -avP -e \"ssh \$H\" models experiments preprocessing utils \$H:/root/autodl-tmp/   # 或 tar 整目录（A.4）"
  echo "  scp \$H:...  # 见上面逐项"
  echo "  scp -P <端口> -i <私钥> data/omnifall_dinov2_giant_frame.npz data/omnifall_pose_semantic_accel.npz  root@host:/root/autodl-tmp/data/"
  echo "  scp -r -P <端口> -i <私钥> data/p9e1_frames_val  root@host:/root/autodl-tmp/data/"
  echo "  scp -r -P <端口> -i <私钥> DATASET-omnifall/splits  root@host:/root/autodl-tmp/DATASET-omnifall/"
  echo "  scp -r -P <端口> -i <私钥> DATASET-omnifall/data_files/extracted  root@host:/root/autodl-tmp/DATASET-omnifall/data_files/"
  echo "  scp -r -P <端口> -i <私钥> logs/phase9/e1_full logs/phase11/w2_1a logs/phase11/w2_4  root@host:/root/autodl-tmp/logs/...  # 只取 *_model.pt"
  echo "  合计 ≈14.6 GB（按实测 8.8 MB/s ≈28 min）"
else
  echo "${GRN}全部就绪 ✅${NC}"
fi
echo
echo "── 就绪后的下一步（务必先冒烟）──"
echo "  1) （离线选点为**全评**方案：48 个 ckpt 直接用 val.csv 全量，无需切子集）"
echo "  2) 冒烟（必须 --val_every 1 逼出 val+存档路径，A.5 教训）："
echo "     HF_HUB_OFFLINE=1 $PY -u experiments/train_p9e1.py --data_mp4 DATASET-omnifall/splits/syn/random/train.csv \\"
echo "       --val_data data/p9e1_frames_val --npz data/omnifall_dinov2_giant_frame.npz \\"
echo "       --epochs 1 --max_steps 5 --val_every 1 --val_n 3 --window 64 --stride 16 --batch 2 \\"
echo "       --seed 42 --workers 0 --save_every_check --out /tmp/p2_smoke"
echo "     核对：出现 [VAL] 与 [CKPT] 行、/tmp/p2_smoke/step_1_model.pt 存在且大小≈41.5MB"
echo "  3) 三臂并行（每卡一臂）："
echo "     CUDA_VISIBLE_DEVICES=0 bash experiments/run_p11_p2.sh anchor &"
echo "     CUDA_VISIBLE_DEVICES=1 bash experiments/run_p11_p2.sh 1a &"
echo "     CUDA_VISIBLE_DEVICES=2 bash experiments/run_p11_p2.sh 4 &"
