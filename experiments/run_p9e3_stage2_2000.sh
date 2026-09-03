#!/bin/bash
# E3@2000：E3 stage2（两阶段时序模型）只用与 E1 完全相同的 2,000 个训练视频。
#
# 目的：受控对比 E1@2000（端到端 LoRA）vs E3@2000（两阶段·冻特征·同数据），
#       拆掉"E3 参考数字其时序头是全量 9,600 视频训的"这一数据量不对称。
# 关键：本 runner 作为 E1 的受控，训练窗口步长须与 E1 对齐（stride 16，2 窗口/视频），
#       否则步长又成第二个变量（E1 训练用 stride16）。要"忠实原 E3 配方(stride8)"的
#       参考数字，把下面 --stride 改回 8 即可（等价 run_p9e3_stage2.sh 只换 train split）。
#
# 训练集：DATASET-omnifall/splits/syn/e1_2000/train.csv = E1 的 2,000 视频
#         （seed-42 抽样，与 prep_p9e1_frames.py 同批同序，首个 = lie_down/lie_down_to_088）。
# val/test：沿用标准 split（评估在全量 test 上进行，与 E1 eval / 原 E3 同口径）。
#
# 前置：data/omnifall_dinov2_finetuned_frame.npz（微调特征全量 NPZ）存在。
#       - 本地有（2.6G）；autodl 上曾因清盘被删，如要 autodl 跑需先 scp 回。
# 运行：冻结特征时序训练很轻，本地 py310 / 4060 即可跑；无需 24G。
#
set -e
cd "$(dirname "$0")/.."

# DATASET-omnifall/ 被 .gitignore（数据目录）。split 不随 git 走：
# train.csv 缺失时用与 prep_p9e1_frames.py 完全相同的 seed-42 抽样重建。
SPLIT_DIR="DATASET-omnifall/splits/syn/e1_2000"
if [ ! -f "$SPLIT_DIR/train.csv" ]; then
  echo "[E3@2000] 重建 E1 同批 2000 训练集 split: $SPLIT_DIR"
  python - <<'PY'
import os, shutil, pandas as pd, numpy as np
base = "DATASET-omnifall/splits/syn"; std = f"{base}/random"; out = f"{base}/e1_2000"
os.makedirs(out, exist_ok=True)
train = list(pd.read_csv(f"{std}/train.csv")["path"].str.strip())
rng = np.random.default_rng(42); rng.shuffle(train); sel = train[:2000]
assert len(sel) == 2000
with open(f"{out}/train.csv", "w", newline="\n") as f:
    f.write("path\n" + "\n".join(sel) + "\n")
for n in ["val.csv", "test.csv"]:
    shutil.copy(f"{std}/{n}", f"{out}/{n}")
print("  split 已生成:", len(sel), "视频, 首个:", sel[0])
PY
fi

echo "[E3@2000] 时序训练（微调特征，仅 E1 同批 2000 视频）| $(pwd)"

python experiments/train_phase6.py \
    --bridge \
    --use_diff \
    --window_size 64 --stride 16 --eval_stride 8 \
    --frame_npz data/omnifall_dinov2_finetuned_frame.npz \
    --splits_dir DATASET-omnifall/splits/syn/e1_2000 \
    --exp_tag phase9/e3_stage2_2000 \
    --epochs 100 --patience 15 \
    --batch_size 32 --lr 1e-3 \
    --seed 42

echo ""
echo "[E3@2000] Done. Results: logs/phase9/e3_stage2_2000/test_results.json"
echo "对比: baseline logs/phase7/p7d_delta, 原E3 logs/phase9/e3_stage2, E1 logs/phase9/e1"
