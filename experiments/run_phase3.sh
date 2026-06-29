#!/bin/bash
# ============================================================
# Phase 3: 消融实验 + 损失权重优化
# 用法:
#   bash experiments/run_phase3.sh              # 全部实验 (默认并行 3)
#   bash experiments/run_phase3.sh 2            # 并行 2 个
#   bash experiments/run_phase3.sh 1            # 串行
#   bash experiments/run_phase3.sh --dry-run    # 只打印，不执行
# ============================================================

set -e

# 切换到项目根目录
cd "$(dirname "$0")/.."
echo "========================================"
echo "Phase 3 Experiment Runner"
echo "Project: $(pwd)"
echo "Start:   $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

# 并行数
MAX_PARALLEL="${1:-3}"
DRY_RUN=false
if [ "$1" = "--dry-run" ]; then
    DRY_RUN=true
    MAX_PARALLEL=1
fi

# 通用参数
COMMON="--epochs 100 --batch_size 64 --lr 1e-3 --weight_decay 1e-4 --patience 15 --seed 42 --num_workers 4"

# ---- 实验定义 ------------------------------------------------
# 格式: "exp_tag  extra_args"
# Phase 3a: 消融实验 (固定权重 w=[1.0, 0.5, 0.5])
EXP_3A=(
    "phase3/p3a_rgb          "
    "phase3/p3a_diff         --use_diff"
    "phase3/p3a_pose         --pose_npz data/omnifall_pose.npz"
    "phase3/p3a_diff_pose    --use_diff --pose_npz data/omnifall_pose.npz"
)

# Phase 3b: 权重搜索 (RGB+Δclip+Pose)
EXP_3B=(
    "phase3/p3b_fallen_x2    --use_diff --pose_npz data/omnifall_pose.npz --w_cls 1.0 --w_fall 0.5 --w_fallen 1.0"
    "phase3/p3b_fall_reduce  --use_diff --pose_npz data/omnifall_pose.npz --w_cls 1.0 --w_fall 0.3 --w_fallen 0.5"
    "phase3/p3b_balanced     --use_diff --pose_npz data/omnifall_pose.npz --w_cls 1.0 --w_fall 0.3 --w_fallen 1.0"
    "phase3/p3b_fallen_x3    --use_diff --pose_npz data/omnifall_pose.npz --w_cls 1.0 --w_fall 0.5 --w_fallen 1.5"
    "phase3/p3b_equal        --use_diff --pose_npz data/omnifall_pose.npz --w_cls 1.0 --w_fall 1.0 --w_fallen 1.0"
)

# Phase 3c: 跷跷板修复 (RGB+Pose)
EXP_3C=(
    "phase3/p3c_pose_fall_reduce  --pose_npz data/omnifall_pose.npz --w_cls 1.0 --w_fall 0.3 --w_fallen 0.5"
    "phase3/p3c_pose_balanced     --pose_npz data/omnifall_pose.npz --w_cls 1.0 --w_fall 0.3 --w_fallen 1.0"
)

# ---- 工具函数 ------------------------------------------------

run_one() {
    local tag="$1"
    local extra="$2"
    local logdir="logs/${tag}"
    local lockfile="${logdir}/.running"

    echo "[$(date '+%H:%M:%S')] START: $tag"
    mkdir -p "$logdir"
    touch "$lockfile"

    if $DRY_RUN; then
        echo "  [DRY-RUN] python experiments/train_tcn.py $COMMON --exp_tag $tag $extra"
    else
        python experiments/train_tcn.py \
            $COMMON \
            --exp_tag "$tag" \
            $extra
        echo "[$(date '+%H:%M:%S')] DONE:  $tag"
    fi
    rm -f "$lockfile"
}

# 信号量并行执行
run_parallel() {
    local -n exp_array=$1   # 引用数组
    local running=0
    local pids=()

    for entry in "${exp_array[@]}"; do
        # 跳过空行
        [ -z "$entry" ] && continue

        read -r tag extra <<< "$entry"

        # 检查是否已完成 (有 test_results.json 就跳过)
        if [ -f "logs/${tag}/test_results.json" ] && [ ! -f "logs/${tag}/.running" ]; then
            echo "[$(date '+%H:%M:%S')] SKIP: $tag (already completed)"
            continue
        fi

        # 等待直到有空位
        while [ "$running" -ge "$MAX_PARALLEL" ]; do
            # 检查是否有进程完成
            for i in "${!pids[@]}"; do
                if ! kill -0 "${pids[$i]}" 2>/dev/null; then
                    wait "${pids[$i]}" 2>/dev/null || true
                    unset "pids[$i]"
                    ((running--))
                fi
            done
            [ "$running" -ge "$MAX_PARALLEL" ] && sleep 5
        done

        # 启动新实验
        run_one "$tag" "$extra" &
        pids+=($!)
        ((running++))

        sleep 2  # 小延迟避免 I/O 竞争
    done

    # 等待所有剩余进程
    echo "[$(date '+%H:%M:%S')] Waiting for ${running} remaining experiments..."
    for pid in "${pids[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
}

# ---- 执行 ------------------------------------------------

echo ""
echo "Phase 3a: Ablation Study (4 experiments, baseline weights)"
echo "----------------------------------------"
run_parallel EXP_3A

echo ""
echo "Phase 3b: Loss Weight Search on RGB+Δclip+Pose (5 experiments)"
echo "----------------------------------------"
run_parallel EXP_3B

echo ""
echo "Phase 3c: Seesaw Fix on RGB+Pose (2 experiments)"
echo "----------------------------------------"
run_parallel EXP_3C

echo ""
echo "========================================"
echo "Phase 3 Complete!"
echo "End: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

# ---- 汇总分析 ------------------------------------------------
echo ""
echo "Running analysis..."
python experiments/analyze_phase3.py

echo ""
echo "TensorBoard: tensorboard --logdir logs/phase3"
echo "Done."
