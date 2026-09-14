#!/usr/bin/env bash
# Phase 10 / 6.1 — TSM baseline on OmniFall OF-Syn.
# Main table setting: full 9,600-video training split, so it is directly comparable with E1-full.
# Run from the project root on Autodl (RTX 3090, conda env py310). Usage:
#     bash experiments/run_p10_tsm.sh [tag]
set -e
cd "$(dirname "$0")/.."

TAG=${1:-tsm_r50_8f_9600}
TSM_CKPT=${TSM_CKPT:-data/tsm_k400_r50_8f.pth}

if [ ! -f "$TSM_CKPT" ]; then
  echo "[run_p10_tsm] downloading Kinetics-400 TSM weights -> $TSM_CKPT"
  curl -L --retry 3 -o "$TSM_CKPT" \
    https://hanlab18.mit.edu/projects/tsm/models/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth
fi

# Dense frames are a one-off shared asset for every reference model (TSM/X3D/I3D/TimeSformer/ViViT).
if [ ! -d data/ref_frames/fall ]; then
  echo "[run_p10_tsm] extracting dense frames (81 x 256x256 JPEG per video, ~23 GB)"
  python experiments/prep_p10_frames.py --workers "${PREP_WORKERS:-16}"
fi

# Main experiment: same train split (9,600) as the Phase 9 main line, K400-initialised.
#
# Hyper-parameter rationale (see docs/0912实验第十阶段总结.md §3.4):
#   * batch 32 — memory-driven: batch 16 already uses 7.5 GB on the 8 GB local card,
#     so 32 (~15 GB) is the safe setting on a 24 GB 3090; 64 would need ~30 GB.
#   * lr 0.005 — TSM's official recipe is lr 0.02 @ batch 128 and explicitly says to scale
#     the lr linearly with batch size -> 0.02 x 32/128 = 0.005.
#   * epochs 10 — step-matched against the Phase 9 main line (E1-full: 28,800 optimizer steps).
#     9,600 videos x 10 windows = 96,000 clips/epoch; at batch 32 that is 3,000 steps/epoch,
#     so 10 epochs = 30,000 steps ~= E1-full's 28,800 steps.
#   * FAST_SHIFT=1 enables the numerically verified cat-based shift (+8-15% throughput).
#   * On an 8 GB card use BATCH_SIZE=8 (batch 16 saturates VRAM and cuts throughput ~3x).
# Verified numerically equivalent on both the local 4060 and the 3090 (forward diff 0.0,
# grad diff ~1e-11). Measured +8-15% on the 4060 and +7.7% on the 3090, so it is on by default;
# set FAST_SHIFT=0 to run the unmodified official implementation.
FAST_SHIFT_FLAG="--fast_shift"
if [ "${FAST_SHIFT:-1}" = "0" ]; then FAST_SHIFT_FLAG=""; fi

python experiments/train_p10_tsm.py \
  --tag "$TAG" \
  --epochs "${EPOCHS:-10}" \
  --batch_size "${BATCH_SIZE:-32}" \
  --workers "${WORKERS:-8}" \
  --lr "${LR:-0.005}" \
  --val_limit_videos 0 \
  --tune_from "$TSM_CKPT" \
  --test_stride 8 $FAST_SHIFT_FLAG

echo "[run_p10_tsm] done. logs: logs/phase10/$TAG"
