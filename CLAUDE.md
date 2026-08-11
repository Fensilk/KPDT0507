# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Elderly fall detection research project (master's thesis). Uses visual keypoint-based temporal modeling to classify video clips into ternary labels: **fall** (跌倒瞬间), **fallen** (倒地状态), **normal** (正常). The pipeline extracts per-frame visual features (DINOv2 ViT-g / ResNet18) then feeds them into temporal models (TCN, Transformer, LSTM, Mamba) for frame-level classification.

## Environment

- **OS**: Windows 11, Git Bash shell
- **Python**: conda env `py310` (Python 3.10)
- **GPU**: Single RTX 4060 (8GB VRAM)
- **Data**: OmniFall OF-Syn subset — 12,000 synthetic videos, 81 frames each, 16 action classes, AV1-encoded

## Commands

```bash
# Activate environment
conda activate py310

# Train Phase 6 (ternary classification, main training entry point)
python experiments/train_phase6.py --bridge --window_size 64 --stride 8 --exp_tag phase7/my_exp

# Train with Δframe diff features
python experiments/train_phase6.py --bridge --use_diff --window_size 64 --stride 8 --exp_tag phase7/my_delta_exp

# Train with different decoders
python experiments/train_phase6.py --decoder_type lstm --focal_gamma 3
python experiments/train_phase6.py --decoder_type mamba --focal_gamma 5

# Run shell scripts for batched experiments
bash experiments/run_p7d.sh

# Verify preprocessing pipeline
python preprocessing/verify_pipeline.py

# Test dataset loading
python models/dataset.py --longseq
```

There is no `requirements.txt` — dependencies are managed via conda environment `py310`. Key packages: `torch`, `numpy`, `pandas`, `scikit-learn`, `tensorboard`.

## Architecture

### Data Flow

```
OmniFall videos (.mp4, 81 frames each)
  → Feature extraction: DINOv2 ViT-g (1536d) or ResNet18 (512d)
  → Frame-level NPZ: omnifall_dinov2_giant_frame.npz
  → LongSequenceDataset: sliding window (T, D) sequences
  → Temporal model (TCN / Transformer / LSTM / Mamba)
  → Per-frame ternary predictions (fall / fallen / normal)
  → Post-hoc rules: RuleV2 suppresses isolated false fallen predictions
```

### Directory Map

| Directory | Purpose |
|-----------|---------|
| `models/` | Model definitions (`phase6_model.py`, `tcn_model.py`, `transformer_model.py`), dataset loader (`dataset.py`) |
| `experiments/` | Training scripts (`train_phase6.py` is the main entry), analysis scripts, shell runners |
| `utils/` | `metrics.py` (F1, precision, recall), `losses.py` (FocalLoss), `visualization.py` (plots, confusion matrices) |
| `preprocessing/` | Data pipeline: label parsing → clip segmentation → feature extraction (steps 1-8) |
| `DATASET-omnifall/` | Dataset labels, splits, parquet files, video archive |
| `data/` | NPZ feature files, CSVs (too large for git) |
| `logs/` | Training outputs organized by phase: `logs/phase7/p7d_delta/` |
| `docs/` | Chinese-language research plans and reports for each phase |
| `analysis/` | Diagnostic and visualization scripts |

### Key Models

- **`Phase6TernaryModel`** (`models/phase6_model.py`): Current main model. DINOv2 ViT-g (1536d) → Linear(1536→384) → LayerNorm → SinusoidalPE → Temporal Decoder → Linear(384→3). Decoder options: `CausalTransformerDecoder`, `LSTMDecoder`, `MambaDecoder`. Bridge mode (`--bridge`) disables causal masking for bidirectional attention.
- **`TCNModel`** (`models/tcn_model.py`): Earlier model. Causal dilated convolutions with 3 heads (16-class, fall binary, fallen binary). Input: 512d ResNet18 features, T=5 clips.
- **`MultimodalFeatureTransformer`** (`models/transformer_model.py`): Transformer-based model for multimodal feature fusion.

### Dataset Classes

- **`OmniFallTCNDataset`**: Video-level dataset (T=5 clips per video). Uses `omnifall_preprocessed.npz` (512d ResNet18 features).
- **`LongSequenceDataset`**: Frame-level sliding window dataset. Uses `omnifall_dinov2_giant_frame.npz` (1536d ViT-g features). Configurable window size and stride. This is the primary dataset class for Phase 6+.

### Post-Hoc Rules (Phase 7)

RuleV2 (`experiments/eval_rule_c.py`) is critical — it's a zero-training-cost post-processing step that enforces event causality: a `fallen` prediction is only valid if preceded by ≥K2 consecutive `fall` frames within window W. Parameters: K1=2 (context skip), K2=7 (consecutive fall threshold), W=64 (lookback window). This dramatically reduces false `lying→event` misclassification.

### Δframe Feature

When `--use_diff` is enabled, frame-to-frame difference features are concatenated to the base features, doubling the input dimension (e.g., 1536→3072 for ViT-g). Effective only with high-quality features (ViT-g); was detrimental with ViT-L (Phase 4b).

### Label System

- 16-class fine-grained: 0=walk, 1=fall, 2=fallen, 3=sit_down, 4=sitting, 5=lie_down, 6=lying, 7=stand_up, 8=standing, 9=other, 10=kneel_down, 11=kneeling, 12=squat_down, 13=squatting, 14=crawl, 15=jump
- Ternary (Phase 6+): 0=fall, 1=fallen, 2=normal
- Binary fall: `fall_labels` (1=fall, 0=other)
- Binary fallen: `fallen_labels` (1=fallen, 0=other)

### Research Phase Evolution

Phases 1-7 represent the research progression. Key decisions that shaped the codebase:
- Phase 4: DINOv2 ViT-g replaced ResNet18 as the feature extractor (frozen, not fine-tuned)
- Phase 5: Loss weight tuning and DINOv2+TCN module ablation
- Phase 6: Switched from binary to ternary classification with causal temporal modeling
- Phase 7: Shifted from frame-level causality to event-level causality. Bidirectional attention is now considered legitimate (≤2s buffer acceptable). RuleV2 is the current best post-hoc rule.
- **High-priority open direction**: Fine-tuning DINOv2 (currently frozen) — identified as the final bottleneck for distinguishing lying vs fallen.

### Training Conventions

- Runs log to `logs/{exp_tag}/` with `training_history.json`, `test_results.json`, TensorBoard events, and model checkpoints
- `--patience 15` early stopping is standard
- `WeightedRandomSampler` handles class imbalance (ternary or 16-class weights)
- Data split: `DATASET-omnifall/splits/syn/random/` (train/val/test CSVs)

### Code Patterns

- Training scripts use `sys.path.insert(0, ...)` to import from project root — run scripts from project root, not from within `experiments/`
- Shell scripts handle this by `cd "$(dirname "$0")/.."` before invoking Python
- NPZ files use `mmap_mode='r'` for memory-efficient loading
- Windows-specific: `num_workers=0` is recommended for DataLoaders
