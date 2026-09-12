# 项目内容清单（PROJECT CONTENTS）

> **定位**：本文是 `CLAUDE.md` 规定之下的**具体描述层**——回答"项目里**实际有什么**：哪些目录、各有多少
> 文件、多大体积、装的是什么"。
>
> **本文不复述规则与约定。** 以下一律以 `CLAUDE.md` 为准：三元标签定义、读帧方式、运行方式、
> NPZ / DataLoader 约定、**缓存与入库纪律**、Long-Run 任务约定、指标定义与验收约定；目录级"去哪找"的
> 速查地图同样在 `CLAUDE.md`。**本文只补它没有的具体清单与规模。**

## 规模速览

| 目录 / 文件 | 体积 | 规模 |
|---|---|---|
| `data/` | 78G | 10 个子目录 + 16 个文件（§[data](#data--特征与缓存78g)）|
| `DATASET-omnifall/` | 28G | 5 个内容子目录 + 10 份文档/脚本；**自带独立 `.git`**（§[DATASET](#dataset-omnifall--数据集)）|
| `logs/` | 5.1G | 9 个阶段 / 共 103 个实验目录（§[logs](#logs--训练产物)）|
| `reference/` | 1.1G | 6 个上游实现（§[reference](#reference--第三方参考实现)）|
| `analysis/` | 473M | 6 个文件 + 2 个子目录（§[analysis](#analysis--诊断与错误审查)）|
| `docs/` | 372K | 20 份文档 + `superpowers/`（§[docs](#docs--阶段文档)）|
| `models/` | 307K | 10 个 `.py`（9 个模型定义 + `__init__`）（§[models](#models--模型定义)）|
| `preprocessing/` | 244K | 13 个脚本 + 2 份说明（§[preprocessing](#preprocessing--数据管线脚本)）|
| `utils/` | 147K | 4 个工具模块（§[utils](#utils--通用工具)）|
| `experiments/` | 1.2M | 67 个脚本（§[experiments](#experiments--训练--评估--脚本)）|

---

## `docs/` — 阶段文档

20 份 `.md`，覆盖第一～第九阶段：`MMDD实验第N阶段{规划|总结|报告}.md`（0518 起），另有
`0513跌倒检测_研究选题整理.md`、`0516实验数据集预处理.md`、`视觉跌倒检测研究进展汇报（202607）.md`。
子目录 `superpowers/`：实现计划类文档。
**每阶段的结论以该阶段"总结/报告"为准；第九阶段为 `0902实验第九阶段总结.md`（E3 + E1 收官，单一权威文档）。**

## `models/` — 模型定义

| 文件 | 具体内容 |
|---|---|
| `phase6_model.py` | `Phase6TernaryModel` + 各 decoder（bridge transformer 等）——**跨阶段复用的时序头主体** |
| `phase9_e1_model.py` | E1 端到端模型（冻结 ViT-g + LoRA + 时序头） |
| `phase7_model.py` | Phase 7（Δdiff 等）变体 |
| `tcn_model.py`、`transformer_model.py` | 早期时序模型（TCN / Transformer） |
| `cross_attention_fusion_model.py`、`gated_fusion_model.py`、`late_fusion_model.py` | 第八阶段多模态融合的三个变体 |
| `dataset.py` | 滑动窗口数据集：`LongSequenceDataset` + `create_longseq_dataloaders`（train/val/test 三份 csv、可分别指定 stride） |

## `experiments/` — 训练 / 评估 / 脚本

67 个脚本，按前缀分五类（**结果与配置不在这里，看 `logs/` 对应 exp_tag**）：

| 前缀 | 个数 | 具体例 |
|---|:---:|---|
| `train_*.py` | 10 | `train_phase6.py`（主时序入口）、`train_p9e1.py`（E1 端到端）、`train_p9_e3_stage1.py`（E3 帧级）、`train_phase3/4/…` |
| `eval_*.py` | 10 | `eval_p9e1_dual.py`（单次推理出 instance + merged 双口径）、`eval_timeline_metrics.py`（事件级指标）、`eval_p9e3.py`、`eval_p7d_k5.py` |
| `prep_*.py` + `extract_*.py` | 2 + 2 | `prep_p9e1_frames.py`（E1 连续帧缓存）、`prep_p9_frames.py`、`extract_p9_features.py`、`extract_false_positives.py` |
| `run_*.sh`（+ `watch_pid.sh`） | 19 (+1) | `run_p9e1_full.sh`（E1-full 训练 + watchdog）、`run_p9e1_h.sh`（H@6000）、`run_e1full_eval*.sh`（三候选评测）、`run_p7a/p7b/p7c/p7d.sh` |
| `analyze_*.py` + `sweep_*.py` + `m7_*.py` | 5 + 2 + 2 | `analyze_e1_e3_attribution.py`、`sweep_rulev2.py`、`m7_precompute_teacher.py`、`m7_train_student.py` |
| 其它（共享模块与探针） | 14 | **`lora.py`**（共享 LoRA，q/k/v/o，E1/E3 复用）、`check_p9_consistency.py`、`probe_pose*.py`、`_run_train_strided/t16/t32.py`、`summarize_phase6.py`、`dump_test_preds.py`、`confusion_matrix_p4.py`、`run_p7a_event_vote.py` |

## `preprocessing/` — 数据管线脚本

`step1…step10` 顺序流水线（11 个脚本）：`step1_load_ofsyn` → `step2_clip_segmentation` →
`step3_binary_labels` → `step4_extract_features` → `step5_extract_pose` / `step6_aggregate_pose` →
`step7_extract_dinov2` → `step8_aggregate_pose_framelvl` → `step9_derive_pose` / `step9b_derive_pose_accel`
→ `step10_extract_optical_flow`；另有 **`av_decode.py`**（读帧工具）与 `verify_pipeline.py`（管线校验），
以及 `README.md`、`数据预处理流程说明.md`。
**读帧的约定（走哪个解码器、失败判据）见 `CLAUDE.md`，本文不重复。**

## `utils/` — 通用工具

| 文件 | 具体内容 |
|---|---|
| `metrics.py` | F1 / precision / recall 等基础指标 |
| `losses.py` | Focal loss |
| `visualization.py` | 时间线图、混淆矩阵图 |
| `timeline_metrics.py` | **事件级指标**：覆盖率、误报率、逐视频 F1 分布、翻转数、检测延迟、order_correct |

## `analysis/` — 诊断与错误审查

`per_video_errors.py` + `per_video_errors.csv`（逐视频错误明细）、`review_errors/`（6 张错误样本审查图）、
`diagnose_lying_event/`（躺卧误报诊断）、`diagnose_overfit.py`、`_verify_pipeline.py`、
`visualize_timeline_clear.py`、`_compare_improve2.py`。**均为可重跑的诊断产物，不是训练产物。**

## `data/` — 特征与缓存（78G）

### 特征 NPZ（12 个）

| 文件 | 体积 | 具体内容 |
|---|---|---|
| `omnifall_dinov2_giant_frame.npz` | 2.6G | **主源**：12,000 视频 × 80 帧 DINOv2-giant 特征；E1/E3/baseline 共用其 `labels_16` / `video_paths` / `video_start_indices` 作为对齐基准 |
| `omnifall_dinov2_finetuned_frame.npz` | 2.6G | E3 帧级微调后重抽的逐帧特征 |
| `omnifall_dinov2_finetuned_m3_frame.npz` | 2.4G | 方法 3（帧级 LoRA 全量）重抽特征 |
| `omnifall_dinov2_frame.npz` | 1.7G | 早期 DINOv2 特征 |
| `omnifall_frame_preprocessed.npz` | 1.7G | 预处理后的基础帧特征 |
| `omnifall_optical_flow.npz` | 438M | 光流特征 |
| `omnifall_pose_frame.npz` | 308M | 逐帧姿态特征（99d） |
| `omnifall_preprocessed.npz` | 106M | 最早期的预处理特征包 |
| `omnifall_pose_semantic_accel.npz` | 39M | 语义姿态 + 加速度（12d） |
| `omnifall_pose_semantic.npz` | 26M | 语义姿态（8d） |
| `omnifall_pose.npz` | 20M | 原始姿态 |
| `m7_teacher_logits.npz` | 3.1M | 方案 G 蒸馏的 teacher 软标签 |

### 清单 CSV（3 个）

`ofsyn_frame_labels.csv`（62M，逐帧标签）、`ofsyn_clips.csv`（3.5M，切片表）、
`e1_6000_manifest.csv`（139K，H 的 6,000 视频清单）。
**训练/评估清单以 `DATASET-omnifall/splits/` 下的 train/val/test csv 为准。**

### 逐视频特征目录（5 个，各 12,000 文件）

| 目录 | 体积 |
|---|---|
| `features/` | 23G |
| `dinov2_giant_features/` | 2.8G |
| `dinov2_features/` | 1.9G |
| `optical_flow_features/` | 521M |
| `pose_features/` | 427M |

### 实验帧缓存（4 个）

| 目录 | 体积 | 具体内容 |
|---|---|---|
| `p9e1_frames/` | 32G | E1 训练帧缓存（按视频 .pt，80 帧 uint8） |
| `p9e1_frames_val/` | 2.4G | E1 的 val 监控子集（150 视频） |
| `p9_frames/` | 4.0G | E3 抽帧缓存 |
| `p9_features_finetuned/` | 2.4M | E3 微调特征中间产物 |

其它：`cifar-10-batches-py/`（178M，与本项目无关的遗留）。

## `logs/` — 训练产物

`logs/phaseN/<exp_tag>/`。各阶段实验目录数（**仅目录**）：**phase1:13｜phase2:4｜phase3:17｜phase4:15｜
phase5:2｜phase6:17｜phase7:7｜phase8:10｜phase9:19**（合计 104）。

**phase9 的具体实验目录**（19 个）：
`e1_5ep`（E1@2000 五轮）、`e1_6000`（H）、`e1_full`（E1-full 收官）、`e1_earlystop`、`e1_dinov2base`（方案 E）、
`e1_cache_build`、`e1_rulev2`、`e3_stage1`、`e3_stage2`（E3-full）、`e3_stage2_2000`（E3@2000）、`e3_rulev2`、
`m1_v1_s8`、`m1_v2_fw3`（方案 A）、`m3_stage1`、`m3_full_features`（方案 B）、`m7_distill`（方案 G）、
`no_keeplying`、`no_keeplying_fallen`（数据侧前奏 P9/P9b）、`e2_rank_trunc`（E2 = LoRA 秩截断探针，§六）。
零散文件：`e3_stage2_2000_terminal.log`、`m3_stage1.live.log`、`timeline_metrics_cached.json`。

> ⚠ **P9/P9b 有产物缺口（2026-09-12 核实）**：两个目录只剩 ckpt + `test_results.json` +
> `training_history.json`；**其数据侧脚本与过滤 split 从未入库、本地已丢失**
> （`analysis/flag_no_gain_videos.py`、`build_filtered_splits.py`、`review_decisions.csv`、
> `splits/syn/no_keeplying/`），全量 test 对照数字出自同批丢失的
> `analysis/eval_compare_no_keeplying.py` → 该部分**不可复现**。详见
> `docs/0902实验第九阶段总结.md` §7.3。

每个实验目录的典型内容：`training_history.json`（逐 epoch）、`test_results.json`（全量 test）、
`monitor.json`（训练中 val 曲线）、`best_model.pt` / `last_model.pt` / `epoch_N_model.pt`、
TensorBoard `events.*`、运行日志与各评估口径的 `eval_*.json`。

## `DATASET-omnifall/` — 数据集

**自带独立 `.git`**（嵌套仓库，故父仓库忽略整个目录）。自带文档：`README.md`、`STRUCTURE.md`、`LABELS.md`、
`CONFIGS.md`、`statistics.md`、`annotation_remarks.webm`；构建脚本 `omnifall_builder.py`、
`generate_parquet.py`、`prepare_oops_videos.py`、`omnifall_dataset_examples.ipynb`。
**标签体系 / 结构 / 配置以它自带的这几份为准。**

| 子目录 | 体积 | 具体内容 |
|---|---|---|
| `data_files/` | 9.1G | **12,000 个 AV1 mp4**（`extracted/<类别>/*.mp4`）——一切抽帧/解码的源 |
| `labels/` | 5.3M | 16 类逐帧标签派生物 |
| `parquet/` | 16M | 元数据表 |
| `splits/` | 1.6M | 划分：顶层 `cs/`、`cv/`、`syn/`；`syn/` 下有 `random`（随机划分，训练默认）、`cross_age`、`cross_bmi`、`cross_ethnicity`（跨人群）与受控子集 `e1_2000` |
| `videos/` | 1.6M | 辅助视频 / 示例 |

## `reference/` — 第三方参考实现

上游开源代码副本（6 个）：`C3D`、`SlowFast`、`TimeSformer`、`kinetics-i3d`、`scenic`、
`temporal-shift-module`；另有 `_part_test.bin`。约 1.1GB，仅备查阅。

## 工具配置与顶层文件

- `.claude/`（`settings.local.json` + `skills/`）、`.agents/`（`skills/`）、`AGENTS.md` —— agent 工具配置。
- `CLAUDE.md` —— **规范层**：目录速查地图 + 跨阶段稳定约定 + Long-Run 任务约定 + 指标定义与验收约定。
- `PROJECT_CONTENTS.md` —— 本文（具体描述层）。

---

## 未入库的大体积内容（事实清单）

| 内容 | 体积 | 位置 |
|---|---|---|
| 逐视频特征目录（5 个） | 28G | `data/{features,dinov2_giant_features,dinov2_features,optical_flow_features,pose_features}/` |
| 实验帧缓存（4 个） | 38G | `data/{p9e1_frames,p9e1_frames_val,p9_frames,p9_features_finetuned}/` |
| 特征 NPZ / CSV | ~12G | `data/*.npz`、`data/*.csv` |
| 数据集 | 28G | `DATASET-omnifall/` |
| 第三方参考实现 | 1.1G | `reference/` |
| 工具配置 | 小 | `.claude/`、`.agents/`、`AGENTS.md` |
| 未归档的历史实验产物 | ~1.5G | `logs/phase9/` 下除 `e1_full/` 外的实验目录 |

**入库与否的判定规则见 `CLAUDE.md`「缓存纪律」**（本文只列事实，不重复规则）。
