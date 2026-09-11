# 项目结构总览（PROJECT STRUCTURE）

> 本文档回答"**每个文件夹里有什么**"，用于快速定位。
> - 想知道"**当前在做什么实验、最新结论、哪个模型/规则最优**" → 看 `docs/` **最新一份阶段文档**；
> - 目录级**速查地图**见 `CLAUDE.md` 的「目录地图」；本文是它的展开版。
> - 历史的**逐文件特征清单**（6 万行）已压缩为下方 [`data/`](#data-特征与缓存不入库) 的汇总表——不再逐文件维护。

## 顶层布局

| 目录 / 文件 | 体积 | 是什么 |
|---|---|---|
| `docs/` | 372K | **阶段文档**（规划/总结/报告）+ `superpowers/` 实现计划 |
| `models/` | 307K | 模型定义（每个文件 docstring 即架构说明） |
| `experiments/` | 1.2M | 训练/评估/数据准备/批处理脚本 |
| `preprocessing/` | 244K | 数据管线 step1-10 + AV1 解码 |
| `utils/` | 147K | 指标 / loss / 可视化 / 时间线指标 |
| `analysis/` | 473M | 诊断与错误审查脚本 + 可视化物 |
| `data/` | 78G | 特征 NPZ 与各实验帧缓存（**不入库**） |
| `logs/` | 5.1G | 训练产物（`logs/phaseN/exp_tag/`） |
| `DATASET-omnifall/` | 28G | 数据集根（**不入库**，自带 README/STRUCTURE/LABELS/CONFIGS） |
| `reference/` | 1.1G | 第三方参考实现（**不入库**） |
| `.claude/`、`.agents/` | 小 | agent 工具配置（**不入库**） |
| `CLAUDE.md` | — | agent 工作指引 + **跨阶段稳定约定**（目录地图、长任务约定、指标约定） |
| `AGENTS.md` | — | 同类工具配置（不入库） |
| `DATA_MANIFEST.md` | — | 本文：项目结构总览 |

---

## `docs/` — 阶段文档

按 `MMDD实验第N阶段{规划|总结|报告}.md` 命名，一个阶段一组（第九阶段规划 + 总结）。**最新实验结论一律看这里最新的一份**。
`superpowers/`：实现计划类文档。

## `models/` — 模型定义

| 文件 | 内容 |
|---|---|
| `phase6_model.py` | **跨阶段主体**：`Phase6TernaryModel` + 各 decoder（bridge transformer 等） |
| `phase9_e1_model.py` | E1 端到端（冻结 ViT-g + LoRA + 时序头） |
| `phase7_model.py`、`tcn_model.py`、`transformer_model.py` | 早期时序模型与 Phase 7 变体 |
| `cross_attention_fusion_model.py`、`gated_fusion_model.py`、`late_fusion_model.py` | 第八阶段多模态融合变体 |
| `dataset.py` | 滑动窗口数据集（含 `create_longseq_dataloaders`：train/val/test 滑窗） |

## `experiments/` — 训练 / 评估 / 脚本

按用途命名，**结果与配置看 `logs/` 对应 exp_tag**：

| 前缀 | 用途 | 例 |
|---|---|---|
| `train_*.py` | 训练入口 | `train_phase6.py`（主时序入口）、`train_p9e1.py`（E1 端到端）、`train_p9_e3_stage1.py` |
| `eval_*.py` | 评估 | `eval_p9e1_dual.py`（单次推理双口径）、`eval_timeline_metrics.py`（事件级指标） |
| `prep_*.py` / `extract_*.py` | 数据准备与特征抽取 | `prep_p9e1_frames.py`、`extract_p9_features.py` |
| `run_*.sh` | 批处理 runner（autodl 上跑） | `run_p9e1_full.sh`、`run_p9e1_h.sh`、`run_p7d.sh` |
| `analyze_*.py` / `sweep_*.py` | 归因 / 后处理扫描 | `analyze_e1_e3_attribution.py`、`sweep_rulev2.py` |
| `m7_*.py` | 蒸馏（方案 G） | `m7_precompute_teacher.py`、`m7_train_student.py` |
| `lora.py` | **共享模块**：手写 LoRA(q/k/v/o) | — |

## `preprocessing/` — 数据管线

`step1…step10` 流水线（读视频 → 切片 → 标签 → 特征/姿态/光流），自带 `README.md` 与 `数据预处理流程说明.md`。
**关键**：`av_decode.py` = AV1 视频解码（PyAV 优先）——**autodl 的 cv2 解不了 OmniFall 的 AV1，读帧必须经它**；帧均值 <20 视为解码失败金丝雀。

## `utils/` — 通用工具

`metrics.py`（F1/precision/recall）、`losses.py`（Focal）、`visualization.py`（时间线/混淆矩阵图）、
`timeline_metrics.py`（**事件级指标**：覆盖率/误报率/逐视频 F1/翻转数/检测延迟）。

## `analysis/` — 诊断与错误审查

逐视频错误分析（`per_video_errors.py` + `.csv`）、错误样本审查图（`review_errors/`）、
`diagnose_lying_event/`（躺卧误报诊断）、`diagnose_overfit.py`、时间线可视化。
**生成物可重跑，不是训练产物。**

## `data/` — 特征与缓存（不入库）

**a) 特征 NPZ（主源，`mmap_mode='r'` 载入）**

| 文件 | 内容 |
|---|---|
| `omnifall_dinov2_giant_frame.npz` | **主源**：12,000 视频 × 80 帧 DINOv2-giant 特征（E1/E3/baseline 共用标签与对齐基准） |
| `omnifall_dinov2_finetuned_frame.npz` | E3 帧级微调后重抽特征 |
| `omnifall_dinov2_finetuned_m3_frame.npz` | 方法 3（帧级 LoRA 全量）重抽特征 |
| `omnifall_dinov2_frame.npz`、`omnifall_frame_preprocessed.npz` | 早期 DINOv2 / 预处理特征 |
| `m7_teacher_logits.npz` | 方案 G 蒸馏的 teacher 软标签 |

**b) 清单 CSV**：`ofsyn_clips.csv`、`ofsyn_frame_labels.csv`、`e1_6000_manifest.csv`（H 的 6000 视频清单）
→ 训练清单以 `DATASET-omnifall/splits/` 下的 train/val/test 为准。

**c) 逐视频特征目录**（各约 12,000 个文件，**按 DATA_MANIFEST 旧清单曾逐文件记录，现仅汇总**）：

| 目录 | 文件数 | 体积 |
|---|---|---|
| `features/` | 12,000 | 23G |
| `dinov2_giant_features/` | 12,000 | 2.8G |
| `dinov2_features/` | 12,000 | 1.9G |
| `optical_flow_features/` | 12,000 | 521M |
| `pose_features/` | 12,000 | 427M |

**d) 实验帧缓存**：`p9e1_frames/`（E1 训练帧，32G）、`p9e1_frames_val/`（val 监控 150 视频，2.4G）、
`p9_frames/`（E3 抽帧，4.0G）、`p9_features_finetuned/`（2.4M）。
**e) 无关遗留**：`cifar-10-batches-py/`（已 gitignore）。

## `logs/` — 训练产物

`logs/phaseN/<exp_tag>/`，每阶段一组实验（phase1:13 / phase2:4 / phase3:19 / phase4:16 / phase5:2 /
phase6:17 / phase7:7 / phase8:11 / **phase9:21**）。典型内容：

- `training_history.json`（逐 epoch 指标）、`test_results.json`（全量 test）、`monitor.json`（训练中 val 曲线）
- `best_model.pt` / `last_model.pt` / `epoch_N_model.pt`（**走 git LFS**）、TensorBoard `events.*`
- 运行日志（`*.log`）、`eval_*.json`（各评估口径）

## `DATASET-omnifall/` — 数据集

**自带 `README.md` / `STRUCTURE.md` / `LABELS.md` / `CONFIGS.md` / `statistics.md`——标签体系、结构、配置以它们为准。**

| 子目录 | 内容 |
|---|---|
| `data_files/` | **12,000 个 AV1 mp4**（9.1G）——所有抽帧/解码的源 |
| `splits/` | 划分：`syn/`（`random` 为随机划分，`cs`/`cv` 为跨风格/视角，`cross_age`/`cross_bmi`/`cross_ethnicity` 为跨人群）+ 受控子集 `e1_2000` |
| `labels/` | 16 类逐帧标签派生物 |
| `parquet/` | 元数据表 |
| `videos/` | 辅助视频/示例 |
| 构建脚本 | `omnifall_builder.py`、`generate_parquet.py`、`prepare_oops_videos.py`、示例 notebook |

## `reference/` — 第三方参考实现（不入库）

上游开源代码副本：`C3D`、`SlowFast`、`TimeSformer`、`kinetics-i3d`、`scenic`、`temporal-shift-module`。
仅作查阅，**已加入 `.gitignore`**。

## 顶层文件

- **`CLAUDE.md`**：agent 工作指引 —— 目录地图、跨阶段稳定约定、**Long-Run GPU 任务运行约定**、**指标定义与验收约定**。改工作方式/约定时改这里。
- `AGENTS.md`、`.agents/`、`.claude/`：agent 工具配置（不入库）。

---

## git 入库策略速查

| 内容 | 是否入库 | 说明 |
|---|:---:|---|
| `docs/` `models/` `experiments/` `preprocessing/` `utils/` `analysis/` 的代码与文档 | ✅ | 正常 git |
| `logs/phaseN/**` 的 `.json` / `.log` / TensorBoard | ✅ | 正常 git |
| `logs/phaseN/**/*.pt` | ✅ | **git LFS**（`logs/**/*.pt`，仓库已有 150+ 个） |
| `data/**` 的大部分 | ❌ | 大体积 npz 与逐视频特征、实验帧缓存均 gitignore |
| `DATASET-omnifall/` | ❌ | 数据集（28G）gitignore |
| `reference/`、`.claude/`、`.agents/`、`AGENTS.md` | ❌ | gitignore |
