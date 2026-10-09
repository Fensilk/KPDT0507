# 项目内容清单（PROJECT CONTENTS）

> **定位**：本文是 `CLAUDE.md` 规定之下的**具体描述层**——回答"项目里**实际有什么**：哪些目录、各有多少
> 文件、多大体积、装的是什么"。
>
> **本文不复述规则与约定。** 以下一律以 `CLAUDE.md` 为准：三元标签定义、读帧方式、运行方式、
> NPZ / DataLoader 约定、**缓存与入库纪律**、Long-Run 任务约定、指标定义与验收约定；目录级"去哪找"的
> 速查地图同样在 `CLAUDE.md`。**本文只补它没有的具体清单与规模。**
>
> 最近更新：**2026-10-09**（第十一阶段收尾时的全量复核）。

## 规模速览

| 目录 / 文件 | 体积 | 规模 |
|---|---|---|
| `data/` | ~103G | 11 个子目录 + 20 个文件（§[data](#data--特征与缓存103g)）|
| `DATASET-omnifall/` | 27G | 5 个内容子目录 + 10 份文档/脚本；**自带独立 `.git`**（§[DATASET](#dataset-omnifall--数据集)）|
| `logs/` | 20G | **11 个阶段 / 共 162 个实验目录**（§[logs](#logs--训练产物)）|
| `reference/` | 1.1G | 6 个上游实现 + 6 份**本地留存**文档（§[reference](#reference--上游实现--本地留存区)）|
| `analysis/` | 474M | 19 个脚本 + 3 个 csv + 2 个子目录（§[analysis](#analysis--诊断与错误审查)）|
| `docs/` | 940K | **31 份文档** + `superpowers/`（§[docs](#docs--阶段文档)）|
| `experiments/` | 1.7M | **124 个脚本**（66 `.py` + 58 `.sh`）（§[experiments](#experiments--训练--评估--脚本)）|
| `models/` | 347K | 10 个 `.py`（9 个模型定义 + `__init__`）（§[models](#models--模型定义)）|
| `preprocessing/` | 248K | 13 个脚本 + 2 份说明（§[preprocessing](#preprocessing--数据管线脚本)）|
| `utils/` | 203K | 6 个模块（5 个工具 + `__init__`）（§[utils](#utils--通用工具)）|

> 体积口径：`du -sm`（Windows 下按 1 MB = 1024² ）。`analysis/` 的 474M 里 **473M 是
> `diagnose_lying_event/` 的可视化产物**，真正的脚本只有几百 KB。

---

## `docs/` — 阶段文档

31 份 `.md`，覆盖**第一～第十一阶段**：`MMDD实验第N阶段{规划|总结|报告}.md`（0518 起），另有
`0513跌倒检测_研究选题整理.md`、`0516实验数据集预处理.md`。
子目录 `superpowers/plans/`：实现计划类文档。

**各阶段文档链（"结论以该阶段总结/报告为准"）**

| 阶段 | 文档 |
|---|---|
| 一～五 | `0518` 规划 ｜ `0607` 规划+总结 ｜ `0624` 规划 + `0627` 报告 ｜ `0706` 规划+报告 ｜ `0712` 规划+报告 |
| 六～十 | `0717` 规划+报告 ｜ `0726` 规划 + `0804` 总结 ｜ `0811` 规划 + `0822` 总结 ｜ `0831` 规划 + `0902` 总结 ｜ `0911` 规划 + `0912` 总结 |
| **十一** | 见下表（10 份，本阶段文档偏多且有时间序） |

**第十一阶段（Phase 11）的文档链**

| 文档 | 作用 |
|---|---|
| `0925实验第十一阶段规划.md` | **预登记规划**（含 §7.2b 判据勘误、§5.3 指标勘误、附录 A 勘误表） |
| `1006…增补-选点规则冻结.md`、`1007…增补-P2三臂重跑冻结.md` | **预登记原件**（跑前冻结、跑后只增不删）；文本本身是证据 |
| `0928实验第十一阶段进度.md` | 逐日流水 + §2 问题导向复核（★问题 1–5）+ §3 踩坑记录 |
| `0930` / `1004` / `1008` 交接 | 三次会话交接单（上下文用尽时写的） |
| **`1009实验第十一阶段总结（终版）.md`** | ⭐ **本阶段唯一权威总结与阅读入口**，取代上述全部文档作为"读哪份"的答案 |
| `1005实验第十一阶段总结.md`、`1009…增补-P2三臂重跑总结.md` | 被终版取代的历史稿（保留可追溯性） |

> ⚠ **写给导师的汇报**（`视觉跌倒检测研究进展汇报（YYYYMM）.md`）**不在本目录、也不入库** ——
> 见 §[reference](#reference--上游实现--本地留存区)。

## `models/` — 模型定义

| 文件 | 具体内容 |
|---|---|
| `phase6_model.py` | `Phase6TernaryModel` + 各 decoder（bridge transformer 等）——**跨阶段复用的时序头主体**；Phase 11 在此加了 `use_boundary_head` / `multiscale` 两个开关（默认关） |
| `phase9_e1_model.py` | E1 端到端模型（冻结 ViT-g + LoRA + 时序头）；Phase 11 加 `aux` 参数以接外部姿态特征 |
| `phase7_model.py` | Phase 7（Δdiff 等）变体 |
| `tcn_model.py`、`transformer_model.py` | 早期时序模型（TCN / Transformer） |
| `cross_attention_fusion_model.py`、`gated_fusion_model.py`、`late_fusion_model.py` | 第八阶段多模态融合的三个变体 |
| `dataset.py` | 滑动窗口数据集：`LongSequenceDataset` + `create_longseq_dataloaders`；Phase 11 加 `build_boundary_labels` / 难例加权采样 / `report_sampling_distribution`（防退化检查），均默认关 |

## `experiments/` — 训练 / 评估 / 脚本

**124 个脚本**（66 `.py` + 58 `.sh`），**结果与配置不在这里，看 `logs/` 对应 exp_tag**。

| 前缀 | 个数 | 具体例 |
|---|:---:|---|
| `run_*.sh` | 41 | `run_p9e1_full.sh`（E1-full）、`run_p11_w1.sh` / `run_p11_w2.sh`（W1/W2 统一 runner，含 watchdog 与服务器端快照）、`run_p11_p2.sh`（P2 三臂）、`run_p10_*.sh`（Phase 10 视频模型基线） |
| `eval_*.py` | 13 | `eval_p9e1_dual.py`（单次推理出 instance + merged 双口径）、`eval_timeline_metrics.py`（事件级）、`eval_p8_probs.py`（Phase 8 十一臂概率重评） |
| `train_*.py` | 12 | `train_phase6.py`（主时序入口）、`train_p9e1.py`（E1 端到端；Phase 11 加 `--use_boundary_head` / `--pose_npz` / `--aux_static_lambda` / `--save_every_check`） |
| `analyze_*.py` | 6 | `analyze_e1_e3_attribution.py` 等 |
| `p11_*.sh` | 5 | **上机与运维**：`p11_preflight.sh`（启动前本地/远端哈希预检）、`p11_env_check.sh`（任意卡型自检，含算力代际硬闸门）、`p11_health.sh` / `p11_mobile_status.sh`（手机友好健康检查）、`p11_pack_when_done.sh` |
| `p2_*.sh` | 4 | P2 无人值守链：`p2_chain.sh`（训练完成自动接离线阶段）、`p2_extwatch.sh` / `p2_stallwatch.sh`（进程退出 / 停滞监视）、`p2_onboard.sh`（新实例一键自检） |
| `prep_*` / `extract_*` / `sweep_*` / `m7_*` | 3 / 2 / 2 / 2 | 数据准备、特征抽取、规则扫参、方案 G 蒸馏 |
| 其余（共享模块、判定与探针） | ~26 `.py` + ~8 `.sh` | **`lora.py`**（共享 LoRA）、**`judge_w1_arm.py`**（曲线口径判定，调 `analysis/prec_rec_curves.py` 的 `dominance()`）、`run_smoke.sh`（**换卡冒烟"包"入口**，见文件头说明）、`setup_{5090,a100}_w2.sh`、`make_wheelhouse.sh`（离线环境包采集）、`bench_p10_*.py`、`probe_*.py`、`verify_p10_fast_shift.py` |

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
| `timeline_metrics.py` | 事件级指标：覆盖率、误报率、逐视频 F1 分布、翻转数、检测延迟、order_correct |
| **`boundary_metrics.py`** | **Phase 11 新增**：边界定位指标（定义见规划 §5.1）；自带两道验收入口，随指标一起声明了四条已知缺陷 |

## `analysis/` — 诊断与错误审查

19 个脚本 + 3 个 csv + 2 个子目录。

| 类别 | 文件 |
|---|---|
| Phase 11 新增（判读与验收） | `p2_judge.py`（R1/R3 判读）、`verify_p2_judge.py`（**独立复算**：绕开判读链自写 numpy，37 项对照 0 不一致）、`seven3_acceptance.py`（§7.3 两道验收）、`rerun_curves.py`（复跑协同性）、`p2_multickpt_avg.py`、`reeval_val_ckpts.py`（重评 val）、`val_probe_checks.py`（选点四道检验）、`make_val_subset.py` |
| Phase 11 新增（诊断） | `prec_rec_curves.py`（**前沿曲线 + `dominance()`** —— 判定与画图同源）、`verify_boundary_metrics.py`、`verify_val_truncation_bias.py`、`derive_hard_negatives.py`、`p11_hardneg_dose.py`、`p11_profile_dataloader.py`（数据侧 profiler） |
| 早期 | `per_video_errors.py`、`diagnose_overfit.py`、`visualize_timeline_clear.py`、`_verify_pipeline.py`、`_compare_improve2.py` |
| 数据 | `per_video_errors.csv`、`no_gain_flags.csv`、`review_decisions.csv` |
| 子目录 | `review_errors/`（1.7M，6 张错误样本审查图）、`diagnose_lying_event/`（**473M**，躺卧误报诊断的可视化产物） |

**均为可重跑的诊断产物，不是训练产物。**

## `data/` — 特征与缓存（~103G）

### 特征 NPZ（12 个）

| 文件 | 体积 | 具体内容 |
|---|---|---|
| `omnifall_dinov2_giant_frame.npz` | 2.7G | **主源**：12,000 视频 × 80 帧 DINOv2-giant 特征；E1/E3/baseline/A0 共用其 `labels_16` / `video_paths` / `video_start_indices` 作为对齐基准 |
| `omnifall_dinov2_finetuned_frame.npz` | 2.7G | E3 帧级微调后重抽的逐帧特征 |
| `omnifall_dinov2_finetuned_m3_frame.npz` | 2.5G | 方法 3（帧级 LoRA 全量）重抽特征 |
| `omnifall_dinov2_frame.npz` | 1.8G | 早期 DINOv2 特征 |
| `omnifall_frame_preprocessed.npz` | 1.8G | 预处理后的基础帧特征 |
| `omnifall_optical_flow.npz` | 459M | 光流特征 |
| `omnifall_pose_frame.npz` | 322M | 逐帧姿态特征（99d） |
| `omnifall_preprocessed.npz` | 111M | 最早期的预处理特征包 |
| `omnifall_pose_semantic_accel.npz` | **40M** | 语义姿态 + 加速度（12d）—— **Phase 11 臂 4 / Exp-4 的输入** |
| `omnifall_pose_semantic.npz` | 27M | 语义姿态（8d） |
| `omnifall_pose.npz` | 21M | 原始姿态 |
| `m7_teacher_logits.npz` | 3.2M | 方案 G 蒸馏的 teacher 软标签 |

### 清单 CSV（6 个）

`ofsyn_frame_labels.csv`（62M，逐帧标签）、`ofsyn_clips.csv`（3.5M，切片表）、
`e1_6000_manifest.csv`（139K，H 的 6,000 视频清单）、
**`e1_2000_manifest.csv`（46K，A0/W1 的 2,000 视频清单 = `default_rng(42).permutation(train)[:2000]`）**、
**`ref_frames_manifest.csv`（1.3M，Phase 10 抽帧缓存清单）**、
**`val_subset300.csv`（7K，固定 300 视频 val 子集，seed42）**。
**训练/评估清单以 `DATASET-omnifall/splits/` 下的 train/val/test csv 为准。**

### 逐视频特征目录（5 个，各 12,000 文件）

| 目录 | 体积 |
|---|---|
| `features/` | 23G |
| `dinov2_giant_features/` | 2.9G |
| `dinov2_features/` | 1.9G |
| `optical_flow_features/` | 521M |
| `pose_features/` | 427M |

### 实验帧缓存（4 个）

| 目录 | 体积 | 具体内容 |
|---|---|---|
| `p9e1_frames/` | 32G | E1 训练帧缓存（按视频 .pt，80 帧 uint8） |
| `p9e1_frames_val/` | 2.4G | E1 的 val 监控子集（**恰好 150 视频**——数目不符会改变与 E1-full 的可比性） |
| `p9_frames/` | 4.0G | E3 抽帧缓存 |
| `p9_features_finetuned/` | 3.4M | E3 微调特征中间产物 |

### Phase 10 抽帧缓存（1 个）

| 目录 | 体积 | 具体内容 |
|---|---|---|
| `ref_frames/` | ~25G（**97.2 万文件**） | Phase 10 全画面抽帧缓存；`experiments/prep_p10_frames.py` 可重生成，故 gitignored。<br>（由 `data/` 总计 103G 减去其余各项反推，2026-10-09 实测） |

其它：`cifar-10-python.tar.gz`（163M）+ `cifar-10-batches-py/`（178M，与本项目无关的遗留）、
`tsm_k400_r50_8f.pth`（98M，Phase 10 TSM 预训练权重；`*.pth` 不在 LFS 规则里，故 gitignored）。

## `logs/` — 训练产物

`logs/phaseN/<exp_tag>/`。各阶段实验目录数（**仅目录**）：

**phase1:13｜phase2:4｜phase3:17｜phase4:15｜phase5:2｜phase6:17｜phase7:9｜phase8:11｜
phase9:23｜phase10:31｜phase11:20**（合计 **162**）。

各阶段体积：`phase9` 1.9G ｜ `phase10` 2.5G ｜ **`phase11` 11.5G**（其余阶段合计约 4G）。
⚠ phase11 的 11.5G 里 **约 10G 是"按约定留本地、不入库"的 ckpt 池**（见文末）。

### phase11 的实验目录（20 个）

| 组 | 目录 |
|---|---|
| 筛选（W1 @2000，对照 `a0_2000`） | `a0_2000`（锚点）、`w1_1a`、`w1_1b1`、`w1_1c`、`w1_2a`、`w1_4`、`w1_a0cos`（诊断臂） |
| 确认（W2 @9600，**已降级为诊断**） | `w2_1a`、`w2_4` |
| **权威一轮（P2 三臂 + 复跑）** | `p2_anchor`、`p2_1a`、`p2_4`、`p2_anchor_seed43` |
| 判读与选点 | `p2_judge`（R1/R3 判定留痕 + **复跑曲线 `rerun_curves.json`**）、`p2_val_probe`（48×3 个 val 分数 + `selection.json`）、`val_probe`（1006 选点试算）、`sel_sensitivity`（**7 份选点敏感性判定日志** `judge_curve_*.log`）、`prec_rec_reeval`（前沿曲线比较产物 `curves*.{json,png}`） |
| 其它 | `smoke_5090`（换卡冒烟产物）、`_w2_pkg`（W2 打包件，387M，留本地） |

复跑的**逐 ckpt 结果**在 `p2_anchor_seed43/{eval_epoch_1..3,timeline_epoch_1..3}.json`。
零散文件：`w2_boundary_all.log`、`w2_timeline_all.json`、`w2_timeline_metrics.json`。

> ⚠ **一处已更正的路径**：`docs/1009…总结（终版）.md` 与 `…增补-P2三臂重跑总结.md` 原先写
> `logs/phase11/p2_rerun_probe/anchor.json`，**该目录不存在**；复跑数据实际落在上表两处
> （2026-10-09 盘点时发现并已连同文档一起更正）。

### phase9 的实验目录（23 个）

含 `e1_5ep`（E1@2000 五轮）、`e1_6000`（H）、**`e1_full`（E1-full 收官 —— 全阶段唯一的报告口径锚点，
ckpt 走 LFS 入库）**、`e1_earlystop`、`e1_dinov2base`（方案 E）、`e3_stage1/2`、`m1/m3/m7` 诸方案、
`no_keeplying` / `no_keeplying_fallen`（数据侧 P9/P9b）、`e2_rank_trunc` 等。

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

## `reference/` — 上游实现 + 本地留存区

⚠ **本目录整目录 gitignored** —— 它是仓库里那块"**仓库内、但不入库**"的区域，装两类东西：

**(a) 上游开源实现（6 个，约 1.1G，仅备查阅）**：`C3D`、`SlowFast`、`TimeSformer`、`kinetics-i3d`、
`scenic`、`temporal-shift-module`；另有 `_part_test.bin`。

**(b) 本地留存文档（6 份，均不入库）**：

| 文件 | 是什么 |
|---|---|
| `视觉跌倒检测研究进展汇报（202607）.md`、`（202609）.md` | **写给导师的汇报**（`CLAUDE.md` 明令不入库） |
| `E1-report.html` | E1 发布件（`docs/0911实验第十阶段规划.md` 引用它作为基线来源） |
| `.e1-report.publish-src.html` | 上一份的 publish 源（与发布件**内容不同**，非副本） |
| `autodl_gpu_pricing.md` | 换卡报价 + 换卡预测式（两个实测验证点） |
| `A100_W2_上机清单.md` | A100 实例传什么/不传什么 + 上机步骤 |

文档中引用它们时写 **`reference/<名>`**（路径真实，只是不入库）。

## 工具配置与顶层文件

**仓库根已无游离的脚本或文档**（2026-10-09 归置：`_run_w3_boundary.sh` 入 `experiments/`、
E1 报告与两份工程备忘入 `reference/`、`_smoke_5090_pkg/` 删除）。顶层配置类条目如下：

- `.claude/`（`settings.local.json` + `skills/`）、`.agents/`（`skills/`）、`AGENTS.md` —— agent 工具配置。
- `.gitattributes` —— **LFS 规则**：`data/*.npz` 与 `logs/**/*.pt`。
- `.gitignore` —— 缓存与"本地留存区"规则（含 `reference/` 的说明）。
- `CLAUDE.md` —— **规范层**：目录速查地图 + 跨阶段稳定约定 + Long-Run 任务约定 + 指标定义与验收约定。
- `PROJECT_CONTENTS.md` —— 本文（具体描述层）。

---

## 未入库的大体积内容（事实清单）

| 内容 | 体积 | 位置 |
|---|---|---|
| 逐视频特征目录（5 个） | 28G | `data/{features,dinov2_giant_features,dinov2_features,optical_flow_features,pose_features}/` |
| 实验帧缓存（4 个） | 38G | `data/{p9e1_frames,p9e1_frames_val,p9_frames,p9_features_finetuned}/` |
| **Phase 10 抽帧缓存** | ~25G（97 万文件） | `data/ref_frames/` |
| 特征 NPZ / CSV | ~12G | `data/*.npz`、`data/*.csv` |
| 数据集 | 27G | `DATASET-omnifall/` |
| 上游实现 + 本地留存文档 | 1.1G | `reference/` |
| **留本地的实验 ckpt 池** | ~10G | `logs/phase11/p2_{anchor,1a,anchor_seed43}/step_*_model.pt`（各 48 个）等；另 `logs/phase11/_w2_pkg/`（387M 打包件） |
| 未归档的历史实验产物 | ~1.5G | `logs/phase9/` 下除 `e1_full/` 外的实验目录 |
| 工具配置 | 小 | `.claude/`、`.agents/`、`AGENTS.md` |

> **Phase 11 的入库取舍**（按 `CLAUDE.md`「只维护最优模型那一份」）：
> `p2_4`（= 强通过的臂 4）入了**选中 ckpt `step_13800` + best / epoch_1..3 / last + 全部 pv / json / 日志**；
> 其余三臂只入 **pv + eval + timeline + 日志**（ckpt 留本地）；
> 48-ckpt 候选池与 `_w2_pkg` 打包件一律留本地。

**入库与否的判定规则见 `CLAUDE.md`「缓存纪律」与「git push 与 logs 同步」**（本文只列事实，不重复规则）。
