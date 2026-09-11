# CLAUDE.md

Guidance for Claude Code when working in this repository. **本文件只写长期稳定、跨阶段不变的内容**；
任何"当前状态"（哪个模型主用、最优后处理规则、最新结论、阶段沿革）一律不固化在此——按下方目录地图去
对应文件/文档现查，避免随项目推进持续腐烂。

## Project Overview

Elderly fall detection research project (master's thesis). OmniFall OF-Syn synthetic videos → per-frame
visual features → sliding-window temporal models → **per-frame ternary classification**: fall(跌倒瞬间) /
fallen(倒地状态) / normal(正常，含故意躺下 lying)。最难点是 fallen vs lying（静态轴，单帧不可分，依赖时序前因）。

## Environment

- Windows 11 (local) + conda env `py310` (Python 3.10)；无 requirements.txt，依赖由 conda 管理
- **GPU (main compute)**: Autodl RTX 3090 (24GB) 云服务器，SSH 别名 `autodl` 免密直连——主力训练/评估在 autodl
- **GPU (local)**: RTX 4060 (8GB)——本地只做无需 GPU 的事（抽帧、分析、冻结特征时序训练、小脚本/冒烟）
- Data: OmniFall OF-Syn，AV1 编码（详见 `DATASET-omnifall/README.md`）

## 目录地图（按图索骥）

> 总原则：想知道"项目当前在做什么实验、最新结论、现在哪个模型/规则最优" → 看 **`docs/` 最新一份阶段文档**
> + **`logs/` 对应 exp_tag**；想知道某类/某模型细节 → 进对应目录读它的 README/docstring。别在 CLAUDE.md 里找当前状态。

| 目录 | 里面有什么 / 去哪找细节 |
|------|------------------------|
| `docs/` | **阶段文档**，按 `MMDD实验第N阶段{规划\|总结\|报告}.md` 命名（如 `0902实验第九阶段总结.md`）。最新实验结论看最新的文件；`docs/superpowers/` 放实现计划。 |
| `models/` | 模型定义，**每个文件 docstring 即架构说明**（`phase6_model.py` 的 `Phase6TernaryModel`+各 decoder 是跨阶段主体；`phase9_e1_model.py` 是 E1 端到端；`dataset.py` 含滑动窗口数据集）。 |
| `experiments/` | 训练/评估/数据准备/批处理脚本，按阶段命名：`train_*.py`（如 `train_phase6.py` 主时序入口、`train_p9e1.py`）、`eval_*.py`、`run_p*.sh`、`prep_*.py`、`extract_*.py`；共享模块 `lora.py`。结果与配置见 `logs/`。 |
| `preprocessing/` | 数据管线 step1-10，自带 `README.md` + `数据预处理流程说明.md`。**`av_decode.py`：AV1 视频解码，PyAV 优先——autodl 的 cv2 解不了 AV1，必须经它读帧**。 |
| `DATASET-omnifall/` | 数据集根，**自带 `README.md`/`STRUCTURE.md`/`LABELS.md`/`CONFIGS.md`**（标签体系、结构、配置以它们为准）；`data_files/extracted/` 视频；`splits/syn/random/`（train/val/test，另有 `e1_2000` 等受控子集）；`labels/`、`parquet/`。 |
| `data/` | 特征 NPZ 与各实验缓存（gitignored）。主源 `omnifall_dinov2_giant_frame.npz`（keys: features/labels_16/fall_labels/fallen_labels/video_paths/video_start_indices，12000×80 帧）；`omnifall_dinov2_finetuned_frame.npz`(E3)、`p9e1_frames*/`(E1 缓存) 等按实验。 |
| `logs/` | 训练产物，按 `logs/{phase}/{exp_tag}/` 组织（training_history.json / test_results.json / TensorBoard / ckpt）。 |
| `utils/` | `metrics.py`(F1/prec/rec)、`losses.py`(Focal)、`visualization.py`、`timeline_metrics.py`。 |
| `analysis/` | 诊断/可视化脚本（`review_errors/` 等）。 |

## 稳定约定（跨阶段不变）

- **三元标签**：0=fall、1=fallen、2=normal（由 16 类派生，完整 16 类映射见 `DATASET-omnifall/LABELS.md`）
- **读帧**：AV1 视频一律走 `preprocessing/av_decode.py`（PyAV 优先，cv2 兜底）；帧均值 <20 = 解码失败金丝雀
- **运行方式**：脚本从项目根运行，靠 `sys.path.insert(0, ...)` 导入；shell runner 用 `cd "$(dirname "$0")/.."`
- **无 pytest**：任务的"测试"= 小规模冒烟运行验证输出
- **NPZ**：`mmap_mode='r'` 载入；Windows 上 DataLoader 用 `num_workers=0`
- **缓存纪律**：`data/` 下大体积 .pt/.npz 缓存 gitignored，不提交
- **git push 与 logs 同步**：`logs/` 是**日志，原则上应随代码同步提交/推送**（保可复现）。**若体积过大**，
  按本阶段研究进展**只维护"最优模型"的那一份训练日志**（其 ckpt + 关键 `json`/日志），其余实验产物留在本地
  不入库——取舍原则是**优先照顾最优模型的可复现性**，而非把每个实验目录都推上去。
- **实验阶段文档优先**：新增实验结论/阶段性总结写 `docs/`，不反哺进本文件

### 指标定义与验收约定（强制，任何情况适用；教训：`event_f1` 是逐帧二值 F1 却被当"事件级"用了整个阶段，2026-09）

- **先定义、后使用**：任何指标在**首次提出/引用时**必须同时写明完整定义——**分子、分母（适用人群）、
  统计口径、单位**。只给名称或缩写（`event_f1`、`F/N` 等）视为**未定义**，不得进入判读、文档或汇报。
  例：覆盖率 =「含该事件的视频中，模型至少报过一次的比例」；误报率 =「不含该事件的视频中被误报的比例」。
- **命名必须名实相符**：叫 `event` 的指标就得做事件级匹配（IoU/容差/实例计数），帧级的就叫帧级。
  名实不符会误导实验方向（该例在 6,000 数据即饱和，当主指标会得出"数据到 6,000 就够"的反向结论）。
- **新指标必须过两道验收才可采用**：① **GT 自检**——真值当预测喂入须得满分（非事件样本应**排除**而非记 0）；
  ② **负对照**——故意注入错误（全漏/平移/抖动）须单调劣化。两者都要**跑出数**并写进文档。
- **已知缺陷随指标一起声明**：如 `order_correct` 在"全漏"时也为 1.0（必须与 detection_rate 联读）、
  分布桶不互斥等；不声明则后人复用必踩坑。
- **多口径分歧如实并列**：同一模型在不同口径下排序可能相反（instance 按帧数聚合 vs per-video 每视频
  等权，E1-full 与 E3-full 互有胜负）→ **互补口径并列报告，禁止挑对自己有利的那个**。

### Long-Run GPU 任务运行约定（强制，任何情况适用；教训：E1 早停空转 ~7h + 方法3 stage1 NaN 靠碰巧才发现，2026-09）

长 GPU 任务（autodl 训练/评估等）**无论是否无人值守，一律必须遵守**——"有人在会话里"不等于
"真的每时每刻在看"：agent 处理其他任务时同样不会持续盯日志。任何长任务启动后立即配齐以下四项：

1. **启动前先写终止预案**：预估总时长；提前决定"若提前终止（早停/崩溃/OOM/NaN 发散）应自动做什么"，并把预案固化成可执行命令/脚本——**不依赖人工发现后再补救**。例如 E1 早停的自动响应是 `--early_stop_patience 0` 跑满轮重训；训练 loss=nan 应立即停、诊断（降 lr / 查数据）后重启，绝不带病跑完污染下游。
2. **进程级退出监视，而非只盯里程碑**：训练可能"正常完成 / 早停 / 崩溃 / 发散"，除正常完成外都算异常。监视器必须以进程消失（`pgrep`）为信号，进程一退立即判断终止类型并执行预案；**loss=nan 这类"进程还活着但已在产出垃圾"的情况要靠定时检查及早抓**，不能等 epoch 结束。
3. **定时进度检查是强制项**，且**不得只依赖"会话内 + 仅 REPL 空闲才触发"的机制**（如 Claude 会话 cron，它会因会话忙/关而失效）：须由可靠调度承担——服务器端 cron / 启动脚本自带 watchdog / 或显式把"几点出结果、跑完自动执行 X、异常找谁"交接给用户。
4. **结束/异常即时可见**：每个运行给出日志路径 + 现成 tail 命令（Git Bash 与 PowerShell 双版本）+ 本地日志镜像；进程退出即汇报，杜绝数小时空转或发散无人察觉。

## Commands

```bash
conda activate py310

# 主时序训练入口（Phase 6+ 共用）：window/stride/decoder 参数见脚本 --help 或对应阶段 run_*.sh
python experiments/train_phase6.py --bridge --window_size 64 --stride 8 --exp_tag phase7/my_exp

# 批处理实验 / 冒烟
bash experiments/run_p7d.sh
python preprocessing/verify_pipeline.py
python models/dataset.py --longseq
```
