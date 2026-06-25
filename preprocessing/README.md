# 预处理脚本归档

本文件夹包含 OmniFall OF-Syn 数据集预处理流水线的全部代码和说明文档。

## 文件清单

| 文件 | 说明 |
|------|------|
| `数据预处理流程说明.md` | 完整流程说明文档（新人请先阅读此文档） |
| `step1_load_ofsyn.py` | Step 1: 加载与验证原始数据 |
| `step2_clip_segmentation.py` | Step 2: 时间段标注 → clip 序列 |
| `step3_binary_labels.py` | Step 3: 生成 fall/fallen 二值标签 |
| `step4_extract_features.py` | Step 4: ResNet18 特征提取（NVDEC 硬件解码版） |
| `verify_pipeline.py` | 全流程 9 项质量验证 |

## 目录依赖

脚本从此目录运行，通过 `parent.parent` 寻址上一层（项目根目录）的资源：

```
KPDT0507/                    ← 项目根（parent.parent）
├── DATASET-omnifall/        ← OmniFall 数据集
│   └── data_files/
│       └── extracted/       ← 预解压 .mp4 文件（Step 4 输入）
├── data/
│   ├── ofsyn_clips.csv      ← 60,000 clip 标注
│   └── features/            ← 12,000 .pt 特征文件
└── preprocessing/           ← 你在这里
```

## 运行方式

所有步骤已执行完毕，产出文件已就绪。如需重新运行某个步骤：

```bash
# 确认在项目根目录下（或从 preprocessing/ 运行均可，路径已自适应）
conda activate py310
python preprocessing/step1_load_ofsyn.py
python preprocessing/step2_clip_segmentation.py
python preprocessing/step3_binary_labels.py
python preprocessing/step4_extract_features.py
python preprocessing/verify_pipeline.py
```

**Step 4 前置条件**：
- 安装 ffmpeg 并确认包含 `av1_cuvid` 解码器：`ffmpeg -decoders | findstr av1_cuvid`
- 将 `omnifall-synthetic_av1.tar` 解压到 `DATASET-omnifall/data_files/extracted/`（脚本会自动检测）

## 数据现状（截至 2026-05-18）

| 产出 | 路径 | 状态 |
|------|------|------|
| Clip 标注 | `../data/ofsyn_clips.csv` | 60,000 条 ✓ |
| 视频特征 | `../data/features/*.pt` | 12,000 文件 ✓ |
| 汇聚 npz | `../data/omnifall_preprocessed.npz` | 105.8 MB ✓ |
| 9 项验证 | 全部通过 | ✓ |

**Step 4 NVDEC 实测性能**：5 vids/s，全量 12,000 视频约 44 分钟（GPU batch=12, 8 workers, RTX 4060）。
