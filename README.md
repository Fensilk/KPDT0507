# KPDT0507 — TCN 跌倒检测

基于 Temporal Convolutional Network (TCN) 的跌倒检测研究项目，使用 Omnifall 数据集。

## 目录结构

```
├── experiments/         # 实验运行脚本
│   ├── train_tcn.py              # 主训练脚本 (Phase 1 & 2 通用)
│   ├── _run_train_strided.py     # 灵活训练 (T + stride 命令行参数)
│   ├── _run_train_t16.py         # T=16 独立训练
│   ├── _run_train_t32.py         # T=32 独立训练
│   └── _run_improve2.py          # 改进二实验 (差分 + 姿态)
├── analysis/            # 分析与验证脚本
│   ├── diagnose_overfit.py       # 过拟合诊断
│   ├── _compare_improve2.py      # 实验对比
│   ├── _verify_pipeline.py       # 6 层流水线验证
│   └── visualize_timeline_clear.py # 预测时间线可视化
├── models/              # 模型定义
│   ├── tcn_model.py
│   └── dataset.py
├── utils/               # 工具函数 (metrics, visualization)
├── preprocessing/       # 数据预处理流水线 (step1-step6)
├── docs/                # 研究文档
│   ├── 0513跌倒检测_研究选题整理.md
│   ├── 0518实验第一阶段规划.md
│   ├── 0607实验第二阶段规划.md
│   ├── 改进一_数据角度_实验总结.md
│   ├── 改进二_模型角度_实验总结.md
│   └── dataset-preprocessing.md
├── data/                # 数据文件
│   ├── cifar-10-batches-py/      # CIFAR-10 (预训练特征)
│   ├── features/                 # 提取的特征 (.pt)
│   ├── pose_features/            # 姿态特征
│   ├── omnifall_*.npz           # 预处理后的数据
│   └── ofsyn_*.csv              # 标签文件
├── DATASET-omnifall/    # 外部数据集 (git submodule)
├── logs/                # 训练日志和模型检查点
```

## 环境

- Python 3.10+
- PyTorch + CUDA
- 主要依赖: numpy, pandas, matplotlib, tensorboard

## 快速开始

```bash
# 主训练 (Phase 1, clip级)
python experiments/train_tcn.py --epochs 100

# 长序列训练 (Phase 2, T=16, stride=4)
python experiments/_run_train_strided.py 16 4

# 改进二实验 (差分 + 姿态)
python experiments/_run_improve2.py --use_diff --pose_npz data/omnifall_pose.npz

# 流水线验证
python analysis/_verify_pipeline.py

# 过拟合诊断
python analysis/diagnose_overfit.py
```

## 损失函数

```
L_total = L_cls + 0.5 * L_fall + 0.5 * L_fallen
```

三项分别对应：16类动作分类、跌倒二分类、倒地二分类。
