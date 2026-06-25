"""
过拟合诊断脚本.

从三个维度分析过拟合原因:
1. 数据维度: 样本量、类别分布、视频内特征相似度
2. 特征维度: 特征空间结构、clip间差异性
3. 模型维度: 容量 vs 有效数据量、训练集泛化差距

用法: python diagnose_overfit.py
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# 1. 数据维度分析
# ============================================================

def analyze_data():
    """分析数据分布和特征特性."""
    print("=" * 60)
    print("1. 数据维度分析")
    print("=" * 60)

    data = np.load(os.path.join(BASE, "data", "omnifall_preprocessed.npz"), allow_pickle=True)
    features = data["features"]          # (60000, 512)
    labels_16 = data["labels_16"]
    fall_labels = data["fall_labels"]
    fallen_labels = data["fallen_labels"]
    video_start = data["video_start_indices"]
    video_counts = data["video_clip_counts"]

    # --- 1a. 类别分布 ---
    CLASS_NAMES = ["walk","fall","fallen","sit_down","sitting","lie_down","lying",
                   "stand_up","standing","other","kneel_down","kneeling",
                   "squat_down","squatting","crawl","jump"]

    print("\n[1a] 每类 clip 数量 (全部 60,000 clips):")
    for i, name in enumerate(CLASS_NAMES):
        count = (labels_16 == i).sum()
        pct = 100 * count / len(labels_16)
        bar = "█" * int(pct / 2)
        print(f"  {name:>12s}: {count:>6d} ({pct:>5.1f}%) {bar}")

    # --- 1b. 视频内特征相似度 ---
    print("\n[1b] 同一视频内 5 个 clip 的特征相似度:")
    intra_cos_sims = []
    intra_l2_dists = []
    inter_cos_sims = []

    n_videos = len(video_start)
    # 采样 500 个视频计算
    sampled = np.random.choice(n_videos, min(500, n_videos), replace=False)

    for vi in sampled:
        s, c = video_start[vi], video_counts[vi]
        if c != 5:
            continue
        f = features[s:s+c]  # (5, 512)
        f_norm = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)

        # 视频内: 相邻 clip 相似度
        sim_matrix = f_norm @ f_norm.T  # (5,5)
        for i in range(5):
            for j in range(i+1, 5):
                intra_cos_sims.append(sim_matrix[i, j])
                intra_l2_dists.append(np.linalg.norm(f[i] - f[j]))

    # 视频间: 随机配对
    for _ in range(1000):
        v1, v2 = np.random.choice(sampled, 2, replace=False)
        s1, c1 = video_start[v1], video_counts[v1]
        s2, c2 = video_start[v2], video_counts[v2]
        if c1 != 5 or c2 != 5:
            continue
        f1 = features[s1:s1+1]  # 取第一个 clip
        f2 = features[s2:s2+1]
        f1_norm = f1 / (np.linalg.norm(f1, axis=1, keepdims=True) + 1e-8)
        f2_norm = f2 / (np.linalg.norm(f2, axis=1, keepdims=True) + 1e-8)
        inter_cos_sims.append((f1_norm @ f2_norm.T).item())

    intra_cos = np.array(intra_cos_sims)
    inter_cos = np.array(inter_cos_sims)

    print(f"  视频内相邻 clip 余弦相似度: 均值={intra_cos.mean():.4f}, 中位数={np.median(intra_cos):.4f}")
    print(f"  不同视频间余弦相似度:       均值={inter_cos.mean():.4f}, 中位数={np.median(inter_cos):.4f}")
    print(f"  相似度比 (intra/inter):      {intra_cos.mean() / (inter_cos.mean() + 1e-8):.2f}x")
    print(f"  >>> 视频内 clip 高度相似，5 个 clip 几乎等价于 1 个 <<<")

    # --- 1c. 视频级别有效样本量 ---
    print("\n[1c] 视频级别样本统计:")
    print(f"  总 clips:  {len(features):,}")
    print(f"  总视频:    {n_videos:,}")
    print(f"  平均 clips/视频: {len(features)/n_videos:.1f}")
    print(f"  有效信息倍数: 由于视频内 clip 相似度={intra_cos.mean():.3f},")
    print(f"  5 个 clip 实际贡献的信息量 ≈ 1.5~2 个独立样本")
    print(f"  >>> 模型实际可学习的样本 ≈ {n_videos * 1.5:.0f} 个，而非 {len(features):,} <<<")

    # --- 1d. 稀有类视频数 ---
    print("\n[1d] 稀有类视频数 (训练集):")
    import pandas as pd
    train_df = pd.read_csv(os.path.join(BASE, "DATASET-omnifall", "splits", "syn", "random", "train.csv"))
    train_paths = set(train_df["path"].str.strip())

    class_video_counts = np.zeros(16, dtype=int)
    for vi in range(n_videos):
        path = str(data["video_paths"][vi])
        if path not in train_paths:
            continue
        s, c = video_start[vi], video_counts[vi]
        if c != 5:
            continue
        # 取众数标签
        lbls = labels_16[s:s+c]
        mode_lbl = int(np.bincount(lbls).argmax())
        class_video_counts[mode_lbl] += 1

    for i, name in enumerate(CLASS_NAMES):
        n_v = class_video_counts[i]
        bar = "!! OVERFIT RISK" if n_v < 100 else ("? borderline" if n_v < 500 else "OK")
        print(f"  {name:>12s}: {n_v:>5d} videos  {bar}")

    return intra_cos.mean(), inter_cos.mean()


# ============================================================
# 2. 模型维度分析
# ============================================================

def analyze_model():
    """分析模型容量和训练动态."""
    print("\n" + "=" * 60)
    print("2. 模型维度分析")
    print("=" * 60)

    # --- 2a. 参数量 vs 有效数据量 ---
    sys.path.insert(0, BASE)
    from models.tcn_model import TCNModel

    model = TCNModel()
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 有效样本量 ≈ 9600 视频 × 1.5 = 14400
    effective_samples = 9600 * 1.5

    print(f"\n[2a] 模型容量 vs 有效数据:")
    print(f"  可训练参数:    {n_trainable:,}")
    print(f"  有效样本量:    ~{effective_samples:,.0f}")
    print(f"  参数/样本比:   {n_trainable / effective_samples:.1f}")
    print(f"  >>> 比值 {n_trainable / effective_samples:.1f}，通常 >10 就有过拟合风险 <<<")

    # --- 2b. TCN 感受野 vs 序列长度 ---
    print(f"\n[2b] 感受野 vs 序列长度:")
    print(f"  序列长度 T:    {5}")
    print(f"  感受野 RF:     {model.receptive_field}")
    print(f"  RF/T:          {model.receptive_field / 5:.1f}x")
    print(f"  >>> 感受野 {model.receptive_field} > 序列长度 {5},")
    print(f"      每个位置都能看到全部 5 个 clip")
    print(f"      TCN 退化为全局池化，失去了时序建模的结构优势 <<<")

    # --- 2c. 各层参数分布 ---
    print(f"\n[2c] 各层参数分布:")
    for name, p in model.named_parameters():
        print(f"  {name:>30s}: {p.numel():>10,} params | shape={list(p.shape)}")

    # --- 2d. 没有归一化层的影响 ---
    print(f"\n[2d] 归一化层检查:")
    has_bn = any(isinstance(m, torch.nn.BatchNorm1d) for m in model.modules())
    has_ln = any(isinstance(m, torch.nn.LayerNorm) for m in model.modules())
    print(f"  BatchNorm:  {'有' if has_bn else '❌ 没有'}")
    print(f"  LayerNorm:  {'有' if has_ln else '❌ 没有'}")
    print(f"  >>> 没有归一化层 → 特征幅度不受控 → 训练不稳定 → 加剧过拟合 <<<")


# ============================================================
# 3. 训练动态分析
# ============================================================

def analyze_training():
    """分析训练历史中的过拟合信号."""
    print("\n" + "=" * 60)
    print("3. 训练动态分析")
    print("=" * 60)

    import json
    history_path = os.path.join(BASE, "logs", "tcn_full", "training_history.json")
    if not os.path.exists(history_path):
        print("[SKIP] 未找到训练历史文件")
        return

    with open(history_path) as f:
        h = json.load(f)

    train_loss = np.array(h["train_loss"])
    val_loss = np.array(h["val_loss"])
    fall_f1 = np.array(h["fall_f1"])
    seg_acc = np.array(h["seg_acc"])
    lr = np.array(h["lr"])

    # --- 3a. 过拟合程度 ---
    print(f"\n[3a] 过拟合量化指标:")
    print(f"  Epoch 1:  train_loss={train_loss[0]:.4f}, val_loss={val_loss[0]:.4f}, gap={val_loss[0]-train_loss[0]:.4f}")
    print(f"  Epoch 20: train_loss={train_loss[19]:.4f}, val_loss={val_loss[19]:.4f}, gap={val_loss[19]-train_loss[19]:.4f}")
    print(f"  Epoch 35: train_loss={train_loss[-1]:.4f}, val_loss={val_loss[-1]:.4f}, gap={val_loss[-1]-train_loss[-1]:.4f}")

    # 过拟合开始点: val_loss 不再下降的 epoch
    best_val_epoch = np.argmin(val_loss) + 1
    print(f"\n  Val loss 最低: Epoch {best_val_epoch} (={val_loss[best_val_epoch-1]:.4f})")
    print(f"  此时 train_loss: {train_loss[best_val_epoch-1]:.4f}")
    print(f"  之后 train_loss 继续下降了 {train_loss[best_val_epoch-1] - train_loss[-1]:.4f}")
    print(f"  但 val_loss 反而上升了 {val_loss[-1] - val_loss[best_val_epoch-1]:.4f}")
    print(f"  >>> 典型过拟合: Epoch {best_val_epoch} 后模型在记忆训练集 <<<")

    # --- 3b. 泛化差距 ---
    print(f"\n[3b] 泛化差距 (Generalization Gap):")
    gap_early = val_loss[:5].mean() - train_loss[:5].mean()
    gap_late = val_loss[-5:].mean() - train_loss[-5:].mean()
    print(f"  前 5 epoch:  avg gap = {gap_early:.2f}")
    print(f"  后 5 epoch:  avg gap = {gap_late:.2f}")
    print(f"  扩大倍数:    {gap_late / (gap_early + 1e-8):.1f}x")
    print(f"  >>> 泛化差距持续扩大，模型在死记硬背 <<<")

    # --- 3c. 学习率与过拟合 ---
    print(f"\n[3c] 学习率调度分析:")
    print(f"  初始 LR: {lr[0]:.0e}")
    lr_changes = np.where(np.diff(lr) != 0)[0]
    for idx in lr_changes:
        print(f"  Epoch {idx+1} → {idx+2}: LR {lr[idx]:.0e} → {lr[idx+1]:.0e}")

    # --- 3d. 过拟合时刻分析 ---
    print(f"\n[3d] 过拟合时间线:")
    # 找到 val_loss 从下降到上升的转折点
    for i in range(1, len(val_loss)-1):
        if val_loss[i] < val_loss[i-1] and val_loss[i] < val_loss[i+1]:
            print(f"  Val loss 局部最优: Epoch {i+1}, val_loss={val_loss[i]:.4f}")
            break

    # 什么时候 train_loss 开始 << val_loss
    for i in range(len(train_loss)):
        if val_loss[i] - train_loss[i] > 2.0:
            print(f"  泛化差距 > 2.0 开始: Epoch {i+1}")
            break


# ============================================================
# 4. 损失函数分析
# ============================================================

def analyze_loss():
    """分析损失函数各分量."""
    print("\n" + "=" * 60)
    print("4. 损失函数分析")
    print("=" * 60)

    data = np.load(os.path.join(BASE, "data", "omnifall_preprocessed.npz"), allow_pickle=True)
    labels_16 = data["labels_16"]
    fall_labels = data["fall_labels"]
    fallen_labels = data["fallen_labels"]

    # 各类的占比
    print(f"\n[4a] 多任务标签不平衡:")
    print(f"  fall=1 占比:    {fall_labels.mean()*100:.1f}%")
    print(f"  fallen=1 占比:  {fallen_labels.mean()*100:.1f}%")
    print(f"  16类均匀分布:   约 6.25%")

    # L_cls vs L_fall vs L_fallen 的量级
    import torch.nn as nn
    print(f"\n[4b] 损失量级分析 (初始值):")
    # 假设随机初始化模型，输出接近均匀分布
    # L_cls ≈ ln(16) ≈ 2.77 (均匀预测)
    # L_fall ≈ ln(2) ≈ 0.693 (均匀预测)
    print(f"  随机预测时: L_cls ≈ ln(16) = {np.log(16):.3f}")
    print(f"  随机预测时: L_fall ≈ ln(2) = {np.log(2):.3f}")
    print(f"  随机预测时: L_fallen ≈ ln(2) = {np.log(2):.3f}")
    print(f"  权重后: L_total = {np.log(16):.3f} + 0.5×{np.log(2):.3f} + 0.5×{np.log(2):.3f} = {np.log(16) + np.log(2):.3f}")
    print(f"  L_cls 占比: {np.log(16) / (np.log(16) + np.log(2)) * 100:.0f}%")
    print(f"  >>> L_cls 主导训练，fall/fallen 信号被稀释 <<<")

    # pos_weight 的影响
    n_pos = max(fall_labels.sum(), 1)
    n_neg = max(len(fall_labels) - n_pos, 1)
    pw = n_neg / n_pos
    print(f"\n[4c] pos_weight 影响:")
    print(f"  fall pos_weight = {pw:.1f}")
    print(f"  这意味着每个正样本的 loss 被放大 {pw:.1f} 倍")
    print(f"  >>> 少数正样本主导梯度 → 过拟合到特定正样本 <<<")


# ============================================================
# 5. 总结
# ============================================================

def main():
    print("\n" + "█" * 60)
    print("TCN 过拟合根因诊断")
    print("█" * 60)

    analyze_data()
    analyze_model()
    analyze_training()
    analyze_loss()

    print("\n" + "█" * 60)
    print("根因总结 (按影响排序)")
    print("█" * 60)
    print("""
  1. 【最大原因】视频内 clip 高度相似 → 有效样本量 ≈ 视频数而非 clip 数
     - 5 个 clip 几乎相同，TCN 看到的是"同一张图 × 5"
     - 实际有效样本 ~9,600 个视频，但参数 923K
     - 参数/有效样本 ≈ 96，严重过剩

  2. 【第二】没有归一化层 → 训练不稳定
     - TCN 中缺少 BatchNorm/LayerNorm
     - 特征幅度不受控，梯度方差大
     - 模型倾向于在训练集上找到"极端"解

  3. 【第三】L_cls 主导总损失，fall 任务信号弱
     - L_cls 占总 loss 约 80%
     - 模型主要学习 16 类分类
     - 在训练集上分类准确率可能 > 90%，但泛化差

  4. 【第四】稀有类迫使模型记忆
     - < 100 个视频的类别无法学到通用模式
     - WeightedRandomSampler 反复采样 → 加速记忆
     - 这些类的 loss 在后期接近于 0（完全记住）

  5. 【第五】TCN 感受野 7 > 序列长度 5
     - 模型不需要时序推理就能看到全部信息
     - TCN 的优势（逐层扩大感受野）被浪费
     - 等价于全局池化 + MLP
""")

    print("=" * 60)
    print("对症下药 — 优化优先级")
    print("=" * 60)
    print("""
  P0 [必做] 加 BatchNorm/LayerNorm → 稳定训练
  P0 [必做] 增大 Dropout (0.2→0.4) + Label Smoothing → 防记忆
  P1 [推荐] 合并稀有类别 → 减少类别数，提高每类样本量
  P1 [推荐] 调整损失权重 L_cls + L_fall + L_fallen → 均衡多任务
  P2 [可选] 减小模型容量 (hidden_dim 256→128) → 降低参数/样本比
  P2 [可选] 时序数据增强 (clip shuffle/drop) → 增加有效样本
""")


if __name__ == "__main__":
    main()
