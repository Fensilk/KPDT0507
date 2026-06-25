"""
TCN 跌倒检测 — 第一阶段实验训练脚本.

实现老师指定的架构和损失函数:
    L_total = L_cls + 0.5 * L_fall + 0.5 * L_fallen

验证 5 项指标:
    1. loss 正常下降
    2. seg_acc > 随机水平 (6.25%)
    3. fall_f1 可计算
    4. fallen_f1 可计算
    5. 预测时间线: standing -> fall -> fallen

用法:
    python train_tcn.py
    python train_tcn.py --epochs 50 --batch_size 32 --lr 5e-4
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# 将项目根目录加入 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.tcn_model import TCNModel
from models.dataset import create_dataloaders, create_longseq_dataloaders
from utils.metrics import compute_all_metrics
from utils.visualization import (
    CLASS_NAMES,
    plot_timeline_comparison,
    plot_confusion_matrix,
    plot_training_curves,
)


# ============================================================
# 配置
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="TCN Fall Detection — Phase 1 Experiment"
    )
    # 数据路径
    parser.add_argument(
        "--npz", type=str,
        default="data/omnifall_preprocessed.npz",
        help="NPZ 文件路径",
    )
    parser.add_argument(
        "--clips_csv", type=str,
        default="data/ofsyn_clips.csv",
        help="Clips CSV 路径",
    )
    parser.add_argument(
        "--splits_dir", type=str,
        default="DATASET-omnifall/splits/syn/random",
        help="Split CSV 目录",
    )

    # 日志
    parser.add_argument(
        "--log_dir", type=str,
        default="logs/tcn_syn_001",
        help="TensorBoard + checkpoint 输出目录",
    )

    # 训练超参数
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15,
                        help="早停耐心值")
    parser.add_argument("--lr_patience", type=int, default=6,
                        help="学习率衰减耐心值")

    # 模型
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--tcn_dropout", type=float, default=0.2)

    # Phase 2: 长序列参数
    parser.add_argument("--window_size", type=int, default=None,
                        help="长序列窗口大小 T (e.g., 16, 32). "
                             "未指定时使用 Phase 1 的 clip 级数据集 (T=5)")
    parser.add_argument("--stride", type=int, default=1,
                        help="滑窗步长 (仅与 --window_size 配合使用)")
    parser.add_argument("--frame_npz", type=str,
                        default="data/omnifall_frame_preprocessed.npz",
                        help="帧级 NPZ 路径 (仅与 --window_size 配合使用)")

    # 改进二: 特征增强
    parser.add_argument("--use_diff", action="store_true",
                        help="拼接相邻 clip 差分特征 (RGB + Δclip → 1024-dim)")
    parser.add_argument("--pose_npz", type=str, default=None,
                        help="姿态特征 NPZ 路径 (e.g., data/omnifall_pose.npz)")
    parser.add_argument("--exp_tag", type=str, default=None,
                        help="实验标签 (用于 log 目录命名, 覆盖 --log_dir)")

    # 系统
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader workers (Windows 建议 0)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto",
                        help="设备: auto / cuda / cpu")
    parser.add_argument("--no_weighted_sampler", action="store_true",
                        help="禁用加权采样")

    return parser.parse_args()


def set_seed(seed: int):
    """固定随机种子."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


# ============================================================
# 损失函数
# ============================================================

def build_criterions(train_dataset, device: torch.device) -> dict:
    """
    构建三个损失函数, 带类别权重.

    - L_cls: CrossEntropyLoss with class weights
    - L_fall: BCEWithLogitsLoss with pos_weight
    - L_fallen: BCEWithLogitsLoss with pos_weight

    Works with both OmniFallTCNDataset (label_16_seqs attribute)
    and LongSequenceDataset (samples from NPZ directly).
    """
    # 统计训练集标签分布 — 兼容两种数据集类型
    if hasattr(train_dataset, 'label_16_seqs'):
        # OmniFallTCNDataset (Phase 1)
        all_labels_16 = torch.cat(
            [seq for seq in train_dataset.label_16_seqs]
        ).numpy()
        all_fall = torch.cat(
            [seq for seq in train_dataset.fall_seqs]
        ).numpy()
        all_fallen = torch.cat(
            [seq for seq in train_dataset.fallen_seqs]
        ).numpy()
    else:
        # LongSequenceDataset (Phase 2): sample subset for efficiency
        print("[INFO] Sampling training labels for loss weights (LongSequenceDataset)...")
        n_samples = min(len(train_dataset), 5000)
        step = max(1, len(train_dataset) // n_samples)
        labels_list, fall_list, fallen_list = [], [], []
        for i in range(0, len(train_dataset), step):
            sample = train_dataset[i]
            labels_list.append(sample["labels_16"])
            fall_list.append(sample["fall_labels"])
            fallen_list.append(sample["fallen_labels"])
        all_labels_16 = torch.cat(labels_list).numpy()
        all_fall = torch.cat(fall_list).numpy()
        all_fallen = torch.cat(fallen_list).numpy()

    # 16 类权重: 反频率
    class_counts = np.bincount(all_labels_16, minlength=16).astype(np.float32)
    class_counts = np.where(class_counts == 0, 1.0, class_counts)  # 防除零
    cls_weights = len(all_labels_16) / (16 * class_counts)
    cls_weights = torch.FloatTensor(cls_weights).to(device)

    # fall pos_weight: 负样本/正样本
    n_pos_fall = max(all_fall.sum(), 1)
    n_neg_fall = max(len(all_fall) - n_pos_fall, 1)
    fall_pos_weight = torch.FloatTensor([n_neg_fall / n_pos_fall]).to(device)

    # fallen pos_weight
    n_pos_fallen = max(all_fallen.sum(), 1)
    n_neg_fallen = max(len(all_fallen) - n_pos_fallen, 1)
    fallen_pos_weight = torch.FloatTensor([n_neg_fallen / n_pos_fallen]).to(device)

    criterions = {
        "cls": nn.CrossEntropyLoss(weight=cls_weights),
        "fall": nn.BCEWithLogitsLoss(pos_weight=fall_pos_weight),
        "fallen": nn.BCEWithLogitsLoss(pos_weight=fallen_pos_weight),
    }

    print(f"[INFO] Class weights (16): {cls_weights.cpu().numpy().round(2)}")
    print(f"[INFO] Fall pos_weight: {fall_pos_weight.item():.2f}")
    print(f"[INFO] Fallen pos_weight: {fallen_pos_weight.item():.2f}")

    return criterions


# ============================================================
# 训练 / 验证一个 Epoch
# ============================================================

def train_epoch(
    model: nn.Module,
    loader,
    criterions: dict,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = len(loader)
    log_every = max(1, n_batches // 10)  # Log ~10 times per epoch

    import time as _time
    t_start = _time.time()
    for i, batch in enumerate(loader):
        features = batch["features"].to(device)          # (B, T, 512)
        labels_16 = batch["labels_16"].to(device)        # (B, T)
        fall_gt = batch["fall_labels"].to(device)         # (B, T)
        fallen_gt = batch["fallen_labels"].to(device)     # (B, T)

        B, T = labels_16.shape

        # 前向传播
        logits_cls, logits_fall, logits_fallen = model(features)
        # logits_cls: (B, T, 16), logits_fall: (B, T, 1), logits_fallen: (B, T, 1)

        # 损失计算
        loss_cls = criterions["cls"](
            logits_cls.reshape(B * T, -1),    # (B*T, 16)
            labels_16.reshape(B * T),          # (B*T,)
        )
        loss_fall = criterions["fall"](
            logits_fall.reshape(B * T),        # (B*T,)
            fall_gt.reshape(B * T).float(),    # (B*T,)
        )
        loss_fallen = criterions["fallen"](
            logits_fallen.reshape(B * T),
            fallen_gt.reshape(B * T).float(),
        )

        # 总损失: 老师指定的权重
        loss = loss_cls + 0.5 * loss_fall + 0.5 * loss_fallen

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

        # Progress log
        if (i + 1) % log_every == 0 or i == 0:
            elapsed = _time.time() - t_start
            eta = elapsed / (i + 1) * (n_batches - i - 1)
            print(f"  [{i+1:>6d}/{n_batches}] "
                  f"{(i+1)/n_batches*100:4.0f}% | "
                  f"loss={total_loss/(i+1):.4f} | "
                  f"elapsed={elapsed:.0f}s | eta={eta:.0f}s", flush=True)

    return total_loss / n_batches


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    loader,
    criterions: dict,
    device: torch.device,
) -> dict:
    model.eval()

    total_loss = 0.0
    n_batches = len(loader)

    # 收集所有预测和标签
    all_pred_cls = []
    all_pred_fall = []
    all_pred_fallen = []
    all_gt_cls = []
    all_gt_fall = []
    all_gt_fallen = []

    for batch in loader:
        features = batch["features"].to(device)
        labels_16 = batch["labels_16"].to(device)
        fall_gt = batch["fall_labels"].to(device)
        fallen_gt = batch["fallen_labels"].to(device)

        B, T = labels_16.shape

        logits_cls, logits_fall, logits_fallen = model(features)

        # 损失
        loss_cls = criterions["cls"](
            logits_cls.reshape(B * T, -1), labels_16.reshape(B * T)
        )
        loss_fall = criterions["fall"](
            logits_fall.reshape(B * T), fall_gt.reshape(B * T).float()
        )
        loss_fallen = criterions["fallen"](
            logits_fallen.reshape(B * T), fallen_gt.reshape(B * T).float()
        )
        loss = loss_cls + 0.5 * loss_fall + 0.5 * loss_fallen
        total_loss += loss.item()

        # 预测
        pred_cls = logits_cls.argmax(dim=-1)         # (B, T)
        pred_fall = (torch.sigmoid(logits_fall) > 0.5).long().squeeze(-1)    # (B, T)
        pred_fallen = (torch.sigmoid(logits_fallen) > 0.5).long().squeeze(-1)

        all_pred_cls.append(pred_cls.cpu().numpy().ravel())
        all_pred_fall.append(pred_fall.cpu().numpy().ravel())
        all_pred_fallen.append(pred_fallen.cpu().numpy().ravel())
        all_gt_cls.append(labels_16.cpu().numpy().ravel())
        all_gt_fall.append(fall_gt.cpu().numpy().ravel())
        all_gt_fallen.append(fallen_gt.cpu().numpy().ravel())

    # 拼接
    pred_cls = np.concatenate(all_pred_cls)
    pred_fall = np.concatenate(all_pred_fall)
    pred_fallen = np.concatenate(all_pred_fallen)
    gt_cls = np.concatenate(all_gt_cls)
    gt_fall = np.concatenate(all_gt_fall)
    gt_fallen = np.concatenate(all_gt_fallen)

    # 计算指标
    metrics = compute_all_metrics(
        pred_cls, pred_fall, pred_fallen,
        gt_cls, gt_fall, gt_fallen,
        CLASS_NAMES,
    )
    metrics["loss"] = total_loss / n_batches

    return metrics


# ============================================================
# 主训练循环
# ============================================================

def main():
    args = parse_args()

    # 基础目录
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(base_dir)

    # exp_tag 覆盖 log_dir
    if args.exp_tag:
        args.log_dir = f"logs/{args.exp_tag}"

    # 创建输出目录
    os.makedirs(args.log_dir, exist_ok=True)

    # 设备
    device = get_device(args.device)
    print(f"[INFO] Using device: {device}")
    set_seed(args.seed)

    # --- 数据加载 ---
    print("\n" + "=" * 60)
    print("Loading data...")
    print("=" * 60)

    npz_path = os.path.join(base_dir, args.npz)
    clips_csv = os.path.join(base_dir, args.clips_csv)
    splits_dir = os.path.join(base_dir, args.splits_dir)
    frame_npz_path = os.path.join(base_dir, args.frame_npz)

    use_longseq = args.window_size is not None

    if use_longseq:
        # Phase 2: 长序列滑窗数据集
        print(f"[INFO] Long sequence mode: T={args.window_size}, stride={args.stride}")
        # Auto-reduce batch size for longer sequences to avoid OOM
        effective_batch = args.batch_size
        if args.window_size >= 32 and args.batch_size >= 64:
            effective_batch = max(16, args.batch_size // 4)
        elif args.window_size >= 16 and args.batch_size >= 64:
            effective_batch = max(24, args.batch_size // 2)
        if effective_batch != args.batch_size:
            print(f"[INFO] Auto-reducing batch_size: {args.batch_size} -> {effective_batch}")

        train_loader, val_loader, test_loader = create_longseq_dataloaders(
            npz_path=frame_npz_path,
            splits_dir=splits_dir,
            window_size=args.window_size,
            stride=args.stride,
            batch_size=effective_batch,
            num_workers=args.num_workers,
            use_weighted_sampler=not args.no_weighted_sampler,
        )
    else:
        # Phase 1: 原有 clip 级数据集 (向后兼容)
        pose_npz_path = None
        if args.pose_npz:
            pose_npz_path = os.path.join(base_dir, args.pose_npz)
        train_loader, val_loader, test_loader = create_dataloaders(
            npz_path=npz_path,
            clips_csv_path=clips_csv,
            splits_dir=splits_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_weighted_sampler=not args.no_weighted_sampler,
            use_diff=args.use_diff,
            pose_npz_path=pose_npz_path,
        )

    # --- 模型 ---
    print("\n" + "=" * 60)
    print("Building model...")
    print("=" * 60)

    if use_longseq:
        model = TCNModel(
            input_dim=512,
            hidden_dim=args.hidden_dim,
            kernel_size=3,
            window_size=args.window_size,
            tcn_dropout=args.tcn_dropout,
            num_classes=16,
        ).to(device)
    else:
        # Phase 1: input_dim 从数据集获取 (支持 use_diff / pose_npz)
        input_dim = train_loader.dataset.feature_dim
        model = TCNModel(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            num_blocks=2,
            kernel_size=3,
            dilations=[1, 2],
            tcn_dropout=args.tcn_dropout,
            num_classes=16,
        ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model params: {n_params:,}")
    print(f"[INFO] Dilations: {model.dilations}")
    print(f"[INFO] Receptive field: {model.receptive_field}")

    # --- 损失函数 ---
    criterions = build_criterions(train_loader.dataset, device)

    # --- 优化器和调度器 ---
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=args.lr_patience,
        factor=0.5,
    )

    # --- 日志 ---
    writer = SummaryWriter(log_dir=args.log_dir)
    history = {
        "train_loss": [],
        "val_loss": [],
        "seg_acc": [],
        "fall_f1": [],
        "fallen_f1": [],
        "lr": [],
    }

    best_fall_f1 = 0.0
    best_epoch = 0
    patience_counter = 0

    # --- 训练循环 ---
    print("\n" + "=" * 60)
    print("Training started")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        # 训练
        train_loss = train_epoch(model, train_loader, criterions, optimizer, device)

        # 验证
        val_metrics = validate_epoch(model, val_loader, criterions, device)

        # 学习率调度
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_metrics["fall_f1"])

        # 记录历史
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["seg_acc"].append(val_metrics["seg_acc"])
        history["fall_f1"].append(val_metrics["fall_f1"])
        history["fallen_f1"].append(val_metrics["fallen_f1"])
        history["lr"].append(current_lr)

        # TensorBoard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
        writer.add_scalar("Metrics/seg_acc", val_metrics["seg_acc"], epoch)
        writer.add_scalar("Metrics/fall_f1", val_metrics["fall_f1"], epoch)
        writer.add_scalar("Metrics/fallen_f1", val_metrics["fallen_f1"], epoch)
        writer.add_scalar("Metrics/avg_f1", val_metrics["avg_f1"], epoch)
        writer.add_scalar("LR", current_lr, epoch)

        # 终端输出
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | "
            f"Seg Acc: {val_metrics['seg_acc']:.4f} | "
            f"Fall F1: {val_metrics['fall_f1']:.4f} | "
            f"Fallen F1: {val_metrics['fallen_f1']:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        # 早停 + 保存最佳模型
        if val_metrics["fall_f1"] > best_fall_f1:
            best_fall_f1 = val_metrics["fall_f1"]
            best_epoch = epoch
            patience_counter = 0

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_fall_f1": best_fall_f1,
                "val_metrics": val_metrics,
                "args": vars(args),
            }
            torch.save(checkpoint, os.path.join(args.log_dir, "tcn_best.pt"))
            print(f"  [BEST] New best model saved! (Fall F1={best_fall_f1:.4f})")
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"\n[INFO] Early stopping at epoch {epoch} "
                  f"(best epoch: {best_epoch}, best Fall F1: {best_fall_f1:.4f})")
            break

    # 保存最终模型
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        os.path.join(args.log_dir, "tcn_last.pt"),
    )

    # 保存训练历史
    with open(os.path.join(args.log_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # 绘制训练曲线
    plot_training_curves(
        history,
        os.path.join(args.log_dir, "training_curves.png"),
    )

    writer.close()

    # ============================================================
    # 测试集评估
    # ============================================================
    print("\n" + "=" * 60)
    print("Evaluating on test set...")
    print("=" * 60)

    # 加载最佳模型
    checkpoint = torch.load(
        os.path.join(args.log_dir, "tcn_best.pt"),
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_metrics = validate_epoch(model, test_loader, criterions, device)

    print(f"\n{'='*60}")
    print("TEST RESULTS")
    print(f"{'='*60}")
    print(f"  seg_acc:       {test_metrics['seg_acc']:.4f}  "
          f"(random baseline: {test_metrics['random_baseline']:.4f})")
    print(f"  fall_f1:       {test_metrics['fall_f1']:.4f}")
    print(f"  fall_precision:{test_metrics['fall_precision']:.4f}")
    print(f"  fall_recall:   {test_metrics['fall_recall']:.4f}")
    print(f"  fallen_f1:     {test_metrics['fallen_f1']:.4f}")
    print(f"  fallen_precision:{test_metrics['fallen_precision']:.4f}")
    print(f"  fallen_recall: {test_metrics['fallen_recall']:.4f}")
    print(f"  avg_f1:        {test_metrics['avg_f1']:.4f}")
    print(f"\n  --- Classification Report (16-class) ---")
    print(test_metrics["cls_report"])

    # 保存测试结果
    with open(os.path.join(args.log_dir, "test_results.json"), "w") as f:
        results_to_save = {
            k: v for k, v in test_metrics.items()
            if k != "cls_report"
        }
        results_to_save["cls_report"] = test_metrics["cls_report"]
        json.dump(results_to_save, f, indent=2, default=str)

    # ============================================================
    # 预测时间线可视化 (老师要求第 5 项)
    # ============================================================
    print("\n" + "=" * 60)
    print("Generating prediction timelines...")
    print("=" * 60)

    timeline_dir = os.path.join(args.log_dir, "timelines")
    os.makedirs(timeline_dir, exist_ok=True)

    # 从测试集中找几个包含 fall 事件的视频
    test_dataset = test_loader.dataset
    fall_video_indices = []

    for idx in range(len(test_dataset)):
        sample = test_dataset[idx]
        if sample["fall_labels"].sum() > 0 and sample["fallen_labels"].sum() > 0:
            fall_video_indices.append(idx)
        if len(fall_video_indices) >= 5:
            break

    if len(fall_video_indices) == 0:
        print("[WARN] No videos with fall+fallen found in test set, using first 5")
        fall_video_indices = list(range(min(5, len(test_dataset))))

    for vi, dataset_idx in enumerate(fall_video_indices):
        sample = test_dataset[dataset_idx]
        features = sample["features"].unsqueeze(0).to(device)  # (1, 5, 512)

        with torch.no_grad():
            logits_cls, logits_fall, logits_fallen = model(features)

        pred_cls = logits_cls.argmax(-1).squeeze(0).cpu().numpy()        # (5,)
        pred_fall = torch.sigmoid(logits_fall).squeeze(0).squeeze(-1).cpu().numpy()  # (5,)
        pred_fallen = torch.sigmoid(logits_fallen).squeeze(0).squeeze(-1).cpu().numpy()
        gt_cls = sample["labels_16"].numpy()
        gt_fall = sample["fall_labels"].numpy()
        gt_fallen = sample["fallen_labels"].numpy()

        plot_timeline_comparison(
            gt_labels=gt_cls,
            pred_labels=pred_cls,
            gt_fall=gt_fall,
            pred_fall=pred_fall,
            gt_fallen=gt_fallen,
            pred_fallen=pred_fallen,
            save_path=os.path.join(timeline_dir, f"video_{vi:03d}.png"),
            video_idx=dataset_idx,
        )

    # ============================================================
    # 总结
    # ============================================================
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"  Log directory: {args.log_dir}")
    print(f"  Input dim: {train_loader.dataset.feature_dim}")
    if args.use_diff:
        print(f"  Feature: RGB + Δclip diff (512 + 512)")
    if args.pose_npz:
        print(f"  Feature: + Pose ({args.pose_npz})")
    if use_longseq:
        print(f"  Window size (T): {args.window_size}, Stride: {args.stride}")
        print(f"  Train samples: {len(train_loader.dataset):,}")
    print(f"  Best epoch: {best_epoch}")
    print(f"  Best Fall F1 (val): {best_fall_f1:.4f}")
    print(f"  ---")
    print(f"  Teacher's 5 requirements:")
    req1 = "PASS" if history["train_loss"][-1] < history["train_loss"][0] else "CHECK"
    req2 = "PASS" if test_metrics["seg_acc"] > test_metrics["random_baseline"] else "FAIL"
    req3 = "PASS" if test_metrics["fall_f1"] > 0 else "FAIL"
    req4 = "PASS" if test_metrics["fallen_f1"] > 0 else "FAIL"
    req5 = "DONE (see timelines/ dir)"

    print(f"  1. Loss decreases:    {req1}")
    print(f"  2. seg_acc > random:  {req2} ({test_metrics['seg_acc']:.4f} > {test_metrics['random_baseline']:.4f})")
    print(f"  3. fall_f1 computed:  {req3} (={test_metrics['fall_f1']:.4f})")
    print(f"  4. fallen_f1 computed:{req4} (={test_metrics['fallen_f1']:.4f})")
    print(f"  5. Timeline plots:    {req5}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
