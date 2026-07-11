"""
Transformer 跌倒检测 — 第四阶段实验训练脚本.

使用 DINOv2 帧级特征 + MultimodalFeatureTransformer 时序建模。
固定使用 LongSequenceDataset (帧级滑窗)，默认 T=32, stride=8。

损失函数: L_total = w_cls * L_cls + w_fall * L_fall + w_fallen * L_fallen
早停: 基于 val fall_f1, patience=15.
FP16: torch.cuda.amp.autocast 加速.

用法:
    python experiments/train_transformer.py
    python experiments/train_transformer.py --exp_tag phase4/p4a_dino_baseline
    python experiments/train_transformer.py --frame_npz data/omnifall_dinov2_frame.npz
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.transformer_model import MultimodalFeatureTransformer
from models.dataset import create_longseq_dataloaders
from utils.metrics import compute_all_metrics
from utils.visualization import (
    CLASS_NAMES,
    plot_training_curves,
)


# ============================================================
# 配置
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Transformer Fall Detection — Phase 4 Experiment"
    )
    # 数据路径
    parser.add_argument("--frame_npz", type=str,
                        default="data/omnifall_dinov2_frame.npz",
                        help="帧级 NPZ 路径 (DINOv2 或 ResNet18)")
    parser.add_argument("--splits_dir", type=str,
                        default="DATASET-omnifall/splits/syn/random",
                        help="Split CSV 目录")
    parser.add_argument("--window_size", type=int, default=32,
                        help="序列长度 T (default: 32)")
    parser.add_argument("--stride", type=int, default=8,
                        help="滑窗步长 (default: 8)")

    # 辅助特征
    parser.add_argument("--use_diff", action="store_true",
                        help="拼接帧间差分特征 (Δframe)")
    parser.add_argument("--pose_npz", type=str, default=None,
                        help="帧级 Pose NPZ 路径")

    # 日志
    parser.add_argument("--log_dir", type=str,
                        default="logs/phase4/p4a_dino_baseline",
                        help="TensorBoard + checkpoint 输出目录")
    parser.add_argument("--exp_tag", type=str, default=None,
                        help="实验标签 (覆盖 --log_dir)")

    # 训练超参数
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size (T=32 默认 32)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lr_patience", type=int, default=6)

    # 模型超参数
    parser.add_argument("--hidden_dim", type=int, default=384)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--num_heads", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--num_classes", type=int, default=16,
                        help="输出类别数 (default: 16; 11 if --remap_rare)")
    parser.add_argument("--remap_rare", action="store_true",
                        help="将稀有类 (10-15) 合并为第10类, 16类→11类")

    # 损失权重
    parser.add_argument("--w_cls", type=float, default=1.0)
    parser.add_argument("--w_fall", type=float, default=0.5)
    parser.add_argument("--w_fallen", type=float, default=0.5)

    # 系统
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--no_weighted_sampler", action="store_true")
    parser.add_argument("--no_fp16", action="store_true",
                        help="禁用 FP16 autocast")
    parser.add_argument("--fast", action="store_true",
                        help="启用 cuDNN benchmark")

    return parser.parse_args()


def set_seed(seed: int, fast: bool = False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if fast:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
        else:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


# ============================================================
# 损失函数
# ============================================================

def build_criterions(train_dataset, device: torch.device, remap: bool = False) -> dict:
    """Build weighted loss functions from training label distribution."""
    # Sample subset for efficiency
    print("[INFO] Sampling training labels for loss weights...")
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

    # Remap rare classes if requested
    if remap:
        REMAP = {11: 10, 12: 10, 13: 10, 14: 10, 15: 10}
        for k, v in REMAP.items():
            all_labels_16[all_labels_16 == k] = v

    num_cls = 11 if remap else 16
    class_counts = np.bincount(all_labels_16, minlength=num_cls).astype(np.float32)
    class_counts = np.where(class_counts == 0, 1.0, class_counts)
    cls_weights = len(all_labels_16) / (num_cls * class_counts)
    cls_weights = torch.FloatTensor(cls_weights).to(device)

    # Fall pos_weight
    n_pos_fall = max(all_fall.sum(), 1)
    n_neg_fall = max(len(all_fall) - n_pos_fall, 1)
    fall_pos_weight = torch.FloatTensor([n_neg_fall / n_pos_fall]).to(device)

    # Fallen pos_weight
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

def train_epoch(model, loader, criterions, optimizer, device,
                w_cls=1.0, w_fall=0.5, w_fallen=0.5, use_fp16=True,
                scaler=None, remap_fn=None) -> float:
    model.train()
    total_loss = 0.0
    n_batches = len(loader)
    log_every = max(1, n_batches // 10)

    import time as _time
    t_start = _time.time()
    for i, batch in enumerate(loader):
        features = batch["features"].to(device)
        labels_16 = batch["labels_16"].to(device)
        if remap_fn is not None:
            labels_16 = remap_fn(labels_16)
        fall_gt = batch["fall_labels"].to(device)
        fallen_gt = batch["fallen_labels"].to(device)
        B, T = labels_16.shape

        optimizer.zero_grad()

        if use_fp16 and scaler is not None:
            with torch.cuda.amp.autocast():
                logits_cls, logits_fall, logits_fallen = model(features)
                loss_cls = criterions["cls"](
                    logits_cls.reshape(B * T, -1), labels_16.reshape(B * T))
                loss_fall = criterions["fall"](
                    logits_fall.reshape(B * T), fall_gt.reshape(B * T).float())
                loss_fallen = criterions["fallen"](
                    logits_fallen.reshape(B * T), fallen_gt.reshape(B * T).float())
                loss = w_cls * loss_cls + w_fall * loss_fall + w_fallen * loss_fallen
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits_cls, logits_fall, logits_fallen = model(features)
            loss_cls = criterions["cls"](
                logits_cls.reshape(B * T, -1), labels_16.reshape(B * T))
            loss_fall = criterions["fall"](
                logits_fall.reshape(B * T), fall_gt.reshape(B * T).float())
            loss_fallen = criterions["fallen"](
                logits_fallen.reshape(B * T), fallen_gt.reshape(B * T).float())
            loss = w_cls * loss_cls + w_fall * loss_fall + w_fallen * loss_fallen
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()

        if (i + 1) % log_every == 0 or i == 0:
            elapsed = _time.time() - t_start
            eta = elapsed / (i + 1) * (n_batches - i - 1)
            print(f"  [{i+1:>6d}/{n_batches}] "
                  f"{(i+1)/n_batches*100:4.0f}% | "
                  f"loss={total_loss/(i+1):.4f} | "
                  f"elapsed={elapsed:.0f}s | eta={eta:.0f}s", flush=True)

    return total_loss / n_batches


@torch.no_grad()
def validate_epoch(model, loader, criterions, device,
                   w_cls=1.0, w_fall=0.5, w_fallen=0.5,
                   remap_fn=None) -> dict:
    model.eval()

    total_loss = 0.0
    n_batches = len(loader)
    all_pred_cls, all_pred_fall, all_pred_fallen = [], [], []
    all_gt_cls, all_gt_fall, all_gt_fallen = [], [], []

    for batch in loader:
        features = batch["features"].to(device)
        labels_16 = batch["labels_16"].to(device)
        if remap_fn is not None:
            labels_16 = remap_fn(labels_16)
        fall_gt = batch["fall_labels"].to(device)
        fallen_gt = batch["fallen_labels"].to(device)
        B, T = labels_16.shape

        logits_cls, logits_fall, logits_fallen = model(features)

        loss_cls = criterions["cls"](
            logits_cls.reshape(B * T, -1), labels_16.reshape(B * T))
        loss_fall = criterions["fall"](
            logits_fall.reshape(B * T), fall_gt.reshape(B * T).float())
        loss_fallen = criterions["fallen"](
            logits_fallen.reshape(B * T), fallen_gt.reshape(B * T).float())
        loss = w_cls * loss_cls + w_fall * loss_fall + w_fallen * loss_fallen
        total_loss += loss.item()

        pred_cls = logits_cls.argmax(dim=-1)
        pred_fall = (torch.sigmoid(logits_fall) > 0.5).long().squeeze(-1)
        pred_fallen = (torch.sigmoid(logits_fallen) > 0.5).long().squeeze(-1)

        all_pred_cls.append(pred_cls.cpu().numpy().ravel())
        all_pred_fall.append(pred_fall.cpu().numpy().ravel())
        all_pred_fallen.append(pred_fallen.cpu().numpy().ravel())
        all_gt_cls.append(labels_16.cpu().numpy().ravel())
        all_gt_fall.append(fall_gt.cpu().numpy().ravel())
        all_gt_fallen.append(fallen_gt.cpu().numpy().ravel())

    pred_cls = np.concatenate(all_pred_cls)
    pred_fall = np.concatenate(all_pred_fall)
    pred_fallen = np.concatenate(all_pred_fallen)
    gt_cls = np.concatenate(all_gt_cls)
    gt_fall = np.concatenate(all_gt_fall)
    gt_fallen = np.concatenate(all_gt_fallen)

    metrics = compute_all_metrics(
        pred_cls, pred_fall, pred_fallen,
        gt_cls, gt_fall, gt_fallen, CLASS_NAMES)
    metrics["loss"] = total_loss / n_batches

    return metrics


# ============================================================
# 主训练循环
# ============================================================

def main():
    args = parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(base_dir)

    if args.exp_tag:
        args.log_dir = f"logs/{args.exp_tag}"

    os.makedirs(args.log_dir, exist_ok=True)

    device = get_device(args.device)
    print(f"[INFO] Using device: {device}")
    set_seed(args.seed, fast=args.fast)

    # --- 数据加载 ---
    print("\n" + "=" * 60)
    print("Loading data...")
    print("=" * 60)

    frame_npz_path = os.path.join(base_dir, args.frame_npz)
    splits_dir = os.path.join(base_dir, args.splits_dir)
    pose_npz_path = None
    if args.pose_npz:
        pose_npz_path = os.path.join(base_dir, args.pose_npz)

    train_loader, val_loader, test_loader = create_longseq_dataloaders(
        npz_path=frame_npz_path,
        splits_dir=splits_dir,
        window_size=args.window_size,
        stride=args.stride,
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

    input_dim = train_loader.dataset.feature_dim
    print(f"[INFO] Input dim: {input_dim} (from dataset)")

    # Label remapping: rare classes (10-15) → 10, 16→11 classes
    if args.remap_rare:
        args.num_classes = 11
        def remap_labels(labels):
            """Remap classes 10-15 → 10 (rare→other), classes 0-9 unchanged."""
            mask = labels >= 11
            labels = labels.clone()
            labels[mask] = 10
            return labels
    else:
        def remap_labels(labels):
            return labels

    model = MultimodalFeatureTransformer(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        max_len=args.window_size + 10,
        num_classes=args.num_classes,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model params: {n_params:,}")
    print(f"[INFO] Architecture: {args.num_layers}L/{args.hidden_dim}d, "
          f"{args.num_heads} heads, dropout={args.dropout}")
    print(f"[INFO] Classes: {args.num_classes}" + (" (remapped 16→11)" if args.remap_rare else ""))
    print(f"[INFO] Window: T={args.window_size}, stride={args.stride}")

    # --- 损失函数 ---
    criterions = build_criterions(train_loader.dataset, device, remap=args.remap_rare)

    # --- 优化器 ---
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=args.lr_patience, factor=0.5)

    # --- FP16 scaler ---
    use_fp16 = not args.no_fp16 and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_fp16 else None
    print(f"[INFO] FP16 autocast: {'enabled' if use_fp16 else 'disabled'}")

    # --- 日志 ---
    writer = SummaryWriter(log_dir=args.log_dir)
    history = {
        "train_loss": [], "val_loss": [],
        "seg_acc": [], "fall_f1": [], "fallen_f1": [], "lr": [],
    }

    best_fall_f1 = 0.0
    best_epoch = 0
    patience_counter = 0

    # --- 训练循环 ---
    print("\n" + "=" * 60)
    print("Training started")
    print("=" * 60)

    val_metrics = {}  # initialize for early-stop reference
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model, train_loader, criterions, optimizer, device,
            w_cls=args.w_cls, w_fall=args.w_fall, w_fallen=args.w_fallen,
            use_fp16=use_fp16, scaler=scaler, remap_fn=remap_labels)

        val_metrics = validate_epoch(
            model, val_loader, criterions, device,
            w_cls=args.w_cls, w_fall=args.w_fall, w_fallen=args.w_fallen,
            remap_fn=remap_labels)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_metrics["fall_f1"])

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["seg_acc"].append(val_metrics["seg_acc"])
        history["fall_f1"].append(val_metrics["fall_f1"])
        history["fallen_f1"].append(val_metrics["fallen_f1"])
        history["lr"].append(current_lr)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
        writer.add_scalar("Metrics/seg_acc", val_metrics["seg_acc"], epoch)
        writer.add_scalar("Metrics/fall_f1", val_metrics["fall_f1"], epoch)
        writer.add_scalar("Metrics/fallen_f1", val_metrics["fallen_f1"], epoch)
        writer.add_scalar("Metrics/avg_f1", val_metrics["avg_f1"], epoch)
        writer.add_scalar("LR", current_lr, epoch)

        train_val_gap = train_loss - val_metrics["loss"]
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} "
            f"(gap={train_val_gap:+.3f}) | "
            f"Seg Acc: {val_metrics['seg_acc']:.4f} | "
            f"Fall F1: {val_metrics['fall_f1']:.4f} | "
            f"Fallen F1: {val_metrics['fallen_f1']:.4f} | "
            f"LR: {current_lr:.2e}"
        )

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
            torch.save(checkpoint, os.path.join(args.log_dir, "transformer_best.pt"))
            print(f"  [BEST] New best model saved! (Fall F1={best_fall_f1:.4f})")
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"\n[INFO] Early stopping at epoch {epoch} "
                  f"(best epoch: {best_epoch}, best Fall F1: {best_fall_f1:.4f})")
            break

    # 保存最终模型
    torch.save(
        {"epoch": epoch, "model_state_dict": model.state_dict(),
         "optimizer_state_dict": optimizer.state_dict()},
        os.path.join(args.log_dir, "transformer_last.pt"))

    # 保存训练历史
    with open(os.path.join(args.log_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    plot_training_curves(history, os.path.join(args.log_dir, "training_curves.png"))
    writer.close()

    # ============================================================
    # 测试集评估
    # ============================================================
    print("\n" + "=" * 60)
    print("Evaluating on test set...")
    print("=" * 60)

    checkpoint = torch.load(
        os.path.join(args.log_dir, "transformer_best.pt"),
        map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_metrics = validate_epoch(
        model, test_loader, criterions, device,
        w_cls=args.w_cls, w_fall=args.w_fall, w_fallen=args.w_fallen)

    print(f"\n{'='*60}")
    print("TEST RESULTS")
    print(f"{'='*60}")
    print(f"  seg_acc:       {test_metrics['seg_acc']:.4f}  "
          f"(random: {test_metrics['random_baseline']:.4f})")
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
    results_to_save = {k: v for k, v in test_metrics.items() if k != "cls_report"}
    results_to_save["cls_report"] = test_metrics["cls_report"]
    results_to_save["config"] = {
        "input_dim": train_loader.dataset.feature_dim,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "dropout": args.dropout,
        "use_diff": args.use_diff,
        "pose_npz": args.pose_npz is not None,
        "window_size": args.window_size,
        "stride": args.stride,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "n_params": n_params,
        "epochs_run": len(history["train_loss"]),
        "best_epoch": best_epoch,
    }
    with open(os.path.join(args.log_dir, "test_results.json"), "w") as f:
        json.dump(results_to_save, f, indent=2, default=str)

    # ============================================================
    # 总结
    # ============================================================
    train_val_gap_final = (history["train_loss"][-1] - history["val_loss"][-1]
                           if history["train_loss"] else 0)
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"  Log directory: {args.log_dir}")
    print(f"  Input dim: {input_dim}")
    print(f"  Architecture: {args.num_layers}L/{args.hidden_dim}d, "
          f"dropout={args.dropout}")
    print(f"  Window: T={args.window_size}, stride={args.stride}")
    print(f"  Train windows: {len(train_loader.dataset):,}")
    print(f"  Best epoch: {best_epoch}")
    print(f"  Best Fall F1 (val): {best_fall_f1:.4f}")
    print(f"  Test Fall F1: {test_metrics['fall_f1']:.4f}")
    print(f"  Train-Val gap: {train_val_gap_final:+.4f}")
    print(f"  FP16: {'enabled' if use_fp16 else 'disabled'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
