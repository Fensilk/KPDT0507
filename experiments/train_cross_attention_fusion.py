"""
Cross-attention 融合训练脚本（Phase 8 公平实验: ViT-g 通过 cross-attention 查询姿态）。

与 train_gated_fusion.py 的区别：模型用 CrossAttentionFusionModel
（主特征 query、姿态 key/value 的 cross-attention，残差安全网），
aux 在模型内显式追加 Δ姿态（速度），不 concat。

用法：
    python experiments/train_cross_attention_fusion.py \
      --bridge --use_diff \
      --pose_npz data/omnifall_pose_frame.npz \
      --window_size 64 --stride 8 \
      --exp_tag phase8/p8_crossattn_pose
"""
import os
import sys
import json
import argparse
import time as _time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cross_attention_fusion_model import CrossAttentionFusionModel
from models.dataset import create_longseq_dataloaders
from utils.metrics import compute_ternary_metrics
from utils.losses import build_focal_loss, build_ce_loss
from utils.visualization import (
    TERNARY_CLASS_NAMES,
    plot_training_curves_ternary,
    plot_confusion_matrix,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Cross-attention fusion training")
    parser.add_argument("--frame_npz", type=str,
                        default="data/omnifall_dinov2_giant_frame.npz")
    parser.add_argument("--splits_dir", type=str,
                        default="DATASET-omnifall/splits/syn/random")
    parser.add_argument("--window_size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=384)
    parser.add_argument("--num_heads", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--bridge", action="store_true",
                        help="Bidirectional + CE (matching p7d_delta baseline)")
    parser.add_argument("--use_diff", action="store_true")
    parser.add_argument("--pose_npz", type=str, default=None,
                        help="Pose NPZ (raw 99d) for cross-attention aux")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lr_patience", type=int, default=6)
    parser.add_argument("--exp_tag", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--no_fp16", action="store_true")
    return parser.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def train_epoch(model, loader, criterion, optimizer, device, use_fp16, scaler):
    model.train()
    total_loss = 0.0
    n_batches = len(loader)
    log_every = max(1, n_batches // 10)
    t_start = _time.time()
    for i, batch in enumerate(loader):
        features = batch["features"].to(device)
        aux = batch["aux_features"].to(device)
        ternary_gt = batch["ternary_labels"].to(device)
        B, T = ternary_gt.shape
        optimizer.zero_grad()
        if use_fp16 and scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(features, aux)
                loss = criterion(logits.reshape(B * T, -1), ternary_gt.reshape(B * T))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(features, aux)
            loss = criterion(logits.reshape(B * T, -1), ternary_gt.reshape(B * T))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total_loss += loss.item()
        if (i + 1) % log_every == 0 or i == 0:
            elapsed = _time.time() - t_start
            eta = elapsed / (i + 1) * (n_batches - i - 1)
            print(f"  [{i+1:>6d}/{n_batches}] "
                  f"{(i+1)/n_batches*100:4.0f}% | loss={total_loss/(i+1):.4f} | "
                  f"elapsed={elapsed:.0f}s | eta={eta:.0f}s", flush=True)
    return total_loss / n_batches


@torch.no_grad()
def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n_batches = len(loader)
    all_pred, all_gt = [], []
    for batch in loader:
        features = batch["features"].to(device)
        aux = batch["aux_features"].to(device)
        ternary_gt = batch["ternary_labels"].to(device)
        B, T = ternary_gt.shape
        logits = model(features, aux)
        loss = criterion(logits.reshape(B * T, -1), ternary_gt.reshape(B * T))
        total_loss += loss.item()
        pred = logits.argmax(dim=-1)
        all_pred.append(pred.cpu().numpy().ravel())
        all_gt.append(ternary_gt.cpu().numpy().ravel())
    pred_all = np.concatenate(all_pred)
    gt_all = np.concatenate(all_gt)
    metrics = compute_ternary_metrics(pred_all, gt_all)
    metrics["loss"] = total_loss / n_batches
    return metrics


def main():
    args = parse_args()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(base_dir)

    if args.exp_tag:
        log_dir = f"logs/{args.exp_tag}"
    else:
        log_dir = "logs/phase8/crossattn_pose"
    os.makedirs(log_dir, exist_ok=True)

    device = get_device(args.device)
    print(f"[INFO] Using device: {device}")
    set_seed(args.seed)

    print("\nLoading data...")
    train_loader, val_loader, test_loader = create_longseq_dataloaders(
        npz_path=os.path.join(base_dir, args.frame_npz),
        splits_dir=os.path.join(base_dir, args.splits_dir),
        window_size=args.window_size,
        stride=args.stride,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_weighted_sampler=True,
        use_diff=args.use_diff,
        pose_npz_path=args.pose_npz,
        return_aux_separately=True,
        use_ternary_sampler=True,
    )

    main_dim = train_loader.dataset.feature_dim
    aux_dim = train_loader.dataset.pose_all.shape[1]
    print(f"\n[INFO] main_dim={main_dim} (ViT-g + diff), aux_dim={aux_dim} (pose)")

    model = CrossAttentionFusionModel(
        main_dim=main_dim,
        aux_dim=aux_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
        causal=False,  # bridge
        max_len=args.window_size + 10,
        num_classes=3,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] CrossAttentionFusionModel params: {n_params:,}")

    if args.bridge:
        loss_info = build_ce_loss(train_loader.dataset, device)
    else:
        loss_info = build_focal_loss(train_loader.dataset, gamma=2.0, device=device)
    criterion = loss_info["criterion"]

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=args.lr_patience, factor=0.5)

    use_fp16 = not args.no_fp16 and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_fp16 else None
    print(f"[INFO] FP16: {'enabled' if use_fp16 else 'disabled'}")

    writer = SummaryWriter(log_dir=log_dir)
    history = {"train_loss": [], "val_loss": [], "ternary_acc": [], "fall_f1": [],
               "fallen_f1": [], "normal_f1": [], "avg_f1": [], "lr": [], "attn_norm": []}
    best_fall_f1 = 0.0
    best_epoch = 0
    patience_counter = 0

    print("\nTraining started")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, use_fp16, scaler)
        val_metrics = validate_epoch(model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_metrics["fall_f1"])

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["ternary_acc"].append(val_metrics["ternary_acc"])
        history["fall_f1"].append(val_metrics["fall_f1"])
        history["fallen_f1"].append(val_metrics["fallen_f1"])
        history["normal_f1"].append(val_metrics["normal_f1"])
        history["avg_f1"].append(val_metrics["avg_f1"])
        history["lr"].append(current_lr)
        history["attn_norm"].append(getattr(model, "_attn_norm", 0.0))

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
        writer.add_scalar("Metrics/avg_f1", val_metrics["avg_f1"], epoch)
        writer.add_scalar("Metrics/fall_f1", val_metrics["fall_f1"], epoch)
        writer.add_scalar("Metrics/fallen_f1", val_metrics["fallen_f1"], epoch)
        writer.add_scalar("CrossAttn/attn_norm", history["attn_norm"][-1], epoch)

        print(f"Epoch {epoch:3d} | Train Loss {train_loss:.4f} | Val Loss {val_metrics['loss']:.4f} | "
              f"Fall F1 {val_metrics['fall_f1']:.4f} | Fallen F1 {val_metrics['fallen_f1']:.4f} | "
              f"Avg F1 {val_metrics['avg_f1']:.4f} | attn_norm={history['attn_norm'][-1]:.3f} | LR {current_lr:.2e}")

        if val_metrics["fall_f1"] > best_fall_f1:
            best_fall_f1 = val_metrics["fall_f1"]
            best_epoch = epoch
            patience_counter = 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "best_fall_f1": best_fall_f1, "args": vars(args)},
                       os.path.join(log_dir, "best_model.pt"))
        else:
            patience_counter += 1
        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} (best {best_epoch}, Fall F1 {best_fall_f1:.4f})")
            break

    torch.save({"epoch": epoch, "model_state_dict": model.state_dict()},
               os.path.join(log_dir, "last_model.pt"))
    with open(os.path.join(log_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    plot_training_curves_ternary(history, os.path.join(log_dir, "training_curves.png"))
    writer.close()

    # ── Test ───────────────────────────────────────────
    print("\nEvaluating on test set...")
    ck = torch.load(os.path.join(log_dir, "best_model.pt"), map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    test_metrics = validate_epoch(model, test_loader, criterion, device)

    print(f"\nTEST RESULTS")
    print(f"  ternary_acc: {test_metrics['ternary_acc']:.4f}")
    print(f"  fall_f1:     {test_metrics['fall_f1']:.4f}")
    print(f"  fallen_f1:   {test_metrics['fallen_f1']:.4f}")
    print(f"  normal_f1:   {test_metrics['normal_f1']:.4f}")
    print(f"  avg_f1:      {test_metrics['avg_f1']:.4f}")
    print(test_metrics["cls_report"])

    cm = np.array(test_metrics["confusion_matrix"])
    plot_confusion_matrix(cm, TERNARY_CLASS_NAMES,
                          os.path.join(log_dir, "confusion_matrix.png"),
                          title=f"Cross-Attn Fusion — Avg F1={test_metrics['avg_f1']:.3f}",
                          normalize=True)

    results_to_save = {k: v for k, v in test_metrics.items()
                       if k not in ("cls_report", "confusion_matrix")}
    results_to_save["cls_report"] = test_metrics["cls_report"]
    results_to_save["confusion_matrix"] = test_metrics["confusion_matrix"]
    results_to_save["config"] = {
        "main_dim": main_dim, "aux_dim": aux_dim, "hidden_dim": args.hidden_dim,
        "num_heads": args.num_heads, "dropout": args.dropout,
        "window_size": args.window_size, "stride": args.stride,
        "batch_size": args.batch_size, "lr": args.lr, "bridge": args.bridge,
        "use_diff": args.use_diff, "pose_npz": args.pose_npz,
        "n_params": n_params, "epochs_run": len(history["train_loss"]),
        "best_epoch": best_epoch,
        "final_attn_norm": history["attn_norm"][-1] if history["attn_norm"] else 0.0,
    }
    with open(os.path.join(log_dir, "test_results.json"), "w") as f:
        json.dump(results_to_save, f, indent=2, default=str)

    print(f"\nDone. Logs in {log_dir}")


if __name__ == "__main__":
    main()
