"""
Phase 6: Ternary Fall Detection — Training Script.

Single-task ternary classification (fall / fallen / normal) with
swappable temporal decoders and Focal Loss.

Architecture:
    ViT-g (1536d) → Linear(1536→384) → LayerNorm → PE → [Decoder] → Linear(384→3)

Decoders: Causal Transformer (1L) / LSTM (2L) / Mamba (2L)
Loss: Focal Loss (γ ∈ {2,3,5}) or CE (bridge experiment)
Data: T=64, stride=8, ~36K windows

Usage:
    # Bridge experiment
    python experiments/train_phase6.py --bridge --exp_tag phase6/bridge

    # Causal Transformer + Focal γ=2
    python experiments/train_phase6.py --decoder_type transformer --focal_gamma 2

    # LSTM + Focal γ=3
    python experiments/train_phase6.py --decoder_type lstm --num_layers 2 --focal_gamma 3

    # Mamba + Focal γ=5
    python experiments/train_phase6.py --decoder_type mamba --num_layers 2 --focal_gamma 5
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

from models.phase6_model import Phase6TernaryModel
from models.dataset import create_longseq_dataloaders
from utils.metrics import compute_ternary_metrics
from utils.losses import build_focal_loss, build_ce_loss
from utils.visualization import (
    TERNARY_CLASS_NAMES,
    plot_training_curves_ternary,
    plot_confusion_matrix,
)


# ============================================================
# Config
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 6: Ternary Fall Detection — Causal Decoder Comparison"
    )
    # ── Data ──────────────────────────────────────────────
    parser.add_argument("--frame_npz", type=str,
                        default="data/omnifall_dinov2_giant_frame.npz",
                        help="Frame-level NPZ (ViT-g recommended)")
    parser.add_argument("--splits_dir", type=str,
                        default="DATASET-omnifall/splits/syn/random",
                        help="Split CSV directory")
    parser.add_argument("--window_size", type=int, default=64,
                        help="Sequence length T (Phase 6 default: 64)")
    parser.add_argument("--stride", type=int, default=8,
                        help="Sliding window stride (default: 8)")

    # ── Model ─────────────────────────────────────────────
    parser.add_argument("--decoder_type", type=str, default="transformer",
                        choices=["transformer", "lstm", "mamba"],
                        help="Temporal decoder type")
    parser.add_argument("--causal", action="store_true", default=True,
                        help="Use causal masking (default: True)")
    parser.add_argument("--no_causal", action="store_false", dest="causal",
                        help="Disable causal masking → bidirectional")
    parser.add_argument("--hidden_dim", type=int, default=384)
    parser.add_argument("--num_layers", type=int, default=1,
                        help="Decoder layers (transformer:1, lstm/mamba:2)")
    parser.add_argument("--num_heads", type=int, default=6,
                        help="Attention heads (transformer only)")
    parser.add_argument("--dropout", type=float, default=0.3)

    # ── Loss ──────────────────────────────────────────────
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                        help="Focal Loss gamma (0 = standard CE)")
    parser.add_argument("--bridge", action="store_true",
                        help="Bridge experiment: bidirectional + CE (no Focal)")

    # ── Training ──────────────────────────────────────────
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience (on fall_f1)")
    parser.add_argument("--lr_patience", type=int, default=6,
                        help="LR scheduler patience")

    # ── Logging ───────────────────────────────────────────
    parser.add_argument("--log_dir", type=str,
                        default="logs/phase6/default")
    parser.add_argument("--exp_tag", type=str, default=None,
                        help="Experiment tag → overrides --log_dir to logs/{exp_tag}")

    # ── System ────────────────────────────────────────────
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--no_weighted_sampler", action="store_true")
    parser.add_argument("--no_fp16", action="store_true",
                        help="Disable FP16 autocast")
    parser.add_argument("--fast", action="store_true",
                        help="Enable cuDNN benchmark")

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
# Training / Validation
# ============================================================

def train_epoch(model, loader, criterion, optimizer, device,
                use_fp16=True, scaler=None) -> float:
    model.train()
    total_loss = 0.0
    n_batches = len(loader)
    log_every = max(1, n_batches // 10)

    t_start = _time.time()
    for i, batch in enumerate(loader):
        features = batch["features"].to(device)
        ternary_gt = batch["ternary_labels"].to(device)
        B, T = ternary_gt.shape

        optimizer.zero_grad()

        if use_fp16 and scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(features)  # (B, T, 3)
                loss = criterion(logits.reshape(B * T, -1),
                                 ternary_gt.reshape(B * T))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(features)
            loss = criterion(logits.reshape(B * T, -1),
                             ternary_gt.reshape(B * T))
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
def validate_epoch(model, loader, criterion, device) -> dict:
    model.eval()

    total_loss = 0.0
    n_batches = len(loader)
    all_pred, all_gt = [], []

    for batch in loader:
        features = batch["features"].to(device)
        ternary_gt = batch["ternary_labels"].to(device)
        B, T = ternary_gt.shape

        logits = model(features)  # (B, T, 3)

        loss = criterion(logits.reshape(B * T, -1),
                         ternary_gt.reshape(B * T))
        total_loss += loss.item()

        pred = logits.argmax(dim=-1)  # (B, T)
        all_pred.append(pred.cpu().numpy().ravel())
        all_gt.append(ternary_gt.cpu().numpy().ravel())

    pred_all = np.concatenate(all_pred)
    gt_all = np.concatenate(all_gt)

    metrics = compute_ternary_metrics(pred_all, gt_all)
    metrics["loss"] = total_loss / n_batches

    return metrics


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(base_dir)

    # Bridge mode overrides
    if args.bridge:
        args.causal = False
        args.focal_gamma = 0.0  # signals "use CE"
        if args.exp_tag is None:
            args.exp_tag = "phase6/bridge"

    if args.exp_tag:
        args.log_dir = f"logs/{args.exp_tag}"

    os.makedirs(args.log_dir, exist_ok=True)

    device = get_device(args.device)
    print(f"[INFO] Using device: {device}")
    set_seed(args.seed, fast=args.fast)

    # ── Data ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Loading data...")
    print("=" * 60)

    frame_npz_path = os.path.join(base_dir, args.frame_npz)
    splits_dir = os.path.join(base_dir, args.splits_dir)

    train_loader, val_loader, test_loader = create_longseq_dataloaders(
        npz_path=frame_npz_path,
        splits_dir=splits_dir,
        window_size=args.window_size,
        stride=args.stride,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_weighted_sampler=not args.no_weighted_sampler,
        use_diff=False,
        pose_npz_path=None,
        use_ternary_sampler=True,  # Phase 6: 3-class balanced sampling
    )

    # ── Model ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Building model...")
    print("=" * 60)

    input_dim = train_loader.dataset.feature_dim
    print(f"[INFO] Input dim: {input_dim} (from dataset)")

    # Set default num_layers per decoder type if not explicitly overridden
    if args.num_layers == 1 and args.decoder_type in ("lstm", "mamba"):
        args.num_layers = 2
        print(f"[INFO] Defaulting num_layers to 2 for {args.decoder_type}")

    model = Phase6TernaryModel(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        decoder_type=args.decoder_type,
        causal=args.causal,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        mlp_ratio=4.0,
        max_len=args.window_size + 10,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    causal_str = "causal" if args.causal else "bidirectional"
    print(f"[INFO] Model params: {n_params:,}")
    print(f"[INFO] Decoder: {args.decoder_type} | {causal_str} | "
          f"{args.num_layers}L/{args.hidden_dim}d | dropout={args.dropout}")
    print(f"[INFO] Window: T={args.window_size}, stride={args.stride}")

    # ── Loss ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Building loss function...")
    print("=" * 60)

    if args.bridge or args.focal_gamma == 0.0:
        loss_info = build_ce_loss(train_loader.dataset, device)
        print("[INFO] Using CrossEntropyLoss (bridge experiment)")
    else:
        loss_info = build_focal_loss(
            train_loader.dataset, gamma=args.focal_gamma, device=device
        )
        print(f"[INFO] Using FocalLoss (gamma={args.focal_gamma})")

    criterion = loss_info["criterion"]

    # ── Optimizer ──────────────────────────────────────
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=args.lr_patience, factor=0.5
    )

    # ── FP16 ───────────────────────────────────────────
    use_fp16 = not args.no_fp16 and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_fp16 else None
    print(f"[INFO] FP16 autocast: {'enabled' if use_fp16 else 'disabled'}")

    # ── Logger ─────────────────────────────────────────
    writer = SummaryWriter(log_dir=args.log_dir)
    history = {
        "train_loss": [], "val_loss": [],
        "ternary_acc": [], "fall_f1": [], "fallen_f1": [],
        "normal_f1": [], "avg_f1": [], "lr": [],
    }

    best_fall_f1 = 0.0
    best_epoch = 0
    patience_counter = 0

    # ── Training Loop ──────────────────────────────────
    print("\n" + "=" * 60)
    print("Training started")
    print("=" * 60)

    val_metrics = {}
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device,
            use_fp16=use_fp16, scaler=scaler,
        )

        val_metrics = validate_epoch(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_metrics["fall_f1"])

        # Update history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["ternary_acc"].append(val_metrics["ternary_acc"])
        history["fall_f1"].append(val_metrics["fall_f1"])
        history["fallen_f1"].append(val_metrics["fallen_f1"])
        history["normal_f1"].append(val_metrics["normal_f1"])
        history["avg_f1"].append(val_metrics["avg_f1"])
        history["lr"].append(current_lr)

        # TensorBoard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
        writer.add_scalar("Metrics/ternary_acc", val_metrics["ternary_acc"], epoch)
        writer.add_scalar("Metrics/fall_f1", val_metrics["fall_f1"], epoch)
        writer.add_scalar("Metrics/fallen_f1", val_metrics["fallen_f1"], epoch)
        writer.add_scalar("Metrics/normal_f1", val_metrics["normal_f1"], epoch)
        writer.add_scalar("Metrics/avg_f1", val_metrics["avg_f1"], epoch)
        writer.add_scalar("LR", current_lr, epoch)

        train_val_gap = train_loss - val_metrics["loss"]
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} "
            f"(gap={train_val_gap:+.3f}) | "
            f"Acc: {val_metrics['ternary_acc']:.4f} | "
            f"Fall F1: {val_metrics['fall_f1']:.4f} | "
            f"Fallen F1: {val_metrics['fallen_f1']:.4f} | "
            f"Normal F1: {val_metrics['normal_f1']:.4f} | "
            f"Avg F1: {val_metrics['avg_f1']:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        # Early stopping
        if val_metrics["fall_f1"] > best_fall_f1:
            best_fall_f1 = val_metrics["fall_f1"]
            best_epoch = epoch
            patience_counter = 0

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_fall_f1": best_fall_f1,
                "val_metrics": {k: v for k, v in val_metrics.items()
                                if k != "cls_report"},
                "args": vars(args),
            }
            torch.save(
                checkpoint,
                os.path.join(args.log_dir, "best_model.pt"),
            )
            print(f"  [BEST] New best model saved! (Fall F1={best_fall_f1:.4f})")
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"\n[INFO] Early stopping at epoch {epoch} "
                  f"(best epoch: {best_epoch}, best Fall F1: {best_fall_f1:.4f})")
            break

    # Save last model
    torch.save(
        {"epoch": epoch, "model_state_dict": model.state_dict(),
         "optimizer_state_dict": optimizer.state_dict()},
        os.path.join(args.log_dir, "last_model.pt"),
    )

    # Save training history
    with open(os.path.join(args.log_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    plot_training_curves_ternary(
        history,
        os.path.join(args.log_dir, "training_curves.png"),
    )
    writer.close()

    # ============================================================
    # Test Set Evaluation
    # ============================================================
    print("\n" + "=" * 60)
    print("Evaluating on test set...")
    print("=" * 60)

    checkpoint = torch.load(
        os.path.join(args.log_dir, "best_model.pt"),
        map_location=device, weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_metrics = validate_epoch(model, test_loader, criterion, device)

    print(f"\n{'='*60}")
    print("TEST RESULTS")
    print(f"{'='*60}")
    print(f"  ternary_acc:     {test_metrics['ternary_acc']:.4f}  "
          f"(random: {test_metrics['random_baseline']:.4f})")
    print(f"  fall_f1:         {test_metrics['fall_f1']:.4f}")
    print(f"  fall_precision:  {test_metrics['fall_precision']:.4f}")
    print(f"  fall_recall:     {test_metrics['fall_recall']:.4f}")
    print(f"  fallen_f1:       {test_metrics['fallen_f1']:.4f}")
    print(f"  fallen_precision:{test_metrics['fallen_precision']:.4f}")
    print(f"  fallen_recall:   {test_metrics['fallen_recall']:.4f}")
    print(f"  normal_f1:       {test_metrics['normal_f1']:.4f}")
    print(f"  normal_precision:{test_metrics['normal_precision']:.4f}")
    print(f"  normal_recall:   {test_metrics['normal_recall']:.4f}")
    print(f"  avg_f1 (macro):  {test_metrics['avg_f1']:.4f}")
    print(f"\n  --- Classification Report (3-class) ---")
    print(test_metrics["cls_report"])

    # ── Confusion Matrix ───────────────────────────────
    cm = np.array(test_metrics["confusion_matrix"])
    cm_path = os.path.join(args.log_dir, "confusion_matrix.png")
    plot_confusion_matrix(
        cm, TERNARY_CLASS_NAMES, cm_path,
        title=f"Ternary Confusion Matrix — {os.path.basename(args.log_dir)}\n"
              f"Fall F1={test_metrics['fall_f1']:.3f}  "
              f"Fallen F1={test_metrics['fallen_f1']:.3f}  "
              f"Normal F1={test_metrics['normal_f1']:.3f}",
        normalize=True,
    )

    cm_path_raw = os.path.join(args.log_dir, "confusion_matrix_counts.png")
    plot_confusion_matrix(
        cm, TERNARY_CLASS_NAMES, cm_path_raw,
        title=f"Ternary Confusion Matrix (Counts) — {os.path.basename(args.log_dir)}",
        normalize=False,
        log_scale=True,
    )

    # ── Save Results ───────────────────────────────────
    results_to_save = {
        k: v for k, v in test_metrics.items()
        if k not in ("cls_report", "confusion_matrix")
    }
    results_to_save["cls_report"] = test_metrics["cls_report"]
    results_to_save["confusion_matrix"] = test_metrics["confusion_matrix"]
    results_to_save["config"] = {
        "input_dim": input_dim,
        "hidden_dim": args.hidden_dim,
        "decoder_type": args.decoder_type,
        "causal": args.causal,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "dropout": args.dropout,
        "window_size": args.window_size,
        "stride": args.stride,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "focal_gamma": args.focal_gamma,
        "bridge": args.bridge,
        "n_params": n_params,
        "epochs_run": len(history["train_loss"]),
        "best_epoch": best_epoch,
    }
    with open(os.path.join(args.log_dir, "test_results.json"), "w") as f:
        json.dump(results_to_save, f, indent=2, default=str)

    # ── Summary ────────────────────────────────────────
    train_val_gap_final = (
        history["train_loss"][-1] - history["val_loss"][-1]
        if history["train_loss"] else 0
    )
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"  Log directory:   {args.log_dir}")
    print(f"  Decoder:         {args.decoder_type} ({causal_str})")
    print(f"  Input dim:       {input_dim}")
    print(f"  Window:          T={args.window_size}, stride={args.stride}")
    print(f"  Train windows:   {len(train_loader.dataset):,}")
    print(f"  Best epoch:      {best_epoch}")
    print(f"  Best Fall F1:    {best_fall_f1:.4f} (val)")
    print(f"  Test Fall F1:    {test_metrics['fall_f1']:.4f}")
    print(f"  Test Fallen F1:  {test_metrics['fallen_f1']:.4f}")
    print(f"  Test Normal F1:  {test_metrics['normal_f1']:.4f}")
    print(f"  Test Avg F1:     {test_metrics['avg_f1']:.4f}")
    print(f"  Train-Val gap:   {train_val_gap_final:+.4f}")
    print(f"  FP16:            {'enabled' if use_fp16 else 'disabled'}")
    print(f"  Focal gamma:     {args.focal_gamma if not args.bridge else 'N/A (CE)'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
