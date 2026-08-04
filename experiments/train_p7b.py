"""
P7b: Event-gated ternary detection — Training script.

Architecture:
    DINOv2 ViT-g (1536d) → PE → Bidirectional Transformer (T=64)
        → ├─ Head_ternary (→3)  +  └─ Head_event (→1)

Training:
    L = CE(ternary_mod, gt) + λ × BCE(event_score, event_gt)
    where event_gt[f] = 1 if GT fall exists in [f-64, f]

Usage:
    python experiments/train_p7b.py --exp_tag phase7/p7b_event_gate
    python experiments/train_p7b.py --event_lambda 0.3 --exp_tag phase7/p7b_el03
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

from models.phase7_model import P7EventGateModel, build_event_labels
from models.dataset import create_longseq_dataloaders
from utils.metrics import compute_ternary_metrics
from utils.losses import build_ce_loss, build_focal_loss
from utils.visualization import (
    TERNARY_CLASS_NAMES,
    plot_training_curves_ternary,
    plot_confusion_matrix,
)

CLASS_NAMES_16 = [
    "walk", "fall", "fallen", "sit_down", "sitting", "lie_down", "lying",
    "stand_up", "standing", "other", "kneel_down", "kneeling",
    "squat_down", "squatting", "crawl", "jump",
]


# ═══════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="P7b: Event-gated ternary detection")
    p.add_argument("--frame_npz", default="data/omnifall_dinov2_giant_frame.npz")
    p.add_argument("--splits_dir", default="DATASET-omnifall/splits/syn/random")
    p.add_argument("--window_size", type=int, default=64)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--hidden_dim", type=int, default=384)
    p.add_argument("--num_layers", type=int, default=1)
    p.add_argument("--num_heads", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--lr_patience", type=int, default=6)
    p.add_argument("--event_lambda", type=float, default=0.5,
                   help="Weight of BCE event loss relative to CE ternary loss")
    p.add_argument("--no_fp16", action="store_true")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--log_dir", default="logs/phase7/p7b_event_gate")
    p.add_argument("--exp_tag", default=None,
                   help="Override log_dir to logs/{exp_tag}")
    return p.parse_args()


def set_seed(seed: int, fast: bool = False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if fast:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(dev: str):
    return torch.device("cuda" if torch.cuda.is_available() and dev == "auto" else dev)


# ═══════════════════════════════════════════════════════════════
# Training / Validation
# ═══════════════════════════════════════════════════════════════

def train_epoch(model, loader, criterion_ce, event_lambda, optimizer,
                device, use_fp16=True, scaler=None):
    """Joint training with CE(ternary) + lambda * BCE(event)."""
    model.train()
    total_loss = 0.0
    n_batches = len(loader)
    log_every = max(1, n_batches // 10)
    t_start = _time.time()

    for i, batch in enumerate(loader):
        features = batch["features"].to(device)
        ternary_gt = batch["ternary_labels"].to(device)
        fall_gt = batch["fall_labels"].to(device)     # (B, T) binary
        B, T = ternary_gt.shape

        # Build event labels from GT fall
        event_gt = build_event_labels(fall_gt, window_size=T).to(device)

        optimizer.zero_grad()

        if use_fp16 and scaler is not None:
            with torch.cuda.amp.autocast():
                logits, event_score, event_logits_raw = model(features)
                # Modulate fallen logit (out-of-place for FP16 safety)
                logits_mod = torch.stack([
                    logits[:, :, 0],
                    logits[:, :, 1] * event_score,
                    logits[:, :, 2],
                ], dim=-1)
                # Joint loss
                loss_ce = criterion_ce(
                    logits_mod.reshape(B * T, -1),
                    ternary_gt.reshape(B * T),
                )
                loss_event = nn.functional.binary_cross_entropy_with_logits(
                    event_logits_raw.reshape(B * T),
                    event_gt.reshape(B * T),
                )
                loss = loss_ce + event_lambda * loss_event
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, event_score, event_logits_raw = model(features)
            logits_mod = torch.stack([
                logits[:, :, 0],
                logits[:, :, 1] * event_score,
                logits[:, :, 2],
            ], dim=-1)
            loss_ce = criterion_ce(
                logits_mod.reshape(B * T, -1),
                ternary_gt.reshape(B * T),
            )
            loss_event = nn.functional.binary_cross_entropy_with_logits(
                event_logits_raw.reshape(B * T),
                event_gt.reshape(B * T),
            )
            loss = loss_ce + event_lambda * loss_event
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
def validate_epoch(model, loader, criterion_ce, event_lambda, device,
                   compute_16class_breakdown=False):
    """Validate with event-modulated forward + 16-class breakdown."""
    model.eval()
    total_loss = 0.0
    n_batches = len(loader)
    all_pred, all_gt, all_event_score, all_event_gt = [], [], [], []
    all_l16, all_pred_u16 = [], []  # for 16-class breakdown

    for batch in loader:
        features = batch["features"].to(device)
        ternary_gt = batch["ternary_labels"].to(device)
        fall_gt = batch["fall_labels"].to(device)
        labels_16 = batch["labels_16"]
        B, T = ternary_gt.shape

        logits, event_score, event_logits_raw = model(features)
        event_gt = build_event_labels(fall_gt, window_size=T).to(device)

        # Modulated forward for evaluation (out-of-place)
        logits_mod = torch.stack([
            logits[:, :, 0],
            logits[:, :, 1] * event_score,
            logits[:, :, 2],
        ], dim=-1)

        loss_ce = criterion_ce(logits_mod.reshape(B * T, -1), ternary_gt.reshape(B * T))
        loss_event = nn.functional.binary_cross_entropy_with_logits(
            event_logits_raw.reshape(B * T), event_gt.reshape(B * T))
        total_loss += (loss_ce + event_lambda * loss_event).item()

        pred = logits_mod.argmax(dim=-1)
        all_pred.append(pred.cpu().numpy().ravel())
        all_gt.append(ternary_gt.cpu().numpy().ravel())
        all_event_score.append(event_score.cpu().numpy().ravel())
        all_event_gt.append(event_gt.cpu().numpy().ravel())
        all_l16.append(labels_16.numpy().ravel())
        all_pred_u16.append(pred.cpu().numpy().ravel())

    pred_all = np.concatenate(all_pred)
    gt_all = np.concatenate(all_gt)
    event_score_all = np.concatenate(all_event_score)
    event_gt_all = np.concatenate(all_event_gt)

    metrics = compute_ternary_metrics(pred_all, gt_all)
    metrics["loss"] = total_loss / n_batches

    # Event gate metrics
    event_pred_bin = (event_score_all > 0.5).astype(int)
    from sklearn.metrics import precision_score, recall_score, f1_score
    metrics["event_precision"] = float(precision_score(event_gt_all, event_pred_bin, zero_division=0))
    metrics["event_recall"] = float(recall_score(event_gt_all, event_pred_bin, zero_division=0))
    metrics["event_f1"] = float(f1_score(event_gt_all, event_pred_bin, zero_division=0))

    # 16-class breakdown for lying/lie_down analysis
    if compute_16class_breakdown:
        l16_all = np.concatenate(all_l16)
        pred_all_u16 = np.concatenate(all_pred_u16)
        breakdown = {}
        for c in range(16):
            mask = l16_all == c
            n = int(mask.sum())
            if n == 0: continue
            breakdown[CLASS_NAMES_16[c]] = {
                "total": n,
                "pred_fall": int((pred_all_u16[mask] == 0).sum()),
                "pred_fallen": int((pred_all_u16[mask] == 1).sum()),
                "pred_normal": int((pred_all_u16[mask] == 2).sum()),
            }
        metrics["_breakdown"] = breakdown

    return metrics


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(base_dir)

    if args.exp_tag:
        args.log_dir = f"logs/{args.exp_tag}"
    os.makedirs(args.log_dir, exist_ok=True)

    device = get_device(args.device)
    print(f"[P7b] Device: {device}")
    print(f"[P7b] Event lambda: {args.event_lambda}")
    print(f"[P7b] Log dir: {args.log_dir}")
    set_seed(args.seed, fast=True)

    # ── Data ──
    print("\n" + "=" * 60)
    print("[P7b] Loading data...")
    print("=" * 60)

    train_loader, val_loader, test_loader = create_longseq_dataloaders(
        npz_path=os.path.join(base_dir, args.frame_npz),
        splits_dir=os.path.join(base_dir, args.splits_dir),
        window_size=args.window_size,
        stride=args.stride,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_weighted_sampler=True,
        use_diff=False,
        pose_npz_path=None,
        use_ternary_sampler=True,
    )

    input_dim = train_loader.dataset.feature_dim
    print(f"[P7b] Input dim: {input_dim}")
    print(f"[P7b] Train windows: {len(train_loader.dataset):,}")

    # ── Model ──
    print("\n" + "=" * 60)
    print("[P7b] Building model...")
    print("=" * 60)

    model = P7EventGateModel(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        decoder_type="transformer",
        causal=False,  # bidirectional (bridge mode)
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        max_len=args.window_size + 10,
        event_lambda=args.event_lambda,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[P7b] Model params: {n_params:,} (+385 event head vs Bridge)")
    print(f"[P7b] Decoder: transformer | bidirectional | "
          f"{args.num_layers}L/{args.hidden_dim}d | dropout={args.dropout}")
    print(f"[P7b] Window: T={args.window_size}, stride={args.stride}")

    # ── Loss ──
    loss_info = build_ce_loss(train_loader.dataset, device)
    criterion_ce = loss_info["criterion"]
    print(f"[P7b] Ternary loss: CE + pos_weight")
    print(f"[P7b] Event loss: BCE × λ={args.event_lambda}")

    # ── Optimizer ──
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=args.lr_patience, factor=0.5)

    use_fp16 = not args.no_fp16 and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_fp16 else None
    print(f"[P7b] FP16: {'enabled' if use_fp16 else 'disabled'}")

    # ── Logger ──
    writer = SummaryWriter(log_dir=args.log_dir)
    history = {"train_loss": [], "val_loss": [], "val_fall_f1": [],
               "val_fallen_f1": [], "val_avg_f1": [], "val_event_f1": [], "lr": []}

    best_fall_f1 = 0.0
    best_epoch = 0
    patience_counter = 0

    # ── Training Loop ──
    print("\n" + "=" * 60)
    print("[P7b] Training started")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        t0 = _time.time()
        train_loss = train_epoch(
            model, train_loader, criterion_ce, args.event_lambda,
            optimizer, device, use_fp16=use_fp16, scaler=scaler,
        )
        val_metrics = validate_epoch(
            model, val_loader, criterion_ce, args.event_lambda, device,
        )
        epoch_time = _time.time() - t0

        # Log
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_fall_f1"].append(val_metrics["fall_f1"])
        history["val_fallen_f1"].append(val_metrics["fallen_f1"])
        history["val_avg_f1"].append(val_metrics["avg_f1"])
        history["val_event_f1"].append(val_metrics["event_f1"])
        history["lr"].append(optimizer.param_groups[0]["lr"])

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
        writer.add_scalar("F1/fall", val_metrics["fall_f1"], epoch)
        writer.add_scalar("F1/fallen", val_metrics["fallen_f1"], epoch)
        writer.add_scalar("F1/avg", val_metrics["avg_f1"], epoch)
        writer.add_scalar("F1/event_gate", val_metrics["event_f1"], epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        current_fall_f1 = val_metrics["fall_f1"]

        if current_fall_f1 > best_fall_f1:
            best_fall_f1 = current_fall_f1
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_fall_f1": best_fall_f1,
                "args": vars(args),
            }, os.path.join(args.log_dir, "best_model.pt"))
        else:
            patience_counter += 1

        print(
            f"[P7b] Epoch {epoch:>3d} | "
            f"T={train_loss:.3f} V={val_metrics['loss']:.3f} | "
            f"fall_f1={val_metrics['fall_f1']:.4f} "
            f"fallen_f1={val_metrics['fallen_f1']:.4f} "
            f"avg_f1={val_metrics['avg_f1']:.4f} | "
            f"event_f1={val_metrics['event_f1']:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.1e} | "
            f"time={epoch_time:.0f}s | "
            f"best@{best_epoch} ({best_fall_f1:.4f})"
        )

        scheduler.step(current_fall_f1)

        if patience_counter >= args.patience:
            print(f"\n[P7b] Early stopping at epoch {epoch}")
            break

    # ── Save last ──
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "best_fall_f1": best_fall_f1,
        "args": vars(args),
    }, os.path.join(args.log_dir, "last_model.pt"))

    with open(os.path.join(args.log_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # ── Load best & test ──
    print("\n" + "=" * 60)
    print("[P7b] Loading best model for test evaluation...")
    print("=" * 60)

    best_ckpt = torch.load(
        os.path.join(args.log_dir, "best_model.pt"),
        map_location=device, weights_only=False,
    )
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    test_metrics = validate_epoch(
        model, test_loader, criterion_ce, args.event_lambda, device,
        compute_16class_breakdown=True,
    )

    breakdown = test_metrics.pop("_breakdown", {})

    # Print key results
    print("\n" + "=" * 60)
    print("[P7b] TEST RESULTS")
    print("=" * 60)
    print(f"  ternary_acc:     {test_metrics['ternary_acc']:.4f}")
    print(f"  fall_f1:         {test_metrics['fall_f1']:.4f}")
    print(f"  fall_precision:  {test_metrics['fall_precision']:.4f}")
    print(f"  fall_recall:     {test_metrics['fall_recall']:.4f}")
    print(f"  fallen_f1:       {test_metrics['fallen_f1']:.4f}")
    print(f"  fallen_precision:{test_metrics['fallen_precision']:.4f}")
    print(f"  fallen_recall:   {test_metrics['fallen_recall']:.4f}")
    print(f"  normal_f1:       {test_metrics['normal_f1']:.4f}")
    print(f"  avg_f1:          {test_metrics['avg_f1']:.4f}")
    print(f"  --- Event Gate ---")
    print(f"  event_f1:        {test_metrics['event_f1']:.4f}")
    print(f"  event_precision: {test_metrics['event_precision']:.4f}")
    print(f"  event_recall:    {test_metrics['event_recall']:.4f}")

    # Compare with p6_bridge
    print(f"\n{'='*60}")
    print("[P7b] COMPARISON WITH P6_BRIDGE")
    print("=" * 60)
    bridge_path = os.path.join(base_dir, "logs/phase6/p6_bridge/test_results.json")
    if os.path.exists(bridge_path):
        with open(bridge_path) as f:
            bridge = json.load(f)
        for key in ["fall_f1", "fall_precision", "fall_recall",
                     "fallen_f1", "fallen_precision", "fallen_recall",
                     "avg_f1", "ternary_acc"]:
            bv = bridge.get(key, 0)
            pv = test_metrics.get(key, 0)
            delta = pv - bv
            print(f"  {key:<22s}: Bridge={bv:.4f}  P7b={pv:.4f}  Δ={delta:+.4f}")

    # Lying breakdown
    print(f"\n{'='*60}")
    print("[P7b] LYING / LIE_DOWN BREAKDOWN")
    print("=" * 60)
    for cls in ["lying", "lie_down"]:
        if cls in breakdown:
            bd = breakdown[cls]
            print(f"  {cls:>12s}: total={bd['total']:,}  "
                  f"→fallen={bd['pred_fallen']:,} ({bd['pred_fallen']/bd['total']*100:.1f}%)  "
                  f"→fall={bd['pred_fall']:,} ({bd['pred_fall']/bd['total']*100:.1f}%)")

    # Confusion
    print(f"\n{'='*60}")
    print("[P7b] CONFUSION (row-normalized)")
    print("=" * 60)
    cm = np.array(test_metrics["confusion_matrix"])
    for i, name in enumerate(["fall", "fallen", "normal"]):
        rs = cm[i].sum() or 1
        print(f"  {name:>8s}: [{cm[i,0]/rs:.1%}f {cm[i,1]/rs:.1%}fd {cm[i,2]/rs:.1%}n]")

    # Save
    results = {
        "config": vars(args),
        **{k: v for k, v in test_metrics.items() if k != "confusion_matrix"},
        "confusion_matrix": test_metrics["confusion_matrix"],
        "16class_breakdown": breakdown,
        "n_params": n_params,
        "best_epoch": best_epoch,
        "best_val_fall_f1": best_fall_f1,
    }
    with open(os.path.join(args.log_dir, "test_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Plots
    try:
        plot_training_curves_ternary(
            history, os.path.join(args.log_dir, "training_curves.png"))
        plot_confusion_matrix(
            np.array(test_metrics["confusion_matrix"]),
            TERNARY_CLASS_NAMES,
            os.path.join(args.log_dir, "confusion_matrix.png"),
            title=f"P7b Event Gate — {os.path.basename(args.log_dir)}",
            normalize=True,
        )
    except Exception as e:
        print(f"[WARN] Plotting failed: {e}")

    print(f"\n[P7b] Done. Results: {args.log_dir}/")
    print(f"[P7b] Best epoch: {best_epoch} (val_fall_f1={best_fall_f1:.4f})")

    return test_metrics


if __name__ == "__main__":
    main()
