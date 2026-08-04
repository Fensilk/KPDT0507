"""
P7c: Binary Event Detection — Training script.

Same as p6_bridge (bidirectional T=64, CE loss) except:
  - Output: Linear(384→2) instead of Linear(384→3)
  - Labels: event (fall ∪ fallen)=0, normal=1

This is a capacity ablation: does removing the fall/fallen distinction
improve event detection precision?

Usage:
    python experiments/train_p7c_binary.py --exp_tag phase7/p7c_binary
"""

import os, sys, json, argparse, time as _time
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.phase6_model import Phase6TernaryModel
from models.dataset import create_longseq_dataloaders
from utils.visualization import plot_training_curves_ternary, plot_confusion_matrix

CLASS_NAMES_16 = [
    "walk", "fall", "fallen", "sit_down", "sitting", "lie_down", "lying",
    "stand_up", "standing", "other", "kneel_down", "kneeling",
    "squat_down", "squatting", "crawl", "jump",
]
BINARY_NAMES = ["event", "normal"]


def parse_args():
    p = argparse.ArgumentParser(description="P7c: Binary Event Detection")
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
    p.add_argument("--no_fp16", action="store_true")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--log_dir", default="logs/phase7/p7c_binary")
    p.add_argument("--exp_tag", default=None)
    return p.parse_args()


def set_seed(seed): torch.manual_seed(seed); np.random.seed(seed)


def get_device(d): return torch.device("cuda" if torch.cuda.is_available() and d=="auto" else d)


def train_epoch(model, loader, criterion, optimizer, device, use_fp16, scaler):
    model.train(); total_loss = 0; nb = len(loader); le = max(1, nb//10)
    t0 = _time.time()
    for i, batch in enumerate(loader):
        features = batch["features"].to(device)
        fall_l = batch["fall_labels"]; fallen_l = batch["fallen_labels"]
        binary_gt = (1 - ((fall_l == 1) | (fallen_l == 1)).long()).to(device)  # 0=event, 1=normal
        B, T = binary_gt.shape
        optimizer.zero_grad()
        if use_fp16 and scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(features)   # (B, T, 2)
                loss = criterion(logits.reshape(B*T, -1), binary_gt.reshape(B*T))
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
        else:
            logits = model(features)
            loss = criterion(logits.reshape(B*T, -1), binary_gt.reshape(B*T))
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total_loss += loss.item()
        if (i+1)%le==0 or i==0:
            e = _time.time()-t0; eta = e/(i+1)*(nb-i-1)
            print(f"  [{i+1:>6d}/{nb}] {(i+1)/nb*100:4.0f}% | loss={total_loss/(i+1):.4f} | {e:.0f}s | eta={eta:.0f}s", flush=True)
    return total_loss/nb


@torch.no_grad()
def validate_epoch(model, loader, criterion, device, compute_breakdown=False):
    model.eval(); total_loss = 0; nb = len(loader)
    all_pred, all_gt, all_l16 = [], [], []
    for batch in loader:
        features = batch["features"].to(device)
        fall_l = batch["fall_labels"]; fallen_l = batch["fallen_labels"]
        binary_gt = (1 - ((fall_l == 1) | (fallen_l == 1)).long()).to(device)
        B, T = binary_gt.shape
        logits = model(features)
        loss = criterion(logits.reshape(B*T, -1), binary_gt.reshape(B*T))
        total_loss += loss.item()
        pred = logits.argmax(dim=-1)
        all_pred.append(pred.cpu().numpy().ravel())
        all_gt.append(binary_gt.cpu().numpy().ravel())
        if compute_breakdown:
            all_l16.append(batch["labels_16"].numpy().ravel())

    pa = np.concatenate(all_pred); ga = np.concatenate(all_gt)
    m = {
        "loss": total_loss/nb,
        "accuracy": float((pa==ga).mean()),
        "event_precision": float(precision_score(ga, pa, pos_label=0, zero_division=0)),
        "event_recall": float(recall_score(ga, pa, pos_label=0, zero_division=0)),
        "event_f1": float(f1_score(ga, pa, pos_label=0, zero_division=0)),
        "normal_precision": float(precision_score(ga, pa, pos_label=1, zero_division=0)),
        "normal_recall": float(recall_score(ga, pa, pos_label=1, zero_division=0)),
        "avg_f1": (float(f1_score(ga, pa, pos_label=0, zero_division=0)) +
                   float(f1_score(ga, pa, pos_label=1, zero_division=0))) / 2,
        "confusion_matrix": confusion_matrix(ga, pa, labels=[0,1]).tolist(),
    }
    if compute_breakdown:
        l16a = np.concatenate(all_l16)
        bd = {}
        for c in range(16):
            mask = l16a == c; n = int(mask.sum())
            if n == 0: continue
            bd[CLASS_NAMES_16[c]] = {
                "total": n,
                "pred_event": int((pa[mask]==0).sum()),
                "pred_normal": int((pa[mask]==1).sum()),
            }
        m["_breakdown"] = bd
    return m


def main():
    args = parse_args(); BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(BASE)
    if args.exp_tag: args.log_dir = f"logs/{args.exp_tag}"
    os.makedirs(args.log_dir, exist_ok=True)
    device = get_device(args.device); set_seed(args.seed)
    print(f"[P7c] Binary Event Detection | Device: {device} | Log: {args.log_dir}")

    # Data
    train_loader, val_loader, test_loader = create_longseq_dataloaders(
        npz_path=os.path.join(BASE, args.frame_npz),
        splits_dir=os.path.join(BASE, args.splits_dir),
        window_size=args.window_size, stride=args.stride,
        batch_size=args.batch_size, num_workers=args.num_workers,
        use_weighted_sampler=True, use_diff=False,
        pose_npz_path=None, use_ternary_sampler=True,
    )
    print(f"[P7c] Train windows: {len(train_loader.dataset):,}")

    # Model
    model = Phase6TernaryModel(
        input_dim=train_loader.dataset.feature_dim, hidden_dim=args.hidden_dim,
        decoder_type="transformer", causal=False, num_layers=args.num_layers,
        num_heads=args.num_heads, dropout=args.dropout,
        max_len=args.window_size+10, num_classes=2,
    ).to(device)
    n = sum(p.numel() for p in model.parameters())
    print(f"[P7c] Params: {n:,} | T={args.window_size}, stride={args.stride}")

    # Loss (class weights: event is rarer, give higher weight)
    event_frames = int(np.sum(train_loader.dataset.fall_labels_all) + np.sum(train_loader.dataset.fallen_labels_all))
    total_frames = len(train_loader.dataset.fall_labels_all)
    normal_frames = total_frames - event_frames
    event_weight = normal_frames / max(event_frames, 1)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(
        [event_weight, 1.0], device=device))
    print(f"[P7c] Event frames: {event_frames:,} (weight={event_weight:.1f}), "
          f"Normal: {normal_frames:,} (weight=1.0)")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=args.lr_patience, factor=0.5)
    use_fp16 = not args.no_fp16 and device.type=="cuda"
    scaler = torch.cuda.amp.GradScaler() if use_fp16 else None

    # Training
    best_ef1 = 0; best_ep = 0; patience = 0
    writer = SummaryWriter(log_dir=args.log_dir)
    print(f"\n[P7c] Training...")
    for epoch in range(1, args.epochs+1):
        t0 = _time.time()
        tl = train_epoch(model, train_loader, criterion, optimizer, device, use_fp16, scaler)
        vm = validate_epoch(model, val_loader, criterion, device)
        et = _time.time()-t0

        writer.add_scalar("Loss/train", tl, epoch); writer.add_scalar("Loss/val", vm["loss"], epoch)
        writer.add_scalar("F1/event", vm["event_f1"], epoch)

        ef1 = vm["event_f1"]
        if ef1 > best_ef1:
            best_ef1 = ef1; best_ep = epoch; patience = 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "best_event_f1": best_ef1, "args": vars(args)},
                       os.path.join(args.log_dir, "best_model.pt"))
        else: patience += 1

        print(f"[P7c] Ep {epoch:>3d} | T={tl:.3f} V={vm['loss']:.3f} | "
              f"event_f1={vm['event_f1']:.4f} prec={vm['event_precision']:.4f} "
              f"rec={vm['event_recall']:.4f} | best@{best_ep} ({best_ef1:.4f}) | {et:.0f}s")

        scheduler.step(ef1)
        if patience >= args.patience:
            print(f"\n[P7c] Early stop ep {epoch}"); break

    torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "args": vars(args)},
               os.path.join(args.log_dir, "last_model.pt"))

    # Test
    print(f"\n[P7c] Testing with best model (ep {best_ep})...")
    ckpt = torch.load(os.path.join(args.log_dir, "best_model.pt"), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    tm = validate_epoch(model, test_loader, criterion, device, compute_breakdown=True)
    bd = tm.pop("_breakdown", {})

    print(f"\n{'='*60}")
    print(f"[P7c] TEST RESULTS")
    print(f"{'='*60}")
    for k in ["accuracy", "event_f1", "event_precision", "event_recall",
              "normal_precision", "normal_recall", "avg_f1"]:
        print(f"  {k:<20s}: {tm[k]:.4f}")

    # Compare with Bridge (binary perspective)
    bp = os.path.join(BASE, "logs/phase6/p6_bridge/test_results.json")
    if os.path.exists(bp):
        with open(bp) as f: br = json.load(f)
        cm_b = np.array(br["confusion_matrix"])
        # bridge ternary → binary: event = fall+fallen cols, normal = normal col
        # GT event = fall+fallen row, normal = normal row
        b_event_gt = cm_b[0] + cm_b[1]  # fall + fallen GT rows
        b_normal_gt = cm_b[2]            # normal GT row
        # Binary pred: event = col0+col1 (fall+fallen pred), normal = col2
        b_event_pred_event = b_event_gt[0] + b_event_gt[1]   # GT event → pred event
        b_event_pred_normal = b_event_gt[2]                    # GT event → pred normal
        b_normal_pred_event = b_normal_gt[0] + b_normal_gt[1] # GT normal → pred event
        b_normal_pred_normal = b_normal_gt[2]                  # GT normal → pred normal
        b_ep = b_event_pred_event / max(b_event_pred_event+b_normal_pred_event, 1)
        b_er = b_event_pred_event / max(b_event_pred_event+b_event_pred_normal, 1)
        b_ef1 = 2*b_ep*b_er/(b_ep+b_er) if (b_ep+b_er)>0 else 0
        print(f"\n{'='*60}")
        print(f"[P7c] COMPARISON: Bridge (binary view) vs Binary model")
        print(f"{'='*60}")
        for k, bv, pv in [("event_precision", b_ep, tm["event_precision"]),
                           ("event_recall", b_er, tm["event_recall"]),
                           ("event_f1", b_ef1, tm["event_f1"])]:
            print(f"  {k:<20s}: Bridge={bv:.4f}  Binary={pv:.4f}  Δ={pv-bv:+.4f}")

    # 16-class breakdown
    print(f"\n{'='*60}")
    print(f"[P7c] 16-CLASS → EVENT BREAKDOWN")
    print(f"{'='*60}")
    for cls in ["lying", "lie_down", "sit_down", "sitting", "stand_up", "other", "walk", "standing"]:
        if cls in bd:
            d = bd[cls]; r = d["pred_event"]/d["total"]*100
            print(f"  {cls:>12s}: {d['pred_event']:>6,d}/{d['total']:>6,d} ({r:.1f}%) → event")

    # Confusion
    print(f"\n{'='*60}")
    print(f"[P7c] CONFUSION (row-normalized)")
    print(f"{'='*60}")
    cm = np.array(tm["confusion_matrix"])
    for i, name in enumerate(BINARY_NAMES):
        rs = cm[i].sum() or 1
        print(f"  {name:>8s}: [{cm[i,0]/rs:.1%} event, {cm[i,1]/rs:.1%} normal]")

    # Save
    results = {"config": vars(args), **{k:v for k,v in tm.items() if k!="_breakdown"},
               "16class_breakdown": bd, "n_params": n, "best_epoch": best_ep,
               "best_val_event_f1": best_ef1}
    with open(os.path.join(args.log_dir, "test_results.json"), "w") as f: json.dump(results, f, indent=2)
    print(f"\n[P7c] Saved: {args.log_dir}/test_results.json")


if __name__ == "__main__":
    main()
