"""
改进二实验脚本: 差分特征 + 姿态特征.
用法:
    python _run_improve2.py --use_diff                           # 2a: RGB+Diff
    python _run_improve2.py --pose_npz data/omnifall_pose.npz     # 2b: RGB+Pose
    python _run_improve2.py --use_diff --pose_npz data/omnifall_pose.npz  # 2ab
"""
import sys, os, json, time, argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch, torch.nn as nn, torch.optim as optim
import numpy as np
from models.tcn_model import TCNModel
from models.dataset import create_dataloaders
from utils.metrics import compute_all_metrics
from utils.visualization import CLASS_NAMES


def parse_args():
    p = argparse.ArgumentParser(description="Improvement 2 Experiment")
    p.add_argument("--use_diff", action="store_true",
                   help="Use adjacent clip differential features")
    p.add_argument("--pose_npz", type=str, default=None,
                   help="Path to pose NPZ")
    p.add_argument("--tag", type=str, default=None,
                   help="Experiment tag (auto-generated if not set)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    # --- Tag & log dir ---
    if args.tag:
        tag = args.tag
    else:
        parts = ["imp2"]
        if args.use_diff:
            parts.append("diff")
        if args.pose_npz:
            parts.append("pose")
        tag = "_".join(parts)

    log_dir = f"logs/{tag}"
    os.makedirs(log_dir, exist_ok=True)

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Improvement 2 Experiment: {tag}")
    print(f"  use_diff={args.use_diff}, pose_npz={args.pose_npz}")
    print(f"  Device: {device}, Log: {log_dir}")

    # --- Data ---
    print("\n[1/4] Loading data...")
    train_loader, val_loader, test_loader = create_dataloaders(
        npz_path="data/omnifall_preprocessed.npz",
        clips_csv_path="data/ofsyn_clips.csv",
        splits_dir="DATASET-omnifall/splits/syn/random",
        batch_size=args.batch_size,
        num_workers=0,
        use_weighted_sampler=True,
        use_diff=args.use_diff,
        pose_npz_path=args.pose_npz,
    )
    input_dim = train_loader.dataset.feature_dim
    print(f"  Train: {len(train_loader.dataset)} videos, input_dim={input_dim}")

    # --- Model ---
    print("[2/4] Building model...")
    model = TCNModel(
        input_dim=input_dim,
        hidden_dim=256,
        num_blocks=2,
        kernel_size=3,
        dilations=[1, 2],
        tcn_dropout=0.2,
        num_classes=16,
    ).to(device)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}, "
          f"RF={model.receptive_field}")

    # --- Criterions ---
    print("[3/4] Building criterions...")
    train_dataset = train_loader.dataset
    # Compute class weights from training set
    all_l16 = torch.cat([s for s in train_dataset.label_16_seqs]).numpy()
    all_fall = torch.cat([s for s in train_dataset.fall_seqs]).numpy()
    all_fallen = torch.cat([s for s in train_dataset.fallen_seqs]).numpy()

    class_counts = np.bincount(all_l16, minlength=16).astype(np.float32)
    class_counts = np.where(class_counts == 0, 1.0, class_counts)
    cls_w = torch.FloatTensor(len(all_l16) / (16 * class_counts)).to(device)
    fall_pw = torch.FloatTensor(
        [max(len(all_fall) - all_fall.sum(), 1) / max(all_fall.sum(), 1)]
    ).to(device)
    fallen_pw = torch.FloatTensor(
        [max(len(all_fallen) - all_fallen.sum(), 1) / max(all_fallen.sum(), 1)]
    ).to(device)

    criterions = {
        "cls": nn.CrossEntropyLoss(weight=cls_w),
        "fall": nn.BCEWithLogitsLoss(pos_weight=fall_pw),
        "fallen": nn.BCEWithLogitsLoss(pos_weight=fallen_pw),
    }

    # --- Training ---
    print(f"[4/4] Training ({args.epochs} epochs max)...")
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=6, factor=0.5
    )

    history = {
        "train_loss": [], "val_loss": [],
        "seg_acc": [], "fall_f1": [], "fallen_f1": [],
    }
    best_fall_f1, best_epoch, patience = 0, 0, 0

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        t0 = time.time()
        train_loss = 0.0
        for batch in train_loader:
            features = batch["features"].to(device)
            labels_16 = batch["labels_16"].to(device)
            fall_gt = batch["fall_labels"].to(device)
            fallen_gt = batch["fallen_labels"].to(device)
            B, TT = labels_16.shape
            logits_cls, logits_fall, logits_fallen = model(features)
            loss = (
                criterions["cls"](logits_cls.reshape(B * TT, -1), labels_16.reshape(B * TT))
                + 0.5 * criterions["fall"](logits_fall.reshape(B * TT), fall_gt.reshape(B * TT).float())
                + 0.5 * criterions["fallen"](logits_fallen.reshape(B * TT), fallen_gt.reshape(B * TT).float())
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        all_pred_cls, all_pred_fall, all_pred_fallen = [], [], []
        all_gt_cls, all_gt_fall, all_gt_fallen = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].to(device)
                labels_16 = batch["labels_16"].to(device)
                fall_gt = batch["fall_labels"].to(device)
                fallen_gt = batch["fallen_labels"].to(device)
                B, TT = labels_16.shape
                logits_cls, logits_fall, logits_fallen = model(features)
                loss = (
                    criterions["cls"](logits_cls.reshape(B * TT, -1), labels_16.reshape(B * TT))
                    + 0.5 * criterions["fall"](logits_fall.reshape(B * TT), fall_gt.reshape(B * TT).float())
                    + 0.5 * criterions["fallen"](logits_fallen.reshape(B * TT), fallen_gt.reshape(B * TT).float())
                )
                val_loss += loss.item()
                all_pred_cls.append(logits_cls.argmax(-1).cpu().numpy().ravel())
                all_pred_fall.append((torch.sigmoid(logits_fall) > 0.5).long().squeeze(-1).cpu().numpy().ravel())
                all_pred_fallen.append((torch.sigmoid(logits_fallen) > 0.5).long().squeeze(-1).cpu().numpy().ravel())
                all_gt_cls.append(labels_16.cpu().numpy().ravel())
                all_gt_fall.append(fall_gt.cpu().numpy().ravel())
                all_gt_fallen.append(fallen_gt.cpu().numpy().ravel())
        val_loss /= len(val_loader)

        metrics = compute_all_metrics(
            np.concatenate(all_pred_cls), np.concatenate(all_pred_fall), np.concatenate(all_pred_fallen),
            np.concatenate(all_gt_cls), np.concatenate(all_gt_fall), np.concatenate(all_gt_fallen),
            CLASS_NAMES,
        )
        scheduler.step(metrics["fall_f1"])

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["seg_acc"].append(metrics["seg_acc"])
        history["fall_f1"].append(metrics["fall_f1"])
        history["fallen_f1"].append(metrics["fallen_f1"])

        print(f"Epoch {epoch:2d}: train_loss={train_loss:.4f} ({time.time() - t0:.0f}s) | "
              f"val_loss={val_loss:.4f} | seg_acc={metrics['seg_acc']:.4f} | "
              f"fall_f1={metrics['fall_f1']:.4f} | fallen_f1={metrics['fallen_f1']:.4f} | "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

        if metrics["fall_f1"] > best_fall_f1:
            best_fall_f1 = metrics["fall_f1"]
            best_epoch = epoch
            patience = 0
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "best_fall_f1": best_fall_f1, "val_metrics": metrics},
                f"{log_dir}/tcn_best.pt",
            )
            print(f"  [BEST] saved!")
        else:
            patience += 1
            if patience >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # --- Test ---
    print(f"\n[5/5] Testing (best epoch={best_epoch})...")
    ckpt = torch.load(f"{log_dir}/tcn_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_pred_cls, all_pred_fall, all_pred_fallen = [], [], []
    all_gt_cls, all_gt_fall, all_gt_fallen = [], [], []
    test_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device)
            labels_16 = batch["labels_16"].to(device)
            fall_gt = batch["fall_labels"].to(device)
            fallen_gt = batch["fallen_labels"].to(device)
            B, TT = labels_16.shape
            logits_cls, logits_fall, logits_fallen = model(features)
            loss = (
                criterions["cls"](logits_cls.reshape(B * TT, -1), labels_16.reshape(B * TT))
                + 0.5 * criterions["fall"](logits_fall.reshape(B * TT), fall_gt.reshape(B * TT).float())
                + 0.5 * criterions["fallen"](logits_fallen.reshape(B * TT), fallen_gt.reshape(B * TT).float())
            )
            test_loss += loss.item()
            all_pred_cls.append(logits_cls.argmax(-1).cpu().numpy().ravel())
            all_pred_fall.append((torch.sigmoid(logits_fall) > 0.5).long().squeeze(-1).cpu().numpy().ravel())
            all_pred_fallen.append((torch.sigmoid(logits_fallen) > 0.5).long().squeeze(-1).cpu().numpy().ravel())
            all_gt_cls.append(labels_16.cpu().numpy().ravel())
            all_gt_fall.append(fall_gt.cpu().numpy().ravel())
            all_gt_fallen.append(fallen_gt.cpu().numpy().ravel())

    test_metrics = compute_all_metrics(
        np.concatenate(all_pred_cls), np.concatenate(all_pred_fall), np.concatenate(all_pred_fallen),
        np.concatenate(all_gt_cls), np.concatenate(all_gt_fall), np.concatenate(all_gt_fallen),
        CLASS_NAMES,
    )
    test_metrics["loss"] = test_loss / len(test_loader)

    # Save results
    with open(f"{log_dir}/training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(f"{log_dir}/test_results.json", "w") as f:
        r = {k: v for k, v in test_metrics.items() if k != "cls_report"}
        r["cls_report"] = test_metrics["cls_report"]
        json.dump(r, f, indent=2)

    print(f"\nResults ({tag}):")
    print(f"  Best epoch:   {best_epoch}")
    print(f"  seg_acc:      {test_metrics['seg_acc']:.4f}")
    print(f"  fall_f1:      {test_metrics['fall_f1']:.4f}")
    print(f"  fallen_f1:    {test_metrics['fallen_f1']:.4f}")
    print(f"  avg_f1:       {test_metrics['avg_f1']:.4f}")
    print(f"  Test loss:    {test_metrics['loss']:.4f}")
    print(f"\n{tag} COMPLETE!")


if __name__ == "__main__":
    main()
