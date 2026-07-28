"""
Analyze fallen vs lying confusion in detail.

Loads the bridge model, runs inference on the test set, and breaks down
"normal -> fallen" errors by original 16-class label to quantify how much
of the confusion comes from lying / lie_down frames.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.phase6_model import Phase6TernaryModel
from models.dataset import create_longseq_dataloaders

# 16-class names
CLASS_NAMES = [
    "walk", "fall", "fallen",
    "sit_down", "sitting", "lie_down", "lying",
    "stand_up", "standing", "other",
    "kneel_down", "kneeling", "squat_down", "squatting",
    "crawl", "jump",
]

TERNARY_NAMES = ["fall", "fallen", "normal"]


def analyze(exp_dir: str, base_dir: str):
    """
    Load model and run test inference, collecting per-frame:
    - original 16-class label
    - ternary ground truth
    - ternary prediction
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # Load test_results.json for config
    results_path = os.path.join(base_dir, exp_dir, "test_results.json")
    with open(results_path) as f:
        results = json.load(f)
    config = results["config"]

    print(f"[INFO] Model config: decoder={config['decoder_type']}, "
          f"causal={config['causal']}, T={config['window_size']}")

    # Build model
    model = Phase6TernaryModel(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        decoder_type=config["decoder_type"],
        causal=config["causal"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        dropout=config["dropout"],
        max_len=config["window_size"] + 10,
    ).to(device)

    # Load checkpoint
    ckpt_path = os.path.join(base_dir, exp_dir, "best_model.pt")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Build test loader
    frame_npz = "data/omnifall_dinov2_giant_frame.npz"
    splits_dir = "DATASET-omnifall/splits/syn/random"

    _, _, test_loader = create_longseq_dataloaders(
        npz_path=os.path.join(base_dir, frame_npz),
        splits_dir=os.path.join(base_dir, splits_dir),
        window_size=config["window_size"],
        stride=config["stride"],
        batch_size=config["batch_size"],
        num_workers=0,
        use_weighted_sampler=False,
        use_diff=False,
        pose_npz_path=None,
        use_ternary_sampler=False,
    )

    # ── Inference ──────────────────────────────────────
    print(f"[INFO] Running inference on {len(test_loader)} batches...")

    # Collect: list of (label_16, ternary_gt, ternary_pred)
    per_frame = {"label_16": [], "ternary_gt": [], "ternary_pred": []}

    with torch.no_grad():
        for bi, batch in enumerate(test_loader):
            features = batch["features"].to(device)
            labels_16 = batch["labels_16"].numpy()  # (B, T)
            ternary_gt = batch["ternary_labels"].numpy()  # (B, T)
            B, T = labels_16.shape

            logits = model(features)  # (B, T, 3)
            pred = logits.argmax(dim=-1).cpu().numpy()  # (B, T)

            per_frame["label_16"].append(labels_16.ravel())
            per_frame["ternary_gt"].append(ternary_gt.ravel())
            per_frame["ternary_pred"].append(pred.ravel())

            if (bi + 1) % 50 == 0:
                print(f"  [{bi+1}/{len(test_loader)}]")

    label_16 = np.concatenate(per_frame["label_16"])
    ternary_gt = np.concatenate(per_frame["ternary_gt"])
    ternary_pred = np.concatenate(per_frame["ternary_pred"])

    total_frames = len(label_16)
    print(f"[INFO] Total frames: {total_frames:,}")
    print(f"[INFO] GT distribution: fall={np.sum(ternary_gt==0):,}, "
          f"fallen={np.sum(ternary_gt==1):,}, normal={np.sum(ternary_gt==2):,}")

    # ── Breakdown: 16-class GT → ternary prediction ───
    print("\n" + "=" * 80)
    print("16-CLASS GT → TERNARY PREDICTION BREAKDOWN")
    print("=" * 80)

    header = f"{'16-Class GT':>12s} | {'Total':>8s} | {'→fall':>8s} {'(rate)':>7s} | {'→fallen':>8s} {'(rate)':>7s} | {'→normal':>8s} {'(rate)':>7s}"
    print(header)
    print("-" * len(header))

    class_stats = {}
    for cls_idx in range(16):
        mask = label_16 == cls_idx
        n = mask.sum()
        if n == 0:
            continue
        pred_fall = np.sum(ternary_pred[mask] == 0)
        pred_fallen = np.sum(ternary_pred[mask] == 1)
        pred_normal = np.sum(ternary_pred[mask] == 2)

        class_stats[cls_idx] = {
            "name": CLASS_NAMES[cls_idx],
            "total": n,
            "pred_fall": pred_fall,
            "pred_fallen": pred_fallen,
            "pred_normal": pred_normal,
        }
        print(
            f"{CLASS_NAMES[cls_idx]:>12s} | {n:>8,d} | "
            f"{pred_fall:>8,d} ({pred_fall/n*100:>5.1f}%) | "
            f"{pred_fallen:>8,d} ({pred_fallen/n*100:>5.1f}%) | "
            f"{pred_normal:>8,d} ({pred_normal/n*100:>5.1f}%)"
        )

    # ── Key question: normal → fallen breakdown ─────────
    print("\n" + "=" * 80)
    print("NORMAL-ORIGIN CLASSES → FALLEN (MISPREDICTIONS)")
    print("=" * 80)
    print("These are frames whose GT is normal (not fall/fallen)")
    print("but the model predicts them as fallen.")
    print()

    normal_classes = [c for c in class_stats if class_stats[c]["name"] not in ("fall", "fallen")]
    normal_classes.sort(key=lambda c: class_stats[c]["pred_fallen"], reverse=True)

    total_normal_to_fallen = sum(class_stats[c]["pred_fallen"] for c in normal_classes)
    print(f"  Total normal→fallen errors: {total_normal_to_fallen:,}")
    print()
    header2 = f"{'Class':>12s} | {'Total':>8s} | {'→fallen':>8s} | {'% of class':>10s} | {'% of all errors':>14s}"
    print(header2)
    print("-" * len(header2))
    for c in normal_classes:
        s = class_stats[c]
        pct_of_class = s["pred_fallen"] / s["total"] * 100
        pct_of_errors = s["pred_fallen"] / total_normal_to_fallen * 100
        print(
            f"{s['name']:>12s} | {s['total']:>8,d} | "
            f"{s['pred_fallen']:>8,d} | {pct_of_class:>9.1f}% | {pct_of_errors:>13.1f}%"
        )

    # ── Key question: normal → fall breakdown ─────────
    print("\n" + "=" * 80)
    print("NORMAL-ORIGIN CLASSES → FALL (MISPREDICTIONS)")
    print("=" * 80)
    print("These are frames whose GT is normal")
    print("but the model predicts them as fall.")
    print()

    normal_classes_fall = sorted(normal_classes, key=lambda c: class_stats[c]["pred_fall"], reverse=True)
    total_normal_to_fall = sum(class_stats[c]["pred_fall"] for c in normal_classes_fall)
    print(f"  Total normal→fall errors: {total_normal_to_fall:,}")
    print()
    header3 = f"{'Class':>12s} | {'Total':>8s} | {'→fall':>8s} | {'% of class':>10s} | {'% of all errors':>14s}"
    print(header3)
    print("-" * len(header3))
    for c in normal_classes_fall:
        s = class_stats[c]
        pct_of_class = s["pred_fall"] / s["total"] * 100
        pct_of_errors = s["pred_fall"] / total_normal_to_fall * 100
        print(
            f"{s['name']:>12s} | {s['total']:>8,d} | "
            f"{s['pred_fall']:>8,d} | {pct_of_class:>9.1f}% | {pct_of_errors:>13.1f}%"
        )

    # ── Fall & Fallen recall by confusion ──────────
    print("\n" + "=" * 80)
    print("FALL GT — WHERE DOES IT GO?")
    print("=" * 80)
    fall_mask = label_16 == 1  # original label 1 = fall
    n_fall = fall_mask.sum()
    pred_fall_as_fall = np.sum(ternary_pred[fall_mask] == 0)
    pred_fall_as_fallen = np.sum(ternary_pred[fall_mask] == 1)
    pred_fall_as_normal = np.sum(ternary_pred[fall_mask] == 2)
    print(f"  Total fall GT frames: {n_fall:,}")
    print(f"  → fall:   {pred_fall_as_fall:>8,d} ({pred_fall_as_fall/n_fall*100:.1f}%)")
    print(f"  → fallen: {pred_fall_as_fallen:>8,d} ({pred_fall_as_fallen/n_fall*100:.1f}%)")
    print(f"  → normal: {pred_fall_as_normal:>8,d} ({pred_fall_as_normal/n_fall*100:.1f}%)")

    print("\n" + "=" * 80)
    print("FALLEN GT — WHERE DOES IT GO?")
    print("=" * 80)
    fallen_mask = label_16 == 2  # original label 2 = fallen
    n_fallen = fallen_mask.sum()
    pred_fallen_as_fall = np.sum(ternary_pred[fallen_mask] == 0)
    pred_fallen_as_fallen = np.sum(ternary_pred[fallen_mask] == 1)
    pred_fallen_as_normal = np.sum(ternary_pred[fallen_mask] == 2)
    print(f"  Total fallen GT frames: {n_fallen:,}")
    print(f"  → fall:   {pred_fallen_as_fall:>8,d} ({pred_fallen_as_fall/n_fallen*100:.1f}%)")
    print(f"  → fallen: {pred_fallen_as_fallen:>8,d} ({pred_fallen_as_fallen/n_fallen*100:.1f}%)")
    print(f"  → normal: {pred_fallen_as_normal:>8,d} ({pred_fallen_as_normal/n_fallen*100:.1f}%)")

    # ── lying 视频的特殊分析 ─────────────────────────
    # 在一个 lying 视频中，窗口可能包含 lie_down → lying
    # lying 帧没有 fall 前因，模型最容易把它们误判为 fallen
    print("\n" + "=" * 80)
    print("SUMMARY: THE lying / lie_down PROBLEM")
    print("=" * 80)
    lying_total = class_stats[6]["total"]    # label=6 = lying
    lying_to_fallen = class_stats[6]["pred_fallen"]
    lying_to_fall = class_stats[6]["pred_fall"]
    liedown_total = class_stats[5]["total"]  # label=5 = lie_down
    liedown_to_fallen = class_stats[5]["pred_fallen"]
    liedown_to_fall = class_stats[5]["pred_fall"]

    print(f"  lying frames:     {lying_total:,} total")
    print(f"    → fallen:       {lying_to_fallen:,} ({lying_to_fallen/lying_total*100:.1f}%)")
    print(f"    → fall:          {lying_to_fall:,} ({lying_to_fall/lying_total*100:.1f}%)")
    print(f"  lie_down frames:  {liedown_total:,} total")
    print(f"    → fallen:       {liedown_to_fallen:,} ({liedown_to_fallen/liedown_total*100:.1f}%)")
    print(f"    → fall:          {liedown_to_fall:,} ({liedown_to_fall/liedown_total*100:.1f}%)")

    combined_errors = lying_to_fallen + lying_to_fall + liedown_to_fallen + liedown_to_fall
    print(f"\n  lying + lie_down contribute {combined_errors:,} normal→non-normal errors")
    if total_normal_to_fallen + total_normal_to_fall > 0:
        total_non_normal = total_normal_to_fallen + total_normal_to_fall
        print(f"  That's {combined_errors/total_non_normal*100:.1f}% "
              f"of ALL normal→(fall+fallen) errors ({total_non_normal:,} total)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="logs/phase6/p6_bridge")
    args = parser.parse_args()

    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(BASE)
    analyze(args.exp_dir, BASE)
