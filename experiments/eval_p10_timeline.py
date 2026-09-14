"""
Timeline (event-level) metrics for Phase 10 reference models.

Input : test_dense_preds.npz produced by
        `python experiments/train_p10_tsm.py --test_dense --eval_only_ckpt <ckpt>`
        (keys: `paths` = video paths, `pred` = (N, 81) int8 ternary predictions, -1 = uncovered)
Output: timeline_results.json = the same event-level panel used in the Phase 9 E1 report
        (per-video fall/fallen F1, coverage, false-alarm rate, flips/video, order correctness),
        plus instance-level metrics computed on exactly the same frames.

Usage:
    python experiments/eval_p10_timeline.py --dump logs/phase10/tsm_r50_8f_9600/test_dense_preds.npz
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)

from utils.timeline_metrics import aggregate_timeline_metrics, compute_video_timeline_metrics  # noqa: E402
from utils.metrics import compute_ternary_metrics  # noqa: E402

FRAME_LABEL_CSV = os.path.join(PROJ_ROOT, "data", "ofsyn_frame_labels.csv")


def load_gt_labels():
    """video path -> np.ndarray(n_frames,) of ternary labels (0=fall, 1=fallen, 2=normal)."""
    table = {}
    with open(FRAME_LABEL_CSV, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            label16 = int(row["label"])
            tri = 0 if label16 == 1 else (1 if label16 == 2 else 2)
            table.setdefault(row["path"], []).append((int(row["frame_idx"]), tri))
    return {p: np.array([t for _, t in sorted(pairs)], dtype=np.int64) for p, pairs in table.items()}


def ternary_metrics_safe(pred, gt):
    try:
        return compute_ternary_metrics(pred, gt)
    except ValueError:
        out = {}
        for idx, name in enumerate(["fall", "fallen", "normal"]):
            out["%s_precision" % name] = float(precision_score(gt == idx, pred == idx, zero_division=0))
            out["%s_recall" % name] = float(recall_score(gt == idx, pred == idx, zero_division=0))
            out["%s_f1" % name] = float(f1_score(gt == idx, pred == idx, zero_division=0))
        out["avg_f1"] = (out["fall_f1"] + out["fallen_f1"] + out["normal_f1"]) / 3.0
        out["ternary_acc"] = float((pred == gt).mean())
        out["confusion_matrix"] = confusion_matrix(gt, pred, labels=[0, 1, 2]).tolist()
        out["cls_report"] = classification_report(gt, pred, labels=[0, 1, 2],
                                                  target_names=["fall", "fallen", "normal"], zero_division=0)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="test_dense_preds.npz")
    ap.add_argument("--out", default=None, help="output json (default: alongside the dump)")
    args = ap.parse_args()

    data = np.load(args.dump, allow_pickle=True)
    paths = [str(p) for p in data["paths"]]
    pred_mat = data["pred"]
    gt_table = load_gt_labels()

    per_video = []
    all_pred, all_gt = [], []
    missing = 0
    for i, path in enumerate(paths):
        gt = gt_table.get(path)
        if gt is None:
            missing += 1
            continue
        pred = pred_mat[i]
        valid = pred >= 0
        if not valid.any():
            continue
        p = pred[valid].astype(np.int64)
        g = gt[: len(pred)][valid]
        all_pred.append(p)
        all_gt.append(g)
        per_video.append(
            compute_video_timeline_metrics(
                pred_fall=(p == 0).astype(np.int64),
                pred_fallen=(p == 1).astype(np.int64),
                gt_fall=(g == 0).astype(np.int64),
                gt_fallen=(g == 1).astype(np.int64),
            )
        )

    timeline = aggregate_timeline_metrics(per_video)
    instance = {}
    if all_pred:
        pred_all = np.concatenate(all_pred)
        gt_all = np.concatenate(all_gt)
        instance = ternary_metrics_safe(pred_all, gt_all)
        instance["n_instances"] = int(len(gt_all))

    result = {
        "dump": args.dump,
        "n_videos": len(paths),
        "n_videos_missing_gt": missing,
        "instance": instance,
        "timeline": timeline,
    }
    out = args.out or os.path.join(os.path.dirname(args.dump), "timeline_results.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2, default=float)

    tf = timeline.get("per_video_fall_f1", {})
    tn = timeline.get("per_video_fallen_f1", {})
    ta = timeline.get("per_video_avg_f1", {})
    print("=== timeline (event-level, per-video) ===")
    print("  videos=%d  has_fall=%s  has_fallen=%s" % (timeline.get("n_videos", 0),
                                                       timeline.get("n_has_fall"),
                                                       timeline.get("n_has_fallen")))
    print("  fall_f1=%.4f  fallen_f1=%.4f  avg_f1=%.4f" % (tf.get("mean") or 0.0,
                                                           tn.get("mean") or 0.0,
                                                           ta.get("mean") or 0.0))
    print("  coverage: fall=%.3f  fallen=%.3f" % (timeline.get("fall_detection_rate") or 0.0,
                                                  timeline.get("fallen_detection_rate") or 0.0))
    print("  false alarm: fall=%.3f  fallen=%.3f" % (timeline.get("fall_false_alarm_rate") or 0.0,
                                                     timeline.get("fallen_false_alarm_rate") or 0.0))
    print("  flips/video=%.2f  order_correct=%.3f" % (timeline.get("mean_total_flips") or 0.0,
                                                      timeline.get("order_correct_rate") or 0.0))
    if instance:
        print("=== instance (same frames) ===")
        print("  avg_f1=%.4f acc=%.4f fall_f1=%.4f fallen_f1=%.4f n=%d"
              % (instance.get("avg_f1", 0.0), instance.get("ternary_acc", 0.0),
                 instance.get("fall_f1", 0.0), instance.get("fallen_f1", 0.0),
                 instance.get("n_instances", 0)))
    print("saved -> %s" % out)


if __name__ == "__main__":
    main()
