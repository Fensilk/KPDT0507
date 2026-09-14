"""
Phase 10: put the main line (E1-full) and a reference model on the SAME frame-level protocol.

E1-full is stored in `logs/phase9/e1_full/pv_best.pt` as per-video per-frame merged predictions
(dict: video path -> {"bridge_pred": (80,), "ternary_gt": (80,)}) produced by
`experiments/eval_p9e1_dual.py` (windows 64/stride 8, per-frame majority vote).

A reference model is stored as `test_dense_preds.npz` (paths + (N,80) int8 predictions, -1 = uncovered)
produced by `python experiments/train_p10_tsm.py --test_dense --eval_only_ckpt ...`.

This script computes both panels -- instance (frame-level) and timeline (event-level) -- on the
exact same set of frames, so the two models can be compared without protocol confounds.

Usage:
    python experiments/eval_p10_compare_frames.py \
        --e1 logs/phase9/e1_full/pv_best.pt \
        --ref logs/phase10/tsm_r50_8f_9600/test_dense_preds.npz \
        --ref-name TSM \
        --out logs/phase10/tsm_r50_8f_9600/compare_e1_vs_tsm.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)

from utils.metrics import compute_ternary_metrics  # noqa: E402
from utils.timeline_metrics import aggregate_timeline_metrics, compute_video_timeline_metrics  # noqa: E402


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
        return out


def load_e1(path):
    pv = torch.load(path, map_location="cpu", weights_only=False)
    pred, gt = {}, {}
    for video, d in pv.items():
        pred[video] = np.asarray(d["bridge_pred"], dtype=np.int64)
        gt[video] = np.asarray(d["ternary_gt"], dtype=np.int64)
    return pred, gt


def load_ref(path):
    data = np.load(path, allow_pickle=True)
    paths = [str(p) for p in data["paths"]]
    mat = data["pred"]
    pred = {p: mat[i].astype(np.int64) for i, p in enumerate(paths)}
    return pred


def panels(pred, gt, videos):
    """Instance + timeline panels on the given videos, using only frames both sides cover."""
    all_p, all_g = [], []
    per_video = []
    used_frames = 0
    for v in videos:
        p, g = pred[v], gt[v]
        n = min(len(p), len(g))
        valid = (p[:n] >= 0) & (g[:n] >= 0)
        if not valid.any():
            continue
        pv, gv = p[:n][valid], g[:n][valid]
        all_p.append(pv)
        all_g.append(gv)
        used_frames += len(pv)
        per_video.append(
            compute_video_timeline_metrics(
                pred_fall=(pv == 0).astype(np.int64),
                pred_fallen=(pv == 1).astype(np.int64),
                gt_fall=(gv == 0).astype(np.int64),
                gt_fallen=(gv == 1).astype(np.int64),
            )
        )
    inst = ternary_metrics_safe(np.concatenate(all_p), np.concatenate(all_g)) if all_p else {}
    inst["n_instances"] = int(used_frames)
    tl = aggregate_timeline_metrics(per_video)
    return inst, tl


def fmt_row(name, inst, tl, ref=None):
    def d(a, b):
        return "" if ref is None else "  (%+.3f)" % (a - b)
    lines = [
        "%-10s inst: acc=%.4f avg_f1=%.4f fall_f1=%.4f fallen_f1=%.4f" % (
            name, inst.get("ternary_acc", 0), inst.get("avg_f1", 0),
            inst.get("fall_f1", 0), inst.get("fallen_f1", 0)),
        "%-10s time: fall_f1=%.4f fallen_f1=%.4f avg_f1=%.4f | cov %.3f/%.3f | FA %.3f/%.3f | flips %.2f" % (
            "", tl.get("per_video_fall_f1", {}).get("mean") or 0,
            tl.get("per_video_fallen_f1", {}).get("mean") or 0,
            tl.get("per_video_avg_f1", {}).get("mean") or 0,
            tl.get("fall_detection_rate") or 0, tl.get("fallen_detection_rate") or 0,
            tl.get("fall_false_alarm_rate") or 0, tl.get("fallen_false_alarm_rate") or 0,
            tl.get("mean_total_flips") or 0),
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1", default=os.path.join(PROJ_ROOT, "logs/phase9/e1_full/pv_best.pt"))
    ap.add_argument("--ref", required=True)
    ap.add_argument("--ref-name", default="ref")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    e1_pred, e1_gt = load_e1(args.e1)
    ref_pred = load_ref(args.ref)
    ref_gt = e1_gt  # same ground truth (per-frame ternary)

    common = sorted(set(e1_pred) & set(ref_pred))
    e1_inst, e1_tl = panels(e1_pred, e1_gt, common)
    ref_inst, ref_tl = panels(ref_pred, ref_gt, common)

    print("common videos = %d" % len(common))
    print(fmt_row("E1-full", e1_inst, e1_tl))
    print(fmt_row(args.ref_name, ref_inst, ref_tl))
    print("delta (%s - E1): acc %+.4f | avg_f1 %+.4f | fall_f1 %+.4f | fallen_f1 %+.4f"
          % (args.ref_name,
             ref_inst.get("ternary_acc", 0) - e1_inst.get("ternary_acc", 0),
             ref_inst.get("avg_f1", 0) - e1_inst.get("avg_f1", 0),
             ref_inst.get("fall_f1", 0) - e1_inst.get("fall_f1", 0),
             ref_inst.get("fallen_f1", 0) - e1_inst.get("fallen_f1", 0)))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "n_common_videos": len(common),
                    "e1_full": {"instance": e1_inst, "timeline": e1_tl},
                    args.ref_name: {"instance": ref_inst, "timeline": ref_tl},
                },
                fh, ensure_ascii=False, indent=2, default=float,
            )
        print("saved -> %s" % args.out)


if __name__ == "__main__":
    main()
