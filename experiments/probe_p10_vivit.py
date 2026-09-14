"""
Phase-10 ViViT-B/16x2 T1 baseline: frozen Kinetics-400 backbone + linear probe on the [CLS] token.

Why T1 (docs/0911实验第十阶段规划.md §6.6): the official ViViT release is JAX/Flax + Scenic with a
DMVR/TFRecord pipeline, and a full fine-tune costs ~7 h on one 3090 for a conclusion that would
largely repeat TimeSformer (both are video transformers). The plan therefore asks for a **linear
probe** as a "how good are video-Transformer features" reference point, to be labelled as such.

Recipe (the ViViT paper's linear-probe protocol — NOT HF's own head, which pools through a dense+act):
  * backbone  : google/vivit-b-16x2-kinetics400 (88.65 M), frozen, eval mode, AMP fp16
  * feature   : [CLS] token of the final layer-norm output (768-d) of a 32-frame clip window
  * head      : a single Linear(768 -> 3), class-weighted CE (inverse-frequency, mean-normalised:
                the same rule the fine-tuned baselines use), AdamW, early stop on val avg_f1
  * merging   : identical to the other baselines (probability averaging over overlapping windows)

Two stages, so the expensive forward passes are cached on disk:
  stage features : train (67,200 clips) + val (8,400) + dense test (58,800) -> about 1 h
  stage probe    : fit the linear head, write test_results.json / test_dense_preds.npz -> minutes

Run on the server:
    /root/miniconda3/bin/python -u experiments/probe_p10_vivit.py --stage features --workers 12
    /root/miniconda3/bin/python -u experiments/probe_p10_vivit.py --stage probe
Then the usual finishing steps:
    /root/miniconda3/bin/python -u experiments/eval_p10_timeline.py \
        --dump logs/phase10/vivit/test_dense_preds.npz
    /root/miniconda3/bin/python -u experiments/eval_p10_compare_frames.py \
        --ref logs/phase10/vivit/test_dense_preds.npz --ref-name vivit \
        --out logs/phase10/vivit/compare_e1_vs_vivit.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)
sys.path.insert(0, os.path.join(PROJ_ROOT, "experiments"))

from train_p10_tsm import (  # noqa: E402
    DEFAULT_FRAMES_ROOT,
    OmniFallClipDataset,
    compute_ternary_metrics_safe,
    load_frame_labels,
    read_split,
    set_seed,
)

TAG = "vivit"
OUT_DIR = os.path.join("logs", "phase10", TAG)
FEAT_DIR = os.path.join(OUT_DIR, "features")
NUM_FRAMES = 32            # the K400 checkpoint's own window (8x8 setting, tubelet 2x16x16)
CROP = 224
N_FRAMES_PER_VIDEO = 80
SPLITS = [("train", "train", 8), ("val", "val", 8), ("test_dense", "test", 1)]


def feat_path(name):
    return os.path.join(FEAT_DIR, name + ".npz")


def build_backbone(device):
    """Frozen ViViT-B/16x2 with the official K400 weights (loaded by train_p10_video.build_model)."""
    from train_p10_video import build_model

    model = build_model("vivit")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device)


def extract_split(split, stride, model, device, workers, batch_size, limit_videos=None):
    frame_labels = load_frame_labels()
    videos = list(read_split(split))
    if limit_videos:
        videos = videos[:limit_videos]
    ds = OmniFallClipDataset(videos, frame_labels, DEFAULT_FRAMES_ROOT, n_frames=NUM_FRAMES,
                             stride=stride, train=False, crop=CROP)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    print("[features] %s: %d videos -> %d clips (missing=%d, workers=%d)"
          % (split, len(videos), len(ds), ds.missing, workers), flush=True)

    feats, labels, paths, starts = [], [], [], []
    t0 = time.time()
    seen = 0
    for step, (clip, label, path, start) in enumerate(loader):
        clip = clip.to(device, non_blocking=True)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            hidden = model.net.vivit(pixel_values=clip).last_hidden_state
        feats.append(hidden[:, 0].float().cpu().numpy())      # [CLS] token
        labels.append(np.asarray(label, dtype=np.int16))
        paths.append(np.asarray(path))
        starts.append(np.asarray(start, dtype=np.int16))
        seen += clip.size(0)
        if step % 100 == 0 or seen == len(ds):
            print("  [%s] %5d/%d clips  %.1f clips/s" % (split, seen, len(ds),
                                                         seen / max(1e-6, time.time() - t0)), flush=True)
    return (np.concatenate(feats), np.concatenate(labels), np.concatenate(paths),
            np.concatenate(starts))


def stage_features(args):
    device = torch.device("cuda")
    set_seed(args.seed)
    os.makedirs(FEAT_DIR, exist_ok=True)
    model = build_backbone(device)
    for name, split, stride in SPLITS:
        out = feat_path(name)
        if os.path.exists(out) and not args.overwrite:
            print("[skip] %s already exists (use --overwrite to redo)" % out, flush=True)
            continue
        t0 = time.time()
        feat, label, path, start = extract_split(split, stride, model, device, args.workers,
                                                 args.batch_size, args.limit_videos)
        np.savez(out, feat=feat, label=label, path=path, start=start)
        print("[features] %s -> %s  (%d clips, dim=%d, %.1f min)"
              % (name, out, len(feat), feat.shape[1], (time.time() - t0) / 60.0), flush=True)
        del feat, label, path, start
    torch.cuda.empty_cache()


def class_weights_from(labels):
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=3).astype(np.float64)
    w = counts.sum() / np.maximum(counts, 1.0)
    w = w / w.mean()
    return counts, w


def stage_probe(args):
    device = torch.device("cuda")
    set_seed(args.seed)
    os.makedirs(OUT_DIR, exist_ok=True)

    tr = np.load(feat_path("train"))
    va = np.load(feat_path("val"))
    Xtr = torch.from_numpy(np.ascontiguousarray(tr["feat"])).to(device)
    ytr = torch.from_numpy(tr["label"].astype(np.int64)).to(device)
    Xva = torch.from_numpy(np.ascontiguousarray(va["feat"])).to(device)
    yva_np = va["label"].astype(np.int64)

    counts, class_weights = class_weights_from(tr["label"])
    print("[data] label counts=%s -> weights=%s"
          % (counts.astype(int).tolist(), np.round(class_weights, 3).tolist()), flush=True)

    feat_dim = Xtr.shape[1]
    head = torch.nn.Linear(feat_dim, 3).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device))

    history, best, best_state, no_improve = [], -1.0, None, 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        head.train()
        perm = torch.randperm(Xtr.shape[0], device=device)
        running, seen = 0.0, 0
        for i in range(0, Xtr.shape[0], args.head_batch):
            idx = perm[i:i + args.head_batch]
            logits = head(Xtr[idx])
            loss = criterion(logits, ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += float(loss.item()) * idx.numel()
            seen += idx.numel()
        head.eval()
        with torch.no_grad():
            preds = []
            for i in range(0, Xva.shape[0], 4096):
                preds.append(head(Xva[i:i + 4096]).argmax(dim=1).cpu().numpy())
        val_pred = np.concatenate(preds)
        vm = compute_ternary_metrics_safe(val_pred, yva_np)
        rec = {"epoch": epoch, "train_loss": running / max(1, seen),
               "val_avg_f1": float(vm.get("avg_f1", 0.0)),
               "val_acc": float(vm.get("ternary_acc", 0.0)),
               "val_fall_f1": float(vm.get("fall_f1", 0.0)),
               "val_fallen_f1": float(vm.get("fallen_f1", 0.0)),
               "lr": float(opt.param_groups[0]["lr"])}
        history.append(rec)
        print("[probe ep%d] loss=%.4f val_avg_f1=%.4f val_acc=%.4f"
              % (epoch, rec["train_loss"], rec["val_avg_f1"], rec["val_acc"]), flush=True)
        if rec["val_avg_f1"] > best:
            best, no_improve = rec["val_avg_f1"], 0
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        else:
            no_improve += 1
        sched.step()
        if args.patience and no_improve >= args.patience:
            print("[probe] early stop: no val improvement for %d epoch(s)" % no_improve, flush=True)
            break

    head.load_state_dict(best_state)
    print("[probe] best val avg_f1=%.4f (%.1f min)" % (best, (time.time() - t0) / 60.0), flush=True)
    torch.save({"state_dict": head.state_dict(), "feat_dim": feat_dim, "class_weights": class_weights,
                "args": vars(args)}, os.path.join(OUT_DIR, "linear_head.pt"))
    with open(os.path.join(OUT_DIR, "training_history.json"), "w", encoding="utf-8") as fh:
        json.dump(history, fh, ensure_ascii=False, indent=2)
    with open(os.path.join(OUT_DIR, "args.json"), "w", encoding="utf-8") as fh:
        json.dump({"model": TAG, "tag": TAG, "protocol": "linear_probe_T1",
                   "backbone": "google/vivit-b-16x2-kinetics400",
                   "feature": "cls_token_last_hidden_state", "num_frames": NUM_FRAMES,
                   "crop": CROP, "train_stride": 8, "test_stride": 1,
                   "head": "Linear(%d,3)" % feat_dim, "optimizer": "adamw",
                   "lr": args.lr, "weight_decay": args.wd, "head_epochs": args.epochs,
                   "head_batch": args.head_batch, "patience": args.patience,
                   "class_weights": np.round(class_weights, 4).tolist(),
                   "best_val_avg_f1": best}, fh, ensure_ascii=False, indent=2)

    # ---- dense test: apply the frozen head to the cached features, then merge exactly like the
    # fine-tuned baselines do (average the softmax probabilities of every window covering a frame)
    te = np.load(feat_path("test_dense"))
    probs = []
    with torch.no_grad():
        for i in range(0, te["feat"].shape[0], 8192):
            x = torch.from_numpy(np.ascontiguousarray(te["feat"][i:i + 8192])).to(device)
            probs.append(torch.softmax(head(x).float(), dim=1).cpu().numpy())
    probs = np.concatenate(probs)
    metrics = merge_and_score(te["path"], te["start"], probs, out_dir=OUT_DIR)
    print("[test] avg_f1=%.4f acc=%.4f fall_f1=%.4f fallen_f1=%.4f (n=%d)"
          % (metrics.get("avg_f1", 0.0), metrics.get("ternary_acc", 0.0), metrics.get("fall_f1", 0.0),
             metrics.get("fallen_f1", 0.0), metrics.get("n", 0)), flush=True)
    print("[probe] artefacts -> %s" % OUT_DIR, flush=True)


def merge_and_score(paths, starts, probs, out_dir, n_frames=N_FRAMES_PER_VIDEO, win=NUM_FRAMES,
                    stride=1):
    """Same per-frame merge as train_p10_tsm.evaluate_dense: average window probabilities, argmax."""
    frame_labels = load_frame_labels()
    per_video = {}
    for p, s, pr in zip(paths, starts, probs):
        slot = per_video.setdefault(str(p), {"sum": np.zeros((n_frames, 3), dtype=np.float64),
                                             "cnt": np.zeros(n_frames, dtype=np.int64)})
        s = int(s)
        for t in range(s, min(s + win, n_frames)):
            slot["sum"][t] += pr
            slot["cnt"][t] += 1

    out_paths, rows, preds, gts = [], [], [], []
    for p, slot in per_video.items():
        labels = frame_labels.get(p)
        if labels is None:
            continue
        n = min(n_frames, len(labels))
        covered = slot["cnt"][:n] > 0
        if not covered.any():
            continue
        avg = slot["sum"][:n][covered] / slot["cnt"][:n][covered][:, None]
        aligned = np.full(n, -1, dtype=np.int8)
        aligned[covered] = avg.argmax(axis=1).astype(np.int8)
        row = np.full(n_frames, -1, dtype=np.int8)
        row[:n] = aligned
        out_paths.append(p)
        rows.append(row)
        preds.append(aligned[covered])
        gts.append(labels[:n][covered])

    metrics = compute_ternary_metrics_safe(np.concatenate(preds), np.concatenate(gts))
    metrics["n"] = int(sum(len(x) for x in gts))
    metrics["n_videos"] = int(len(rows))
    metrics["protocol"] = "dense_per_frame"
    metrics["stride"] = stride
    metrics["head"] = "linear_probe_T1"
    np.savez_compressed(os.path.join(out_dir, "test_dense_preds.npz"),
                        paths=np.asarray(out_paths), pred=np.stack(rows))
    with open(os.path.join(out_dir, "test_results.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2, default=float)
    return metrics


def main():
    ap = argparse.ArgumentParser(description="Phase 10 ViViT-B/16x2 T1 linear-probe baseline")
    ap.add_argument("--stage", required=True, choices=["features", "probe", "all"])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=32, help="feature extraction batch")
    ap.add_argument("--head_batch", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=60, help="max head epochs")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit_videos", type=int, default=None, help="debug: cap clips per split")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    if args.stage in ("features", "all"):
        stage_features(args)
    if args.stage in ("probe", "all"):
        stage_probe(args)


if __name__ == "__main__":
    main()
