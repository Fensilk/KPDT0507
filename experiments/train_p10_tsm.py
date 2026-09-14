"""
Phase 10 / 6.1: TSM (Temporal Shift Module) adaptation on OmniFall OF-Syn.

Task: frame-level ternary classification (0=fall, 1=fallen, 2=normal), same mapping as the main line.
Model: official TSN/TSM implementation vendored in reference/temporal-shift-module
       (resnet50 + temporal shift, shift_div=8, shift_place=blockres).
Data : dense JPEG frames produced by experiments/prep_p10_frames.py
       (data/ref_frames/<class>/<video>/img_00001.jpg ... img_00081.jpg).
Sample: one clip = N consecutive frames; its label = the ternary label of the CENTER frame
        (so sliding the clip yields frame-level predictions, comparable with the Phase 6/9 protocol).

Usage (smoke):
    python experiments/train_p10_tsm.py --tag tsm_smoke --limit_videos 200 --epochs 1 --max_steps 60
Usage (full, on Autodl):
    python experiments/train_p10_tsm.py --tag tsm_r50_8f --epochs 20 --batch_size 32 --tune_from <tsm_k400.pth>
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import cv2

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)
sys.path.insert(0, os.path.join(PROJ_ROOT, "reference", "temporal-shift-module"))

from utils.metrics import compute_ternary_metrics  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# --------------------------------------------------------------------------------------
# torchvision compatibility: TSM's ops/models.py calls torchvision.models.resnet50(True/False)
# positionally, which is no longer allowed (torchvision >= 0.13 keyword-only weights=).
# We patch it in-process instead of editing the vendored third-party code.
# --------------------------------------------------------------------------------------
import torchvision.models as tvm  # noqa: E402

_resnet50_orig = tvm.resnet50


def _resnet50_compat(pretrained=False, **kwargs):
    weights = tvm.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    return _resnet50_orig(weights=weights, **kwargs)


tvm.resnet50 = _resnet50_compat  # patched before TSN is constructed

from ops.models import TSN  # noqa: E402

TERNARY_NAMES = ["fall", "fallen", "normal"]
DEFAULT_FRAMES_ROOT = os.path.join(PROJ_ROOT, "data", "ref_frames")
FRAME_LABEL_CSV = os.path.join(PROJ_ROOT, "data", "ofsyn_frame_labels.csv")
SPLIT_ROOT = os.path.join(PROJ_ROOT, "DATASET-omnifall", "splits", "syn", "random")
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ternary_from_16(label16: int) -> int:
    """Same mapping as models/dataset.py (Phase 6)."""
    if label16 == 1:
        return 0  # fall
    if label16 == 2:
        return 1  # fallen
    return 2      # normal (includes lying)


def compute_ternary_metrics_safe(pred, gt):
    """utils.metrics.compute_ternary_metrics, but robust when a small eval subset misses a class.

    The project helper calls sklearn's classification_report without `labels=[0,1,2]`, which raises
    when one of the three classes is absent (e.g. an 8-video validation subset). We fall back to the
    identical formulas with explicit labels, so numbers stay comparable with the main line.
    """
    try:
        return compute_ternary_metrics(pred, gt)
    except ValueError:
        out = {}
        for idx, name in enumerate(["fall", "fallen", "normal"]):
            p, r, f = (
                precision_score(gt == idx, pred == idx, zero_division=0),
                recall_score(gt == idx, pred == idx, zero_division=0),
                f1_score(gt == idx, pred == idx, zero_division=0),
            )
            out["%s_precision" % name] = float(p)
            out["%s_recall" % name] = float(r)
            out["%s_f1" % name] = float(f)
        out["avg_f1"] = (out["fall_f1"] + out["fallen_f1"] + out["normal_f1"]) / 3.0
        out["ternary_acc"] = float((pred == gt).mean())
        out["confusion_matrix"] = confusion_matrix(gt, pred, labels=[0, 1, 2]).tolist()
        out["cls_report"] = classification_report(gt, pred, labels=[0, 1, 2],
                                                  target_names=["fall", "fallen", "normal"], zero_division=0)
        return out


def load_frame_labels():
    """path -> np.ndarray(81,) of ternary labels."""
    table = {}
    with open(FRAME_LABEL_CSV, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            table.setdefault(row["path"], []).append((int(row["frame_idx"]), ternary_from_16(int(row["label"]))))
    out = {}
    for path, pairs in table.items():
        pairs.sort()
        out[path] = np.array([lab for _, lab in pairs], dtype=np.int64)
    return out


def read_split(split: str):
    with open(os.path.join(SPLIT_ROOT, split + ".csv"), newline="", encoding="utf-8") as fh:
        return [row["path"].strip() for row in csv.DictReader(fh)]


class OmniFallClipDataset(Dataset):
    """Clip dataset: N consecutive frames -> ternary label of the center frame."""

    def __init__(self, videos, frame_labels, frames_root, n_frames=8, stride=8, train=True, crop=224,
                 scale=(0.7, 1.0), resize_to=None):
        self.n_frames = n_frames
        self.train = train
        self.crop = crop
        self.scale = scale
        self.resize_to = resize_to   # optional second resize (e.g. C3D needs 112x112)
        self.samples = []
        self.missing = 0

        for path in videos:
            labels = frame_labels.get(path)
            if labels is None:
                continue
            frames_dir = os.path.join(frames_root, path)
            if not os.path.isdir(frames_dir):
                self.missing += 1
                continue
            total = len(labels)
            starts = list(range(0, total - n_frames + 1, stride))
            for start in starts:
                center = start + n_frames // 2
                self.samples.append((path, frames_dir, start, int(labels[center])))

    def __len__(self):
        return len(self.samples)

    def _load_clip(self, frames_dir, start):
        frames = []
        for idx in range(start, start + self.n_frames):
            img = cv2.imread(os.path.join(frames_dir, "img_%05d.jpg" % (idx + 1)))
            if img is None:
                img = np.zeros((self.crop, self.crop, 3), dtype=np.uint8)
            frames.append(img)  # BGR uint8
        return frames

    def _augment(self, frames):
        h, w = frames[0].shape[:2]
        if self.train:
            # random resized crop (scale jitter) + horizontal flip
            for _ in range(10):
                area = h * w
                target = random.uniform(self.scale[0], self.scale[1]) * area
                ratio = random.uniform(3.0 / 4.0, 4.0 / 3.0)
                cw = int(round(np.sqrt(target * ratio)))
                ch = int(round(np.sqrt(target / ratio)))
                if cw <= w and ch <= h:
                    x0 = random.randint(0, w - cw)
                    y0 = random.randint(0, h - ch)
                    break
            else:
                cw, ch, x0, y0 = w, h, 0, 0
            flip = random.random() < 0.5
            out = []
            for f in frames:
                patch = f[y0 : y0 + ch, x0 : x0 + cw]
                patch = cv2.resize(patch, (self.crop, self.crop), interpolation=cv2.INTER_LINEAR)
                if flip:
                    patch = patch[:, ::-1]
                out.append(patch)
        else:
            # center crop
            y0 = max(0, (h - self.crop) // 2)
            x0 = max(0, (w - self.crop) // 2)
            out = [f[y0 : y0 + self.crop, x0 : x0 + self.crop] for f in frames]
        if self.resize_to and self.resize_to != self.crop:
            out = [cv2.resize(f, (self.resize_to, self.resize_to), interpolation=cv2.INTER_LINEAR) for f in out]
        return out

    def __getitem__(self, index):
        path, frames_dir, start, label = self.samples[index]
        frames = self._load_clip(frames_dir, start)
        frames = self._augment(frames)
        arr = np.stack([f[:, :, ::-1].astype(np.float32) / 255.0 for f in frames], axis=0)  # RGB
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(np.ascontiguousarray(arr.transpose(0, 3, 1, 2)))  # (N,3,H,W)
        return tensor, label, path, start


def build_model(args, device):
    model = TSN(
        num_class=3,
        num_segments=args.num_frames,
        modality="RGB",
        base_model="resnet50",
        new_length=1,
        consensus_type="avg",
        dropout=args.dropout,
        is_shift=True,
        shift_div=args.shift_div,
        shift_place="blockres",
        pretrain=None,          # no ImageNet download; use --tune_from for Kinetics weights
        partial_bn=False,
        print_spec=False,
    )

    if args.tune_from:
        ckpt = torch.load(args.tune_from, map_location="cpu")
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        state = {k.replace("module.", ""): v for k, v in state.items()}
        model_dict = model.state_dict()
        # drop classifier weights (different num classes)
        state = {k: v for k, v in state.items() if not k.startswith("new_fc") and not k.startswith("base_model.fc")}
        loadable = {k: v for k, v in state.items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(loadable)
        model.load_state_dict(model_dict)
        print("[tsm] loaded %d/%d tensors from %s" % (len(loadable), len(model_dict), args.tune_from))

    if getattr(args, "fast_shift", False):
        sys.path.insert(0, os.path.join(PROJ_ROOT, "experiments"))
        from fast_tsm_shift import convert_to_fast_shift

        convert_to_fast_shift(model, verbose=True)

    return model.to(device)


def evaluate(model, loader, device, max_batches=None):
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for step, (clip, label, _path, _center) in enumerate(loader):
            clip = clip.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                logits = model(clip)
            preds.append(logits.argmax(dim=1).cpu().numpy())
            gts.append(np.asarray(label))
            if max_batches and step + 1 >= max_batches:
                break
    if not preds:
        return {"avg_f1": 0.0, "ternary_acc": 0.0, "n": 0}
    pred = np.concatenate(preds)
    gt = np.concatenate(gts)
    metrics = compute_ternary_metrics_safe(pred, gt)
    metrics["n"] = int(len(gt))
    return metrics


def evaluate_dense(model, loader, device, n_frames=80, max_batches=None):
    """Dense sliding-window inference with per-frame merging (same idea as the Phase 9 eval).

    Every clip prediction is written to ALL frames inside its window and averaged across the
    overlapping windows, so each frame gets a merged prediction (this is what the main line calls
    the merged/frame-level timeline). Returns (metrics, {video path: (n_frames,) int8, -1 = uncovered}).
    """
    model.eval()
    per_video = {}
    with torch.no_grad():
        for step, (clip, label, path, start) in enumerate(loader):
            clip = clip.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                logits = model(clip)
            probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
            for i, p in enumerate(path):
                slot = per_video.setdefault(
                    p, {"sum": np.zeros((n_frames, 3), dtype=np.float64), "cnt": np.zeros(n_frames, dtype=np.int64)}
                )
                s = int(start[i])
                for t in range(s, min(s + loader.dataset.n_frames, n_frames)):
                    slot["sum"][t] += probs[i]
                    slot["cnt"][t] += 1
            if max_batches and step + 1 >= max_batches:
                break

    frame_labels = load_frame_labels()
    per_video_pred = {}
    preds, gts = [], []
    for path, slot in per_video.items():
        labels = frame_labels.get(path)
        if labels is None:
            continue
        n = min(n_frames, len(labels))
        covered = slot["cnt"][:n] > 0
        if not covered.any():
            continue
        avg = slot["sum"][:n][covered] / slot["cnt"][:n][covered][:, None]
        aligned = np.full(n, -1, dtype=np.int8)
        aligned[covered] = avg.argmax(axis=1).astype(np.int8)
        per_video_pred[path] = aligned
        preds.append(aligned[covered])
        gts.append(labels[:n][covered])
    if not preds:
        return {"avg_f1": 0.0, "ternary_acc": 0.0, "n": 0}, {}
    pred_all = np.concatenate(preds)
    gt_all = np.concatenate(gts)
    metrics = compute_ternary_metrics_safe(pred_all, gt_all)
    metrics["n"] = int(len(gt_all))
    metrics["n_videos"] = int(len(preds))
    return metrics, per_video_pred


def run_test_eval(model, args, frame_labels, device, out_dir, ckpt_path=None):
    """Evaluate the (best) checkpoint on the test split and write the artefacts.

    Writes test_results.json (instance metrics) and, when --test_dense is set,
    test_dense_preds.npz holding per-video per-frame predictions for the timeline metrics
    (see experiments/eval_p10_timeline.py).
    """
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])

    test_videos = read_split("test")
    if args.test_limit_videos:
        rng_test = np.random.default_rng(args.seed)
        test_videos = list(test_videos)
        rng_test.shuffle(test_videos)
        test_videos = test_videos[: args.test_limit_videos]
    test_stride = 1 if args.test_dense else args.test_stride
    test_ds = OmniFallClipDataset(test_videos, frame_labels, args.frames_root,
                                  n_frames=args.num_frames, stride=test_stride, train=False, crop=args.crop,
                                  resize_to=getattr(args, "resize_to", None))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=(device.type == "cuda"))
    print("[test] %d videos -> %d clips (stride=%d)" % (len(test_videos), len(test_ds), test_stride))

    if args.test_dense:
        test_metrics, per_video = evaluate_dense(model, test_loader, device)
        paths = list(per_video.keys())
        pred_mat = np.full((len(paths), 80), -1, dtype=np.int8)
        for i, p in enumerate(paths):
            arr = per_video[p]
            pred_mat[i, : len(arr)] = arr[:80]
        np.savez_compressed(os.path.join(out_dir, "test_dense_preds.npz"),
                            paths=np.array(paths), pred=pred_mat)
        test_metrics["protocol"] = "dense_per_frame"
    else:
        test_metrics = evaluate(model, test_loader, device)
        test_metrics["protocol"] = "center_frame_sparse"

    test_metrics["n_videos"] = len(test_videos)
    test_metrics["stride"] = test_stride
    with open(os.path.join(out_dir, "test_results.json"), "w", encoding="utf-8") as fh:
        json.dump(test_metrics, fh, ensure_ascii=False, indent=2, default=float)
    print("[test] avg_f1=%.4f acc=%.4f fall_f1=%.4f fallen_f1=%.4f (n=%d)"
          % (test_metrics.get("avg_f1", 0.0), test_metrics.get("ternary_acc", 0.0),
             test_metrics.get("fall_f1", 0.0), test_metrics.get("fallen_f1", 0.0), test_metrics.get("n", 0)))
    return test_metrics


def main():
    ap = argparse.ArgumentParser(description="Phase 10 TSM adaptation training")
    ap.add_argument("--tag", default="tsm_smoke")
    ap.add_argument("--frames_root", default=DEFAULT_FRAMES_ROOT)
    ap.add_argument("--num_frames", type=int, default=8)
    ap.add_argument("--train_stride", type=int, default=8)
    ap.add_argument("--val_stride", type=int, default=8,
                    help="stride for the validation split (keep at the evaluation protocol value)")
    ap.add_argument("--crop", type=int, default=224)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--shift_div", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.01, help="lr for the classifier head; backbone gets lr/10")
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--early_stop_patience", type=int, default=0,
                    help="0 = disable; otherwise stop after N epochs without val improvement")
    ap.add_argument("--max_steps", type=int, default=None, help="stop after N optimizer steps (smoke)")
    ap.add_argument("--limit_videos", type=int, default=None, help="use only the first N train videos")
    ap.add_argument("--val_limit_videos", type=int, default=200)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tune_from", default=None, help="TSM Kinetics checkpoint (.pth)")
    ap.add_argument("--balanced_sampler", action="store_true")
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--fast_shift", action="store_true",
                    help="cat-based temporal shift (verified numerically equivalent, ~10%% faster)")
    ap.add_argument("--test_stride", type=int, default=8, help="sliding-window stride for test evaluation")
    ap.add_argument("--test_dense", action="store_true", help="stride=1 sliding windows aggregated per frame (slower)")
    ap.add_argument("--test_limit_videos", type=int, default=None)
    ap.add_argument("--skip_test", action="store_true")
    ap.add_argument("--eval_only_ckpt", default=None,
                    help="skip training: load this checkpoint and only run the test evaluation")
    ap.add_argument("--log_root", default=os.path.join(PROJ_ROOT, "logs", "phase10"))
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(args.log_root, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "args.json"), "w", encoding="utf-8") as fh:
        json.dump(vars(args), fh, ensure_ascii=False, indent=2)

    frame_labels = load_frame_labels()
    train_videos = read_split("train")
    val_videos = read_split("val")
    # Subset selection mirrors experiments/prep_p9e1_frames.py (fresh rng(seed) shuffle then take N),
    # so "--limit_videos 2000" lands on exactly the same videos as the E1@2000 / E3@2000 experiments.
    if args.limit_videos:
        rng = np.random.default_rng(args.seed)
        train_videos = list(train_videos)
        rng.shuffle(train_videos)
        train_videos = train_videos[: args.limit_videos]
    if args.val_limit_videos:
        rng_val = np.random.default_rng(args.seed)
        val_videos = list(val_videos)
        rng_val.shuffle(val_videos)
        val_videos = val_videos[: args.val_limit_videos]

    train_ds = OmniFallClipDataset(train_videos, frame_labels, args.frames_root,
                                   n_frames=args.num_frames, stride=args.train_stride, train=True, crop=args.crop)
    val_ds = OmniFallClipDataset(val_videos, frame_labels, args.frames_root,
                                 n_frames=args.num_frames, stride=args.val_stride, train=False, crop=args.crop)
    print("[data] train clips=%d (missing video dirs=%d), val clips=%d (missing=%d)"
          % (len(train_ds), train_ds.missing, len(val_ds), val_ds.missing))
    if len(train_ds) == 0:
        print("[data] no training clips; run experiments/prep_p10_frames.py first")
        return

    labels = np.array([s[3] for s in train_ds.samples])
    counts = np.bincount(labels, minlength=3).astype(np.float64)
    class_weights = counts.sum() / np.maximum(counts, 1.0)
    class_weights = class_weights / class_weights.mean()
    print("[data] window label counts=%s -> class weights=%s" % (counts.astype(int).tolist(), np.round(class_weights, 3).tolist()))

    sampler = None
    if args.balanced_sampler:
        per_sample = class_weights[labels]
        sampler = WeightedRandomSampler(torch.as_tensor(per_sample, dtype=torch.double), num_samples=len(labels), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler,
                              num_workers=args.workers, pin_memory=(device.type == "cuda"), drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                            pin_memory=(device.type == "cuda"))

    model = build_model(args, device)
    n_params = sum(p.numel() for p in model.parameters())
    print("[model] TSN-TSM resnet50 params=%.2fM, num_frames=%d" % (n_params / 1e6, args.num_frames))

    if args.eval_only_ckpt:
        run_test_eval(model, args, frame_labels, device, out_dir, ckpt_path=args.eval_only_ckpt)
        print("[train] eval-only mode finished; logs -> %s" % out_dir)
        return

    head_params, backbone_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (head_params if name.startswith("new_fc") else backbone_params).append(param)
    optimizer = torch.optim.SGD(
        [{"params": backbone_params, "lr": args.lr / 10.0}, {"params": head_params, "lr": args.lr}],
        momentum=0.9, weight_decay=args.weight_decay, nesterov=False,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device),
                                    label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device.type == "cuda"))

    history = []
    best = -1.0
    epochs_since_improve = 0
    global_step = 0
    t_start = time.time()
    stop = False

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        t_epoch = time.time()
        for clip, label, _path, _center in train_loader:
            clip = clip.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(args.amp and device.type == "cuda")):
                logits = model(clip)
                loss = criterion(logits, label)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            running += float(loss.item()) * label.size(0)
            seen += label.size(0)
            global_step += 1

            if global_step % 20 == 0:
                speed = seen / max(1e-6, time.time() - t_epoch)
                print("  ep%d step%d loss=%.4f (%.1f clips/s)" % (epoch, global_step, running / max(1, seen), speed), flush=True)
            if args.max_steps and global_step >= args.max_steps:
                stop = True
                break

        train_loss = running / max(1, seen)
        val_metrics = evaluate(model, val_loader, device)
        epoch_time = time.time() - t_epoch
        record = {
            "epoch": epoch,
            "step": global_step,
            "train_loss": train_loss,
            "val_avg_f1": val_metrics.get("avg_f1", 0.0),
            "val_acc": val_metrics.get("ternary_acc", 0.0),
            "val_fall_f1": val_metrics.get("fall_f1", 0.0),
            "val_fallen_f1": val_metrics.get("fallen_f1", 0.0),
            "epoch_sec": epoch_time,
            "clips_seen": seen,
        }
        history.append(record)
        print("[epoch %d] loss=%.4f val_avg_f1=%.4f val_acc=%.4f (%.1f min)"
              % (epoch, train_loss, record["val_avg_f1"], record["val_acc"], epoch_time / 60.0), flush=True)

        if record["val_avg_f1"] > best:
            best = record["val_avg_f1"]
            epochs_since_improve = 0
            torch.save({"state_dict": model.state_dict(), "args": vars(args), "epoch": epoch}, os.path.join(out_dir, "best_model.pt"))
        else:
            epochs_since_improve += 1
        # per-epoch snapshot: keeps a one-off long run salvageable if the instance is stopped midway
        torch.save({"state_dict": model.state_dict(), "args": vars(args), "epoch": epoch},
                   os.path.join(out_dir, "epoch_%d_model.pt" % epoch))
        with open(os.path.join(out_dir, "training_history.json"), "w", encoding="utf-8") as fh:
            json.dump(history, fh, ensure_ascii=False, indent=2)

        scheduler.step()
        if args.early_stop_patience and epochs_since_improve >= args.early_stop_patience:
            print("[train] early stop: no val improvement for %d epoch(s)" % epochs_since_improve)
            break
        if stop:
            print("[train] reached max_steps=%s" % args.max_steps)
            break

    total_min = (time.time() - t_start) / 60.0
    print("[train] finished in %.1f min, best val avg_f1=%.4f" % (total_min, best))

    # ---- test evaluation with the best checkpoint (same protocol as the main line) ----
    best_path = os.path.join(out_dir, "best_model.pt")
    if not args.skip_test and os.path.isfile(best_path):
        run_test_eval(model, args, frame_labels, device, out_dir, ckpt_path=best_path)

    with open(os.path.join(out_dir, "training_history.json"), "w", encoding="utf-8") as fh:
        json.dump(history, fh, ensure_ascii=False, indent=2)
    print("[train] logs -> %s" % out_dir)


if __name__ == "__main__":
    main()
