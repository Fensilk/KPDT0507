"""
Phase 10 baselines — pytorchvideo models (X3D / I3D-R50 / SlowFast-R50) on OmniFall.

Same task, data and evaluation protocol as the TSM baseline (see experiments/train_p10_tsm.py):
frame-level ternary classification, clip = N consecutive frames, label = centre-frame label,
Kinetics-400 pretrained backbone, one official recipe per model (no per-model tuning).

The shared pieces (clip dataset, ternary metrics, dense per-frame dump, test evaluation) are
imported from train_p10_tsm so every reference model is scored by exactly the same code.

Usage:
    python experiments/train_p10_video.py --model x3d_s --tag x3d_s --epochs 5 --batch_size 32
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)
sys.path.insert(0, os.path.join(PROJ_ROOT, "experiments"))
sys.path.insert(0, os.path.join(PROJ_ROOT, "reference", "temporal-shift-module"))

from train_p10_tsm import (  # noqa: E402
    DEFAULT_FRAMES_ROOT,
    OmniFallClipDataset,
    evaluate,
    evaluate_dense,
    load_frame_labels,
    read_split,
    run_test_eval,
    set_seed,
)


def replace_classifier(net, num_classes: int = 3):
    """Swap pytorchvideo's 400-class projection for our 3-class one, keeping the pretrained
    pooling/projection stack intact, and remove the head's Softmax so the output is logits
    (pytorchvideo's X3D head applies Softmax before the output pooling, which is wrong for CE)."""
    head = net.blocks[-1]
    if not hasattr(head, "proj") or not isinstance(head.proj, nn.Linear):
        raise RuntimeError("unexpected head type: %s" % type(head).__name__)
    head.proj = nn.Linear(head.proj.in_features, num_classes)
    if getattr(head, "activation", None) is not None and isinstance(head.activation, nn.Softmax):
        head.activation = None
    return head


class SinglePathWrapper(nn.Module):
    """Our dataset yields (B, T, C, H, W); pytorchvideo models want (B, C, T, H, W)."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x.permute(0, 2, 1, 3, 4))


class HFTimeSformerWrapper(nn.Module):
    """HuggingFace TimeSformer (official Kinetics-400 weights ported to `transformers`).

    Input is our dataset's (B, T, C, H, W) — no permute needed (HF uses channel-last videos).
    Our dataset normalises with ImageNet statistics; this checkpoint expects its own
    `image_mean`/`image_std`, so we invert ours and re-apply the model's (exact, dataset untouched).
    """

    IMG_MEAN = (0.485, 0.456, 0.406)
    IMG_STD = (0.229, 0.224, 0.225)

    def __init__(self, net, hf_mean, hf_std):
        super().__init__()
        self.net = net
        self.register_buffer("img_mean", torch.tensor(self.IMG_MEAN).view(1, 1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor(self.IMG_STD).view(1, 1, 3, 1, 1))
        self.register_buffer("hf_mean", torch.tensor(hf_mean).view(1, 1, 3, 1, 1))
        self.register_buffer("hf_std", torch.tensor(hf_std).view(1, 1, 3, 1, 1))

    def forward(self, x):
        x = x * self.img_std + self.img_mean          # back to [0, 1]
        x = (x - self.hf_mean) / self.hf_std          # into the checkpoint's normalisation
        return self.net(pixel_values=x).logits


VIVIT_REPO = "google/vivit-b-16x2-kinetics400"


def _convert_vivit_state_dict(state):
    """Map the released ViViT-B/16x2 K400 checkpoint onto the `transformers>=5` module names.

    transformers 5 renamed ViViT's modules (`encoder.layer.N.attention.attention.query`
    -> `layers.N.attention.q_proj`, `attention.output.dense` -> `attention.o_proj`,
    `intermediate.dense` -> `mlp.fc1`, `output.dense` -> `mlp.fc2`), so the official
    checkpoint only loads through an explicit rename (order matters: the attention-output
    rename has to happen before the generic `output.dense` one).
    """
    out = {}
    for k, v in state.items():
        if k.startswith("classifier."):
            continue                          # 400-class head -> our 3-class one
        nk = k.replace("vivit.encoder.layer.", "vivit.layers.")
        nk = nk.replace("attention.attention.query.", "attention.q_proj.")
        nk = nk.replace("attention.attention.key.", "attention.k_proj.")
        nk = nk.replace("attention.attention.value.", "attention.v_proj.")
        nk = nk.replace("attention.output.dense.", "attention.o_proj.")
        nk = nk.replace("intermediate.dense.", "mlp.fc1.")
        nk = nk.replace("output.dense.", "mlp.fc2.")
        out[nk] = v
    return out


class HFVivitWrapper(nn.Module):
    """HuggingFace ViViT-B/16x2 (official Kinetics-400 checkpoint, 32 frames x 224^2).

    Input is our dataset's (B, T, C, H, W) — channel-last, which is exactly what HF ViViT
    consumes, so no permute is needed. This config carries no `image_mean`/`image_std`, i.e.
    it expects the ImageNet statistics our dataset already applies, so — unlike TimeSformer —
    no re-normalisation is required.
    """

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(pixel_values=x).logits


class SlowFastWrapper(nn.Module):
    """Build the two pathways from one clip: slow = every `alpha`-th frame, fast = all frames."""

    def __init__(self, net, alpha: int = 4):
        super().__init__()
        self.net = net
        self.alpha = alpha

    def forward(self, x):  # x: (B, T, C, H, W)
        x = x.permute(0, 2, 1, 3, 4)          # (B, C, T, H, W)
        fast = x
        slow = x[:, :, :: self.alpha]
        return self.net([slow, fast])


def build_model(name: str, dropout: float = 0.5):
    if name == "c3d":
        from c3d_model import C3D

        return SinglePathWrapper(C3D(num_classes=3, dropout=dropout))

    if name == "timesformer":
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")   # huggingface.co is unreachable here
        from huggingface_hub import hf_hub_download
        from transformers import TimesformerConfig, TimesformerForVideoClassification

        # transformers >=5 refuses to torch.load a `.bin` under torch <2.6 (CVE-2025-32434), so we
        # build the model from the config and load the official K400 state dict by hand
        # (the 400-class head is replaced by our 3-class one, hence `strict=False`).
        cfg = TimesformerConfig.from_pretrained("facebook/timesformer-base-finetuned-k400")
        cfg.num_labels = 3
        net = TimesformerForVideoClassification(cfg)
        ckpt_path = hf_hub_download("facebook/timesformer-base-finetuned-k400", "pytorch_model.bin")
        try:
            state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except Exception:  # older checkpoints may contain non-tensor entries
            state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = {k: v for k, v in state.items() if not k.startswith("classifier.")}  # 400-class head -> ours
        missing, unexpected = net.load_state_dict(state, strict=False)
        head_keys = [k for k in missing if "classifier" in k]
        print("[timesformer] K400 weights loaded (missing=%d, unexpected=%d, head re-init=%d)"
              % (len(missing), len(unexpected), len(head_keys)), flush=True)
        hf_mean = getattr(cfg, "image_mean", [0.45, 0.45, 0.45])
        hf_std = getattr(cfg, "image_std", [0.225, 0.225, 0.225])
        return HFTimeSformerWrapper(net, hf_mean, hf_std)

    if name == "vivit":
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        from huggingface_hub import hf_hub_download
        from transformers import VivitConfig, VivitForVideoClassification

        # same constraint as TimeSformer: transformers>=5 refuses to torch.load a `.bin` under
        # torch<2.6 (CVE-2025-32434) and this repo ships no safetensors, so we build from the
        # config and load the official K400 state dict by hand (with the key rename below).
        cfg = VivitConfig.from_pretrained(VIVIT_REPO)
        cfg.num_labels = 3
        net = VivitForVideoClassification(cfg)
        ckpt_path = hf_hub_download(VIVIT_REPO, "pytorch_model.bin")
        try:
            state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except Exception:
            state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = _convert_vivit_state_dict(state)
        missing, unexpected = net.load_state_dict(state, strict=False)
        head_keys = [k for k in missing if "classifier" in k]
        print("[vivit] K400 weights loaded (missing=%d, unexpected=%d, head re-init=%d)"
              % (len(missing), len(unexpected), len(head_keys)), flush=True)
        return HFVivitWrapper(net)

    import pytorchvideo.models.hub as hub

    if name == "x3d_s":
        net = hub.x3d_s(pretrained=True)
    elif name == "x3d_m":
        net = hub.x3d_m(pretrained=True)
    elif name == "i3d_r50":
        net = hub.i3d_r50(pretrained=True)
    elif name == "slowfast_r50":
        net = hub.slowfast_r50(pretrained=True)
    else:
        raise ValueError("unknown model: %s" % name)

    replace_classifier(net, 3)

    wrapper = SlowFastWrapper(net) if name == "slowfast_r50" else SinglePathWrapper(net)
    return wrapper


def main():
    ap = argparse.ArgumentParser(description="Phase 10 pytorchvideo baseline (X3D / I3D / SlowFast)")
    ap.add_argument("--model", required=True,
                    choices=["x3d_s", "x3d_m", "i3d_r50", "slowfast_r50", "c3d", "timesformer",
                             "vivit"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--frames_root", default=DEFAULT_FRAMES_ROOT)
    ap.add_argument("--num_frames", type=int, default=None, help="default: 16 (x3d/i3d) or 32 (slowfast)")
    ap.add_argument("--train_stride", type=int, default=8)
    ap.add_argument("--val_stride", type=int, default=8)
    ap.add_argument("--crop", type=int, default=224)
    ap.add_argument("--resize_to", type=int, default=None,
                    help="optional second resize after cropping (C3D uses 112)")
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--optimizer", default="sgd", choices=["sgd", "adamw"],
                    help="sgd for the CNN baselines; adamw is the standard choice for ViT fine-tuning")
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--early_stop_patience", type=int, default=2)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--limit_videos", type=int, default=None)
    ap.add_argument("--val_limit_videos", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--balanced_sampler", action="store_true")
    ap.add_argument("--test_stride", type=int, default=8)
    ap.add_argument("--test_dense", action="store_true")
    ap.add_argument("--test_limit_videos", type=int, default=None)
    ap.add_argument("--skip_test", action="store_true")
    ap.add_argument("--eval_only_ckpt", default=None)
    ap.add_argument("--log_root", default=os.path.join(PROJ_ROOT, "logs", "phase10"))
    args = ap.parse_args()

    if args.num_frames is None:
        if args.model == "slowfast_r50":
            args.num_frames = 32
        elif args.model == "timesformer":
            args.num_frames = 8          # matches the K400 checkpoint's num_frames
        elif args.model == "vivit":
            args.num_frames = 32         # matches the K400 checkpoint's num_frames
        elif args.model == "c3d":
            args.num_frames = 16
        else:
            args.num_frames = 16
    if args.model == "c3d" and args.resize_to is None:
        args.resize_to = 112

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(args.log_root, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "args.json"), "w", encoding="utf-8") as fh:
        json.dump(vars(args), fh, ensure_ascii=False, indent=2)

    frame_labels = load_frame_labels()
    train_videos = read_split("train")
    val_videos = read_split("val")
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
                                   n_frames=args.num_frames, stride=args.train_stride, train=True,
                                   crop=args.crop, resize_to=args.resize_to)
    val_ds = OmniFallClipDataset(val_videos, frame_labels, args.frames_root,
                                 n_frames=args.num_frames, stride=args.val_stride, train=False,
                                 crop=args.crop, resize_to=args.resize_to)
    print("[data] model=%s frames=%d | train clips=%d (missing=%d), val clips=%d (missing=%d)"
          % (args.model, args.num_frames, len(train_ds), train_ds.missing, len(val_ds), val_ds.missing), flush=True)

    labels = np.array([s[3] for s in train_ds.samples])
    counts = np.bincount(labels, minlength=3).astype(np.float64)
    class_weights = counts.sum() / np.maximum(counts, 1.0)
    class_weights = class_weights / class_weights.mean()
    print("[data] label counts=%s -> weights=%s" % (counts.astype(int).tolist(), np.round(class_weights, 3).tolist()), flush=True)

    sampler = None
    if args.balanced_sampler:
        sampler = WeightedRandomSampler(torch.as_tensor(class_weights[labels], dtype=torch.double),
                                        num_samples=len(labels), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    model = build_model(args.model, args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print("[model] %s params=%.2fM" % (args.model, n_params / 1e6), flush=True)

    if args.eval_only_ckpt:
        run_test_eval(model, args, frame_labels, device, out_dir, ckpt_path=args.eval_only_ckpt)
        return

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device),
                                    label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    history, best, no_improve, global_step = [], -1.0, 0, 0
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        t_epoch = time.time()
        for clip, label, _path, _start in train_loader:
            clip = clip.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                loss = criterion(model(clip), label)
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
                print("  ep%d step%d loss=%.4f (%.1f clips/s)"
                      % (epoch, global_step, running / max(1, seen), seen / max(1e-6, time.time() - t_epoch)), flush=True)
            if args.max_steps and global_step >= args.max_steps:
                break

        val_metrics = evaluate(model, val_loader, device)
        rec = {
            "epoch": epoch, "step": global_step, "train_loss": running / max(1, seen),
            "val_avg_f1": val_metrics.get("avg_f1", 0.0), "val_acc": val_metrics.get("ternary_acc", 0.0),
            "val_fall_f1": val_metrics.get("fall_f1", 0.0), "val_fallen_f1": val_metrics.get("fallen_f1", 0.0),
            "epoch_sec": time.time() - t_epoch, "clips_seen": seen,
        }
        history.append(rec)
        print("[epoch %d] loss=%.4f val_avg_f1=%.4f val_acc=%.4f (%.1f min)"
              % (epoch, rec["train_loss"], rec["val_avg_f1"], rec["val_acc"], rec["epoch_sec"] / 60.0), flush=True)

        if rec["val_avg_f1"] > best:
            best, no_improve = rec["val_avg_f1"], 0
            torch.save({"state_dict": model.state_dict(), "args": vars(args), "epoch": epoch},
                       os.path.join(out_dir, "best_model.pt"))
        else:
            no_improve += 1
        with open(os.path.join(out_dir, "training_history.json"), "w", encoding="utf-8") as fh:
            json.dump(history, fh, ensure_ascii=False, indent=2)
        scheduler.step()
        if args.early_stop_patience and no_improve >= args.early_stop_patience:
            print("[train] early stop: no val improvement for %d epoch(s)" % no_improve, flush=True)
            break

    print("[train] finished in %.1f min, best val avg_f1=%.4f" % ((time.time() - t_start) / 60.0, best), flush=True)
    best_path = os.path.join(out_dir, "best_model.pt")
    if not args.skip_test and os.path.isfile(best_path):
        run_test_eval(model, args, frame_labels, device, out_dir, ckpt_path=best_path)
    print("[train] logs -> %s" % out_dir, flush=True)


if __name__ == "__main__":
    main()
