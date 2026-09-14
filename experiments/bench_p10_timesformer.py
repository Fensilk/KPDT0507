"""
Pure-GPU throughput benchmark for TimeSformer (ViT-B, 8 frames, 224^2) as a Phase-10 baseline.

Synthetic input, no dataloader, so the number is the model's own cost — used to pick the epoch
budget via the §4.4.1 rule before committing GPU hours.

Run on the server:
    /root/miniconda3/bin/python -u experiments/bench_p10_timesformer.py --batch_sizes 4 8 16 --iters 5
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)
sys.path.insert(0, os.path.join(PROJ_ROOT, "reference", "TimeSformer"))

# skip the pretrained download for the throughput measurement (weights are irrelevant here)
import timesformer.models.vit as vit_mod  # noqa: E402

vit_mod.load_pretrained = lambda *args, **kwargs: None

from timesformer.models.vit import TimeSformer  # noqa: E402


def build(num_frames=8, num_classes=3, img_size=224, attention_type="divided_space_time"):
    return TimeSformer(img_size=img_size, patch_size=16, num_classes=num_classes,
                       num_frames=num_frames, attention_type=attention_type)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_sizes", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    print("GPU: %s, %.1f GB" % (props.name, props.total_memory / 1e9))

    for bs in args.batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = build(args.frames, 3, args.img_size).to(device)
        model.train()
        opt = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        criterion = torch.nn.CrossEntropyLoss()
        # TimeSformer expects channel-first 5D input (B, C, T, H, W)
        x = torch.randn(bs, 3, args.frames, args.img_size, args.img_size, device=device)
        y = torch.randint(0, 3, (bs,), device=device)
        scaler = torch.amp.GradScaler("cuda")
        try:
            for i in range(args.warmup + args.iters):
                if i == args.warmup:
                    torch.cuda.synchronize()
                    t0 = time.time()
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    loss = criterion(model(x), y)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            torch.cuda.synchronize()
            dt = time.time() - t0
            n_params = sum(p.numel() for p in model.parameters())
            print("batch=%-3d  %.2f s for %d iters  ->  %.1f clips/s  peak_mem=%.2f GB  params=%.1f M"
                  % (bs, dt, args.iters, bs * args.iters / dt,
                     torch.cuda.max_memory_allocated() / 1e9, n_params / 1e6), flush=True)
        except torch.cuda.OutOfMemoryError:
            print("batch=%-3d  OOM" % bs, flush=True)
        del model


if __name__ == "__main__":
    main()
