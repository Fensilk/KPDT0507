"""
Pure-GPU throughput benchmark for the Phase 10 TSM model (no dataloader, synthetic input).

Purpose: separate GPU compute cost from the CPU dataloader cost, so we can estimate
full-run wall-clock on the local RTX 4060 and on the cloud RTX 3090.

Usage:
    python experiments/bench_p10_tsm_gpu.py --batch_sizes 8 16 32 --iters 20
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)
sys.path.insert(0, os.path.join(PROJ_ROOT, "reference", "temporal-shift-module"))

import torchvision.models as tvm  # noqa: E402

_r50 = tvm.resnet50


def _resnet50_compat(pretrained=False, **kwargs):
    weights = tvm.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    return _r50(weights=weights, **kwargs)


tvm.resnet50 = _resnet50_compat

from ops.models import TSN  # noqa: E402


def build_model(num_frames=8, is_shift=True):
    return TSN(num_class=3, num_segments=num_frames, modality="RGB", base_model="resnet50",
               new_length=1, consensus_type="avg", dropout=0.5, is_shift=is_shift,
               shift_div=8, shift_place="blockres", pretrain=None, partial_bn=False, print_spec=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_sizes", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--crop", type=int, default=224)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--plain_resnet", action="store_true", help="benchmark a bare ResNet50 on B*frames images")
    ap.add_argument("--channels_last", action="store_true")
    ap.add_argument("--cudnn_benchmark", action="store_true")
    ap.add_argument("--no_shift", action="store_true", help="build the same network without temporal shift")
    ap.add_argument("--fast_shift", action="store_true", help="use the equivalent cat-based shift implementation")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    print("GPU: %s, %.1f GB, sm_%d%d" % (props.name, props.total_memory / 1e9, props.major, props.minor))
    if args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    for bs in args.batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = build_model(args.frames, is_shift=not args.no_shift).to(device)
        if args.fast_shift:
            sys.path.insert(0, os.path.join(PROJ_ROOT, "experiments"))
            from fast_tsm_shift import convert_to_fast_shift

            convert_to_fast_shift(model)
        if args.plain_resnet:
            model = _r50(weights=None).to(device)
        if args.channels_last:
            model = model.to(memory_format=torch.channels_last)
        model.train()
        opt = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        criterion = torch.nn.CrossEntropyLoss()
        if args.plain_resnet:
            x = torch.randn(bs * args.frames, 3, args.crop, args.crop, device=device)
        else:
            x = torch.randn(bs, args.frames, 3, args.crop, args.crop, device=device)
        if args.channels_last and args.plain_resnet:
            x = x.to(memory_format=torch.channels_last)
        if args.plain_resnet:
            y = torch.randint(0, 1000, (bs * args.frames,), device=device)
        else:
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
            cps = bs * args.iters / dt
            print("batch=%-3d  %.2f s for %d iters  ->  %.1f clips/s  peak_mem=%.2f GB"
                  % (bs, dt, args.iters, cps, torch.cuda.max_memory_allocated() / 1e9))
        except torch.cuda.OutOfMemoryError:
            print("batch=%-3d  OOM" % bs)
        del model


if __name__ == "__main__":
    main()
