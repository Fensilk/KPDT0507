"""
Profile the Phase 10 TSM model to locate the cost of the official TemporalShift.

Usage:
    python experiments/profile_p10_tsm.py --is_shift 1
    python experiments/profile_p10_tsm.py --is_shift 0
"""

from __future__ import annotations

import argparse
import os
import sys

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--is_shift", type=int, default=1)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--crop", type=int, default=224)
    ap.add_argument("--iters", type=int, default=3)
    args = ap.parse_args()

    device = torch.device("cuda")
    model = TSN(num_class=3, num_segments=args.frames, modality="RGB", base_model="resnet50",
                new_length=1, consensus_type="avg", dropout=0.5, is_shift=bool(args.is_shift),
                shift_div=8, shift_place="blockres", pretrain=None, partial_bn=False,
                print_spec=False).to(device)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    x = torch.randn(args.batch, args.frames, 3, args.crop, args.crop, device=device)
    y = torch.randint(0, 3, (args.batch,), device=device)
    scaler = torch.amp.GradScaler("cuda")

    def step():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = criterion(model(x), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    for _ in range(3):
        step()
    torch.cuda.synchronize()

    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        for _ in range(args.iters):
            step()
        torch.cuda.synchronize()

    print("=== is_shift=%d ===" % args.is_shift)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=12))


if __name__ == "__main__":
    main()
