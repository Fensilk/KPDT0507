"""
Verify that FastTemporalShift is numerically equivalent to TSM's official TemporalShift,
then benchmark both (module-level and full-model level).

Usage:
    python experiments/verify_p10_fast_shift.py            # equivalence + timing
    python experiments/verify_p10_fast_shift.py --iters 8  # longer timing
"""

from __future__ import annotations

import argparse
import copy
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
from ops.temporal_shift import TemporalShift  # noqa: E402

sys.path.insert(0, os.path.join(PROJ_ROOT, "experiments"))
from fast_tsm_shift import convert_to_fast_shift, shift_fast  # noqa: E402


def build_model(frames=8):
    return TSN(num_class=3, num_segments=frames, modality="RGB", base_model="resnet50",
               new_length=1, consensus_type="avg", dropout=0.5, is_shift=True,
               shift_div=8, shift_place="blockres", pretrain=None, partial_bn=False, print_spec=False)


def check_shift_module():
    """Compare the shift op alone on representative shapes."""
    torch.manual_seed(0)
    cases = [(16, 256, 56, 56, 8), (16, 64, 56, 56, 8), (8, 128, 28, 28, 8), (8, 16, 14, 14, 8)]
    worst = 0.0
    for nt, c, h, w, t in cases:
        x = torch.randn(nt, c, h, w, dtype=torch.float64)
        ref = TemporalShift.shift(x.clone(), t, fold_div=8, inplace=False)
        fast = shift_fast(x.clone(), t, fold_div=8)
        diff = (ref - fast).abs().max().item()
        worst = max(worst, diff)
        print("  shift op  shape=(%d,%d,%d,%d) T=%d  max|diff|=%.3e" % (nt, c, h, w, t, diff))
    return worst


def check_model(device, frames=8, batch=4, crop=224):
    """Compare forward output and input gradients of official vs fast TSN-TSM."""
    torch.manual_seed(42)
    model_ref = build_model(frames).to(device)
    model_fast = copy.deepcopy(model_ref)
    n = convert_to_fast_shift(model_fast)
    print("  converted %d shift modules" % n)

    model_ref.eval()
    model_fast.eval()

    x = torch.randn(batch, frames, 3, crop, crop, device=device)
    x_ref = x.clone().requires_grad_(True)
    x_fast = x.clone().requires_grad_(True)

    y_ref = model_ref(x_ref)
    y_fast = model_fast(x_fast)
    fwd_diff = (y_ref - y_fast).abs().max().item()

    # gradient wrt input through the whole network
    loss_ref = (y_ref ** 2).mean()
    loss_fast = (y_fast ** 2).mean()
    g_ref = torch.autograd.grad(loss_ref, x_ref)[0]
    g_fast = torch.autograd.grad(loss_fast, x_fast)[0]
    bwd_diff = (g_ref - g_fast).abs().max().item()

    # parameter gradient comparison (needs a real backward pass)
    model_ref.zero_grad(set_to_none=True)
    model_fast.zero_grad(set_to_none=True)
    ((model_ref(x.clone()) ** 2).mean()).backward()
    ((model_fast(x.clone()) ** 2).mean()).backward()
    p_ref = dict(model_ref.named_parameters())
    p_fast = dict(model_fast.named_parameters())
    pg_diff = 0.0
    for k in p_ref:
        if k not in p_fast:
            continue
        g1, g2 = p_ref[k].grad, p_fast[k].grad
        if g1 is None or g2 is None:
            continue
        pg_diff = max(pg_diff, (g1 - g2).abs().max().item())
    print("  model forward  max|diff|=%.3e" % fwd_diff)
    print("  input grad     max|diff|=%.3e" % bwd_diff)
    print("  param grad     max|diff|=%.3e" % pg_diff)
    return fwd_diff, bwd_diff


def time_model(device, model, iters, batch=16, frames=8, crop=224, warmup=2):
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    x = torch.randn(batch, frames, 3, crop, crop, device=device)
    y = torch.randint(0, 3, (batch,), device=device)
    scaler = torch.amp.GradScaler("cuda")
    for _ in range(warmup):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = criterion(model(x), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = criterion(model(x), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    torch.cuda.synchronize()
    dt = time.time() - t0
    return batch * iters / dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--only", default="all", choices=["all", "none", "official", "fast"],
                    help="time only one variant in a fresh process (cleanest measurements)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.only != "all":
        from ops.models import TSN as _TSN

        torch.manual_seed(0)
        if args.only == "none":
            model = _TSN(num_class=3, num_segments=8, modality="RGB", base_model="resnet50",
                         new_length=1, consensus_type="avg", dropout=0.5, is_shift=False,
                         shift_div=8, shift_place="blockres", pretrain=None, partial_bn=False,
                         print_spec=False).to(device)
        else:
            model = build_model().to(device)
            if args.only == "fast":
                convert_to_fast_shift(model)
        cps = time_model(device, model, args.iters, batch=args.batch)
        print("RESULT %s: %.1f clips/s" % (args.only, cps))
        return

    print("=== 1) shift op equivalence (float64) ===")
    worst_op = check_shift_module()

    print("=== 2) full-model equivalence ===")
    check_model(device)

    print("=== 3) throughput on %s ===" % torch.cuda.get_device_name(0))
    torch.manual_seed(0)
    m_off = build_model().to(device)
    m_fast = copy.deepcopy(m_off)
    convert_to_fast_shift(m_fast)
    from ops.models import TSN as _TSN

    m_none = _TSN(num_class=3, num_segments=8, modality="RGB", base_model="resnet50",
                  new_length=1, consensus_type="avg", dropout=0.5, is_shift=False,
                  shift_div=8, shift_place="blockres", pretrain=None, partial_bn=False,
                  print_spec=False).to(device)

    for name, model in (("no shift", m_none), ("official shift", m_off), ("fast shift", m_fast)):
        cps = time_model(device, model, args.iters, batch=args.batch)
        print("  %-16s %.1f clips/s" % (name, cps))
        torch.cuda.empty_cache()

    print("worst shift-op diff: %.3e" % worst_op)


if __name__ == "__main__":
    main()
