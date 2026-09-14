"""
Fast, mathematically equivalent replacement for TSM's official TemporalShift.

Official (ops/temporal_shift.py):
    out = torch.zeros_like(x)                       # full-tensor fill
    out[:, :-1, :fold] = x[:, 1:, :fold]            # left shift  -> CopySlices + slice_backward
    out[:, 1:, fold:2fold] = x[:, :-1, fold:2fold]  # right shift -> CopySlices + slice_backward
    out[:, :, 2fold:] = x[:, :, 2fold:]             # passthrough -> CopySlices + slice_backward

Fast (this file): build the same tensor out-of-place with narrow views + cat, which avoids
the full-tensor zero fill and the CopySlices/slice_backward autograd nodes; the backward of
cat/narrow is a cheap copy. Results are elementwise identical (see
experiments/verify_p10_fast_shift.py), so the two are interchangeable numerically.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def shift_fast(x: torch.Tensor, n_segment: int, fold_div: int = 8) -> torch.Tensor:
    nt, c, h, w = x.shape
    n_batch = nt // n_segment
    x5 = x.view(n_batch, n_segment, c, h, w)
    fold = c // fold_div

    left = x5[:, :, :fold]              # channels that shift to earlier frames
    right = x5[:, :, fold : 2 * fold]   # channels that shift to later frames
    rest = x5[:, :, 2 * fold :]         # channels that stay

    zero = x5.new_zeros((n_batch, 1, fold, h, w))
    left_shifted = torch.cat((left[:, 1:], zero), dim=1)    # out[t] = x[t+1], last frame = 0
    right_shifted = torch.cat((zero, right[:, :-1]), dim=1)  # out[t] = x[t-1], first frame = 0

    out = torch.cat((left_shifted, right_shifted, rest), dim=2)
    return out.reshape(nt, c, h, w)


class FastTemporalShift(nn.Module):
    """Same interface as ops.temporal_shift.TemporalShift, faster implementation."""

    def __init__(self, net, n_segment: int = 3, n_div: int = 8, **kwargs):
        super().__init__()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div

    def forward(self, x):
        return self.net(shift_fast(x, self.n_segment, self.fold_div))


def convert_to_fast_shift(model: nn.Module, verbose: bool = False) -> int:
    """Replace every official TemporalShift wrapper inside `model` with FastTemporalShift.

    The wrapped submodule (`net`) is preserved, so parameters / state_dict are unchanged.
    """
    from ops.temporal_shift import TemporalShift

    replaced = 0
    for module in model.modules():
        for name, child in list(module.named_children()):
            if isinstance(child, TemporalShift):
                setattr(module, name, FastTemporalShift(child.net, child.n_segment, child.fold_div))
                replaced += 1
    if verbose:
        print("=> converted %d TemporalShift modules to FastTemporalShift" % replaced)
    return replaced
