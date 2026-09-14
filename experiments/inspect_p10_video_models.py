"""
Inspect the pytorchvideo model zoo entries we plan to use as Phase-10 baselines:
input clip length / crop size, head structure, and how to swap the classifier to 3 classes.

Run on the server:
    /root/miniconda3/bin/python -u experiments/inspect_p10_video_models.py
"""

import torch
import torch.nn as nn
import pytorchvideo.models.hub as hub


def describe(name, model, shapes):
    print("=" * 70)
    print(name)
    head = model.blocks[-1]
    print("  head:", type(head).__name__)
    for attr in ("proj", "activation", "pool", "dropout", "output"):
        if hasattr(head, attr):
            mod = getattr(head, attr)
            info = type(mod).__name__
            if isinstance(mod, (nn.Linear, nn.Conv3d)):
                info += "  out_features=%d" % mod.out_features
            print("    .%s: %s" % (attr, info))
    n_params = sum(p.numel() for p in model.parameters())
    print("  params: %.2f M" % (n_params / 1e6))
    model.eval()
    for shape in shapes:
        try:
            with torch.no_grad():
                out = model(torch.zeros(*shape))
            print("  forward %s -> %s" % (tuple(shape), tuple(out.shape)))
        except Exception as exc:  # noqa: BLE001
            print("  forward %s -> FAILED: %s" % (tuple(shape), str(exc)[:120]))


def main():
    print("pytorchvideo", __import__("pytorchvideo").__version__, "| torch", torch.__version__)

    m = hub.x3d_s(pretrained=False)
    describe("x3d_s", m, [(1, 3, 13, 160, 160), (1, 3, 16, 224, 224)])

    m = hub.x3d_m(pretrained=False)
    describe("x3d_m", m, [(1, 3, 16, 224, 224)])

    m = hub.slowfast_r50(pretrained=False)
    describe("slowfast_r50", m, [(1, 3, 32, 224, 224), (1, 3, 16, 224, 224)])

    m = hub.i3d_r50(pretrained=False)
    describe("i3d_r50", m, [(1, 3, 8, 224, 224), (1, 3, 16, 224, 224)])


if __name__ == "__main__":
    main()
