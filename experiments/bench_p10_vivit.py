"""
Pure-GPU feasibility + throughput benchmark for ViViT-B/16x2 (32 frames, 224^2), Phase-10 baseline.

Route decision (docs/0911实验第十阶段规划.md §6.6): the official ViViT release is JAX/Flax + Scenic
with a DMVR/TFRecord data pipeline, which would mean a new environment plus a TFRecord writer.
This script instead uses the HuggingFace port of the same Kinetics-400 checkpoint
(`google/vivit-b-16x2-kinetics400`) — the identical shortcut that already worked for TimeSformer.

Measures, on synthetic input (no dataloader, so the numbers are the model's own cost):
  * TRAIN  = one AMP training step (that is the T2 fine-tune cost), and
  * EVAL   = one AMP forward pass in eval mode (that is the T1 linear-probe cost),
plus peak memory per batch size, so the §4.4.1 budget rule can be decided on evidence.

Run on the server:
    /root/miniconda3/bin/python -u experiments/bench_p10_vivit.py \
        --batch_sizes 2 4 8 --iters 3 --warmup 1 --load_weights
"""

from __future__ import annotations

import argparse
import os

# has to be set before huggingface_hub is imported (hf.co is unreachable from the server)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import time  # noqa: E402

import torch  # noqa: E402

CKPT = "google/vivit-b-16x2-kinetics400"


def build(num_frames=None, img_size=224, num_classes=3):
    from transformers import VivitConfig, VivitForVideoClassification

    cfg = VivitConfig.from_pretrained(CKPT)
    cfg.num_labels = num_classes
    if num_frames is not None:
        cfg.num_frames = num_frames
    if img_size is not None:
        cfg.image_size = img_size
    return cfg, VivitForVideoClassification(cfg)


def load_official_weights(net):
    """Same trick as the TimeSformer adapter: `transformers>=5` refuses to `torch.load` a `.bin`
    checkpoint under torch<2.6 (CVE-2025-32434), so build from the config and load the official
    K400 state dict by hand, dropping the 400-class head."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(CKPT, "pytorch_model.bin")
    print("[vivit] checkpoint file: %s (%.1f MB)"
          % (path, os.path.getsize(path) / 1e6), flush=True)
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        state = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state" in state:
        state = state["model_state"]
    state = {k: v for k, v in state.items() if not k.startswith("classifier.")}
    missing, unexpected = net.load_state_dict(state, strict=False)
    head = [k for k in missing if "classifier" in k]
    print("[vivit] K400 weights loaded (missing=%d, unexpected=%d, head re-init=%d)"
          % (len(missing), len(unexpected), len(head)), flush=True)
    if unexpected:
        print("[vivit] unexpected keys (first 5): %s" % unexpected[:5], flush=True)


def timed(model, x, y, train, iters, warmup, device):
    criterion = torch.nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9) if train else None
    scaler = torch.amp.GradScaler("cuda", enabled=train)
    model.train() if train else model.eval()
    torch.cuda.synchronize()
    t0 = None
    for i in range(warmup + iters):
        if i == warmup:
            torch.cuda.synchronize()
            t0 = time.time()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out = model(x).logits
            loss = criterion(out, y) if train else None
        if train:
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
    torch.cuda.synchronize()
    return time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_sizes", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--frames", type=int, default=None, help="default: the checkpoint's own window")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--load_weights", action="store_true")
    ap.add_argument("--skip_train", action="store_true", help="only measure the T1 (frozen) cost")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    print("GPU: %s, %.1f GB" % (props.name, props.total_memory / 1e9), flush=True)

    cfg, model = build(args.frames, args.img_size)
    n_params = sum(p.numel() for p in model.parameters())
    n_backbone = sum(p.numel() for n, p in model.named_parameters() if not n.startswith("classifier"))
    print("[config] num_frames=%s image_size=%s tubelet_size=%s patch_size=%s hidden=%s "
          "image_mean=%s image_std=%s"
          % (getattr(cfg, "num_frames", None), getattr(cfg, "image_size", None),
             getattr(cfg, "tubelet_size", None), getattr(cfg, "patch_size", None),
             getattr(cfg, "hidden_size", None),
             getattr(cfg, "image_mean", None), getattr(cfg, "image_std", None)), flush=True)
    print("[model] params=%.2f M (backbone %.2f M + head %.3f M)"
          % (n_params / 1e6, n_backbone / 1e6, (n_params - n_backbone) / 1e6), flush=True)
    if args.load_weights:
        load_official_weights(model)
    del model

    for bs in args.batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        cfg, model = build(args.frames, args.img_size)
        model = model.to(device)
        # HF ViViT consumes channel-last video tensors (B, T, C, H, W)
        x = torch.randn(bs, cfg.num_frames, 3, cfg.image_size, cfg.image_size, device=device)
        y = torch.randint(0, 3, (bs,), device=device)

        if not args.skip_train:
            try:
                dt = timed(model, x, y, True, args.iters, args.warmup, device)
                print("TRAIN  batch=%-3d %5.2f s / %d iters -> %6.1f clips/s  peak=%.2f GB"
                      % (bs, dt, args.iters, bs * args.iters / dt,
                         torch.cuda.max_memory_allocated() / 1e9), flush=True)
            except torch.cuda.OutOfMemoryError:
                print("TRAIN  batch=%-3d OOM" % bs, flush=True)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        try:
            with torch.no_grad():
                dt = timed(model, x, y, False, args.iters, args.warmup, device)
            print("EVAL   batch=%-3d %5.2f s / %d iters -> %6.1f clips/s  peak=%.2f GB"
                  % (bs, dt, args.iters, bs * args.iters / dt,
                     torch.cuda.max_memory_allocated() / 1e9), flush=True)
        except torch.cuda.OutOfMemoryError:
            print("EVAL   batch=%-3d OOM" % bs, flush=True)
        del model, x, y
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
