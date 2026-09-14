"""
Benchmark for Phase 10 frame extraction: PyAV software decode vs NVDEC (CUDA) hardware decode,
with and without JPEG encoding, so we can see where the extraction time actually goes.

Usage:
    python experiments/bench_p10_decode.py --n 15 --mode all
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import av
import cv2
import numpy as np

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_ROOT = os.path.join(PROJ_ROOT, "DATASET-omnifall", "data_files", "extracted")


def collect_videos(n):
    out = []
    for cls in sorted(os.listdir(VIDEO_ROOT)):
        cls_dir = os.path.join(VIDEO_ROOT, cls)
        if not os.path.isdir(cls_dir):
            continue
        for name in sorted(os.listdir(cls_dir))[:3]:
            if name.endswith(".mp4"):
                out.append(os.path.join(cls_dir, name))
            if len(out) >= n:
                return out
    return out[:n]


def decode_software(path, n_frames=81):
    container = av.open(path)
    frames = []
    try:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) >= n_frames:
                break
    finally:
        container.close()
    return frames


def decode_hwaccel(path, n_frames=81, hwaccel="cuda"):
    # PyAV >= 14 wants the hwaccel passed through options (a plain string is rejected).
    container = av.open(path, options={"hwaccel": hwaccel})
    frames = []
    try:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))  # downloads from GPU to CPU
            if len(frames) >= n_frames:
                break
    finally:
        container.close()
    return frames


def resize_center_crop(frame, size=256):
    h, w = frame.shape[:2]
    scale = size / float(min(h, w))
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    y0, x0 = (nh - size) // 2, (nw - size) // 2
    return resized[y0 : y0 + size, x0 : x0 + size]


def bench(name, videos, decode_fn, with_jpeg=False, jpeg_out=None):
    t0 = time.time()
    total = 0
    for path in videos:
        frames = decode_fn(path)
        total += len(frames)
        if with_jpeg:
            for i, f in enumerate(frames, start=1):
                img = resize_center_crop(f)
                cv2.imwrite(
                    os.path.join(jpeg_out, "img_%05d.jpg" % i),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                    [int(cv2.IMWRITE_JPEG_QUALITY), 92],
                )
    dt = time.time() - t0
    print(
        "%-22s videos=%d frames=%d  %.1f s  %.2f vid/s  %.1f frame/s"
        % (name, len(videos), total, dt, len(videos) / dt, total / dt)
    )
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--mode", default="all", choices=["all", "sw", "hw"])
    args = ap.parse_args()

    videos = collect_videos(args.n)
    print("benchmark videos: %d (AV1 1280x720, 81 frames each)" % len(videos))
    if not videos:
        print("no videos found")
        return

    tmp_out = os.path.join(PROJ_ROOT, "data", "_bench_jpeg")
    os.makedirs(tmp_out, exist_ok=True)

    if args.mode in ("all", "sw"):
        bench("decode CPU (software)", videos, decode_software)
        bench("decode CPU + JPEG", videos, decode_software, with_jpeg=True, jpeg_out=tmp_out)
    if args.mode in ("all", "hw"):
        try:
            bench("decode NVDEC (cuda)", videos, decode_hwaccel)
            bench("decode NVDEC + JPEG", videos, decode_hwaccel, with_jpeg=True, jpeg_out=tmp_out)
        except Exception as exc:  # noqa: BLE001
            print("NVDEC path failed: %s: %s" % (type(exc).__name__, exc))


if __name__ == "__main__":
    main()
