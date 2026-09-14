"""
Phase 10 dense frame extraction for reference video models (TSM / X3D / I3D / TimeSformer / ViViT).

Output layout (TSM convention):
    data/ref_frames/<class>/<video>/img_00001.jpg ... img_00081.jpg

Source videos: DATASET-omnifall/data_files/extracted/<class>/<video>.mp4
(1280x720 / 16 fps / 81 frames / AV1, decoded with PyAV because cv2 cannot read AV1).
Each frame: resize short side to --size (default 256), then center-crop to size x size, saved as JPEG.
Models resize/crop further themselves (224 / 160 / 112).
Resumable: a video whose output dir already holds the full 81 frames is skipped.

Usage:
    python experiments/prep_p10_frames.py --workers 8
    python experiments/prep_p10_frames.py --workers 8 --limit-per-class 20
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)
sys.path.insert(0, os.path.join(PROJ_ROOT, "preprocessing"))

from av_decode import read_rgb_frames  # noqa: E402

VIDEO_ROOT = os.path.join(PROJ_ROOT, "DATASET-omnifall", "data_files", "extracted")
SPLIT_ROOT = os.path.join(PROJ_ROOT, "DATASET-omnifall", "splits", "syn", "random")
DEFAULT_OUT = os.path.join(PROJ_ROOT, "data", "ref_frames")
N_FRAMES = 81


def resize_center_crop(frame, size: int):
    """Resize so the short side equals size, then center-crop to size x size."""
    h, w = frame.shape[:2]
    scale = size / float(min(h, w))
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    y0 = (nh - size) // 2
    x0 = (nw - size) // 2
    return resized[y0 : y0 + size, x0 : x0 + size]


def squash_resize(frame, size: int):
    """Resize the FULL frame to size x size (aspect distorted, but nothing is cropped).

    This matches the geometry the main line uses for DINOv2 features
    (preprocessing/step7_extract_dinov2.py resizes the whole frame to 224x224), so the reference
    models see the same field of view instead of losing ~44% of the frame width to a center crop.
    """
    return cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)


def extract_one(job):
    rel_path, out_root, size, quality, overwrite, geometry = job
    src = os.path.join(VIDEO_ROOT, rel_path + ".mp4")
    out_dir = os.path.join(out_root, rel_path)

    if not os.path.isfile(src):
        return {"path": rel_path, "status": "missing_video", "n": 0}

    if not overwrite and os.path.isdir(out_dir):
        existing = [f for f in os.listdir(out_dir) if f.endswith(".jpg")]
        if len(existing) == N_FRAMES:
            return {"path": rel_path, "status": "skip", "n": N_FRAMES}

    frames = read_rgb_frames(src, N_FRAMES)
    if not frames:
        return {"path": rel_path, "status": "decode_failed", "n": 0}

    os.makedirs(out_dir, exist_ok=True)
    for idx, frame in enumerate(frames, start=1):
        img = squash_resize(frame, size) if geometry == "squash" else resize_center_crop(frame, size)
        cv2.imwrite(
            os.path.join(out_dir, "img_%05d.jpg" % idx),
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), quality],
        )
    return {"path": rel_path, "status": "ok", "n": len(frames)}


def collect_paths(splits, limit, limit_per_class, classes):
    paths = []
    for split in splits:
        csv_path = os.path.join(SPLIT_ROOT, split + ".csv")
        with open(csv_path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                paths.append(row["path"].strip())

    if classes:
        keep = set(classes)
        paths = [p for p in paths if p.split("/")[0] in keep]

    if limit_per_class:
        counter = {}
        picked = []
        for p in paths:
            cls = p.split("/")[0]
            if counter.get(cls, 0) < limit_per_class:
                counter[cls] = counter.get(cls, 0) + 1
                picked.append(p)
        paths = picked
    elif limit:
        paths = paths[:limit]

    seen = set()
    uniq = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def main():
    ap = argparse.ArgumentParser(description="Phase 10 dense frame extraction")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--classes", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--limit-per-class", type=int, default=None)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--quality", type=int, default=92)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--geometry", default="crop", choices=["crop", "squash"],
                    help="crop: short-side resize + center crop (TSM official); "
                         "squash: resize the FULL frame to size x size, nothing cropped (matches the main line's FOV)")
    args = ap.parse_args()

    paths = collect_paths(args.splits, args.limit, args.limit_per_class, args.classes)
    print("[prep_p10_frames] %d videos, workers=%d, size=%d" % (len(paths), args.workers, args.size))
    if not paths:
        return

    jobs = [(p, args.out, args.size, args.quality, args.overwrite, args.geometry) for p in paths]
    os.makedirs(args.out, exist_ok=True)

    t0 = time.time()
    stats = {"ok": 0, "skip": 0, "missing_video": 0, "decode_failed": 0}
    done = 0
    manifest_rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(extract_one, j) for j in jobs]
        for fut in as_completed(futures):
            res = fut.result()
            stats[res["status"]] = stats.get(res["status"], 0) + 1
            done += 1
            if res["status"] in ("ok", "skip"):
                manifest_rows.append(
                    {
                        "path": res["path"],
                        "class": res["path"].split("/")[0],
                        "frames_dir": os.path.join(args.out, res["path"]),
                        "n_frames": res["n"],
                    }
                )
            if done % 50 == 0 or done == len(jobs):
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0.0
                print(
                    "  %d/%d  %.1f vid/s  ok=%d skip=%d decode_failed=%d missing=%d"
                    % (done, len(jobs), rate, stats["ok"], stats["skip"], stats["decode_failed"], stats["missing_video"]),
                    flush=True,
                )

    print("[prep_p10_frames] done in %.1f min, %s" % ((time.time() - t0) / 60.0, stats))
    if manifest_rows:
        manifest = os.path.join(PROJ_ROOT, "data", "ref_frames_manifest.csv")
        manifest_exists = os.path.isfile(manifest)
        mode = "a" if manifest_exists else "w"
        with open(manifest, mode, newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["path", "class", "frames_dir", "n_frames"])
            if not manifest_exists:
                writer.writeheader()
            writer.writerows(manifest_rows)
        print("[prep_p10_frames] manifest -> %s (+%d rows)" % (manifest, len(manifest_rows)))


if __name__ == "__main__":
    main()
