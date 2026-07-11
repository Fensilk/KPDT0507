"""
Step 7: Extract DINOv2 ViT-L/14 frame-level features from OF-Syn videos.

Uses facebook/dinov2-giant (ViT-g/14, 1536-dim) to extract per-frame CLS token.
Each video (80 frames) → [80, 1536] feature matrix.
Aggregates to frame-level NPZ compatible with LongSequenceDataset.

Output per-video: data/dinov2_giant_features/{category}/{video}.pt
    features:   (80, 1536) float16
    labels:     (80,) int64
    fall_label: (80,) int64
    fallen_label:(80,) int64

Aggregated NPZ: data/omnifall_dinov2_giant_frame.npz
    features:           (960000, 1536) float16
    labels_16:          (960000,) int64
    fall_labels:        (960000,) int64
    fallen_labels:      (960000,) int64
    video_paths:        (12000,) object
    video_clip_counts:  (12000,) int64 (all 80)
    video_start_indices:(12000,) int64

Checkpointing: skips videos already in data/dinov2_giant_features/.

Usage:
    python preprocessing/step7_extract_dinov2.py
    python preprocessing/step7_extract_dinov2.py --category fall  # single category
    python preprocessing/step7_extract_dinov2.py --batch_size 16
"""

import os
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_DIR = os.path.join(ROOT, "DATASET-omnifall", "data_files", "extracted")
PT_OUTPUT_DIR = os.path.join(ROOT, "data", "dinov2_giant_features")
NPZ_OUTPUT_PATH = os.path.join(ROOT, "data", "omnifall_dinov2_giant_frame.npz")
FRAME_CSV_PATH = os.path.join(ROOT, "data", "ofsyn_frame_labels.csv")
N_FRAMES = 80  # frames per video

# Default args (module-level for multiprocessing compatibility)
DEFAULT_MODEL_NAME = "facebook/dinov2-giant"


# DINOv2 preprocessing constants (bypasses AutoImageProcessor, network-independent)
DINOV2_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DINOV2_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DINOV2_SIZE = (224, 224)


def preprocess_frames(frames: list, device: torch.device) -> torch.Tensor:
    """
    Manual preprocessing for DINOv2: resize, normalize, convert to tensor.
    Network-independent — no AutoImageProcessor needed.
    """
    import torch.nn.functional as F

    # Resize all frames to 224x224
    resized = [cv2.resize(f, DINOV2_SIZE, interpolation=cv2.INTER_LINEAR) for f in frames]

    # Stack to (B, H, W, C), convert to (B, C, H, W), normalize
    batch = np.stack(resized, axis=0).astype(np.float32) / 255.0  # (B, H, W, C)
    batch = np.transpose(batch, (0, 3, 1, 2))                     # (B, C, H, W)
    batch = (batch - DINOV2_MEAN[None, :, None, None]) / DINOV2_STD[None, :, None, None]

    return torch.from_numpy(batch).to(device)


def extract_dinov2_features(video_path: str, model, device,
                            batch_size: int = 8) -> np.ndarray:
    """
    Extract DINOv2 CLS token for all frames in a video.

    Args:
        video_path: path to .mp4 file.
        model: DINOv2 ViT-L model (fp16, eval mode).
        device: torch device.
        batch_size: frames per batch.

    Returns:
        features: (80, 1024) float16 numpy array, or None on failure.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    # Pad/trim to exactly N_FRAMES
    if len(frames) < N_FRAMES:
        while len(frames) < N_FRAMES:
            frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))
    frames = frames[:N_FRAMES]

    # Process in batches
    all_cls = []
    for i in range(0, N_FRAMES, batch_size):
        batch_frames = frames[i:i + batch_size]
        pixel_values = preprocess_frames(batch_frames, device)  # (B, 3, 224, 224)

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                outputs = model(pixel_values)
                cls_token = outputs.last_hidden_state[:, 0, :]  # (B, 1024)
            all_cls.append(cls_token.cpu().to(torch.float16).numpy())

    return np.concatenate(all_cls, axis=0).astype(np.float16)


def process_video(video_path: str, rel_path: str, labels_dict: dict,
                  model, device, batch_size: int = 8):
    """
    Process a single video.

    Args:
        video_path: absolute path to .mp4.
        rel_path: relative path like "fall/fall_ch_001.mp4".
        labels_dict: maps rel_path -> DataFrame row subset.
        model: DINOv2 model (already loaded).
        device: torch device.
        batch_size: frames per batch.

    Returns:
        (rel_path, success: bool, message: str)
    """
    out_path = os.path.join(PT_OUTPUT_DIR, rel_path.replace(".mp4", ".pt"))

    # Checkpoint
    if os.path.exists(out_path):
        return (rel_path, True, "skipped")

    try:
        features = extract_dinov2_features(video_path, model, device, batch_size)
        if features is None:
            return (rel_path, False, "frame extraction failed")

        # Get labels from CSV (CSV path has no .mp4 extension)
        csv_key = rel_path.replace(".mp4", "")
        rows = labels_dict.get(csv_key)
        if rows is None:
            return (rel_path, False, f"labels not found in CSV (key: {csv_key})")

        labels = rows["label"].values.astype(np.int64)
        fall_label = rows["fall_label"].values.astype(np.int64)
        fallen_label = rows["fallen_label"].values.astype(np.int64)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save({
            "features": torch.from_numpy(features),
            "labels": torch.from_numpy(labels),
            "fall_label": torch.from_numpy(fall_label),
            "fallen_label": torch.from_numpy(fallen_label),
        }, out_path)

        return (rel_path, True, "OK")
    except Exception as e:
        return (rel_path, False, str(e))


def aggregate_to_npz(pt_dir: str, npz_path: str):
    """Aggregate all per-video .pt files into a single frame-level NPZ."""
    import glob as _glob
    pt_files = sorted(_glob.glob(os.path.join(pt_dir, "**", "*.pt"), recursive=True))

    if len(pt_files) == 0:
        print("[ERROR] No .pt files found!")
        return

    n_videos = len(pt_files)
    features_list, labels_list, fall_list, fallen_list = [], [], [], []
    video_paths = []
    video_frame_counts = np.empty(n_videos, dtype=np.int64)
    video_start_indices = np.empty(n_videos, dtype=np.int64)

    offset = 0
    for i, ptf in enumerate(pt_files):
        rel = os.path.relpath(ptf, pt_dir).replace("\\", "/").replace(".pt", "")
        d = torch.load(ptf, map_location="cpu", weights_only=False)

        feats = d["features"].numpy()
        features_list.append(feats)
        labels_list.append(d["labels"].numpy())
        fall_list.append(d["fall_label"].numpy())
        fallen_list.append(d["fallen_label"].numpy())
        video_paths.append(rel)
        video_start_indices[i] = offset
        video_frame_counts[i] = len(d["labels"])
        offset += len(d["labels"])

        if (i + 1) % 2000 == 0:
            print(f"  Aggregated {i+1:,}/{n_videos:,} videos...")

    # Stack
    all_features = np.concatenate(features_list, axis=0)
    all_labels = np.concatenate(labels_list, axis=0)
    all_fall = np.concatenate(fall_list, axis=0)
    all_fallen = np.concatenate(fallen_list, axis=0)

    # Save NPZ
    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    np.savez_compressed(
        npz_path,
        features=all_features,
        labels_16=all_labels,
        fall_labels=all_fall,
        fallen_labels=all_fallen,
        video_paths=np.array(video_paths, dtype=object),
        video_clip_counts=video_frame_counts,
        video_start_indices=video_start_indices,
    )

    size_mb = os.path.getsize(npz_path) / 1024 / 1024
    print(f"\n[DONE] NPZ saved: {npz_path}")
    print(f"  Videos: {n_videos}, Frames: {offset}")
    print(f"  Features: [{offset}, {all_features.shape[1]}], dtype={all_features.dtype}")
    print(f"  File size: {size_mb:.1f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Extract DINOv2 ViT-L/14 frame-level features"
    )
    parser.add_argument("--category", type=str, default=None,
                        help="Process single category (default: all)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Frames per DINOv2 inference batch")
    parser.add_argument("--skip_aggregation", action="store_true",
                        help="Skip NPZ aggregation (extraction only)")
    args = parser.parse_args()

    # --- Collect video paths ---
    categories = sorted(os.listdir(VIDEO_DIR))
    if args.category:
        if args.category not in categories:
            print(f"[ERROR] Category '{args.category}' not found. Available: {categories}")
            return
        categories = [args.category]

    all_videos = []
    for cat in categories:
        cat_dir = os.path.join(VIDEO_DIR, cat)
        if not os.path.isdir(cat_dir):
            continue
        for vname in sorted(os.listdir(cat_dir)):
            if vname.endswith(".mp4"):
                rel = f"{cat}/{vname}"
                all_videos.append((os.path.join(cat_dir, vname), rel))

    # --- Load frame labels ---
    print(f"Loading frame labels from {FRAME_CSV_PATH}...")
    frame_df = pd.read_csv(FRAME_CSV_PATH)
    labels_dict = {}
    for path, grp in frame_df.groupby("path"):
        labels_dict[path] = grp.sort_values("frame_idx")

    # --- Checkpoint counts ---
    already_done = sum(
        1 for _, rel in all_videos
        if os.path.exists(os.path.join(PT_OUTPUT_DIR, rel.replace(".mp4", ".pt")))
    )
    n_total = len(all_videos)
    n_todo = n_total - already_done

    print(f"Total videos: {n_total:,}")
    print(f"Already done: {already_done:,}")
    print(f"Remaining:    {n_todo:,}")
    print(f"Categories:   {categories}")
    print(f"Batch size:   {args.batch_size}")

    if n_todo == 0:
        print("[DONE] All videos already processed!")
        if not args.skip_aggregation:
            aggregate_to_npz(PT_OUTPUT_DIR, NPZ_OUTPUT_PATH)
        return

    # --- Load DINOv2 model ONCE ---
    print(f"\nLoading DINOv2 model: {DEFAULT_MODEL_NAME}...")
    from transformers import AutoModel
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = AutoModel.from_pretrained(DEFAULT_MODEL_NAME).to(device)
    model = model.half().eval()
    print("Model loaded (manual preprocessing, no AutoImageProcessor needed).")

    # Estimate time
    est_per_video = 1.5  # seconds
    est_total = n_todo * est_per_video
    print(f"Est. time:    {est_total/60:.0f} min (~{est_total/3600:.1f}h)")

    # --- Process ---
    print(f"\nProcessing...")
    t0 = time.time()
    n_ok, n_fail, n_skip = 0, 0, 0

    for i, (vp, rel) in enumerate(all_videos):
        rel_path, success, msg = process_video(
            vp, rel, labels_dict, model, device, args.batch_size)
        if success:
            if msg == "skipped":
                n_skip += 1
            else:
                n_ok += 1
        else:
            n_fail += 1
            print(f"  [FAIL] {rel_path}: {msg}", flush=True)

        done = i + 1
        if done % 100 == 0 or done == n_total:
            elapsed = time.time() - t0
            rate = (n_ok + n_skip) / elapsed if elapsed > 0 else 0
            eta = (n_todo - n_ok) / rate - elapsed if rate > 0 else 0
            eta = max(0, eta)
            print(f"  [{done:>6d}/{n_total}] "
                  f"{100*done/n_total:5.1f}% | "
                  f"OK={n_ok} SKIP={n_skip} FAIL={n_fail} | "
                  f"{rate:.2f} vid/s | "
                  f"elapsed={elapsed/60:.0f}min | "
                  f"eta={eta/60:.0f}min",
                  flush=True)

    total_time = time.time() - t0
    print(f"\n{'='*60}")
    print(f"DONE! Total time: {total_time/60:.1f} min ({total_time/3600:.2f}h)")
    print(f"Successful: {n_ok}, Skipped: {n_skip}, Failed: {n_fail}")
    print(f"Output dir: {PT_OUTPUT_DIR}")

    # --- Aggregate to NPZ ---
    if not args.skip_aggregation:
        print(f"\n{'='*60}")
        print("Aggregating to NPZ...")
        print(f"{'='*60}")
        aggregate_to_npz(PT_OUTPUT_DIR, NPZ_OUTPUT_PATH)


if __name__ == "__main__":
    main()
