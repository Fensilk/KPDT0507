"""
Step 8: Aggregate per-frame pose .pt files into frame-level NPZ.

Reads data/pose_features/**/*.pt (80 frames x 99 landmarks per video),
preserves frame-level resolution (NO clip pooling), and saves as:
    data/omnifall_pose_frame.npz

Unlike step6 (which pools 80 frames → 5 clips via mean), this script
keeps all 80 frames per video for direct compatibility with the
frame-level DINOv2 NPZ and LongSequenceDataset.

NPZ keys:
    pose_features:       (960000, 99)  float32
    video_paths:         (12000,)      object (str)
    video_clip_counts:   (12000,)      int64 (all 80)
    video_start_indices: (12000,)      int64 (0, 80, 160, ...)

Ordering matches omnifall_dinov2_frame.npz (alphabetical by path).

Usage:
    python preprocessing/step8_aggregate_pose_framelvl.py
"""

import os
import sys
import glob
import argparse
from pathlib import Path

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POSE_DIR = os.path.join(ROOT, "data", "pose_features")
OUTPUT_PATH = os.path.join(ROOT, "data", "omnifall_pose_frame.npz")
REF_NPZ_PATH = os.path.join(ROOT, "data", "omnifall_frame_preprocessed.npz")

FRAMES_PER_VIDEO = 80  # 80 frames per video, NO clip aggregation


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate per-frame pose to frame-level NPZ"
    )
    parser.add_argument("--pose_dir", type=str, default=POSE_DIR,
                        help="Directory of pose .pt files")
    parser.add_argument("--output", type=str, default=OUTPUT_PATH,
                        help="Output NPZ path")
    parser.add_argument("--ref_npz", type=str, default=REF_NPZ_PATH,
                        help="Reference NPZ (for ordering check)")
    args = parser.parse_args()

    # --- Collect all .pt files ---
    pt_files = sorted(glob.glob(os.path.join(args.pose_dir, "**", "*.pt"), recursive=True))
    print(f"Found {len(pt_files):,} .pt files in {args.pose_dir}")

    if len(pt_files) == 0:
        print("[ERROR] No .pt files found! Run step5_extract_pose.py first.")
        return

    # --- Build video path list ---
    video_paths = []
    for ptf in pt_files:
        rel = os.path.relpath(ptf, args.pose_dir).replace("\\", "/")
        rel = rel.replace(".pt", ".mp4")
        video_paths.append(rel)

    # Sort by path (alphabetical) - matches existing NPZ convention
    sorted_pairs = sorted(zip(video_paths, pt_files), key=lambda x: x[0])
    video_paths = [p[0] for p in sorted_pairs]
    pt_files = [p[1] for p in sorted_pairs]

    n_videos = len(video_paths)
    print(f"Videos: {n_videos:,}")
    print(f"Sample paths: {video_paths[:3]}")

    # --- Aggregate per video (frame-level, no pooling) ---
    all_frame_features = []
    skip_count = 0

    for vi, (vpath, ptf) in enumerate(zip(video_paths, pt_files)):
        data = torch.load(ptf, map_location="cpu", weights_only=False)
        landmarks = data["landmarks"].numpy()  # (80, 99)

        if landmarks.shape[0] != FRAMES_PER_VIDEO:
            print(f"[WARN] {vpath}: expected {FRAMES_PER_VIDEO} frames, got {landmarks.shape[0]}")
            if landmarks.shape[0] < FRAMES_PER_VIDEO:
                pad = np.zeros((FRAMES_PER_VIDEO - landmarks.shape[0], landmarks.shape[1]),
                               dtype=np.float32)
                landmarks = np.concatenate([landmarks, pad], axis=0)
            else:
                landmarks = landmarks[:FRAMES_PER_VIDEO]

        # KEY DIFFERENCE from step6: keep all 80 frames, no reshape+mean pooling
        all_frame_features.append(landmarks)  # (80, 99)

        if (vi + 1) % 1000 == 0:
            print(f"  Processed {vi+1:,}/{n_videos:,} videos...")

    # --- Build NPZ arrays ---
    pose_features = np.concatenate(all_frame_features, axis=0).astype(np.float32)
    video_clip_counts = np.full(n_videos, FRAMES_PER_VIDEO, dtype=np.int64)
    video_start_indices = np.arange(0, n_videos * FRAMES_PER_VIDEO,
                                    FRAMES_PER_VIDEO, dtype=np.int64)
    video_paths_arr = np.array(video_paths, dtype=object)

    print(f"\nPose features:  {pose_features.shape} ({pose_features.dtype})")
    print(f"Video paths:    {len(video_paths_arr)}")

    # --- Cross-check with reference NPZ ordering ---
    if os.path.exists(args.ref_npz):
        ref = np.load(args.ref_npz, allow_pickle=True)
        ref_paths = [str(p).replace("\\", "/") for p in ref["video_paths"]]

        matches = sum(1 for a, b in zip(video_paths, ref_paths) if a == b)
        print(f"\nPath match with reference NPZ: {matches}/{n_videos}")

        if matches != n_videos:
            print("[WARN] Ordering mismatch with reference NPZ!")
            ref_idx = {p: i for i, p in enumerate(ref_paths)}
            reorder = []
            missing = 0
            for vp in video_paths:
                if vp in ref_idx:
                    reorder.append(ref_idx[vp])
                else:
                    missing += 1
            if missing == 0 and len(reorder) == n_videos:
                print("  Reordering pose features to match reference NPZ...")
                pose_features = pose_features.reshape(n_videos, FRAMES_PER_VIDEO, -1)
                pose_features = pose_features[np.argsort(reorder)]
                pose_features = pose_features.reshape(n_videos * FRAMES_PER_VIDEO, -1)
                video_paths_arr = np.array(ref_paths, dtype=object)
                print("  Reordered OK!")
            else:
                print(f"  Cannot reorder: {missing} paths missing from reference")
    else:
        print(f"\n[INFO] No reference NPZ at {args.ref_npz} -- skipping cross-check")

    # --- Save ---
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez_compressed(
        args.output,
        pose_features=pose_features,
        video_paths=video_paths_arr,
        video_clip_counts=video_clip_counts,
        video_start_indices=video_start_indices,
    )
    size_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f"\nSaved: {args.output}")
    print(f"  pose_features:       {pose_features.shape}")
    print(f"  video_paths:         {len(video_paths_arr)}")
    print(f"  video_clip_counts:   all {FRAMES_PER_VIDEO}")
    print(f"  video_start_indices: 0..{n_videos * FRAMES_PER_VIDEO - FRAMES_PER_VIDEO}, step {FRAMES_PER_VIDEO}")
    print(f"  File size: {size_mb:.1f} MB")

    # Quick stats
    nonzero_rate = (pose_features.sum(axis=1) > 0).mean() * 100
    print(f"  Non-zero frames:     {nonzero_rate:.1f}%")
    print(f"\nDONE!")


if __name__ == "__main__":
    main()
