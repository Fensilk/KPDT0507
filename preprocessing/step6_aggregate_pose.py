"""
Step 6: Aggregate per-frame pose .pt files into clip-level NPZ.

Reads data/pose_features/**/*.pt (80 frames x 99 landmarks per video),
pools into 5 clips of 16 frames each (mean), and saves as:
    data/omnifall_pose.npz

NPZ keys:
    pose_features:      (60000, 99)  float32
    video_paths:        (12000,)     object (str)
    video_clip_counts:  (12000,)     int64 (all 5)
    video_start_indices:(12000,)     int64 (0, 5, 10, ...)

Ordering matches omnifall_preprocessed.npz (alphabetical by path).

Usage:
    python preprocessing/step6_aggregate_pose.py
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
OUTPUT_PATH = os.path.join(ROOT, "data", "omnifall_pose.npz")
REF_NPZ_PATH = os.path.join(ROOT, "data", "omnifall_preprocessed.npz")

CLIPS_PER_VIDEO = 5
FRAMES_PER_CLIP = 16  # 5 * 16 = 80 frames


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate per-frame pose to clip-level NPZ"
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
        # Convert Windows path to forward-slash relative path
        rel = os.path.relpath(ptf, args.pose_dir).replace("\\", "/")
        # Remove .pt extension
        rel = rel.replace(".pt", ".mp4")
        video_paths.append(rel)

    # Sort by path (alphabetical) - matches existing NPZ convention
    sorted_pairs = sorted(zip(video_paths, pt_files), key=lambda x: x[0])
    video_paths = [p[0] for p in sorted_pairs]
    pt_files = [p[1] for p in sorted_pairs]

    n_videos = len(video_paths)
    print(f"Videos: {n_videos:,}")
    print(f"Sample paths: {video_paths[:3]}")

    # --- Aggregate per video ---
    all_clip_features = []
    skip_count = 0

    for vi, (vpath, ptf) in enumerate(zip(video_paths, pt_files)):
        data = torch.load(ptf, map_location="cpu", weights_only=False)
        landmarks = data["landmarks"].numpy()  # should be (80, 99)

        if landmarks.shape[0] != 80:
            print(f"[WARN] {vpath}: expected 80 frames, got {landmarks.shape[0]}")
            if landmarks.shape[0] < 80:
                # Pad
                pad = np.zeros((80 - landmarks.shape[0], landmarks.shape[1]), dtype=np.float32)
                landmarks = np.concatenate([landmarks, pad], axis=0)
            else:
                landmarks = landmarks[:80]

        # Pool: 5 clips x 16 frames -> mean per clip
        # Reshape: (80, 99) -> (5, 16, 99) -> mean over dim 1 -> (5, 99)
        clip_feats = landmarks.reshape(CLIPS_PER_VIDEO, FRAMES_PER_CLIP, -1).mean(axis=1)
        all_clip_features.append(clip_feats)

        if (vi + 1) % 1000 == 0:
            print(f"  Processed {vi+1:,}/{n_videos:,} videos...")

    # --- Build NPZ arrays ---
    pose_features = np.concatenate(all_clip_features, axis=0).astype(np.float32)
    video_clip_counts = np.full(n_videos, CLIPS_PER_VIDEO, dtype=np.int64)
    video_start_indices = np.arange(0, n_videos * CLIPS_PER_VIDEO, CLIPS_PER_VIDEO, dtype=np.int64)
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
            print(f"  First 3 ref:   {ref_paths[:3]}")
            print(f"  First 3 pose:  {video_paths[:3]}")
            # Try to reorder
            ref_idx = {p: i for i, p in enumerate(ref_paths)}
            reorder = []
            missing = 0
            for vp in video_paths:
                if vp in ref_idx:
                    reorder.append(ref_idx[vp])
                else:
                    missing += 1
            if missing == 0 and len(reorder) == n_videos:
                print(f"  Reordering pose features to match reference NPZ...")
                pose_features = pose_features.reshape(n_videos, CLIPS_PER_VIDEO, -1)[np.argsort(reorder)]
                pose_features = pose_features.reshape(n_videos * CLIPS_PER_VIDEO, -1)
                video_paths_arr = np.array(ref_paths, dtype=object)
                print(f"  Reordered OK!")
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
    print(f"\nSaved: {args.output}")
    print(f"  pose_features:       {pose_features.shape}")
    print(f"  video_paths:         {len(video_paths_arr)}")
    print(f"  video_clip_counts:   all {CLIPS_PER_VIDEO}")
    print(f"  video_start_indices: 0..{n_videos * CLIPS_PER_VIDEO - CLIPS_PER_VIDEO}, step {CLIPS_PER_VIDEO}")

    # Quick stats
    nonzero_rate = (pose_features.sum(axis=1) > 0).mean() * 100
    print(f"  Non-zero clips:      {nonzero_rate:.1f}%")
    print(f"\nDONE!")


if __name__ == "__main__":
    main()
