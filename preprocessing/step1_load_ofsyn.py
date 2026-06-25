"""
Step 1: Load and verify OF-Syn (OF-Synthetic) data from OmniFall.

Reads labels/splits/video metadata, confirms data integrity,
and prints summaries to validate the dataset is usable.
"""
import os
import sys
from pathlib import Path

import random

import pandas as pd

# --- Config ---
OMNIFALL_ROOT = Path(__file__).resolve().parent.parent / "DATASET-omnifall"
LABEL_CSV = OMNIFALL_ROOT / "labels" / "of-syn.csv"
META_CSV = OMNIFALL_ROOT / "videos" / "metadata.csv"
VIDEO_ARCHIVE = OMNIFALL_ROOT / "data_files" / "omnifall-synthetic_av1.tar"
FRAMEWISE_ARCHIVE = OMNIFALL_ROOT / "data_files" / "syn_frame_wise_labels.tar.zst"
SPLIT_DIR = OMNIFALL_ROOT / "splits" / "syn" / "random"

# 16-class activity taxonomy
LABEL_NAMES = [
    "walk",         # 0
    "fall",         # 1
    "fallen",       # 2
    "sit_down",     # 3
    "sitting",      # 4
    "lie_down",     # 5
    "lying",        # 6
    "stand_up",     # 7
    "standing",     # 8
    "other",        # 9
    "kneel_down",   # 10
    "kneeling",     # 11
    "squat_down",   # 12
    "squatting",    # 13
    "crawl",        # 14
    "jump",         # 15
]

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_labels(path):
    """Load OF-Syn label CSV and add a readable label_name column."""
    df = pd.read_csv(path)
    df["label_name"] = df["label"].map(lambda x: LABEL_NAMES[int(x)])  # pyright: ignore[reportArgumentType]
    df["duration"] = df["end"] - df["start"]
    return df


def load_splits(split_dir):
    """Load train/val/test split CSVs. Returns dict split_name -> DataFrame of paths."""
    splits = {}
    for name in ("train", "val", "test"):
        fp = split_dir / f"{name}.csv"
        if fp.exists():
            splits[name] = pd.read_csv(fp)
        else:
            print(f"  WARNING: split file not found: {fp}")
    return splits


def merge_splits_with_labels(splits, labels_df):
    """Join each split's paths with the label dataframe."""
    merged = {}
    for name, split_df in splits.items():
        m = split_df.merge(labels_df, on="path", how="left")
        # Check for unmatched paths
        unmatched = m[m["label"].isna()]
        if len(unmatched) > 0:
            print(f"  WARNING: {len(unmatched)} paths in {name} split have NO labels")
        merged[name] = m
    return merged


def summarize_labels(labels_df):
    """Print label-level statistics."""
    n_segments = len(labels_df)
    n_videos = labels_df["path"].nunique()
    print(f"  Total temporal segments: {n_segments:,}")
    print(f"  Unique videos:           {n_videos:,}")
    print(f"  Avg segments per video:  {n_segments / n_videos:.2f}")

    # Label distribution
    print(f"\n  {'Label':>12s}  {'Count':>6s}  {'Pct':>6s}")
    print(f"  {'-'*12}  {'-'*6}  {'-'*6}")
    counts = labels_df["label_name"].value_counts()
    for name, cnt in counts.items():
        print(f"  {name:>12s}  {cnt:>6d}  {100*cnt/n_segments:5.1f}%")
    return counts


def summarize_splits(merged_splits):
    """Print per-split statistics."""
    print()
    for name, m in merged_splits.items():
        n_videos = m["path"].nunique()
        n_segments = len(m)
        has_fall = (m["label_name"] == "fall").any()
        fall_videos = m[m["label_name"] == "fall"]["path"].nunique()
        print(f"  {name:>5s}: {n_videos:>5d} videos, {n_segments:>6d} segments, "
              f"has_fall={has_fall}, {fall_videos} videos contain 'fall'")


def verify_metadata(labels_df, meta_csv):
    """Cross-check label paths against video metadata."""
    meta = pd.read_csv(meta_csv)
    label_paths = set(labels_df["path"].unique())
    meta_paths = set(meta["path"].unique())

    in_both = label_paths & meta_paths
    only_label = label_paths - meta_paths
    only_meta = meta_paths - label_paths

    print(f"\n  Label unique paths:  {len(label_paths):,}")
    print(f"  Metadata paths:      {len(meta_paths):,}")
    print(f"  Intersection:        {len(in_both):,}")
    if only_label:
        print(f"  WARNING: {len(only_label)} paths in labels but NOT in metadata")
    if only_meta:
        print(f"  WARNING: {len(only_meta)} paths in metadata but NOT in labels")

    # Check demographic coverage in labels
    if "age_group" in labels_df.columns:
        print("\n  Age distribution in labels:")
        for v, c in labels_df["age_group"].value_counts().items():
            print(f"    {v}: {c:,}")


def check_archives():
    """Report video and frame-wise archive status."""
    print()
    for name, path in [("Video archive", VIDEO_ARCHIVE),
                        ("Frame-wise labels", FRAMEWISE_ARCHIVE)]:
        if path.exists():
            size_gb = path.stat().st_size / (1024**3)
            print(f"  {name}: EXISTS ({size_gb:.1f} GB)  -> {path}")
        else:
            print(f"  {name}: NOT FOUND  -> {path}")


def show_examples(merged_splits, n=5):
    """Print N example video timelines with human-readable label sequences."""
    print(f"\n{'='*70}")
    print(f"  EXAMPLE TIMELINES ({n} videos)")
    print(f"{'='*70}")

    # Pick examples from train split that contain 'fall'
    train = merged_splits.get("train")
    if train is None:
        return

    fall_paths = train[train["label_name"] == "fall"]["path"].unique()
    chosen = random.sample(list(fall_paths), min(n, len(fall_paths)))

    for path in chosen:
        segs = train[train["path"] == path].sort_values("start")
        print(f"\n  {path}")
        for _, seg in segs.iterrows():
            label = seg["label_name"]
            print(f"    [{seg['start']:6.2f}s - {seg['end']:6.2f}s]  {label}")


def main():
    print("=" * 70)
    print("  Step 1: Load and Verify OF-Syn Data")
    print("=" * 70)

    # 1. Load labels
    print("\n--- 1. Loading labels ---")
    labels_df = load_labels(LABEL_CSV)
    print(f"  Columns: {list(labels_df.columns)}")

    # 2. Summarize
    print("\n--- 2. Label summary ---")
    summarize_labels(labels_df)

    # 3. Load splits
    print("\n--- 3. Loading splits ---")
    splits = load_splits(SPLIT_DIR)

    # 4. Merge and summarize
    merged = merge_splits_with_labels(splits, labels_df)
    print("\n--- 4. Split summary ---")
    summarize_splits(merged)

    # 5. Verify against metadata
    print("\n--- 5. Metadata cross-check ---")
    verify_metadata(labels_df, META_CSV)

    # 6. Check archives
    print("\n--- 6. Archive status ---")
    check_archives()

    # 7. Show examples
    show_examples(merged, n=5)

    print("\n" + "=" * 70)
    print("  Step 1 complete - data verified successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
