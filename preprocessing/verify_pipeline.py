"""
Pipeline Verification: check all 12,000 .pt files for correctness.

Checks:
  1. File count = 12,000
  2. All keys present: features, labels, fall_label, fallen_label
  3. Shapes: features=[5,512], labels=[5], fall_label=[5], fallen_label=[5]
  4. No NaN / Inf in features
  5. Feature stats (per-video mean/std distribution)
  6. Label values: labels in [0..15], fall_label in {0,1}, fallen_label in {0,1}
  7. Cross-check: labels match data/ofsyn_clips.csv
  8. Fall/fallen consistency: fall_label matches where label==1, etc.
  9. Per-split counts match CSV
"""
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "features"
CLIPS_CSV = Path(__file__).resolve().parent.parent / "data" / "ofsyn_clips.csv"
FRAME_CSV = Path(__file__).resolve().parent.parent / "data" / "ofsyn_frame_labels.csv"

# Check for --frame_level flag
USE_FRAME_LEVEL = "--frame_level" in sys.argv
EXPECTED_FRAMES = 80 if USE_FRAME_LEVEL else 5   # frames/clips per video
NPZ_NAME = ("omnifall_frame_preprocessed.npz" if USE_FRAME_LEVEL
            else "omnifall_preprocessed.npz")
LABEL_CSV_PATH = FRAME_CSV if USE_FRAME_LEVEL else CLIPS_CSV
LABEL_SORT_COL = "frame_idx" if USE_FRAME_LEVEL else "clip_idx"

print(f"[CONFIG] Use frame-level: {USE_FRAME_LEVEL}")
print(f"[CONFIG] Expected features per video: {EXPECTED_FRAMES}")
print(f"[CONFIG] NPZ output: {NPZ_NAME}")

LABEL_NAMES = [
    "walk", "fall", "fallen", "sit_down", "sitting",
    "lie_down", "lying", "stand_up", "standing", "other",
    "kneel_down", "kneeling", "squat_down", "squatting", "crawl", "jump",
]


def check_file_count():
    pts = sorted(OUTPUT_DIR.glob("**/*.pt"))
    n = len(pts)
    print(f"\n1. File count: {n}")
    if n == 12000:
        print("   OK - exactly 12,000 .pt files")
    else:
        print(f"   WARNING - expected 12,000, got {n}")
    return pts


def check_shapes_and_keys(pt_files):
    print(f"\n2. Shape & key check (sampling {min(200, len(pt_files))} files)...")
    bad = []
    shapes_counter = Counter()
    # Sample uniformly
    step = max(1, len(pt_files) // 200)
    sample = pt_files[::step][:200]
    for fp in sample:
        try:
            d = torch.load(fp, weights_only=False)
            for k in ["features", "labels", "fall_label", "fallen_label"]:
                if k not in d:
                    bad.append((fp.stem, f"missing key: {k}"))
            shapes_counter[tuple(d["features"].shape)] += 1
            shapes_counter[("labels", tuple(d["labels"].shape))] += 1
            shapes_counter[("fall_label", tuple(d["fall_label"].shape))] += 1
            shapes_counter[("fallen_label", tuple(d["fallen_label"].shape))] += 1

            # Verify expected shape
            if d["features"].shape != (EXPECTED_FRAMES, 512):
                bad.append((fp.stem, f"wrong features shape: {d['features'].shape}"))
            if d["labels"].shape != (EXPECTED_FRAMES,):
                bad.append((fp.stem, f"wrong labels shape: {d['labels'].shape}"))
        except Exception as e:
            bad.append((fp.stem, str(e)))

    for shape, count in shapes_counter.most_common(10):
        print(f"   {shape}: {count}")
    if bad:
        print(f"   BAD FILES: {bad[:10]}")
        return False
    print("   OK - all sampled files have correct keys and shapes")
    return True


def check_nan_inf(pt_files):
    print(f"\n3. NaN / Inf check (sampling 500 files)...")
    step = max(1, len(pt_files) // 500)
    sample = pt_files[::step][:500]
    nan_count = 0
    inf_count = 0
    for fp in sample:
        d = torch.load(fp, weights_only=False)
        feats = d["features"]
        if torch.isnan(feats).any():
            nan_count += 1
        if torch.isinf(feats).any():
            inf_count += 1
    print(f"   NaN: {nan_count}, Inf: {inf_count}")
    if nan_count == 0 and inf_count == 0:
        print("   OK - no NaN or Inf in features")
    else:
        print("   FAIL")


def check_feature_stats(pt_files):
    print(f"\n4. Feature statistics (sampling 500 files)...")
    step = max(1, len(pt_files) // 500)
    sample = pt_files[::step][:500]
    means = []
    stds = []
    for fp in sample:
        d = torch.load(fp, weights_only=False)
        feats = d["features"].float()
        means.append(feats.mean().item())
        stds.append(feats.std().item())
    means = np.array(means)
    stds = np.array(stds)
    print(f"   Per-video mean:  min={means.min():.4f}  max={means.max():.4f}  "
          f"avg={means.mean():.4f}  std={means.std():.4f}")
    print(f"   Per-video std:   min={stds.min():.4f}  max={stds.max():.4f}  "
          f"avg={stds.mean():.4f}  std={stds.std():.4f}")
    if means.mean() > 0 and stds.mean() > 0.1:
        print("   OK - features appear normal (non-degenerate)")
    else:
        print("   WARNING - features may be degenerate")


def check_label_values(pt_files):
    print(f"\n5. Label value ranges (all files)...")
    bad_label = 0
    bad_fall = 0
    bad_fallen = 0
    for fp in pt_files:
        d = torch.load(fp, weights_only=False)
        if d["labels"].min() < 0 or d["labels"].max() > 15:
            bad_label += 1
        if not ((d["fall_label"] == 0) | (d["fall_label"] == 1)).all():
            bad_fall += 1
        if not ((d["fallen_label"] == 0) | (d["fallen_label"] == 1)).all():
            bad_fallen += 1
    print(f"   label out of range [0..15]: {bad_label}")
    print(f"   fall_label not binary:      {bad_fall}")
    print(f"   fallen_label not binary:    {bad_fallen}")
    if bad_label == 0 and bad_fall == 0 and bad_fallen == 0:
        print("   OK - all label values valid")


def check_label_distribution(pt_files):
    print(f"\n6. Label distribution in .pt files...")
    label_counter = Counter()
    fall_count = 0
    fallen_count = 0
    total_clips = 0
    for fp in pt_files:
        d = torch.load(fp, weights_only=False)
        labels = d["labels"].tolist()
        total_clips += len(labels)
        for l in labels:
            label_counter[l] += 1
        fall_count += d["fall_label"].sum().item()
        fallen_count += d["fallen_label"].sum().item()

    print(f"   Total clips: {total_clips}")
    print(f"   {'Label':>12s}  {'Count':>7s}  {'Pct':>6s}")
    print(f"   {'-'*12}  {'-'*7}  {'-'*6}")
    for label_id in sorted(label_counter.keys()):
        cnt = label_counter[label_id]
        print(f"   {LABEL_NAMES[label_id]:>12s}  {cnt:>7d}  {100*cnt/total_clips:5.1f}%")
    print(f"   fall_label=1:   {int(fall_count):,}")
    print(f"   fallen_label=1: {int(fallen_count):,}")


def cross_check_csv(pt_files):
    print(f"\n7. Cross-check against CSV (sampling 300 files)...")
    csv_df = pd.read_csv(LABEL_CSV_PATH)
    step = max(1, len(pt_files) // 300)
    sample = pt_files[::step][:300]
    mismatches = 0
    for fp in sample:
        rel = fp.relative_to(OUTPUT_DIR).with_suffix("").as_posix()
        d = torch.load(fp, weights_only=False)
        csv_rows = csv_df[csv_df["path"] == rel].sort_values(LABEL_SORT_COL)  # pyright: ignore[reportCallIssue]
        if len(csv_rows) != EXPECTED_FRAMES:
            mismatches += 1
            continue
        if not (d["labels"].numpy() == csv_rows["label"].values).all():
            mismatches += 1
        if not (d["fall_label"].numpy() == csv_rows["fall_label"].values).all():
            mismatches += 1
        if not (d["fallen_label"].numpy() == csv_rows["fallen_label"].values).all():
            mismatches += 1
    print(f"   Mismatches in sample: {mismatches}")
    if mismatches == 0:
        print("   OK - sampled .pt labels match CSV")
    else:
        print("   FAIL")


def check_consistency(pt_files):
    print(f"\n8. Internal consistency: fall_label=1 iff label==1, etc.")
    bad_fall = 0
    bad_fallen = 0
    for fp in pt_files:
        d = torch.load(fp, weights_only=False)
        expected_fall = (d["labels"] == 1).long()
        expected_fallen = (d["labels"] == 2).long()
        if not (d["fall_label"] == expected_fall).all():
            bad_fall += 1
        if not (d["fallen_label"] == expected_fallen).all():
            bad_fallen += 1
    print(f"   fall_label inconsistency: {bad_fall}")
    print(f"   fallen_label inconsistency: {bad_fallen}")
    if bad_fall == 0 and bad_fallen == 0:
        print("   OK - binary labels consistent with 16-class labels")


def check_split_counts(pt_files):
    print(f"\n9. Per-split video counts...")
    label_df = pd.read_csv(LABEL_CSV_PATH)
    # Build path -> split mapping
    path_to_split = dict(zip(label_df["path"], label_df["split"]))
    split_counts = Counter()
    for fp in pt_files:
        rel = fp.relative_to(OUTPUT_DIR).with_suffix("").as_posix()
        split = path_to_split.get(rel, "unknown")
        split_counts[split] += 1
    for s in ["train", "val", "test"]:
        print(f"   {s}: {split_counts[s]:,} videos")
    print(f"   Expected from CSV: train={label_df[label_df['split']=='train']['path'].nunique()}, "
          f"val={label_df[label_df['split']=='val']['path'].nunique()}, "
          f"test={label_df[label_df['split']=='test']['path'].nunique()}")


def aggregate_to_npz(pt_files):
    """Aggregate all .pt files into a single .npz file.

    Output data/omnifall_preprocessed.npz with keys:
      features:          float32 [N_clips, 512]
      labels_16:         int64   [N_clips]
      fall_labels:       int64   [N_clips]
      fallen_labels:     int64   [N_clips]
      clips_times:       float32 [N_clips, 2]  (start_time, end_time)
      video_paths:       object  [N_videos]
      video_clip_counts: int64   [N_videos]
      video_start_indices: int64 [N_videos]
    """
    print(f"\n10. Aggregating .pt files into .npz...")

    label_csv = pd.read_csv(LABEL_CSV_PATH)
    path_to_rows = {}
    for path, grp in label_csv.groupby("path"):
        path_to_rows[path] = grp.sort_values(LABEL_SORT_COL)

    pt_files_sorted = sorted(
        pt_files, key=lambda p: p.relative_to(OUTPUT_DIR).with_suffix("").as_posix()
    )

    n_videos = len(pt_files_sorted)
    features_list, labels_list, fall_list, fallen_list, times_list = [], [], [], [], []
    video_paths = []
    video_frame_counts = np.empty(n_videos, dtype=np.int64)
    video_start_indices = np.empty(n_videos, dtype=np.int64)

    offset = 0
    for i, fp in enumerate(pt_files_sorted):
        rel = fp.relative_to(OUTPUT_DIR).with_suffix("").as_posix()
        d = torch.load(fp, weights_only=False)

        feats = d["features"].numpy().astype(np.float32)
        n_entries = len(d["labels"])

        rows = path_to_rows.get(rel)
        if rows is not None and len(rows) == n_entries:
            times = rows[["start_time", "end_time"]].values.astype(np.float32)
        else:
            times = np.zeros((n_entries, 2), dtype=np.float32)

        features_list.append(feats)
        labels_list.append(d["labels"].numpy().astype(np.int64))
        fall_list.append(d["fall_label"].numpy().astype(np.int64))
        fallen_list.append(d["fallen_label"].numpy().astype(np.int64))
        times_list.append(times)
        video_paths.append(rel)
        video_start_indices[i] = offset
        video_frame_counts[i] = n_entries
        offset += n_entries

    npz_path = OUTPUT_DIR.parent / NPZ_NAME
    np.savez_compressed(
        npz_path,
        features=np.concatenate(features_list, axis=0),
        labels_16=np.concatenate(labels_list, axis=0),
        fall_labels=np.concatenate(fall_list, axis=0),
        fallen_labels=np.concatenate(fallen_list, axis=0),
        clips_times=np.concatenate(times_list, axis=0),
        video_paths=np.array(video_paths, dtype=object),
        video_clip_counts=video_frame_counts,
        video_start_indices=video_start_indices,
    )

    size_mb = npz_path.stat().st_size / 1024 / 1024
    unit = "frames" if USE_FRAME_LEVEL else "clips"
    print(f"   Output: {npz_path}")
    print(f"   Videos: {n_videos}, {unit.capitalize()}: {offset}")
    print(f"   Features: [{offset}, 512], dtype=float32")
    print(f"   File size: {size_mb:.1f} MB")
    print(f"   Keys: features, labels_16, fall_labels, fallen_labels, "
          f"clips_times, video_paths, video_clip_counts, video_start_indices")


def main():
    print("=" * 60)
    print("  Pipeline Verification")
    print("=" * 60)

    pt_files = check_file_count()
    if not pt_files:
        print("No .pt files found!")
        return

    check_shapes_and_keys(pt_files)
    check_nan_inf(pt_files)
    check_feature_stats(pt_files)
    check_label_values(pt_files)
    check_label_distribution(pt_files)
    cross_check_csv(pt_files)
    check_consistency(pt_files)
    check_split_counts(pt_files)
    aggregate_to_npz(pt_files)

    print(f"\n{'='*60}")
    print(f"  Verification complete.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
