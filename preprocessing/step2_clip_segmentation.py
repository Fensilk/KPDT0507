"""
Step 2: Clip Segmentation and Label Assignment

Converts OF-Syn temporal segment-level annotations into clip-level label sequences.
Each 81-frame video is split into 5 non-overlapping 16-frame clips (~1.0s each).
Labels are assigned by matching each clip's center time to the corresponding segment.
"""
import os
from pathlib import Path

import pandas as pd

# --- Config ---
OMNIFALL_ROOT = Path(__file__).resolve().parent.parent / "DATASET-omnifall"
LABEL_CSV = OMNIFALL_ROOT / "labels" / "of-syn.csv"
SPLIT_DIR = OMNIFALL_ROOT / "splits" / "syn" / "random"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_CSV = OUTPUT_DIR / "ofsyn_clips.csv"
OUTPUT_FRAME_CSV = OUTPUT_DIR / "ofsyn_frame_labels.csv"

CLIP_FRAMES = 16       # frames per clip
TOTAL_FRAMES = 81      # total frames per OF-Syn video
FPS = 16.0              # standardized FPS (81 frames / 5.0625s)
FRAME_DURATION = 1.0 / FPS
CLIP_DURATION = CLIP_FRAMES * FRAME_DURATION  # 1.0s
CLIPS_PER_VIDEO = TOTAL_FRAMES // CLIP_FRAMES  # 5
EFFECTIVE_FRAMES = CLIPS_PER_VIDEO * CLIP_FRAMES  # 80 (trim last frame)

LABEL_NAMES = [
    "walk", "fall", "fallen", "sit_down", "sitting",
    "lie_down", "lying", "stand_up", "standing", "other",
    "kneel_down", "kneeling", "squat_down", "squatting", "crawl", "jump",
]


def load_labels(path):
    """Load OF-Syn label CSV."""
    df = pd.read_csv(path)
    df["label_name"] = df["label"].map(lambda x: LABEL_NAMES[int(x)])  # pyright: ignore[reportArgumentType]
    return df


def load_splits(split_dir):
    """Load train/val/test split CSVs with split column."""
    frames = []
    for split_name in ("train", "val", "test"):
        fp = split_dir / f"{split_name}.csv"
        if fp.exists():
            sdf = pd.read_csv(fp)
            sdf["split"] = split_name
            frames.append(sdf)
    return pd.concat(frames, ignore_index=True)


def assign_label_to_clip(center_time, segments):
    """
    Find the segment whose time interval contains center_time.
    Returns (label_id, label_name, is_matched).
    is_matched is False only when no segment covers center_time.
    """
    for _, seg in segments.iterrows():
        if seg["start"] <= center_time < seg["end"]:
            return int(seg["label"]), LABEL_NAMES[int(seg["label"])], True  # pyright: ignore[reportArgumentType]
    return 9, "other", False


def generate_clips(labels_df, splits_df):
    """
    For each video path in splits, generate clips and assign labels.
    Returns a DataFrame with one row per clip.
    """
    # Group labels by video path for fast lookup
    path_to_segments = {}
    for path, group in labels_df.groupby("path"):
        path_to_segments[path] = group.sort_values("start")  # pyright: ignore[reportCallIssue]

    rows = []
    no_match_count = 0
    total_clips = 0

    for _, split_row in splits_df.iterrows():
        path = split_row["path"]
        split = split_row["split"]
        segments = path_to_segments.get(path)

        if segments is None:
            print(f"  WARNING: {path} in splits but not in labels")
            continue

        for clip_idx in range(CLIPS_PER_VIDEO):
            start_frame = clip_idx * CLIP_FRAMES
            end_frame = start_frame + CLIP_FRAMES
            start_time = start_frame * FRAME_DURATION
            end_time = end_frame * FRAME_DURATION
            center_time = (start_time + end_time) / 2.0

            label, label_name, matched = assign_label_to_clip(center_time, segments)

            if not matched:
                no_match_count += 1

            total_clips += 1
            rows.append({
                "path": path,
                "split": split,
                "clip_idx": clip_idx,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_time": round(start_time, 4),
                "end_time": round(end_time, 4),
                "center_time": round(center_time, 4),
                "label": label,
                "label_name": label_name,
            })

    clips_df = pd.DataFrame(rows)
    n_other = (clips_df["label_name"] == "other").sum()
    print(f"  Total clips generated: {total_clips}")
    print(f"  Label 'other' (class 9): {n_other} "
          f"({100*n_other/total_clips:.1f}%)")
    print(f"  Truly unmatched (no segment covers center): {no_match_count} "
          f"({100*no_match_count/total_clips:.2f}%)" if total_clips else "")
    return clips_df


def generate_frame_labels(labels_df, splits_df):
    """
    For each video path in splits, generate per-frame labels for all 80 frames.

    Uses frame center time to assign labels from temporal segments.
    Returns a DataFrame with one row per frame (80 per video).
    """
    # Group labels by video path for fast lookup
    path_to_segments = {}
    for path, group in labels_df.groupby("path"):
        path_to_segments[path] = group.sort_values("start")  # pyright: ignore[reportCallIssue]

    rows = []
    no_match_count = 0
    total_frames = 0

    for _, split_row in splits_df.iterrows():
        path = split_row["path"]
        split = split_row["split"]
        segments = path_to_segments.get(path)

        if segments is None:
            print(f"  WARNING: {path} in splits but not in labels")
            continue

        for frame_idx in range(EFFECTIVE_FRAMES):
            # Center time of this frame
            center_time = (frame_idx + 0.5) * FRAME_DURATION
            start_time = frame_idx * FRAME_DURATION
            end_time = (frame_idx + 1) * FRAME_DURATION

            label, label_name, matched = assign_label_to_clip(center_time, segments)

            if not matched:
                no_match_count += 1

            total_frames += 1
            rows.append({
                "path": path,
                "split": split,
                "frame_idx": frame_idx,
                "start_time": round(start_time, 4),
                "end_time": round(end_time, 4),
                "center_time": round(center_time, 4),
                "label": label,
                "label_name": label_name,
                "fall_label": 1 if label == 1 else 0,
                "fallen_label": 1 if label == 2 else 0,
            })

    frames_df = pd.DataFrame(rows)
    n_other = (frames_df["label_name"] == "other").sum()
    print(f"  Total frames generated: {total_frames}")
    print(f"  Label 'other' (class 9): {n_other} "
          f"({100*n_other/total_frames:.1f}%)")
    print(f"  Truly unmatched (no segment covers center): {no_match_count} "
          f"({100*no_match_count/total_frames:.2f}%)" if total_frames else "")
    return frames_df


def summarize_frame_labels(frames_df):
    """Print frame-level summary statistics."""
    print(f"\n  Total frames:     {len(frames_df):,}")
    print(f"  Unique videos:    {frames_df['path'].nunique():,}")
    print(f"  Frames per video: {EFFECTIVE_FRAMES}")

    print(f"\n  Per-split frame counts:")
    for split_name in ["train", "val", "test"]:
        sdf = frames_df[frames_df["split"] == split_name]
        print(f"    {split_name:>5s}: {len(sdf):,} frames, "
              f"{sdf['path'].nunique():,} videos")

    print(f"\n  Frame-level label distribution:")
    print(f"  {'Label':>12s}  {'Count':>7s}  {'Pct':>6s}")
    print(f"  {'-'*12}  {'-'*7}  {'-'*6}")
    counts = frames_df["label_name"].value_counts()
    for name, cnt in counts.items():
        print(f"  {name:>12s}  {cnt:>7d}  {100*cnt/len(frames_df):5.1f}%")


def show_frame_sequences(frames_df, n=5):
    """Print N example frame-level label sequences with transitions."""
    print(f"\n{'='*65}")
    print(f"  EXAMPLE FRAME-LEVEL SEQUENCES ({n} videos)")
    print(f"{'='*65}")

    path_groups = frames_df.groupby("path")
    diverse_paths = []
    for path, grp in path_groups:
        labels = grp.sort_values("frame_idx")["label_name"].tolist()  # pyright: ignore[reportCallIssue]
        if len(set(labels)) > 1:
            diverse_paths.append((path, len(set(labels))))

    diverse_paths.sort(key=lambda x: -x[1])
    chosen = diverse_paths[:min(n, len(diverse_paths))]

    for path, n_unique in chosen:
        segs = frames_df[frames_df["path"] == path].sort_values("frame_idx")  # pyright: ignore[reportCallIssue]
        labels_list = segs["label_name"].tolist()
        # Show transitions compactly: collapse consecutive same labels
        transitions = []
        prev = None
        for i, lbl in enumerate(labels_list):
            if lbl != prev:
                transitions.append(f"[f{i}] {lbl}")
                prev = lbl
        print(f"\n  {path}  ({n_unique} unique labels, {len(transitions)} transitions)")
        print(f"  {' | '.join(transitions)}")


def summarize(clips_df):
    """Print clip-level summary statistics."""
    print(f"\n  Total clips:      {len(clips_df):,}")
    print(f"  Unique videos:    {clips_df['path'].nunique():,}")
    print(f"  Clips per video:  {CLIPS_PER_VIDEO}")

    print(f"\n  Per-split clip counts:")
    for split_name in ["train", "val", "test"]:
        sdf = clips_df[clips_df["split"] == split_name]
        print(f"    {split_name:>5s}: {len(sdf):,} clips, "
              f"{sdf['path'].nunique():,} videos")

    print(f"\n  Clip-level label distribution:")
    print(f"  {'Label':>12s}  {'Count':>7s}  {'Pct':>6s}")
    print(f"  {'-'*12}  {'-'*7}  {'-'*6}")
    counts = clips_df["label_name"].value_counts()
    for name, cnt in counts.items():
        print(f"  {name:>12s}  {cnt:>7d}  {100*cnt/len(clips_df):5.1f}%")


def show_sequences(clips_df, splits_df, n=5):
    """Print N example clip-level label sequences."""
    print(f"\n{'='*65}")
    print(f"  EXAMPLE CLIP SEQUENCES ({n} videos)")
    print(f"{'='*65}")

    # Find videos with at least one dynamic transition (not all same label)
    path_groups = clips_df.groupby("path")
    diverse_paths = []
    for path, grp in path_groups:
        labels = grp.sort_values("clip_idx")["label_name"].tolist()  # pyright: ignore[reportCallIssue]
        if len(set(labels)) > 1:
            diverse_paths.append((path, len(set(labels))))

    diverse_paths.sort(key=lambda x: -x[1])
    chosen = diverse_paths[:min(n, len(diverse_paths))]

    for path, n_unique in chosen:
        segs = clips_df[clips_df["path"] == path].sort_values("clip_idx")  # pyright: ignore[reportCallIssue]
        sequence = " -> ".join(segs["label_name"].tolist())
        print(f"\n  {path}  ({n_unique} unique labels)")
        print(f"  {sequence}")


def main():
    print("=" * 65)
    print("  Step 2: Clip Segmentation and Label Assignment")
    print("=" * 65)

    # 1. Load labels
    print("\n--- 1. Loading labels ---")
    labels_df = load_labels(LABEL_CSV)
    print(f"  {len(labels_df)} segments, {labels_df['path'].nunique()} videos")

    # 2. Load splits
    print("\n--- 2. Loading splits ---")
    splits_df = load_splits(SPLIT_DIR)
    print(f"  {len(splits_df)} total paths across splits")

    # 3. Generate clips
    print("\n--- 3. Generating clips ---")
    print(f"  Clip size: {CLIP_FRAMES} frames ({CLIP_DURATION:.2f}s), "
          f"FPS: {FPS}, Clips per video: {CLIPS_PER_VIDEO}")
    clips_df = generate_clips(labels_df, splits_df)

    # 4. Summary
    print("\n--- 4. Clip summary ---")
    summarize(clips_df)

    # 5. Show example sequences
    show_sequences(clips_df, splits_df, n=8)

    # 6. Save clip-level
    print(f"\n--- 5. Saving clip-level ---")
    clips_df.to_csv(OUTPUT_CSV, index=False)
    print(f"  Saved {len(clips_df):,} clips to {OUTPUT_CSV}")
    print(f"  File size: {OUTPUT_CSV.stat().st_size / 1024:.1f} KB")

    # 7. Generate frame-level labels
    print(f"\n{'='*65}")
    print("  Frame-Level Label Generation")
    print(f"{'='*65}")
    print(f"\n--- 6. Generating frame-level labels ---")
    print(f"  Effective frames per video: {EFFECTIVE_FRAMES}")
    print(f"  Frame duration: {FRAME_DURATION:.4f}s")
    frames_df = generate_frame_labels(labels_df, splits_df)

    print(f"\n--- 7. Frame-level summary ---")
    summarize_frame_labels(frames_df)

    # 8. Show example frame sequences
    show_frame_sequences(frames_df, n=5)

    # 9. Save frame-level
    print(f"\n--- 8. Saving frame-level ---")
    frames_df.to_csv(OUTPUT_FRAME_CSV, index=False)
    print(f"  Saved {len(frames_df):,} frames to {OUTPUT_FRAME_CSV}")
    print(f"  File size: {OUTPUT_FRAME_CSV.stat().st_size / 1024:.1f} KB")

    print(f"\n{'='*65}")
    print(f"  Step 2 complete - clip and frame sequences saved.")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
