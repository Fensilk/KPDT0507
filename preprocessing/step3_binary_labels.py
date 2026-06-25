"""
Step 3: Generate fall / fallen binary labels from clip-level 16-class labels.

Input:  data/ofsyn_clips.csv (from Step 2)
Output: data/ofsyn_clips.csv (updated with fall_label and fallen_label columns)

Rules:
  - fall_label   = 1 if label == 1 (fall),  else 0
  - fallen_label = 1 if label == 2 (fallen), else 0
"""
import random
from pathlib import Path

import pandas as pd

LABEL_NAMES = [
    "walk", "fall", "fallen", "sit_down", "sitting",
    "lie_down", "lying", "stand_up", "standing", "other",
    "kneel_down", "kneeling", "squat_down", "squatting", "crawl", "jump",
]

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"
CLIPS_CSV = OUTPUT_DIR / "ofsyn_clips.csv"


def main():
    print("=" * 55)
    print("  Step 3: Generate fall / fallen Binary Labels")
    print("=" * 55)

    # 1. Load
    print("\n--- 1. Loading clips ---")
    df = pd.read_csv(CLIPS_CSV)
    print(f"  {len(df):,} clips, {df['path'].nunique():,} videos")

    # 2. Generate binary labels
    print("\n--- 2. Generating binary labels ---")
    df["fall_label"] = (df["label"] == 1).astype(int)
    df["fallen_label"] = (df["label"] == 2).astype(int)

    n_fall = df["fall_label"].sum()
    n_fallen = df["fallen_label"].sum()
    print(f"  fall_label=1:   {n_fall} clips ({100*n_fall/len(df):.1f}%)")
    print(f"  fallen_label=1: {n_fallen} clips ({100*n_fallen/len(df):.1f}%)")

    # Verify counts match original label distribution
    n_orig_fall = (df["label"] == 1).sum()
    n_orig_fallen = (df["label"] == 2).sum()
    assert n_fall == n_orig_fall, f"Mismatch: {n_fall} vs {n_orig_fall}"
    assert n_fallen == n_orig_fallen, f"Mismatch: {n_fallen} vs {n_orig_fallen}"
    print("  Verified: binary labels match original label distribution.")

    # 3. Show examples
    print("\n--- 3. Example clip sequences with binary labels ---")
    print(f"  {'clip_idx':>8s}  {'label':>10s}  {'fall_label':>10s}  {'fallen_label':>12s}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*12}")

    # Pick a few diverse examples
    fall_paths = df[df["label"] == 1]["path"].unique()
    example_paths = random.sample(list(fall_paths), min(3, len(fall_paths)))
    for path in example_paths:
        rows = df[df["path"] == path].sort_values("clip_idx") # pyright: ignore[reportCallIssue]
        print(f"\n  {path}:")
        seq_16 = []
        seq_fall = []
        seq_fallen = []
        for _, r in rows.iterrows():
            ln = LABEL_NAMES[int(r["label"])]
            seq_16.append(ln)
            seq_fall.append(str(r["fall_label"]))
            seq_fallen.append(str(r["fallen_label"]))
        print(f"    16-class:  {' -> '.join(seq_16)}")
        print(f"    fall:      {' -> '.join(seq_fall)}")
        print(f"    fallen:    {' -> '.join(seq_fallen)}")

    # 4. Save
    print(f"\n--- 4. Saving ---")
    cols = list(df.columns)
    # Move binary labels next to label columns for readability
    label_idx = cols.index("label_name")
    binary_cols = ["fall_label", "fallen_label"]
    new_cols = cols[:label_idx+1] + binary_cols + [c for c in cols[label_idx+1:] if c not in binary_cols]
    df = df[new_cols]
    df.to_csv(CLIPS_CSV, index=False)
    print(f"  Updated {CLIPS_CSV}")
    print(f"  Columns: {new_cols}")

    # 5. Per-split summary
    print(f"\n--- 5. Per-split statistics ---")
    print(f"  {'split':>6s}  {'clips':>7s}  {'fall=1':>7s}  {'fallen=1':>9s}")
    print(f"  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*9}")
    for split_name in ["train", "val", "test"]:
        sdf = df[df["split"] == split_name]
        print(f"  {split_name:>6s}  {len(sdf):>7,d}  "
              f"{sdf['fall_label'].sum():>7,d}  {sdf['fallen_label'].sum():>9,d}")

    print(f"\n{'='*55}")
    print(f"  Step 3 complete - binary labels added.")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
