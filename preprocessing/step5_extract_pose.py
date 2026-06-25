"""
Step 5: Extract MediaPipe Pose landmarks from OF-Syn videos.

Uses MediaPipe Pose (video tracking mode, static_image_mode=False) to
extract 33 body landmarks (x, y, visibility = 99-dim) per frame.
Saves per-video .pt files.

Key optimization: static_image_mode=False → ~12ms/frame (vs 22ms for static).
ThreadPoolExecutor is used because:
  - Each MediaPipe instance internally uses XNNPACK multi-threading
  - Python GIL limits thread parallelism but MediaPipe releases GIL during inference
  - ThreadPoolExecutor has high spawn overhead on Windows

Output: data/pose_features/{category}/{video_name}.pt
    Each file: {"landmarks": (80, 99) float32, "path": str}

Performance: ~0.9s/video single-thread (with video tracking mode).
With 4 workers: ~2-3 vid/s. 12,000 videos ≈ 1-2h.
Checkpointing: skips videos already processed.

Usage:
    python preprocessing/step5_extract_pose.py                    # all categories
    python preprocessing/step5_extract_pose.py --workers 4        # default
"""

import os
import sys
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch

# === Module-level constants (needed for pickling in ThreadPoolExecutor) ===
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_DIR = os.path.join(ROOT, "DATASET-omnifall", "data_files", "extracted")
OUTPUT_DIR = os.path.join(ROOT, "data", "pose_features")
N_FRAMES = 80  # match feature extraction: 5 clips x 16 frames


def extract_pose_from_video(video_path: str) -> np.ndarray:
    """
    Extract MediaPipe Pose landmarks from a single video.
    Must be at module level for ThreadPoolExecutor pickling.

    Returns: (80, 99) float32 array, or None on failure.
    """
    import cv2
    import mediapipe as mp

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    # Pad/trim to exactly N_FRAMES
    if len(frames) < N_FRAMES:
        while len(frames) < N_FRAMES:
            frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))
    frames = frames[:N_FRAMES]

    # Video tracking mode: ~12ms/frame (vs ~22ms for static_image_mode=True)
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,  # video tracking mode (faster, uses temporal info)
        model_complexity=0,       # lite model
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    landmarks_list = []
    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = pose.process(rgb)

        if results.pose_landmarks:
            lms = []
            for lm in results.pose_landmarks.landmark:
                lms.extend([lm.x, lm.y, lm.visibility])
            landmarks_list.append(lms)
        else:
            landmarks_list.append([0.0] * 99)

    pose.close()
    return np.array(landmarks_list, dtype=np.float32)


def process_video(args_tuple):
    """
    Process a single video. Must be at module level for pickling.

    Args:
        args_tuple: (video_path, rel_path) where rel_path = "category/name.mp4"

    Returns:
        (rel_path, success: bool, error_msg: str)
    """
    video_path, rel_path = args_tuple
    out_path = os.path.join(OUTPUT_DIR, rel_path.replace(".mp4", ".pt"))

    # Checkpoint
    if os.path.exists(out_path):
        return (rel_path, True, "skipped")

    try:
        landmarks = extract_pose_from_video(video_path)
        if landmarks is None:
            return (rel_path, False, "extraction failed")

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save({
            "landmarks": torch.from_numpy(landmarks),
            "path": rel_path,
        }, out_path)

        return (rel_path, True, "OK")
    except Exception as e:
        return (rel_path, False, str(e))


def main():
    parser = argparse.ArgumentParser(description="Extract MediaPipe Pose landmarks")
    parser.add_argument("--category", type=str, default=None,
                        help="Process single category")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers (threads). "
                             "Default=4.")
    args = parser.parse_args()

    # Collect all video paths
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
                vpath = os.path.join(cat_dir, vname)
                rel = f"{cat}/{vname}"
                all_videos.append((vpath, rel))

    # Count already processed
    already_done = sum(
        1 for _, rel in all_videos
        if os.path.exists(os.path.join(OUTPUT_DIR, rel.replace(".mp4", ".pt")))
    )

    n_total = len(all_videos)
    n_todo = n_total - already_done
    print(f"Total videos: {n_total:,}")
    print(f"Already done: {already_done:,}")
    print(f"Remaining:    {n_todo:,}")
    print(f"Categories:   {categories}")
    print(f"Workers:      {args.workers} threads")

    if n_todo == 0:
        print("[DONE] All videos already processed!")
        return

    # Estimate
    est_per_video = 0.9  # seconds with video tracking mode
    est_total = n_todo * est_per_video / args.workers
    print(f"Est. time:    {est_total/60:.0f} min (~{est_total/3600:.1f}h) "
          f"with {args.workers} processes")

    # Process
    print(f"\nProcessing...")
    t0 = time.time()
    n_ok, n_fail, n_skip = 0, 0, 0

    # ThreadPoolExecutor: each process has its own Python interpreter,
    # GIL, and MediaPipe instance. No GIL contention, no XNNPACK cross-talk.
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_video, v): v for v in all_videos}

        for i, future in enumerate(as_completed(futures)):
            rel_path, success, msg = future.result()
            if success:
                if msg == "skipped":
                    n_skip += 1
                else:
                    n_ok += 1
            else:
                n_fail += 1
                print(f"  [FAIL] {rel_path}: {msg}", flush=True)

            # Progress every 200 videos
            done = i + 1
            if done % 200 == 0 or done == n_total:
                elapsed = time.time() - t0
                rate = (n_ok + n_skip) / elapsed if elapsed > 0 else 0
                eta = n_todo / rate - elapsed if rate > 0 else 0
                print(f"  [{done:>6d}/{n_total}] "
                      f"{100*done/n_total:5.1f}% | "
                      f"OK={n_ok} SKIP={n_skip} FAIL={n_fail} | "
                      f"{rate:.1f} vid/s | "
                      f"elapsed={elapsed/60:.0f}min | "
                      f"eta={eta/60:.0f}min",
                      flush=True)

    total_time = time.time() - t0
    print(f"\n{'='*60}")
    print(f"DONE! Total time: {total_time/60:.1f} min ({total_time/3600:.2f}h)")
    print(f"Successful: {n_ok}, Skipped: {n_skip}, Failed: {n_fail}")
    print(f"Output dir: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
