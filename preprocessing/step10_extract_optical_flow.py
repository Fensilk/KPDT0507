"""
Step 10: Extract Farneback optical flow features from OF-Syn videos.

对每对相邻帧计算稠密光流，空间池化到 8×8 网格 → 128 维特征，
与 ViT-g 帧特征（80 帧/视频）对齐。

输出 data/omnifall_optical_flow.npz:
    pose_features:      (960000, 128) float32  ← 复用 dataset.py 的辅助特征加载机制
    video_paths:        (12000,) object
    video_clip_counts:  (12000,) int64 (all 80)
    video_start_indices:(12000,) int64

用法：
    python preprocessing/step10_extract_optical_flow.py
    python preprocessing/step10_extract_optical_flow.py --workers 8
"""
import os
import sys
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_DIR = os.path.join(ROOT, "DATASET-omnifall", "data_files", "extracted")
PT_OUTPUT_DIR = os.path.join(ROOT, "data", "optical_flow_features")
NPZ_OUTPUT_PATH = os.path.join(ROOT, "data", "omnifall_optical_flow.npz")

N_FRAMES = 80
FLOW_SIZE = (256, 256)   # 计算光流前下采样分辨率（加速）
POOL_GRID = 8            # 空间池化网格 8×8
FLOW_DIM = POOL_GRID * POOL_GRID * 2  # 128


def extract_flow_from_video(video_path):
    """
    读视频，计算 backward Farneback 光流，空间池化到 8×8×2=128 维。

    Returns: (80, 128) float32 array, or None on failure.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, FLOW_SIZE, interpolation=cv2.INTER_LINEAR)
        frames.append(gray)
    cap.release()

    if len(frames) < N_FRAMES:
        while len(frames) < N_FRAMES:
            frames.append(frames[-1] if frames else np.zeros(FLOW_SIZE, dtype=np.uint8))
    frames = frames[:N_FRAMES]

    flow_feats = np.zeros((N_FRAMES, FLOW_DIM), dtype=np.float32)
    for t in range(1, N_FRAMES):
        flow = cv2.calcOpticalFlowFarneback(
            frames[t - 1], frames[t], None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )  # (H, W, 2)
        # 空间池化到 8×8
        flow_pooled = cv2.resize(flow, (POOL_GRID, POOL_GRID),
                                 interpolation=cv2.INTER_AREA)  # (8, 8, 2)
        flow_feats[t] = flow_pooled.reshape(-1)
    # flow[0] 保持 0（backward flow 无前帧）

    return flow_feats


def process_video(args_tuple):
    video_path, rel_path = args_tuple
    out_path = os.path.join(PT_OUTPUT_DIR, rel_path.replace(".mp4", ".pt"))

    if os.path.exists(out_path):
        return rel_path, True, "skipped"

    try:
        feats = extract_flow_from_video(video_path)
        if feats is None:
            return rel_path, False, "frame read failed"

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.save(out_path.replace(".pt", ".npy"), feats)  # 直接存 npy 更简单
        return rel_path, True, "OK"
    except Exception as e:
        return rel_path, False, str(e)


def aggregate_to_npz():
    import glob
    npy_files = sorted(glob.glob(os.path.join(PT_OUTPUT_DIR, "**", "*.npy"), recursive=True))
    if not npy_files:
        print("[ERROR] No .npy files found!")
        return

    n_videos = len(npy_files)
    features_list = []
    video_paths = []
    video_clip_counts = np.empty(n_videos, dtype=np.int64)
    video_start_indices = np.empty(n_videos, dtype=np.int64)

    offset = 0
    for i, npyf in enumerate(npy_files):
        rel = os.path.relpath(npyf, PT_OUTPUT_DIR).replace("\\", "/").replace(".npy", ".mp4")
        feats = np.load(npyf)  # (80, 128)
        features_list.append(feats)
        video_paths.append(rel)
        video_clip_counts[i] = feats.shape[0]
        video_start_indices[i] = offset
        offset += feats.shape[0]
        if (i + 1) % 2000 == 0:
            print(f"  Aggregated {i+1:,}/{n_videos:,} videos...")

    all_features = np.concatenate(features_list, axis=0).astype(np.float32)

    np.savez_compressed(
        NPZ_OUTPUT_PATH,
        pose_features=all_features,  # 复用 dataset.py 的辅助特征 key
        video_paths=np.array(video_paths, dtype=object),
        video_clip_counts=video_clip_counts,
        video_start_indices=video_start_indices,
    )
    size_mb = os.path.getsize(NPZ_OUTPUT_PATH) / 1024 / 1024
    print(f"\n[DONE] NPZ saved: {NPZ_OUTPUT_PATH}")
    print(f"  Videos: {n_videos}, Frames: {offset}")
    print(f"  Features: [{offset}, {all_features.shape[1]}], size {size_mb:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Extract Farneback optical flow features")
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--skip_aggregation", action="store_true")
    args = parser.parse_args()

    categories = sorted(os.listdir(VIDEO_DIR))
    if args.category:
        categories = [args.category]

    all_videos = []
    for cat in categories:
        cat_dir = os.path.join(VIDEO_DIR, cat)
        if not os.path.isdir(cat_dir):
            continue
        for vname in sorted(os.listdir(cat_dir)):
            if vname.endswith(".mp4"):
                all_videos.append((os.path.join(cat_dir, vname), f"{cat}/{vname}"))

    n_total = len(all_videos)
    already_done = sum(1 for _, rel in all_videos
                       if os.path.exists(os.path.join(PT_OUTPUT_DIR, rel.replace(".mp4", ".npy"))))
    n_todo = n_total - already_done
    print(f"Total videos: {n_total:,} | Done: {already_done:,} | Todo: {n_todo:,}")
    print(f"Workers: {args.workers}")

    if n_todo == 0:
        print("[DONE] All videos processed!")
        if not args.skip_aggregation:
            aggregate_to_npz()
        return

    print(f"Processing with {args.workers} workers...")
    t0 = time.time()
    n_ok = n_fail = n_skip = 0

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_video, v): v for v in all_videos}
        for i, fut in enumerate(as_completed(futures)):
            rel, success, msg = fut.result()
            if success:
                if msg == "skipped":
                    n_skip += 1
                else:
                    n_ok += 1
            else:
                n_fail += 1
                print(f"  [FAIL] {rel}: {msg}", flush=True)

            done = i + 1
            if done % 500 == 0 or done == n_total:
                elapsed = time.time() - t0
                rate = (n_ok + n_skip) / elapsed if elapsed > 0 else 0
                eta = (n_todo - n_ok - n_skip) / rate - elapsed if rate > 0 else 0
                print(f"  [{done:>6d}/{n_total}] {100*done/n_total:5.1f}% | "
                      f"OK={n_ok} SKIP={n_skip} FAIL={n_fail} | "
                      f"{rate:.1f} vid/s | eta={max(0, eta)/60:.0f}min", flush=True)

    print(f"\nDONE! Time {time.time()-t0:.0f}s | OK={n_ok} SKIP={n_skip} FAIL={n_fail}")
    if not args.skip_aggregation:
        print("\nAggregating to NPZ...")
        aggregate_to_npz()


if __name__ == "__main__":
    main()
