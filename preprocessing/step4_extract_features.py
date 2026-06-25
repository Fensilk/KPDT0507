"""
Step 4: Clip Feature Extraction with ResNet18 (NVDEC Hardware Decode)

Uses ffmpeg + NVDEC (av1_cuvid) to hardware-decode AV1 on the GPU,
then pipes raw RGB frames to Python for tensor conversion.
ResNet18 inference runs on the same GPU via PyTorch CUDA.

NVDEC and CUDA cores are separate hardware units — they don't compete.

Requires: ffmpeg with av1_cuvid decoder in PATH, pre-extracted .mp4 files.

Output: data/features/{path}.pt with keys:
  features:     [5, 512]   clip-level ResNet18 features
  labels:       [5]        16-class action label per clip
  fall_label:   [5]        binary fall label
  fallen_label: [5]        binary fallen label
"""
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

import random

import numpy as np
import pandas as pd
import torch
from torchvision.models import resnet18, ResNet18_Weights

# --- Config ---
OMNIFALL_ROOT = Path(__file__).resolve().parent.parent / "DATASET-omnifall"
EXTRACTED_ROOT = OMNIFALL_ROOT / "data_files" / "extracted"
CLIPS_CSV = Path(__file__).resolve().parent.parent / "data" / "ofsyn_clips.csv"
FRAME_CSV = Path(__file__).resolve().parent.parent / "data" / "ofsyn_frame_labels.csv"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def _find_ffmpeg():
    """Locate ffmpeg binary: check PATH first, then winget install path."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    # Winget BtbN FFmpeg installs here by default
    winget_root = Path(os.path.expandvars(
        r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\BtbN.FFmpeg.GPL_"
        r"Microsoft.Winget.Source_8wekyb3d8bbwe"
    ))
    if winget_root.exists():
        candidates = sorted(winget_root.glob("ffmpeg-*"), reverse=True)
        for c in candidates:
            exe = c / "bin" / "ffmpeg.exe"
            if exe.exists():
                return str(exe)
    return "ffmpeg"

FFMPEG = _find_ffmpeg()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 8               # ffmpeg subprocess threads
GPU_BATCH = 12                 # videos per GPU forward
SUBMIT_AHEAD = GPU_BATCH * 4  # decode jobs kept in flight

FRAME_SIZE = 224 * 224 * 3    # bytes per RGB24 frame

MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)

LABEL_NAMES = [
    "walk", "fall", "fallen", "sit_down", "sitting",
    "lie_down", "lying", "stand_up", "standing", "other",
    "kneel_down", "kneeling", "squat_down", "squatting", "crawl", "jump",
]


def build_model():
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity() # pyright: ignore[reportAttributeAccessIssue]
    model.eval()
    model.to(DEVICE)
    return model


_TIMING_BUF = []  # (ffmpeg_ms, numpy_ms, n_frames)

def decode_and_resize_hw(video_path):
    """ffmpeg NVDEC AV1 decode + GPU resize → [N,3,224,224] CPU tensor.
    Returns (tensor, video_path_str) or (None, video_path_str) on failure.
    """
    try:
        t0 = time.perf_counter()
        proc = subprocess.run(
            [
                FFMPEG, "-y",
                "-hwaccel", "cuda",
                "-c:v", "av1_cuvid",
                "-i", str(video_path),
                "-vf", "scale=224:224,format=rgb24",
                "-f", "rawvideo",
                "pipe:1",
            ],
            capture_output=True,
            timeout=30,
        )
        t1 = time.perf_counter()
        stdout = proc.stdout or b""
        if proc.returncode != 0 or len(stdout) == 0:
            return None, str(video_path)

        n_frames = len(stdout) // FRAME_SIZE
        if n_frames == 0:
            return None, str(video_path)

        raw = np.frombuffer(stdout[:n_frames * FRAME_SIZE], dtype=np.uint8)
        frames = raw.reshape(n_frames, 224, 224, 3)

        tensor = torch.from_numpy(frames.copy()).float() / 255.0  # [N,224,224,3]
        tensor = tensor.permute(0, 3, 1, 2)                      # [N,3,224,224]
        t2 = time.perf_counter()

        if len(_TIMING_BUF) < 200:
            _TIMING_BUF.append(((t1 - t0) * 1000, (t2 - t1) * 1000, n_frames))
        return tensor, str(video_path)
    except Exception:
        return None, str(video_path)


@torch.no_grad()
def extract_frame_features(frame_feats):
    """Pad/trim to 80 frames, keep per-frame features -> [80, 512]."""
    n = frame_feats.shape[0]
    if n < 80:
        pad = frame_feats[-1:].repeat(80 - n, 1)
        frame_feats = torch.cat([frame_feats, pad], dim=0)
    else:
        frame_feats = frame_feats[:80]
    return frame_feats  # [80, 512]


@torch.no_grad()
def process_gpu_batch(model, batch, path_to_info):
    """GPU forward pass for one batch of decoded videos. Saves .pt files."""
    tensors = [t for t, _ in batch]
    # Convert Windows paths back to CSV-relative paths (e.g. fall/fall_ch_001)
    paths = [
        Path(p).relative_to(EXTRACTED_ROOT).with_suffix("").as_posix()
        for _, p in batch
    ]

    big = torch.cat(tensors, dim=0).to(DEVICE)
    big = (big - MEAN) / STD

    feats = model(big)  # [total_frames, 512]

    offset = 0
    saved = 0
    for path, tensor in zip(paths, tensors):
        nf = tensor.shape[0]
        frame_feats = extract_frame_features(feats[offset:offset + nf])   # [80, 512]
        offset += nf

        info = path_to_info[path]
        result = {
            "features": frame_feats.cpu(),
            "labels": info["labels"],           # [80]
            "fall_label": info["fall_label"],    # [80]
            "fallen_label": info["fallen_label"], # [80]
        }

        out_subdir = OUTPUT_DIR / os.path.dirname(path)
        out_subdir.mkdir(parents=True, exist_ok=True)
        torch.save(result, OUTPUT_DIR / f"{path}.pt")
        saved += 1

    del big, feats
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return saved


def main():
    print("=" * 65)
    print("  Step 4: Clip Feature Extraction (NVDEC Hardware Decode)")
    print(f"  FFmpeg:  {FFMPEG}")
    print(f"  Device:  {DEVICE}")
    print(f"  GPU batch: {GPU_BATCH}, Workers: {NUM_WORKERS}")
    print("=" * 65)

    # 0. Verify ffmpeg & extracted files
    print("\n--- 0. Checking prerequisites ---")
    proc = subprocess.run([FFMPEG, "-decoders"], capture_output=True, text=True,
                          timeout=10)
    if "av1_cuvid" in proc.stdout:
        print("  av1_cuvid decoder: OK")
    else:
        print("  FATAL: av1_cuvid decoder not found. "
              "Install ffmpeg with NVDEC support (e.g. winget install BtbN.FFmpeg.GPL).")
        return

    mp4_count = sum(1 for _ in EXTRACTED_ROOT.rglob("*.mp4"))
    print(f"  Extracted .mp4 files: {mp4_count:,}")
    if mp4_count == 0:
        print("  FATAL: No .mp4 files under", EXTRACTED_ROOT)
        print("  Run extract_tar.py first.")
        return

    # 1. Model
    print("\n--- 1. Building ResNet18 ---")
    model = build_model()
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 2. Clips
    print("\n--- 2. Loading clip annotations ---")
    clips_df = pd.read_csv(CLIPS_CSV)
    unique_paths = sorted(clips_df["path"].unique())
    n_videos = len(unique_paths)
    print(f"  {len(clips_df):,} clips, {n_videos:,} videos")

    # Load frame-level labels for per-frame annotation
    print("\n--- 2b. Loading frame-level labels ---")
    frames_df = pd.read_csv(FRAME_CSV)
    print(f"  {len(frames_df):,} frame labels, {frames_df['path'].nunique():,} videos")

    path_to_info = {}
    for path in unique_paths:
        finfo = frames_df[frames_df["path"] == path].sort_values("frame_idx")  # pyright: ignore[reportCallIssue]
        if len(finfo) != 80:
            print(f"  [WARN] {path}: expected 80 frames, got {len(finfo)}")
        path_to_info[path] = {
            "labels": torch.tensor(finfo["label"].values, dtype=torch.long),
            "fall_label": torch.tensor(finfo["fall_label"].values, dtype=torch.long),
            "fallen_label": torch.tensor(finfo["fallen_label"].values, dtype=torch.long),
        }

    # 3. Resume
    print("\n--- 3. Checking existing files ---")
    existing = set()
    stale = 0
    for pt_file in OUTPUT_DIR.glob("**/*.pt"):
        rel = pt_file.relative_to(OUTPUT_DIR)
        # Validate shape — old clip-level (5,512) files need re-extraction
        try:
            d = torch.load(pt_file, weights_only=False)
            if d["features"].shape == (80, 512) and d["labels"].shape == (80,):
                existing.add(rel.with_suffix("").as_posix())
            else:
                stale += 1
        except Exception:
            stale += 1
    pending = [p for p in unique_paths if p not in existing]
    n_done = len(existing)
    n_pending = len(pending)
    print(f"  Done: {n_done:,}, Pending: {n_pending:,}, Stale (re-extract): {stale}")

    if not pending:
        print("  All done.")
        return

    # 4. Pipeline: CPU pool runs ffmpeg NVDEC, main thread runs GPU batches
    print(f"\n--- 4. Extracting features (NVDEC pipeline) ---")
    t_start = time.time()
    processed = 0
    skipped = 0
    last_log = t_start

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {}   # {future: csv_path}
        next_idx = 0
        buffer = []    # [(tensor, full_path), ...]

        # Submit initial window of decode jobs
        for i in range(min(SUBMIT_AHEAD, len(pending))):
            path = pending[i]
            vid_path = EXTRACTED_ROOT / f"{path}.mp4"
            futures[executor.submit(decode_and_resize_hw, vid_path)] = path
            next_idx = i + 1

        while futures or buffer:
            # Collect completed decode jobs
            if futures:
                done, not_done = wait(futures, return_when=FIRST_COMPLETED)
                for f in done:
                    path = futures[f]
                    tensor, full_path = f.result()
                    if tensor is not None:
                        buffer.append((tensor, full_path))
                    else:
                        skipped += 1
                futures = {f: futures[f] for f in not_done}

            # Refill submission window
            while len(futures) < SUBMIT_AHEAD and next_idx < len(pending):
                path = pending[next_idx]
                vid_path = EXTRACTED_ROOT / f"{path}.mp4"
                futures[executor.submit(decode_and_resize_hw, vid_path)] = path
                next_idx += 1

            # GPU: drain buffer in full batches
            while len(buffer) >= GPU_BATCH:
                batch = buffer[:GPU_BATCH]
                del buffer[:GPU_BATCH]
                processed += process_gpu_batch(model, batch, path_to_info)

            # If no more futures and no more pending jobs, drain partial buffer
            if not futures and next_idx >= len(pending) and buffer:
                processed += process_gpu_batch(model, buffer, path_to_info)
                buffer.clear()

            # Progress every 15s (NVDEC is faster, so log more often)
            now = time.time()
            if now - last_log >= 15:
                elapsed = now - t_start
                done_total = n_done + processed + skipped
                rate = processed / elapsed if elapsed > 0 else 0
                remaining = n_pending - processed - skipped
                eta = remaining / rate if rate > 0 else 0
                vram = torch.cuda.memory_allocated() / 1e9 if DEVICE.type == "cuda" else 0
                print(f"  [{done_total:>5d}/{n_videos}] {rate:.0f} vids/s, "
                      f"ETA {eta/60:.0f} min, VRAM {vram:.1f}GB, "
                      f"done={n_done+processed} skip={skipped}")
                sys.stdout.flush()
                last_log = now

        # Process final partial buffer
        if buffer:
            processed += process_gpu_batch(model, buffer, path_to_info)

    t_total = time.time() - t_start
    print(f"\n  Done: {processed:,} processed, {skipped} skipped")
    if processed > 0:
        print(f"  Total time: {t_total/60:.1f} min ({t_total/processed:.2f}s/video)")

    # Timing breakdown (first 200 samples)
    if _TIMING_BUF:
        ffmpeg_ms = np.array([t[0] for t in _TIMING_BUF])
        numpy_ms = np.array([t[1] for t in _TIMING_BUF])
        total_ms = ffmpeg_ms + numpy_ms
        print(f"\n  --- Decode timing breakdown (n={len(_TIMING_BUF)}) ---")
        print(f"  {'Stage':>12s}  {'mean':>8s}  {'median':>8s}  {'min':>8s}  {'max':>8s}")
        print(f"  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
        print(f"  {'ffmpeg':>12s}  {ffmpeg_ms.mean():>7.1f}ms  {np.median(ffmpeg_ms):>7.1f}ms  {ffmpeg_ms.min():>7.1f}ms  {ffmpeg_ms.max():>7.1f}ms")
        print(f"  {'numpy/torch':>12s}  {numpy_ms.mean():>7.1f}ms  {np.median(numpy_ms):>7.1f}ms  {numpy_ms.min():>7.1f}ms  {numpy_ms.max():>7.1f}ms")
        print(f"  {'total':>12s}  {total_ms.mean():>7.1f}ms  {np.median(total_ms):>7.1f}ms  {total_ms.min():>7.1f}ms  {total_ms.max():>7.1f}ms")
        ffmpeg_pct = ffmpeg_ms.sum() / total_ms.sum() * 100
        print(f"  ffmpeg share: {ffmpeg_pct:.0f}%")

    # 5. Verify
    print(f"\n--- 5. Verification ---")
    sample_path = unique_paths[random.randint(0, len(unique_paths) - 1)]
    data = torch.load(OUTPUT_DIR / f"{sample_path}.pt", weights_only=False)
    feats = data["features"]
    labels = data["labels"]
    print(f"  {sample_path}: features={list(feats.shape)}")  # should be [80, 512]
    print(f"    labels ({len(labels)}): {labels.tolist()[:10]}... (showing first 10)")
    print(f"    names:  {[LABEL_NAMES[l] for l in labels.tolist()[:10]]}...")
    print(f"    fall sum={data['fall_label'].sum().item()}, "
          f"fallen sum={data['fallen_label'].sum().item()}")
    print(f"    feat mean={feats.mean():.4f}, std={feats.std():.4f}")

    n_saved = len(list(OUTPUT_DIR.glob("**/*.pt")))
    print(f"  Total .pt files: {n_saved}")

    print(f"\n{'='*65}")
    print(f"  Step 4 complete.")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
