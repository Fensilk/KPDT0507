"""
OmniFall TCN 数据集加载.

将预处理的 NPZ 数据按视频分组为 (T=5, 512) 序列,
配合 split CSV 划分 train/val/test.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


class OmniFallTCNDataset(Dataset):
    """
    OmniFall 数据集, 按视频分组为 5-clip 序列.

    每个样本是一个完整的视频: features (5, 512), labels (5,).
    """

    def __init__(
        self,
        npz_path: str,
        clips_csv_path: str,
        split_csv_path: str,
        use_diff: bool = False,
        pose_npz_path: str = None,
    ):
        """
        Args:
            npz_path: omnifall_preprocessed.npz 路径.
            clips_csv_path: ofsyn_clips.csv 路径 (用于 split 信息).
            split_csv_path: train/val/test.csv 路径 (确定本 split 的视频).
            use_diff: 是否拼接相邻 clip 差分特征 (Δclip).
            pose_npz_path: 姿态特征 NPZ 路径 (可选, 用于 Pose+TCN).
        """
        self.use_diff = use_diff
        self.feature_dim = 512  # base RGB dim

        # --- 加载 NPZ ---
        print(f"[INFO] Loading NPZ from {npz_path}...")
        data = np.load(npz_path, allow_pickle=True)
        self.features_all = data["features"]           # (60000, 512)
        self.labels_16_all = data["labels_16"]          # (60000,)
        self.fall_labels_all = data["fall_labels"]      # (60000,)
        self.fallen_labels_all = data["fallen_labels"]  # (60000,)

        # --- 加载姿态特征 (可选) ---
        self.pose_all = None
        if pose_npz_path is not None:
            print(f"[INFO] Loading pose NPZ from {pose_npz_path}...")
            pose_data = np.load(pose_npz_path, allow_pickle=True)
            self.pose_all = pose_data["pose_features"]  # (60000, pose_dim)
            self.feature_dim += self.pose_all.shape[1]
            print(f"[INFO] Pose dim: {self.pose_all.shape[1]}, "
                  f"total input dim: {self.feature_dim}")

        if use_diff:
            self.feature_dim += 512  # diff duplicates RGB dim
            print(f"[INFO] Using Δclip diff features, total input dim: {self.feature_dim}")

        # --- 加载 clips CSV: 获取每个视频的 split ---
        clips_df = pd.read_csv(clips_csv_path)

        # 获取每个 video 的 split 归属 (取第一个 clip 的 split, 同一视频的所有 clip split 相同)
        video_splits = clips_df.groupby("path")["split"].first()

        # --- 加载 split CSV: 确定本 split 包含哪些视频 ---
        split_df = pd.read_csv(split_csv_path)
        split_paths = set(split_df["path"].str.strip())

        # --- 按视频分组 ---
        # NPZ 中 video_paths 的顺序就是 start_indices 对应的顺序
        # 直接使用 video_start_indices + video_clip_counts 重建
        video_start_indices = data["video_start_indices"]
        video_clip_counts = data["video_clip_counts"]

        self.feature_seqs = []
        self.label_16_seqs = []
        self.fall_seqs = []
        self.fallen_seqs = []
        self.video_paths = []

        n_videos = len(video_start_indices)
        skipped = 0

        for vi in range(n_videos):
            path = str(data["video_paths"][vi])

            # 检查是否在本 split 中, 且 clips_csv 中 split 一致
            if path not in split_paths:
                continue
            if path in video_splits.index:
                if video_splits[path] != Path(split_csv_path).stem:
                    # split 不匹配, 跳过 (安全校验)
                    # 实际上 split_csv 文件名就是 split 名
                    pass

            count = video_clip_counts[vi]
            if count != 5:
                skipped += 1
                continue  # 跳过非标准 clip 数的视频

            start = video_start_indices[vi]
            end = start + count

            # --- Base RGB features ---
            feat = torch.from_numpy(self.features_all[start:end].copy()).float()  # (5, 512)

            # --- 2a: Δclip 差分特征 ---
            if self.use_diff:
                # diff[t] = feat[t] - feat[t-1], diff[0] = 0
                diff = torch.zeros_like(feat)
                diff[1:] = feat[1:] - feat[:-1]
                feat = torch.cat([feat, diff], dim=-1)  # (5, 1024)

            # --- 2b: 姿态特征 ---
            if self.pose_all is not None:
                pose_feat = torch.from_numpy(
                    self.pose_all[start:end].copy()
                ).float()  # (5, pose_dim)
                feat = torch.cat([feat, pose_feat], dim=-1)

            self.feature_seqs.append(feat)
            self.label_16_seqs.append(
                torch.from_numpy(self.labels_16_all[start:end].copy()).long()
            )
            self.fall_seqs.append(
                torch.from_numpy(self.fall_labels_all[start:end].copy()).long()
            )
            self.fallen_seqs.append(
                torch.from_numpy(self.fallen_labels_all[start:end].copy()).long()
            )
            self.video_paths.append(path)

        if skipped > 0:
            print(f"[WARN] Skipped {skipped} videos with non-5 clip counts")

        print(
            f"[INFO] Loaded {len(self)} videos for split "
            f"'{Path(split_csv_path).stem}' "
            f"| Fall: {self._count_positive(self.fall_seqs):.1f}% "
            f"| Fallen: {self._count_positive(self.fallen_seqs):.1f}%"
        )

    @staticmethod
    def _count_positive(seqs: list) -> float:
        total = sum(s.numel() for s in seqs)
        if total == 0:
            return 0.0
        pos = sum((s == 1).sum().item() for s in seqs)
        return 100.0 * pos / total

    def __len__(self) -> int:
        return len(self.feature_seqs)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            features: (5, 512) float32
            labels_16: (5,) int64
            fall_labels: (5,) int64
            fallen_labels: (5,) int64
        """
        return {
            "features": self.feature_seqs[idx],
            "labels_16": self.label_16_seqs[idx],
            "fall_labels": self.fall_seqs[idx],
            "fallen_labels": self.fallen_seqs[idx],
        }


# 修正 Path 引用
from pathlib import Path


def _get_sample_weights(dataset: OmniFallTCNDataset) -> torch.Tensor:
    """
    为 WeightedRandomSampler 计算样本权重.

    基于 16 类标签的多数类来确定权重, 使每个类被等概率采样.
    对每个视频, 取其众数标签作为该视频的代表标签.

    Args:
        dataset: OmniFallTCNDataset 实例.
    Returns:
        weights: (N,) float32 权重数组.
    """
    # 每个视频取第一个 clip 的标签作为代表 (或取众数)
    labels = torch.tensor([
        int(torch.mode(seq).values.item()) for seq in dataset.label_16_seqs
    ])

    class_counts = torch.bincount(labels, minlength=16).float()
    class_weights = 1.0 / (class_counts + 1e-6)
    weights = class_weights[labels]
    return weights


def create_dataloaders(
    npz_path: str,
    clips_csv_path: str,
    splits_dir: str,
    batch_size: int = 64,
    num_workers: int = 0,
    use_weighted_sampler: bool = True,
    use_diff: bool = False,
    pose_npz_path: str = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建 train/val/test DataLoader.

    Args:
        npz_path: NPZ 文件路径.
        clips_csv_path: clips CSV 路径.
        splits_dir: split CSV 目录 (包含 train.csv, val.csv, test.csv).
        batch_size: 批次大小.
        num_workers: DataLoader 工作线程数 (Windows 建议 0).
        use_weighted_sampler: 是否在 train 上使用加权采样.
        use_diff: 是否拼接相邻 clip 差分特征.
        pose_npz_path: 姿态特征 NPZ 路径 (可选).

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_dataset = OmniFallTCNDataset(
        npz_path=npz_path,
        clips_csv_path=clips_csv_path,
        split_csv_path=os.path.join(splits_dir, "train.csv"),
        use_diff=use_diff,
        pose_npz_path=pose_npz_path,
    )
    val_dataset = OmniFallTCNDataset(
        npz_path=npz_path,
        clips_csv_path=clips_csv_path,
        split_csv_path=os.path.join(splits_dir, "val.csv"),
        use_diff=use_diff,
        pose_npz_path=pose_npz_path,
    )
    test_dataset = OmniFallTCNDataset(
        npz_path=npz_path,
        clips_csv_path=clips_csv_path,
        split_csv_path=os.path.join(splits_dir, "test.csv"),
        use_diff=use_diff,
        pose_npz_path=pose_npz_path,
    )

    # 训练集加权采样
    if use_weighted_sampler:
        weights = _get_sample_weights(train_dataset)
        sampler = WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def test_dataset():
    """测试数据集加载."""
    BASE = r"c:\Users\Lizhe\Documents\KPDT0507"
    npz_path = os.path.join(BASE, "data", "omnifall_preprocessed.npz")
    clips_csv = os.path.join(BASE, "data", "ofsyn_clips.csv")
    splits_dir = os.path.join(BASE, "DATASET-omnifall", "splits", "syn", "random")

    if not os.path.exists(npz_path):
        print(f"[SKIP] NPZ not found at {npz_path}")
        return

    loaders = create_dataloaders(
        npz_path=npz_path,
        clips_csv_path=clips_csv,
        splits_dir=splits_dir,
        batch_size=8,
        num_workers=0,
    )

    for name, loader in zip(["train", "val", "test"], loaders):
        batch = next(iter(loader))
        f = batch["features"]
        print(
            f"[{name}] features: {f.shape} | "
            f"labels_16: {batch['labels_16'].shape} | "
            f"fall: {batch['fall_labels'].shape} | "
            f"fallen: {batch['fallen_labels'].shape}"
        )

        # 验证 shapes
        assert f.shape == (8, 5, 512), f"Expected (8,5,512) got {f.shape}"
        assert batch["labels_16"].shape == (8, 5)
        assert batch["fall_labels"].shape == (8, 5)
        assert batch["fallen_labels"].shape == (8, 5)

    print("[PASS] All dataset tests passed!")


# ============================================================
# Long Sequence Dataset (Phase 2: sliding window over frames)
# ============================================================

class LongSequenceDataset(Dataset):
    """
    Frame-level dataset with configurable sliding window.

    Loads frame-level NPZ, applies sliding window to produce
    (T, 512) sequences with (T,) labels per sample.

    Each video has 80 frames. A sliding window of size T with
    stride S produces (80 - T) // S + 1 sequences per video.
    """

    def __init__(
        self,
        npz_path: str,
        split_csv_path: str,
        window_size: int = 16,
        stride: int = 1,
        use_diff: bool = False,
        use_accel: bool = False,
        pose_npz_path: str = None,
    ):
        """
        Args:
            npz_path: Frame-level NPZ path (omnifall_frame_preprocessed.npz).
            split_csv_path: train/val/test.csv path.
            window_size: T — number of frames per sequence.
            stride: Sliding window stride (default 1 = max overlap).
            use_diff: If True, concat frame-to-frame diff features (Δframe).
            use_accel: If True, concat second-order diff features (acceleration).
            pose_npz_path: Optional frame-level pose NPZ path.
        """
        self.window_size = window_size
        self.stride = stride
        self.use_diff = use_diff
        self.use_accel = use_accel

        # --- Load NPZ ---
        print(f"[INFO] Loading frame-level NPZ from {npz_path}...")
        data = np.load(npz_path, allow_pickle=True, mmap_mode='r')
        self.features_all = data["features"]             # (N_frames, D)
        self.labels_16_all = data["labels_16"]            # (N_frames,)
        self.fall_labels_all = data["fall_labels"]        # (N_frames,)
        self.fallen_labels_all = data["fallen_labels"]    # (N_frames,)

        self.base_feature_dim = self.features_all.shape[1]
        self.feature_dim = self.base_feature_dim

        # --- Load pose features (optional) ---
        self.pose_all = None
        if pose_npz_path is not None:
            print(f"[INFO] Loading frame-level pose NPZ from {pose_npz_path}...")
            pose_data = np.load(pose_npz_path, allow_pickle=True, mmap_mode='r')
            self.pose_all = pose_data["pose_features"]  # (N_frames, pose_dim)
            self.feature_dim += self.pose_all.shape[1]
            print(f"[INFO] Pose dim: {self.pose_all.shape[1]}, "
                  f"total input dim: {self.feature_dim}")

        if use_diff:
            self.feature_dim += self.base_feature_dim  # diff duplicates base dim
            print(f"[INFO] Using Δframe diff features, total input dim: {self.feature_dim}")
        if use_accel:
            self.feature_dim += self.base_feature_dim  # accel duplicates base dim
            print(f"[INFO] Using second-order diff (accel) features, total input dim: {self.feature_dim}")

        video_start_indices = data["video_start_indices"]
        video_clip_counts = data["video_clip_counts"]
        video_paths_arr = data["video_paths"]

        # --- Load split ---
        split_df = pd.read_csv(split_csv_path)
        split_paths = set(split_df["path"].str.strip())

        # --- Build window index ---
        # Map: flat index -> (video_start_in_npz, window_start_offset)
        self.window_index = []  # list of (npz_start, frame_offset)
        self.video_windows = []  # list of (video_path, n_windows) for debugging

        n_videos = len(video_start_indices)
        skipped = 0

        for vi in range(n_videos):
            path = str(video_paths_arr[vi])

            if path not in split_paths:
                continue

            n_frames = int(video_clip_counts[vi])
            if n_frames < window_size:
                skipped += 1
                continue

            npz_start = int(video_start_indices[vi])
            n_windows = (n_frames - window_size) // stride + 1

            for w in range(n_windows):
                offset = w * stride
                self.window_index.append((npz_start, offset))

            self.video_windows.append((path, n_windows))

        if skipped > 0:
            print(f"[WARN] Skipped {skipped} videos with < {window_size} frames")

        n_total = len(self.window_index)
        n_videos_used = len(self.video_windows)
        expansion = n_total / max(n_videos_used, 1)
        print(
            f"[INFO] Loaded {n_total:,} windows from {n_videos_used:,} videos "
            f"for split '{Path(split_csv_path).stem}' "
            f"(T={window_size}, stride={stride}, {expansion:.1f}x expansion) "
            f"| Fall frames: {self._count_fall_percent():.1f}%"
        )

    def _count_fall_percent(self) -> float:
        """Estimate fall frame percentage using NPZ labels."""
        total = len(self.fall_labels_all)
        if total == 0:
            return 0.0
        pos = int(np.sum(self.fall_labels_all))
        return 100.0 * pos / total

    def __len__(self) -> int:
        return len(self.window_index)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            features: (T, feature_dim) float32
            labels_16: (T,) int64
            fall_labels: (T,) int64
            fallen_labels: (T,) int64
        """
        npz_start, offset = self.window_index[idx]
        s = npz_start + offset
        e = s + self.window_size

        # Base features
        feat = torch.from_numpy(
            self.features_all[s:e].copy()
        ).float()

        # Optional: temporal difference features (Δframe + accel)
        parts = [feat]
        if self.use_diff:
            diff = torch.zeros_like(feat)
            diff[1:] = feat[1:] - feat[:-1]
            parts.append(diff)
        if self.use_accel:
            accel = torch.zeros_like(feat)
            accel[1:-1] = feat[2:] - 2 * feat[1:-1] + feat[:-2]
            parts.append(accel)
        if len(parts) > 1:
            feat = torch.cat(parts, dim=-1)

        # Optional: pose features
        if self.pose_all is not None:
            pose_feat = torch.from_numpy(
                self.pose_all[s:e].copy()
            ).float()
            feat = torch.cat([feat, pose_feat], dim=-1)

        labels_16 = torch.from_numpy(
            self.labels_16_all[s:e].copy()
        ).long()

        # Compute ternary labels on-the-fly for Phase 6
        # 0=fall, 1=fallen, 2=normal
        ternary_labels = torch.full_like(labels_16, fill_value=2)  # default: normal
        ternary_labels[labels_16 == 1] = 0  # fall
        ternary_labels[labels_16 == 2] = 1  # fallen

        return {
            "features": feat,
            "labels_16": labels_16,
            "fall_labels": torch.from_numpy(
                self.fall_labels_all[s:e].copy()
            ).long(),
            "fallen_labels": torch.from_numpy(
                self.fallen_labels_all[s:e].copy()
            ).long(),
            "ternary_labels": ternary_labels,  # Phase 6: (T,) int64
        }


def _get_longseq_sample_weights(dataset: LongSequenceDataset) -> torch.Tensor:
    """
    Calculate sample weights for WeightedRandomSampler.

    Uses center-frame label as a proxy for window majority label.
    Reads directly from mmap'd NPZ array (O(1) per window, no data copy),
    making weight computation for 600k+ windows near-instant.

    Args:
        dataset: LongSequenceDataset instance.
    Returns:
        weights: (N,) float32 weights array.
    """
    n_total = len(dataset)
    half = dataset.window_size // 2

    # mmap O(1) label read — no tensor copy, just 8 bytes per window
    labels = []
    for npz_start, offset in dataset.window_index:
        center_idx = npz_start + offset + half
        labels.append(int(dataset.labels_16_all[center_idx]))

    labels_t = torch.tensor(labels, dtype=torch.long)
    class_counts = torch.bincount(labels_t, minlength=16).float()
    class_weights = 1.0 / (class_counts + 1e-6)
    weights = class_weights[labels_t]
    return weights


def _get_longseq_sample_weights_ternary(dataset: LongSequenceDataset) -> torch.Tensor:
    """
    Calculate sample weights for WeightedRandomSampler using ternary labels.

    Uses center-frame ternary label as proxy for window majority label.
    Ternary classes: 0=fall, 1=fallen, 2=normal.

    Args:
        dataset: LongSequenceDataset instance.
    Returns:
        weights: (N,) float32 weights array.
    """
    n_total = len(dataset)
    half = dataset.window_size // 2

    labels = []
    for npz_start, offset in dataset.window_index:
        center_idx = npz_start + offset + half
        lbl_16 = int(dataset.labels_16_all[center_idx])
        # Map to ternary
        if lbl_16 == 1:
            ternary_lbl = 0
        elif lbl_16 == 2:
            ternary_lbl = 1
        else:
            ternary_lbl = 2
        labels.append(ternary_lbl)

    labels_t = torch.tensor(labels, dtype=torch.long)
    class_counts = torch.bincount(labels_t, minlength=3).float()
    class_weights = 1.0 / (class_counts + 1e-6)
    weights = class_weights[labels_t]
    return weights


def create_longseq_dataloaders(
    npz_path: str,
    splits_dir: str,
    window_size: int = 16,
    stride: int = 1,
    batch_size: int = 64,
    num_workers: int = 0,
    use_weighted_sampler: bool = True,
    use_diff: bool = False,
    use_accel: bool = False,
    pose_npz_path: str = None,
    use_ternary_sampler: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test DataLoaders with sliding window.

    Args:
        npz_path: Frame-level NPZ path.
        splits_dir: Directory with train.csv, val.csv, test.csv.
        window_size: T — sequence length.
        stride: Sliding window stride.
        batch_size: Batch size.
        num_workers: DataLoader workers (0 recommended for Windows).
        use_weighted_sampler: Use WeightedRandomSampler on train.
        use_diff: If True, concat frame-to-frame diff features.
        use_accel: If True, concat second-order diff features (acceleration).
        pose_npz_path: Optional frame-level pose NPZ path.
        use_ternary_sampler: If True, use ternary (3-class) weights
            instead of 16-class weights (Phase 6).

    Returns:
        (train_loader, val_loader, test_loader)
    """
    import os as _os

    train_dataset = LongSequenceDataset(
        npz_path=npz_path,
        split_csv_path=_os.path.join(splits_dir, "train.csv"),
        window_size=window_size,
        stride=stride,
        use_diff=use_diff,
        use_accel=use_accel,
        pose_npz_path=pose_npz_path,
    )
    val_dataset = LongSequenceDataset(
        npz_path=npz_path,
        split_csv_path=_os.path.join(splits_dir, "val.csv"),
        window_size=window_size,
        stride=stride,
        use_diff=use_diff,
        use_accel=use_accel,
        pose_npz_path=pose_npz_path,
    )
    test_dataset = LongSequenceDataset(
        npz_path=npz_path,
        split_csv_path=_os.path.join(splits_dir, "test.csv"),
        window_size=window_size,
        stride=stride,
        use_diff=use_diff,
        use_accel=use_accel,
        pose_npz_path=pose_npz_path,
    )

    # Training set with weighted sampling
    if use_weighted_sampler:
        if use_ternary_sampler:
            weights = _get_longseq_sample_weights_ternary(train_dataset)
        else:
            weights = _get_longseq_sample_weights(train_dataset)
        sampler = WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def test_longseq_dataset():
    """Test LongSequenceDataset with a mock NPZ."""
    BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    npz_path = os.path.join(BASE, "data", "omnifall_frame_preprocessed.npz")
    splits_dir = os.path.join(BASE, "DATASET-omnifall", "splits", "syn", "random")

    if not os.path.exists(npz_path):
        print(f"[SKIP] Frame NPZ not found at {npz_path}")
        print("  Run: python preprocessing/verify_pipeline.py --frame_level")
        return

    for T, stride in [(16, 1), (32, 1), (16, 4)]:
        print(f"\n--- Testing T={T}, stride={stride} ---")
        loaders = create_longseq_dataloaders(
            npz_path=npz_path,
            splits_dir=splits_dir,
            window_size=T,
            stride=stride,
            batch_size=4,
            num_workers=0,
        )

        for name, loader in zip(["train", "val", "test"], loaders):
            batch = next(iter(loader))
            f = batch["features"]
            expected_shape = (4, T, 512)
            assert f.shape == expected_shape, \
                f"[{name}] Expected {expected_shape}, got {f.shape}"
            assert batch["labels_16"].shape == (4, T)
            assert batch["fall_labels"].shape == (4, T)
            assert batch["fallen_labels"].shape == (4, T)
            print(
                f"  [{name}] features: {f.shape} | "
                f"dataset size: {len(loader.dataset):,}"
            )

    print("\n[PASS] All LongSequenceDataset tests passed!")


if __name__ == "__main__":
    import sys
    if "--longseq" in sys.argv:
        test_longseq_dataset()
    else:
        test_dataset()
