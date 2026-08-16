"""
方向 3 探针实验：仅用姿态特征（33 关键点 × xyz = 99 维）训练轻量模型，
测纯姿态特征的三元 F1，判断姿态是否携带 ViT-g 之外的判别信号。

判据：
  - 若纯姿态 avg_f1 明显高于随机基线 0.333（目标 ≥ 0.5），姿态有价值，值得集成
  - 若纯姿态 avg_f1 很低，ViT-g 已隐含姿态信息，方向 3 收益小

用法（本地 GPU/CPU 均可，姿态特征只有 99 维，训练很快）：
    python experiments/probe_pose.py
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.phase6_model import Phase6TernaryModel

POSE_NPZ = "data/omnifall_pose_frame.npz"
VITG_NPZ = "data/omnifall_dinov2_giant_frame.npz"
SPLITS = "DATASET-omnifall/splits/syn/random"

T = 64
STRIDE = 8
HIDDEN_DIM = 128
BATCH = 64
EPOCHS = 40
LR = 1e-3
PATIENCE = 10


class PoseWindowDataset(Dataset):
    """从 pose NPZ + vitg NPZ（标签）构建滑动窗口。"""

    def __init__(self, split: str, pose_feats, fall_labels, fallen_labels,
                 video_paths, vsi, vcc):
        split_df = pd.read_csv(os.path.join(SPLITS, f"{split}.csv"))
        split_paths = set(split_df["path"].str.strip())

        self.pose_feats = pose_feats
        self.windows = []   # (start_offset, )
        self.ternary = []   # (T,) int64

        for vi in range(len(video_paths)):
            base = str(video_paths[vi]).replace(".mp4", "")
            if base not in split_paths:
                continue
            ns = int(vsi[vi]); nf = int(vcc[vi])
            if nf < T:
                continue
            nw = (nf - T) // STRIDE + 1
            for w in range(nw):
                off = ns + w * STRIDE
                self.windows.append(off)
                tg = np.full(T, 2, dtype=np.int64)
                tg[fall_labels[off:off + T] == 1] = 0
                tg[fallen_labels[off:off + T] == 1] = 1
                self.ternary.append(tg)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        off = self.windows[idx]
        feat = torch.from_numpy(self.pose_feats[off:off + T].copy()).float()
        return feat, torch.from_numpy(self.ternary[idx])


def get_weights(dataset):
    labels = torch.tensor([int(np.argmax(np.bincount(t))) for t in dataset.ternary])
    counts = torch.bincount(labels, minlength=3).float()
    w = 1.0 / (counts + 1e-6)
    return w[labels]


def main():
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(BASE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Probe] Device: {device}")

    # ── 1. Load pose + labels ──
    pose = np.load(POSE_NPZ, allow_pickle=True, mmap_mode='r')
    vitg = np.load(VITG_NPZ, allow_pickle=True, mmap_mode='r')
    pose_feats = pose["pose_features"]
    fall_labels = vitg["fall_labels"]
    fallen_labels = vitg["fallen_labels"]
    vp = pose["video_paths"]; vsi = pose["video_start_indices"]; vcc = pose["video_clip_counts"]
    print(f"[Probe] Pose features: {pose_feats.shape}, dim={pose_feats.shape[1]}")

    # ── 2. Build datasets ──
    train_ds = PoseWindowDataset("train", pose_feats, fall_labels, fallen_labels, vp, vsi, vcc)
    val_ds = PoseWindowDataset("val", pose_feats, fall_labels, fallen_labels, vp, vsi, vcc)
    test_ds = PoseWindowDataset("test", pose_feats, fall_labels, fallen_labels, vp, vsi, vcc)
    print(f"[Probe] Windows — train={len(train_ds):,}, val={len(val_ds):,}, test={len(test_ds):,}")

    sampler = WeightedRandomSampler(get_weights(train_ds), num_samples=len(train_ds), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH, sampler=sampler, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH, num_workers=0, pin_memory=True)

    # ── 3. Model ──
    model = Phase6TernaryModel(
        input_dim=pose_feats.shape[1], hidden_dim=HIDDEN_DIM,
        decoder_type="transformer", causal=False, num_layers=1, num_heads=4,
        dropout=0.3, max_len=T + 10, num_classes=3,
    ).to(device)
    print(f"[Probe] Model params: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5)

    # ── 4. Train ──
    best_val_f1 = 0.0
    best_state = None
    patience = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for feat, tg in train_loader:
            feat, tg = feat.to(device), tg.to(device)
            optimizer.zero_grad()
            logits = model(feat)
            loss = criterion(logits.reshape(-1, 3), tg.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # val
        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for feat, tg in val_loader:
                feat, tg = feat.to(device), tg.to(device)
                preds.append(model(feat).argmax(-1).cpu().numpy().ravel())
                gts.append(tg.cpu().numpy().ravel())
        pred = np.concatenate(preds); gt = np.concatenate(gts)
        val_f1 = f1_score(gt, pred, average=None, labels=[0, 1, 2], zero_division=0).mean()
        scheduler.step(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict().copy()
            patience = 0
        else:
            patience += 1
        if epoch % 5 == 0 or epoch == 1:
            print(f"  epoch {epoch:>2}: val avg_f1={val_f1:.4f} (best={best_val_f1:.4f})")
        if patience >= PATIENCE:
            print(f"  early stop at epoch {epoch}")
            break

    # ── 5. Test ──
    model.load_state_dict(best_state)
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for feat, tg in test_loader:
            feat, tg = feat.to(device), tg.to(device)
            preds.append(model(feat).argmax(-1).cpu().numpy().ravel())
            gts.append(tg.cpu().numpy().ravel())
    pred = np.concatenate(preds); gt = np.concatenate(gts)

    f1s = f1_score(gt, pred, average=None, labels=[0, 1, 2], zero_division=0)
    acc = (pred == gt).mean()
    cm = confusion_matrix(gt, pred, labels=[0, 1, 2])

    print(f"\n{'='*60}")
    print("探针实验结果：纯姿态特征三元分类")
    print(f"{'='*60}")
    print(f"  avg_f1 (macro):  {f1s.mean():.4f}  (随机基线 0.333)")
    print(f"  fall_f1:         {f1s[0]:.4f}")
    print(f"  fallen_f1:       {f1s[1]:.4f}")
    print(f"  normal_f1:       {f1s[2]:.4f}")
    print(f"  ternary_acc:     {acc:.4f}")
    print(f"\n  混淆矩阵 (GT行 vs Pred列, 顺序 fall/fallen/normal):")
    print(cm)
    print(f"\n  对比: ViT-g baseline avg_f1=0.764, 纯姿态 {'有价值(≥0.5)' if f1s.mean() >= 0.5 else '价值有限(<0.5)'}")


if __name__ == "__main__":
    main()
