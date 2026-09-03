"""E1 端到端训练：窗口采样 + 逐帧 LoRA 反传 + val 定时检查 + 发散停。

Phase 9 E1（docs/0902实验第九阶段总结.md §E1）唯一与基线 p7d_delta 不同的点：
把两段式（先抽 DINOv2 特征再训练时序头）换成端到端反传——冻结 ViT-g 主干，仅
在注意力 q/k/v/o 上挂 LoRA(r=16) 反传，时序头沿用 Phase6 bridge transformer。
输入是连续帧缓存（uint8, (80,224,224,3)），标签按 video path 从 NPZ 对齐。

关键设计：
  - WindowCache：每视频所有 stride 窗口，帧来自 .pt（uint8），标签来自 NPZ(mmapped)
  - 输入 handoff：batch (B,T,H,W,3) uint8 → /255 → permute(B,T,3,H,W) → mean/std
    归一化。E1Model.forward 不做归一化（见 phase9_e1_model.py docstring），故此处
    必须严格对齐，否则静默错位（NHWC 与 CHW 元素数相同）。
  - 梯度 checkpointing：window=64 × batch=2 = 128 帧/step，必须开启否则显存爆。
  - 类权重：全 80 帧（NPZ labels_16）逆频率，非窗口重叠计数、非 WeightedRandomSampler。
  - 紧凑 checkpoint（R6）：lora_state + temporal_state + temporal_args + lora_rank + args，
    经 E1Model.from_checkpoint 重建。
  - 监控：每 --val_every 步 val_check 写 monitor.json（数组），滚动写 e1_train.log；
    NaN/inf loss 立即中止并写 DIVERGED 标记。

用法（项目根目录，py310）：
    python experiments/train_p9e1.py --window 64 --stride 16 --epochs 3 --batch 2
    # 冒烟（本地 8GB，可能需降 window）：
    python experiments/train_p9e1.py --epochs 1 --val_every 9999 \
        --limit_videos 24 --max_steps 100
"""
import os
import sys
import glob
import json
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "preprocessing"))

from step7_extract_dinov2 import DINOV2_MEAN, DINOV2_STD  # noqa: E402
from models.phase9_e1_model import E1Model             # noqa: E402

# 时序头 kwargs（与 phase9_e1_model.py._DEFAULT_HEAD_ARGS / p7d_delta 基线严格一致）。
# from_checkpoint 会用这些 kwargs 重建 Phase6TernaryModel，input_dim 必须 = 3072。
TEMPORAL_ARGS = dict(
    input_dim=1536 * 2,
    hidden_dim=384,
    decoder_type="transformer",
    causal=False,               # bridge：双向注意力
    num_heads=6,
    dropout=0.3,
    max_len=100,
    num_classes=3,
)


def ternary(l16):
    """16 类 → 三分：1→fall(0), 2→fallen(1), 其余→normal(2)。"""
    return np.where(l16 == 1, 0, np.where(l16 == 2, 1, 2))


# ═══════════════════════════════════════════════════════════
# 窗口数据集
# ═══════════════════════════════════════════════════════════
class WindowCache(Dataset):
    """每视频所有 stride 窗口。帧来自 .pt，标签来自 NPZ（mmap）。

    __init__ 会 torch.load 每个 .pt 读 video 名（真实运行读 ~24GB，可接受：
    autodl NVMe + page cache）。__getitem__ 每次仅 load 一个 .pt 取连续帧切片。
    """

    def __init__(self, cache_dir, npz, window=64, stride=16, limit_videos=None):
        self.files = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))
        if limit_videos is not None:
            self.files = self.files[: limit_videos]
        if not self.files:
            raise ValueError(f"帧缓存为空: {cache_dir}")
        d = np.load(npz, allow_pickle=True, mmap_mode="r")
        self.labels16 = d["labels_16"]
        self.vsi = d["video_start_indices"]
        vp = [str(p) for p in d["video_paths"]]
        self.idx = {p: i for i, p in enumerate(vp)}
        self.window, self.stride = window, stride
        self.videos = []   # 唯一视频 (file, st)，供类权重在全 80 帧上统计
        self.meta = []     # (file, st, ws)
        for f in self.files:
            video = torch.load(f, weights_only=False)["video"]
            if video not in self.idx:
                raise KeyError(f"缓存视频 {video} 不在 NPZ 索引中（标签对齐失败）")
            vi = self.idx[video]
            st = int(self.vsi[vi])
            self.videos.append((f, st))
            for ws in range(0, 80 - window + 1, stride):
                self.meta.append((f, st, ws))

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        f, st, ws = self.meta[i]
        frames = torch.load(f, weights_only=False)["frames"][ws:ws + self.window]
        frames = torch.from_numpy(np.ascontiguousarray(frames))  # (T,H,W,3) uint8
        y = torch.from_numpy(ternary(self.labels16[st + ws: st + ws + self.window])).long()
        return frames, y


def compute_class_weights(cache, device):
    """逆频率类权重，统计每个训练缓存视频的全 80 帧（NPZ labels_16）。"""
    counts = np.zeros(3, dtype=np.float64)
    for _f, st in cache.videos:
        y = ternary(cache.labels16[st: st + 80])
        counts += np.bincount(y, minlength=3).astype(np.float64)
    w = counts.sum() / (3.0 * counts + 1e-9)
    w = w / w.sum() * 3.0
    return torch.tensor(w, dtype=torch.float32, device=device)


def collate_windows(batch):
    """(T,H,W,3) uint8 × B → (B,T,H,W,3) uint8；(T,) → (B,T)。"""
    frames = torch.stack([b[0] for b in batch], dim=0)
    labels = torch.stack([b[1] for b in batch], dim=0)
    return frames, labels


# ═══════════════════════════════════════════════════════════
# 预处理 handoff（关键：E1Model.forward 不做归一化）
# ═══════════════════════════════════════════════════════════
def preprocess_windows(frames, device):
    """(B,T,H,W,3) uint8 → (B,T,3,H,W) fp32，DINOv2 mean/std 归一化。"""
    x = frames.to(device).float().div_(255.0)                 # (B,T,H,W,3) fp32
    x = x.permute(0, 1, 4, 2, 3).contiguous()                 # (B,T,3,H,W)
    mean = torch.as_tensor(DINOV2_MEAN, device=device).view(1, 1, 3, 1, 1)
    std = torch.as_tensor(DINOV2_STD, device=device).view(1, 1, 3, 1, 1)
    return (x - mean) / std


# ═══════════════════════════════════════════════════════════
# 监控：val 定时检查
# ═══════════════════════════════════════════════════════════
def val_check(model, val_cache_dir, npz, window, device, n_videos=150):
    """在 val 视频子集跑端到端，返回 {val_avg_f1, val_fallen_prec, val_fallen_rec}。"""
    model.eval()
    ds = WindowCache(val_cache_dir, npz, window, stride=80)  # 每视频 1 窗口
    all_y, all_p = [], []
    for i in range(min(len(ds), n_videos)):
        fr, y = ds[i]
        x = preprocess_windows(fr.unsqueeze(0), device)      # (1,T,3,H,W)
        with torch.no_grad():
            logits = model(x)
        all_p.append(logits.argmax(-1).cpu().numpy().ravel())
        all_y.append(y.numpy().ravel())
    y = np.concatenate(all_y)
    p = np.concatenate(all_p)
    from sklearn.metrics import f1_score, precision_score, recall_score
    f1 = f1_score(y, p, labels=[0, 1, 2], average=None, zero_division=0)
    prec = precision_score(y, p, labels=[0, 1, 2], average=None, zero_division=0)
    rec = recall_score(y, p, labels=[0, 1, 2], average=None, zero_division=0)
    model.train()
    return {
        "val_avg_f1": float(f1.mean()),
        "val_fallen_prec": float(prec[1]),
        "val_fallen_rec": float(rec[1]),
    }


def save_checkpoint(model, out_dir, name, args):
    """紧凑 checkpoint（R6 格式），可经 E1Model.from_checkpoint 重建。"""
    ckpt = {
        "lora_state": {k: v for k, v in model.state_dict().items()
                       if "lora_" in k},
        "temporal_state": model.temporal.state_dict(),
        "temporal_args": TEMPORAL_ARGS,
        "lora_rank": model.lora_rank,
        "args": vars(args),
    }
    path = os.path.join(out_dir, name)
    torch.save(ckpt, path)
    return path


# ═══════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(BASE, "data", "p9e1_frames"))
    ap.add_argument("--val_data", default=os.path.join(BASE, "data", "p9e1_frames_val"))
    ap.add_argument("--npz", default=os.path.join(BASE, "data", "omnifall_dinov2_giant_frame.npz"))
    ap.add_argument("--out", default=os.path.join(BASE, "logs", "phase9", "e1"))
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val_every", type=int, default=600)
    ap.add_argument("--val_n", type=int, default=150)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--limit_videos", type=int, default=None)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--compile", action="store_true", help="torch.compile（默认关，冒烟不开）")
    ap.add_argument("--early_stop_patience", type=int, default=2,
                    help="连续多少次 val_avg_f1 未创新高即早停；0=关闭早停（跑满 epochs）")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    device = args.device
    log_path = os.path.join(args.out, "e1_train.log")
    monitor_path = os.path.join(args.out, "monitor.json")
    logf = open(log_path, "a")
    try:
        def emit(s):
            print(s)
            logf.write(s + "\n")
            logf.flush()

        # ---- 模型 ----
        emit(f"[1] 加载 facebook/dinov2-giant(fp16) + LoRA(r=16) + Phase6 bridge 头")
        model = E1Model.from_pretrained(model_name="facebook/dinov2-giant",
                                        lora_rank=16, device=device)
        model.vit.gradient_checkpointing_enable()   # 必须：128 帧/step 显存
        emit(f"  梯度 checkpointing: {model.vit.is_gradient_checkpointing}")
        if args.compile:
            model = torch.compile(model)
            emit("  torch.compile: ON")
        trainable = [p for p in model.parameters() if p.requires_grad]
        n_train = sum(p.numel() for p in trainable)
        emit(f"  可训练参数: {n_train / 1e6:.2f}M "
             f"(LoRA + 时序头；主干冻结 fp16)")

        # ---- 数据集 + 类权重 ----
        emit(f"[2] 窗口数据集: {args.data} (window={args.window}, stride={args.stride})")
        ds = WindowCache(args.data, args.npz, args.window, args.stride,
                         limit_videos=args.limit_videos)
        w = compute_class_weights(ds, device)
        emit(f"  训练视频: {len(ds.videos)} | 窗口: {len(ds)} | "
             f"类权重 fall/fallen/normal: {[round(float(x), 3) for x in w.cpu().numpy()]}")
        criterion = nn.CrossEntropyLoss(weight=w)

        optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

        # ---- 训练 ----
        start_time = time.time()
        global_step = 0
        best_f1 = -1.0
        no_improve = 0
        diverged = False
        stop_reason = None
        monitor_records = []

        model.train()
        for ep in range(args.epochs):
            loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                                num_workers=0, collate_fn=collate_windows)
            # 滚动日志区间累计
            log_loss, log_cnt, log_correct, log_total = 0.0, 0, np.zeros(3, np.int64), np.zeros(3, np.int64)
            # val 区间累计（monitor 的 train_loss_avg）
            mon_loss, mon_cnt = 0.0, 0
            for frames, labels in loader:
                x = preprocess_windows(frames, device)
                y = labels.to(device)
                optimizer.zero_grad()
                logits = model(x)
                if global_step == 0 and not torch.isfinite(logits).all():
                    raise RuntimeError("首个前向输出含 NaN/inf（输入 handoff 或模型异常）")
                loss = criterion(logits.reshape(-1, 3), y.reshape(-1))
                if not torch.isfinite(loss):
                    # 发散：立即中止，保留已存 best/last，写 DIVERGED 标记
                    emit(f"[DIVERGED] step {global_step + 1} loss={loss.item():.4f} "
                         f"NaN/inf → 立即中止")
                    save_checkpoint(model, args.out, "last_model.pt", args)
                    with open(os.path.join(args.out, "DIVERGED"), "w") as fm:
                        fm.write(f"diverged at step {global_step + 1} "
                                 f"loss={loss.item():.4f}\n")
                    diverged = True
                    stop_reason = "diverged"
                    break
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()

                global_step += 1
                b = loss.item() * y.numel()
                log_loss += b
                log_cnt += y.numel()
                mon_loss += b
                mon_cnt += y.numel()
                pred = logits.argmax(-1)
                for c in range(3):
                    mask = (y == c)
                    log_correct[c] += (pred[mask] == c).sum().item()
                    log_total[c] += mask.sum().item()

                # 滚动日志
                if global_step % args.log_every == 0:
                    accs = [f"{log_correct[c] / max(log_total[c], 1):.3f}" for c in range(3)]
                    lr = optimizer.param_groups[0]["lr"]
                    emit(f"step {global_step} | loss {log_loss / max(log_cnt, 1):.4f} "
                         f"| fall/fallen/normal acc {'/'.join(accs)} | lr {lr:.2e}")
                    log_loss, log_cnt = 0.0, 0
                    log_correct, log_total = np.zeros(3, np.int64), np.zeros(3, np.int64)

                # val 定时检查
                if global_step % args.val_every == 0:
                    m = val_check(model, args.val_data, args.npz, args.window,
                                  device, n_videos=args.val_n)
                    rec = {
                        "step": global_step,
                        "epoch": ep + 1,
                        "train_loss_avg": mon_loss / max(mon_cnt, 1),
                        "val_avg_f1": m["val_avg_f1"],
                        "val_fallen_prec": m["val_fallen_prec"],
                        "val_fallen_rec": m["val_fallen_rec"],
                        "lr": optimizer.param_groups[0]["lr"],
                        "seconds_elapsed": time.time() - start_time,
                    }
                    monitor_records.append(rec)
                    with open(monitor_path, "w") as fm:
                        json.dump(monitor_records, fm, indent=2)
                    emit(f"[VAL] step {global_step} | f1 {m['val_avg_f1']:.4f} "
                         f"| fallen prec {m['val_fallen_prec']:.4f} "
                         f"rec {m['val_fallen_rec']:.4f} | train_loss {rec['train_loss_avg']:.4f}")
                    if m["val_avg_f1"] > best_f1:
                        best_f1 = m["val_avg_f1"]
                        save_checkpoint(model, args.out, "best_model.pt", args)
                        no_improve = 0
                        emit(f"  [BEST] 新最优 f1={best_f1:.4f} → best_model.pt")
                    else:
                        no_improve += 1
                        if args.early_stop_patience > 0 and no_improve >= args.early_stop_patience:
                            stop_reason = (f"early_stop: 连续 {no_improve} 次 val_avg_f1 "
                                           f"未超过 {best_f1:.4f} (patience={args.early_stop_patience})")
                            emit(f"  [STOP] {stop_reason}")
                            break
                    mon_loss, mon_cnt = 0.0, 0

                if args.max_steps and global_step >= args.max_steps:
                    stop_reason = f"max_steps={args.max_steps} 达到"
                    break

            # epoch 结束
            if diverged or stop_reason == "diverged":
                break
            save_checkpoint(model, args.out, f"epoch_{ep + 1}_model.pt", args)
            emit(f"[EPOCH {ep + 1}] 完成，已存 epoch_{ep + 1}_model.pt")
            if stop_reason:
                break

        # 收尾
        if not diverged:
            save_checkpoint(model, args.out, "last_model.pt", args)
            emit("[DONE] 已存 last_model.pt")
        if stop_reason:
            emit(f"[STOP] 原因: {stop_reason}")
        emit(f"[TOTAL] {global_step} 步, {time.time() - start_time:.1f}s")
    finally:
        logf.close()


if __name__ == "__main__":
    main()
