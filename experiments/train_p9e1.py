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

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "preprocessing"))

from step7_extract_dinov2 import DINOV2_MEAN, DINOV2_STD  # noqa: E402
from models.phase9_e1_model import E1Model             # noqa: E402
from models.dataset import (                            # noqa: E402
    build_boundary_labels,
    CLASS_NAMES,
    CLASS_NAMES_INV,
)

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


def pose_dim_of(pose_npz):
    """读 pose NPZ 的特征维数（用于把 input_dim 加上去）。"""
    if not pose_npz:
        return 0
    d = np.load(pose_npz, allow_pickle=True, mmap_mode="r")
    return int(d["pose_features"].shape[1])


def build_temporal_args(args):
    """TEMPORAL_ARGS + Phase 11 开关。

    ⚠ 必须把新开关写进 temporal_args 并随 checkpoint 保存：from_checkpoint 用它重建
      时序头，`load_state_dict(temporal_state)` 是 **strict** 的——开关不一致会导致
      boundary_head 的键缺失而直接抛错（这是好事：静默不匹配会更难查）。
    ⚠ Exp-4 的 pose 会把 input_dim 从 3072 加到 3072+D_pose，也必须写进 temporal_args，
      否则 from_checkpoint 重建出的 input_proj 形状不符、strict load 直接报错。
    """
    ta = dict(TEMPORAL_ARGS)
    if getattr(args, "use_boundary_head", False):
        ta["use_boundary_head"] = True
    if getattr(args, "multiscale", False):
        ta["multiscale"] = True
    d_pose = pose_dim_of(getattr(args, "pose_npz", None))
    if d_pose:
        # aux_dim 必须显式存进 temporal_args：E1Model.from_pretrained 会按当前骨干
        # 覆写 input_dim（= hidden*2），只靠 input_dim 无法区分"骨干变了"与"多了外部特征"。
        ta["aux_dim"] = d_pose
        ta["input_dim"] = ta["input_dim"] + d_pose
    return ta


def boundary_slice(labels16, st, ws, window, k):
    """窗口的边界标签：**在整片 80 帧上算再切窗**。

    只在窗口内算会漏掉窗口边界外侧 k 帧内的切换点（那种情况下窗口首/尾帧本就
    该标 1）。E1 的每个视频固定 80 帧（已验证），故此处直接按 80 取全片。
    """
    full = ternary(labels16[st: st + 80])
    return build_boundary_labels(full, k)[ws: ws + window].astype(np.float32)


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

    def __init__(self, cache_dir, npz, window=64, stride=16, limit_videos=None,
                 use_boundary=False, boundary_k=3, pose_npz=None):
        # ⚠ 第三返回值槽位是**边界标签或 pose 特征二选一**（本阶段两者从不同时使用；
        #   同时开会被显式拒绝，避免静默取错）。collate_windows 对二者一视同仁。
        if use_boundary and pose_npz:
            raise ValueError("use_boundary 与 pose_npz 不可同时启用（第三返回值槽位互斥）")
        self.use_boundary = use_boundary
        self.boundary_k = boundary_k
        self.pose_all = None
        if pose_npz:
            pd = np.load(pose_npz, allow_pickle=True, mmap_mode="r")
            self.pose_all = pd["pose_features"]
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
        if self.use_boundary:
            b = boundary_slice(self.labels16, st, ws, self.window, self.boundary_k)
            return frames, y, torch.from_numpy(b)
        if self.pose_all is not None:
            pz = self.pose_all[st + ws: st + ws + self.window]
            return frames, y, torch.from_numpy(np.ascontiguousarray(pz)).float()
        return frames, y


# ── mp4 现解码数据集（H@6000 用：无 96GB 磁盘缓存，decode-on-fly；
#    解码+resize 与 prep_p9e1_frames/eval_p9e1 同一路径 → 与 E1@2000 缓存逐字节一致）──
def decode_mp4_frames(mp4, n=80, size=(224, 224)):
    """decode + INTER_LINEAR resize → (80,224,224,3) uint8；失败返回 None。"""
    from av_decode import read_rgb_frames
    frames = read_rgb_frames(mp4, n)
    if not frames:
        return None
    imgs = np.stack([cv2.resize(f, size, interpolation=cv2.INTER_LINEAR) for f in frames])
    return np.ascontiguousarray(imgs).astype(np.uint8)


class Mp4WindowCache(Dataset):
    """训练视频现解码滑窗数据集。与 WindowCache 同接口（.videos/.len/下标），
    DataLoader 多 worker 并行下 decode 成本可忽略。labels 按 video path 对齐 NPZ。"""

    def __init__(self, csv_path, npz, window=64, stride=16, limit_videos=None,
                 video_root=None, use_boundary=False, boundary_k=3, pose_npz=None):
        import pandas as pd
        if use_boundary and pose_npz:
            raise ValueError("use_boundary 与 pose_npz 不可同时启用（第三返回值槽位互斥）")
        self.use_boundary = use_boundary
        self.boundary_k = boundary_k
        self.pose_all = None
        if pose_npz:
            pd_ = np.load(pose_npz, allow_pickle=True, mmap_mode="r")
            self.pose_all = pd_["pose_features"]
        rels = list(pd.read_csv(csv_path)["path"].str.strip())
        if limit_videos is not None:
            rels = rels[: limit_videos]
        if video_root is None:
            video_root = os.path.join(BASE, "DATASET-omnifall", "data_files", "extracted")
        self.video_root = video_root
        d = np.load(npz, allow_pickle=True, mmap_mode="r")
        self.labels16 = d["labels_16"]
        self.vsi = d["video_start_indices"]
        vp = [str(p) for p in d["video_paths"]]
        self.idx = {p: i for i, p in enumerate(vp)}
        self.window, self.stride = window, stride
        self.videos = []   # (rel, st)
        self.meta = []     # (rel, st, ws)
        n_skip = 0
        for rel in rels:
            if rel not in self.idx:
                n_skip += 1; continue
            if not os.path.exists(os.path.join(video_root, rel + ".mp4")):
                n_skip += 1; continue
            st = int(self.vsi[self.idx[rel]])
            self.videos.append((rel, st))
            for ws in range(0, 80 - window + 1, stride):
                self.meta.append((rel, st, ws))
        if n_skip:
            print(f"[Mp4WindowCache] 跳过 {n_skip} 个不在 NPZ/缺失 mp4 的视频", flush=True)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        rel, st, ws = self.meta[i]
        frames = decode_mp4_frames(os.path.join(self.video_root, rel + ".mp4"))
        if frames is None:
            raise RuntimeError(f"[Mp4WindowCache] 解码失败: {rel}")
        fr = torch.from_numpy(np.ascontiguousarray(frames[ws:ws + self.window]))  # (T,H,W,3)
        y = torch.from_numpy(ternary(self.labels16[st + ws: st + ws + self.window])).long()
        if self.use_boundary:
            b = boundary_slice(self.labels16, st, ws, self.window, self.boundary_k)
            return fr, y, torch.from_numpy(b)
        if self.pose_all is not None:
            pz = self.pose_all[st + ws: st + ws + self.window]
            return fr, y, torch.from_numpy(np.ascontiguousarray(pz)).float()
        return fr, y


def compute_class_weights(cache, device):
    """逆频率类权重，统计每个训练缓存视频的全 80 帧（NPZ labels_16）。"""
    counts = np.zeros(3, dtype=np.float64)
    for _f, st in cache.videos:
        y = ternary(cache.labels16[st: st + 80])
        counts += np.bincount(y, minlength=3).astype(np.float64)
    w = counts.sum() / (3.0 * counts + 1e-9)
    w = w / w.sum() * 3.0
    return torch.tensor(w, dtype=torch.float32, device=device)


def emit_profile_table(prof, emit_fn):
    """打印 torch.profiler 的算子归因表，并给出**选卡导向**的判读。

    为什么要有它（2026-10-03）：实测 3090 上 `util` 中位 **100%**（SM 全程在跑 kernel），
    但有效算力只有约 **0.81 TFLOPS ≈ 峰值 71 的 1%** —— 即"**忙但极低效**"，
    典型的访存受限特征。这直接决定该换哪张卡：

      - 若时间花在**矩阵乘 / attention**（算力类）→ 算力受限 → **峰值 TFLOPS 表可用**
        （4090 那档 165 TFLOPS 划算）
      - 若花在 **LayerNorm / elementwise / copy / memcpy**（访存类）→ **访存受限**
        → **4090 的带宽只比 3090 高 8%，买了没用**；只有 A100 那档 HBM
        （1555–2039 GB/s）才值得换

    ⚠ profiler 本身有开销，**绝对时间不可用**，只看**相对占比**。
    """
    emit_fn("")
    emit_fn("=" * 92)
    emit_fn("torch.profiler 算子归因（按 CUDA 时间排序，前 25 项）")
    emit_fn("  ⚠ profiler 有开销 → **绝对时间不可用，只看相对占比**")
    emit_fn("=" * 92)
    try:
        tbl = prof.key_averages().table(sort_by="cuda_time_total", row_limit=25)
        for line in tbl.splitlines():
            emit_fn(line)
    except Exception as e:                                  # noqa: BLE001
        emit_fn(f"  [WARN] 归因表生成失败：{e}")
    emit_fn("")
    emit_fn("判读：")
    emit_fn("  矩阵乘 / attention 类（mm / bmm / addmm / _scaled_dot_product_* / flash）占大头")
    emit_fn("     → **算力受限** → 峰值 TFLOPS 表可用（4090 那档划算）")
    emit_fn("  LayerNorm / elementwise / copy_ / cat / memcpy 占大头")
    emit_fn("     → **访存受限** → 4090 带宽仅 +8%，**买了没用**；只有 A100 那档高带宽值得换")


NORMAL_TERNARY = 2      # 三元标签：0=fall, 1=fallen, 2=normal


def static_aux_loss(logits, y, lam, eps=1e-6):
    """Phase 11 Exp-1c：**显式静态轴辅助损失**。

        L_aux = −λ · mean_{ t : y_t = normal } log(1 − P_t(fallen) + eps)

    **为什么要它**（2026-10-01）：静态轴（`normal→fallen` 误报）是 Phase 8 确认的瓶颈，
    A0 的混淆矩阵里真实为 normal 的帧有 **2883** 帧被报成 fallen（另有 1834 帧被报成 fall）。
    难例采样（Exp-1b）那条路已证**杠杆偏弱**：sampler 抬高 normal 窗口的*频率*，
    而三元 CE 类权重（normal=0.132，仅 fall 的 1/10）压低每个 normal 样本的*贡献*，
    两者反向抵消；且 `lie_down` 全池仅 266 个窗口，加大 α 只是重复采样。
    本项改为**直接作用于目标函数**，且不引入任何新参数、不改模型结构。

    **它到底怎么起作用**（推导 + 已数值验证，2026-10-01）：

    - **损失值** `−log(1−P)` 随 P→1 发散 → 该项的**总损失**由"当前仍被误判为 fallen
      的那批 normal 帧"主导。这是"难例挖掘"效应所在：它作用在全部 normal 帧上
      （本数据集约 3377 个窗口），而不是 Exp-1b 那 266 个。
    - **梯度**对 logit 是 `∂L/∂logit_fallen = P(fallen) ∈ [0,1]` —— **有界、会饱和**，
      不会像 `1/(1−P)` 那样爆。
      ⚠ 设计文档初稿把梯度写成 `∝ 1/(1−P)`，**那是 dL/dP 而非 dL/dlogit**，已更正；
      实测 `g(P=0.5)=0.333`、`g(P=0.95)=0.909`，正是 P。
    - 等价说法：在 normal 帧上，它给"非 fallen"方向**额外加了一个权重 λ 的 BCE**——
      与单纯上调 CE 类权重相比，它更聚焦于高 P(fallen) 的难帧。

    ⚠ **只罚 `fallen`、不罚 `fall`，这是冻结的设计选择，勿改**：
    真值 normal 的帧在"刚跌倒前"本就该有升高的 `fall` 概率，罚它会推迟检测，
    而 `fall` 召回是 §7.2b 的主攻轴。

    ⚠ **已知风险（已冻结进判据 §7.2b 第 ③ 条）**：全局压低 P(fallen) 可能顺带压低
    真 fallen 帧的召回。容差：能力侧要求 `fallen` 前沿不被基准支配且均值 ≥ −0.03；
    落点侧允许逐帧 `fallen_rec` / 逐视频 `fallen_f1` 最多跌 0.15（≈落点 band 量级）。
    """
    p_fallen = torch.softmax(logits, dim=-1)[..., 1]      # (B,T)
    m = (y == NORMAL_TERNARY)                            # (B,T) 真值为 normal 的帧
    if int(m.sum()) == 0:
        # 整批没有 normal 帧：返回 0 但**保持计算图连通**，否则 backward 会报"无梯度"
        return logits.sum() * 0.0
    return -lam * torch.log1p(-p_fallen[m] + eps).mean()


def collate_windows(batch):
    """(T,H,W,3) uint8 × B → (B,T,H,W,3) uint8；(T,) → (B,T)。

    开了 boundary 时 batch 元素是三元组，额外拼出 (B,T) float 的边界标签。
    """
    frames = torch.stack([b[0] for b in batch], dim=0)
    labels = torch.stack([b[1] for b in batch], dim=0)
    if len(batch[0]) > 2:
        bnd = torch.stack([b[2] for b in batch], dim=0)
        return frames, labels, bnd
    return frames, labels


def build_hard_neg_sampler(ds, hard_neg_classes, hard_neg_alpha):
    """Exp-1b 难例采样（E1 路径）。

    ⚠ 与 train_phase6.py 不同：E1 原本用 `shuffle=True`、不做加权采样
      （类不平衡靠 CE 的逆频 class weight 处理）。此处只在显式给出难例池时才
      挂 WeightedRandomSampler，否则返回 None、行为与基线完全一致。
    窗口的"类别"取**中心帧**的 16 类标签（与 train_phase6 的采样器同口径）。
    """
    if not hard_neg_classes or hard_neg_alpha <= 0:
        return None
    from torch.utils.data import WeightedRandomSampler
    half = ds.window // 2
    cls16 = np.array([int(ds.labels16[st + ws + half]) for _r, st, ws in ds.meta])
    ter = np.where(cls16 == 1, 0, np.where(cls16 == 2, 1, 2))
    counts = np.bincount(ter, minlength=3).astype(np.float64)
    w = (1.0 / (counts + 1e-6))[ter]
    mask = np.isin(cls16, list(hard_neg_classes))
    w = w * np.where(mask, 1.0 + hard_neg_alpha, 1.0)
    names = ", ".join(CLASS_NAMES_INV[c] for c in sorted(hard_neg_classes))
    # ⚠ 剂量必须报**两个可比的数**（都按权重占比），不能拿"加权后的权重占比"去比"计数占比"。
    #   2026-10-01 踩过：旧写法印出 "7.48%（原 6.65%）" → 被读成"剂量只到位 12.5%"，
    #   而 6.65% 是**计数**占比、7.48% 是**权重**占比，两者不可比。
    #   正确基线是 α=0 的权重占比（本数据集上 lie_down 只有 2.63%），
    #   对上去是 ×2.84 —— 即 (1+α) 完全按设计生效，问题在**基量太小**，不在剂量没拧。
    share_after = 100.0 * w[mask].sum() / w.sum()
    w0 = w / np.where(mask, 1.0 + hard_neg_alpha, 1.0)      # 还原 α=0 的纯逆频权重
    share_before = 100.0 * w0[mask].sum() / w0.sum()
    print(f"[hard-neg] 难例池 {{{names}}} ×(1+{hard_neg_alpha}) | "
          f"命中窗口 {int(mask.sum())}/{len(cls16)}（按计数 {100*mask.mean():.2f}%）| "
          f"**按权重**占比 {share_before:.2f}% → {share_after:.2f}%"
          f"（×{share_after / max(share_before, 1e-9):.2f}）", flush=True)
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                 num_samples=len(w), replacement=True)


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
def val_check(model, val_cache_dir, npz, window, device, n_videos=150,
              use_boundary=False, pose_npz=None):
    """在 val 视频子集跑端到端，返回 {val_avg_f1, val_fallen_prec, val_fallen_rec}。

    use_boundary=True 时模型 forward 返回 (logits, boundary_logits)，此处只取 logits。
    val 本身不算 boundary 指标（监控信号只作趋势；边界指标在 test 上算，见规划 §6）。
    """
    model.eval()
    ds = WindowCache(val_cache_dir, npz, window, stride=80,   # 每视频 1 窗口
                     pose_npz=pose_npz)
    all_y, all_p = [], []
    for i in range(min(len(ds), n_videos)):
        item = ds[i]
        fr, y = item[0], item[1]
        aux = item[2].unsqueeze(0).to(device) if (pose_npz and len(item) > 2) else None
        x = preprocess_windows(fr.unsqueeze(0), device)      # (1,T,3,H,W)
        with torch.no_grad():
            logits = model(x, aux)
        if use_boundary:
            logits = logits[0]                               # 丢弃 boundary logits
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


def save_checkpoint(model, out_dir, name, args, temporal_args=None):
    """紧凑 checkpoint（R6 格式），可经 E1Model.from_checkpoint 重建。

    temporal_args 必须与**实际构建时**一致（含 Phase 11 开关），否则 from_checkpoint
    重建出的时序头与 temporal_state 不匹配，strict load 会直接抛错。
    """
    ckpt = {
        "lora_state": {k: v for k, v in model.state_dict().items()
                       if "lora_" in k},
        "temporal_state": model.temporal.state_dict(),
        "temporal_args": temporal_args if temporal_args is not None else TEMPORAL_ARGS,
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
    ap.add_argument("--data_mp4", default=None,
                    help="训练改为 mp4 现解码（csv: path 列；H@6000 用，免 96GB 缓存）")
    ap.add_argument("--init_ckpt", default=None,
                    help="resume/热启: 从 E1 紧凑 ckpt（如 epoch_2_model.pt）续训")
    ap.add_argument("--init_epoch", type=int, default=0,
                    help="resume: 已完成的 epoch 数（跳过前 init_epoch 个）")
    ap.add_argument("--workers", type=int, default=0,
                    help="DataLoader workers（现解码建议 ≥8）")
    ap.add_argument("--val_every", type=int, default=600)
    ap.add_argument("--val_n", type=int, default=150)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--limit_videos", type=int, default=None)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    # ── Phase 11 开关（默认关 → 行为与 E1 基线完全一致）──
    ap.add_argument("--use_boundary_head", action="store_true",
                    help="Exp-1a：加 boundary head，损失 = CE + λ·BCE（切换邻域）")
    ap.add_argument("--boundary_lambda", type=float, default=0.3)
    ap.add_argument("--boundary_k", type=int, default=3)
    ap.add_argument("--multiscale", action="store_true",
                    help="Exp-2a：并行短程分支（深度可分离 1D 卷积 k=8/16，零初始化）")
    ap.add_argument("--hard_neg_classes", type=str, default=None,
                    help="Exp-1b：逗号分隔 16 类名，如 'lie_down' 或 'lie_down,other'")
    ap.add_argument("--hard_neg_alpha", type=float, default=2.0)
    ap.add_argument("--profile_steps", type=int, default=0,
                    help="性能诊断：跑 N 步 torch.profiler 归因，打印按 CUDA 时间排序的算子表，"
                         "用于判断瓶颈是**算力**还是**访存**（决定换哪类卡）。"
                         "默认 0=关，**训练行为与基线一字不变**。"
                         "⚠ 只在诊断时开：profiler 本身有开销，会拖慢训练。")
    ap.add_argument("--aux_static_lambda", type=float, default=0.0,
                    help="Exp-1c：显式静态轴辅助损失权重。对**真值为 normal 的帧**惩罚其 "
                         "P(fallen)：L += λ·(−mean log(1−P_fallen))。只罚 fallen、不罚 fall"
                         "（跌倒前的 normal 帧本该有升高的 fall 概率）。默认 0=关，"
                         "此时训练行为与基线**一字不变**。")
    ap.add_argument("--lr_schedule", type=str, default="constant",
                    choices=["constant", "cosine"],
                    help="学习率调度。constant=恒定（E1 原配方，默认，行为不变）；"
                         "cosine=余弦退火到 ≈0。选 cosine 的理由见进度文档："
                         "E1 原配方全程恒定 1e-4，train loss 在约 1 个 epoch 即停止下降，"
                         "之后 val 在一条带里震荡 3600 步——这是'从未退火→模型持续大幅更新'"
                         "的典型征状（Phase 10 §A.6 已在 X3D-S 上受控验证过该机制）。")
    ap.add_argument("--pose_npz", type=str, default=None,
                    help="Exp-4：外部运动特征 NPZ（如 data/omnifall_pose_semantic_accel.npz "
                         "的 12d）。concat 到时序特征尾部，input_dim 自动 +D_pose。")
    ap.add_argument("--compile", action="store_true", help="torch.compile（默认关，冒烟不开）")
    ap.add_argument("--early_stop_patience", type=int, default=2,
                    help="连续多少次 val_avg_f1 未创新高即早停；0=关闭早停（跑满 epochs）")
    ap.add_argument("--save_every_check", action="store_true",
                    help="每次 val 检查都存一份 step_{N}_model.pt（Phase 11 P2 增补："
                         "离线两段式选点需密集候选池；48 份×41.5MB≈2GB/臂，训练计算量不变）")
    ap.add_argument("--model_name", default="facebook/dinov2-giant",
                    help="ViT 骨干（方案E 小骨干: facebook/dinov2-vits14 / vitb14）")
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
        t_args = build_temporal_args(args)
        if args.init_ckpt:
            emit(f"[1] resume: 从 {args.init_ckpt} 续训（跳过前 {args.init_epoch} 个 epoch）")
            model = E1Model.from_checkpoint(args.init_ckpt, device=device,
                                            model_name=args.model_name)
        else:
            emit(f"[1] 加载 {args.model_name}(fp16) + LoRA(r=16) + Phase6 bridge 头")
            model = E1Model.from_pretrained(model_name=args.model_name,
                                            lora_rank=16, device=device,
                                            temporal_args=t_args)
        model.vit.gradient_checkpointing_enable()   # 必须：128 帧/step 显存
        emit(f"  梯度 checkpointing: {model.vit.is_gradient_checkpointing}")
        if args.compile:
            model = torch.compile(model)
            emit("  torch.compile: ON")
        trainable = [p for p in model.parameters() if p.requires_grad]
        n_train = sum(p.numel() for p in trainable)
        emit(f"  可训练参数: {n_train / 1e6:.2f}M "
             f"(LoRA + 时序头；主干冻结 fp16)")
        if args.multiscale:
            emit("  Phase11 Exp-2a 多尺度分支: ON（k=8/16，零初始化 → 起点=基线）")
        if args.use_boundary_head:
            emit(f"  Phase11 Exp-1a boundary head: ON (λ={args.boundary_lambda}, "
                 f"k={args.boundary_k})")

        # ---- 数据集 + 类权重 ----
        if args.data_mp4:
            emit(f"[2] mp4 现解码数据集: {args.data_mp4} (window={args.window}, "
                 f"stride={args.stride}, workers={args.workers})")
            ds = Mp4WindowCache(args.data_mp4, args.npz, args.window, args.stride,
                                limit_videos=args.limit_videos,
                                use_boundary=args.use_boundary_head,
                                boundary_k=args.boundary_k,
                                pose_npz=args.pose_npz)
        else:
            emit(f"[2] 窗口数据集: {args.data} (window={args.window}, stride={args.stride})")
            ds = WindowCache(args.data, args.npz, args.window, args.stride,
                             limit_videos=args.limit_videos,
                             use_boundary=args.use_boundary_head,
                             boundary_k=args.boundary_k,
                             pose_npz=args.pose_npz)
        w = compute_class_weights(ds, device)
        emit(f"  训练视频: {len(ds.videos)} | 窗口: {len(ds)} | "
             f"类权重 fall/fallen/normal: {[round(float(x), 3) for x in w.cpu().numpy()]}")
        criterion = nn.CrossEntropyLoss(weight=w)

        # ---- Phase 11 Exp-1a：boundary loss ----
        bnd_crit = None
        if args.use_boundary_head:
            n_pos = n_neg = 0
            for _r, st, ws in ds.meta:
                b = boundary_slice(ds.labels16, st, ws, args.window, args.boundary_k)
                n_pos += int(b.sum()); n_neg += int(len(b) - b.sum())
            pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
            bnd_crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            emit(f"  boundary loss: λ={args.boundary_lambda} | 正/负帧 {n_pos:,}/{n_neg:,} "
                 f"(正类率 {100*n_pos/max(n_pos+n_neg,1):.2f}%) | "
                 f"pos_weight={float(pos_weight):.2f}")

        # ---- Phase 11 Exp-1b：难例采样 ----
        hard_neg = None
        if args.hard_neg_classes:
            names = [s.strip() for s in args.hard_neg_classes.split(",") if s.strip()]
            bad = [n for n in names if n not in CLASS_NAMES]
            if bad:
                raise SystemExit(f"[ERROR] 未知类别名 {bad}；可用 {sorted(CLASS_NAMES)}")
            hard_neg = [CLASS_NAMES[n] for n in names]
        train_sampler = build_hard_neg_sampler(ds, hard_neg, args.hard_neg_alpha)

        # ---- Phase 11 Exp-1c：显式静态轴辅助损失 ----
        # 只打印、不改结构；但它必须在日志里**自证生效**——
        # 否则事后无法区分"开关没接上"与"接上了但无效"（Exp-1b 就有过
        # "以为调了剂量、其实基线拿错"的教训，见进度文档 §2）。
        if args.aux_static_lambda > 0:
            n_norm = n_tot = 0
            for _r, st, ws in ds.meta:
                yv = ternary(ds.labels16[st + ws: st + ws + args.window])
                n_norm += int((yv == NORMAL_TERNARY).sum())
                n_tot += int(len(yv))
            emit(f"  aux_static loss: λ={args.aux_static_lambda} | 受罚帧 "
                 f"{n_norm:,}/{n_tot:,}（{100*n_norm/max(n_tot,1):.1f}%，真值为 normal）| "
                 f"只罚 P(fallen)、不罚 P(fall) | 损失值 −log(1−P) 由高 P 的难帧主导，"
                 f"梯度 ∂L/∂logit=P 有界")
        else:
            emit("  aux_static loss: 关（λ=0，训练行为与基线一致）")

        optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

        # ---- 训练 ----
        start_time = time.time()
        global_step = 0
        best_f1 = -1.0
        no_improve = 0
        # 性能诊断状态（--profile_steps>0 才用；前 _PF_SKIP 步跳过热身/autotune）
        _prof, _pf_done, _PF_SKIP = None, False, 10
        diverged = False
        stop_reason = None
        monitor_records = []

        model.train()
        start_ep = min(max(args.init_epoch, 0), args.epochs)

        # Phase 11 诊断：余弦退火（默认 constant → 不建 scheduler，行为与原配方完全一致）。
        # ⚠ 必须放在 start_ep 之后——上面那行才定义它（曾把这段插在 optimizer 之后，
        #   导致 UnboundLocalError: start_ep referenced before assignment）。
        sched = None
        if args.lr_schedule == "cosine":
            import math as _math
            _total_steps = (len(ds) // args.batch) * max(args.epochs - start_ep, 1)
            sched = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda st: 0.5 * (1.0 + _math.cos(
                    _math.pi * min(st, _total_steps) / max(_total_steps, 1))),
            )
            emit(f"  LR 调度: cosine 退火，总步数 {_total_steps}（{args.lr:g} → ~0）")

        for ep in range(start_ep, args.epochs):
            loader = DataLoader(
                ds, batch_size=args.batch,
                shuffle=(train_sampler is None), sampler=train_sampler,
                num_workers=args.workers, collate_fn=collate_windows,
                persistent_workers=args.workers > 0)
            # 滚动日志区间累计
            log_loss, log_cnt, log_correct, log_total = 0.0, 0, np.zeros(3, np.int64), np.zeros(3, np.int64)
            # val 区间累计（monitor 的 train_loss_avg）
            mon_loss, mon_cnt = 0.0, 0
            # Exp-1c 辅助项的区间累计（λ=0 时恒为 0，基线日志一字不变）
            log_aux, log_aux_n = 0.0, 0
            for batch in loader:
                # 第三槽位按开关解释：pose 优先（二者互斥，构造数据集时已拒绝同开）
                if args.pose_npz:
                    frames, labels, aux = batch
                    aux = aux.to(device)
                    bnd_gt = None
                elif bnd_crit is not None:
                    frames, labels, bnd_gt = batch
                    bnd_gt = bnd_gt.to(device)
                    aux = None
                else:
                    frames, labels = batch
                    bnd_gt = aux = None
                x = preprocess_windows(frames, device)
                y = labels.to(device)
                optimizer.zero_grad()
                out = model(x, aux)
                if bnd_crit is not None:
                    logits, bnd_logits = out
                else:
                    logits, bnd_logits = out, None
                if global_step == 0 and not torch.isfinite(logits).all():
                    raise RuntimeError("首个前向输出含 NaN/inf（输入 handoff 或模型异常）")
                loss = criterion(logits.reshape(-1, 3), y.reshape(-1))
                if bnd_logits is not None:
                    loss = loss + args.boundary_lambda * bnd_crit(
                        bnd_logits.reshape(-1), bnd_gt.reshape(-1))
                if args.aux_static_lambda > 0:
                    # Exp-1c：直接作用于目标函数（不新增参数、不改结构）
                    aux_term = static_aux_loss(logits, y, args.aux_static_lambda)
                    loss = loss + aux_term
                    log_aux += float(aux_term.detach())
                    log_aux_n += 1
                if not torch.isfinite(loss):
                    # 发散：立即中止，保留已存 best/last，写 DIVERGED 标记
                    emit(f"[DIVERGED] step {global_step + 1} loss={loss.item():.4f} "
                         f"NaN/inf → 立即中止")
                    save_checkpoint(model, args.out, "last_model.pt", args, t_args)
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
                if sched is not None:
                    sched.step()

                # ---- 性能诊断：采 N 步做算子归因（默认关，不影响任何基线行为）----
                if args.profile_steps > 0 and not _pf_done:
                    if _prof is None and global_step >= _PF_SKIP:
                        _prof = torch.profiler.profile(
                            activities=[torch.profiler.ProfilerActivity.CPU,
                                        torch.profiler.ProfilerActivity.CUDA])
                        _prof.start()
                        emit(f"  [profile] 开始采 {args.profile_steps} 步算子归因…")
                    elif _prof is not None and global_step >= _PF_SKIP + args.profile_steps:
                        _prof.stop()
                        emit_profile_table(_prof, emit)
                        _prof, _pf_done = None, True
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
                    # 开了 aux_static 时把辅助项单列出来（不发散时它应平稳下降；
                    # 若它一路不降或反升，说明该项没接上或与主任务冲突 → 早发现）
                    aux_txt = (f" | aux {log_aux / max(log_aux_n, 1):.4f}"
                               if args.aux_static_lambda > 0 else "")
                    emit(f"step {global_step} | loss {log_loss / max(log_cnt, 1):.4f} "
                         f"| fall/fallen/normal acc {'/'.join(accs)} | lr {lr:.2e}{aux_txt}")
                    log_loss, log_cnt, log_aux, log_aux_n = 0.0, 0, 0.0, 0
                    log_correct, log_total = np.zeros(3, np.int64), np.zeros(3, np.int64)

                # val 定时检查
                if global_step % args.val_every == 0:
                    # ⚠ pose_npz 必须一并传下去：漏传会让 val_check 构造出**没有姿态**的
                    #   WindowCache → aux=None → 送进模型的特征是 3072 维，而 input_proj
                    #   是 3084×384 → `RuntimeError: mat1 and mat2 shapes cannot be multiplied`。
                    #   2026-10-02 Exp-4 就是这样在**step 600 的首次 val** 崩溃的（训练步没事，
                    #   因为训练步自己传了 aux）。
                    #   **教训**：本路径的冒烟只跑了 20 步、`val_every=600`，**从未触发 val_check**，
                    #   所以漏传没被发现。**凡改动 val 相关路径，冒烟必须用 `--val_every 1` 逼出它。**
                    m = val_check(model, args.val_data, args.npz, args.window,
                                  device, n_videos=args.val_n,
                                  use_boundary=args.use_boundary_head,
                                  pose_npz=args.pose_npz)
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
                    # 每个检查点存一份 ckpt（Phase 11 P2 增补）。
                    # 为什么：离线两段式选点需要一个**密集候选池**——只存 epoch 边界的话，
                    #   "val 最优点"可能压根没被存下来（1006 实测过：锚点的平滑极大落在未被存的步上）。
                    # 开销：48 份 × 41.5 MB ≈ 2 GB/臂，**训练计算量不变**，只多磁盘 I/O。
                    if args.save_every_check:
                        save_checkpoint(model, args.out, f"step_{global_step}_model.pt", args, t_args)
                        emit(f"  [CKPT] 已存 step_{global_step}_model.pt")
                    if m["val_avg_f1"] > best_f1:
                        best_f1 = m["val_avg_f1"]
                        save_checkpoint(model, args.out, "best_model.pt", args, t_args)
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
            save_checkpoint(model, args.out, f"epoch_{ep + 1}_model.pt", args, t_args)
            emit(f"[EPOCH {ep + 1}] 完成，已存 epoch_{ep + 1}_model.pt")
            if stop_reason:
                break

        # 收尾
        if not diverged:
            save_checkpoint(model, args.out, "last_model.pt", args, t_args)
            emit("[DONE] 已存 last_model.pt")
        if stop_reason:
            emit(f"[STOP] 原因: {stop_reason}")
        emit(f"[TOTAL] {global_step} 步, {time.time() - start_time:.1f}s")
    finally:
        logf.close()


if __name__ == "__main__":
    main()
