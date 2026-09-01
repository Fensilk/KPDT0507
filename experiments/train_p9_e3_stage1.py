"""
E3 Stage1：帧级 LoRA 微调 DINOv2 ViT-g。

手写 LoRA（不依赖 peft）：冻结主干权重，在注意力 q/k/v/o 投影上插低秩 A/B。
逐帧三元分类，loss = CE + 类权重。训练后用 LoRA 状态 + 头重抽特征（extract_p9_features.py）。

用法（项目根目录，py310）：
    python experiments/train_p9_e3_stage1.py --lr 1e-4 --epochs 5 --batch_size 64
"""
import os
import sys
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "preprocessing"))

from step7_extract_dinov2 import DINOV2_MEAN, DINOV2_STD  # noqa: E402


# ═══════════════════════════════════════════════════════════
# 手写 LoRA
# ═══════════════════════════════════════════════════════════
class LoRALinear(nn.Module):
    def __init__(self, original: nn.Linear, r: int = 16, alpha: int = 16):
        super().__init__()
        self.original = original
        self.original.weight.requires_grad_(False)
        if self.original.bias is not None:
            self.original.bias.requires_grad_(False)
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.zeros(r, original.in_features))
        self.lora_B = nn.Parameter(torch.zeros(original.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        base = self.original(x)
        return base + self.scaling * (x @ self.lora_A.t() @ self.lora_B.t())


def inject_lora(model, r=16, alpha=16, targets=("query", "key", "value")):
    """在注意力 q/k/v 和 output.dense 上注入 LoRA。返回可训练参数名列表。"""
    for layer in model.encoder.layer:
        attn = layer.attention.attention
        for name in targets:
            setattr(attn, name, LoRALinear(getattr(attn, name), r, alpha))
        layer.attention.output.dense = LoRALinear(layer.attention.output.dense, r, alpha)


# ═══════════════════════════════════════════════════════════
# 帧级数据集
# ═══════════════════════════════════════════════════════════
class FrameCacheDataset(Dataset):
    """从 data/p9_frames/*.pt 加载全部帧到内存（跑通规模 20k 帧 ≈ 3GB）。"""

    def __init__(self, cache_dir):
        files = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))
        assert files, f"帧缓存为空: {cache_dir}"
        frames, labels = [], []
        for f in files:
            d = torch.load(f, weights_only=False)
            frames.append(torch.from_numpy(d["frames"]))   # (K,H,W,3) uint8
            labels.append(d["labels"])                      # (K,) int64
        self.frames = torch.cat(frames, 0)   # (N,224,224,3)
        self.labels = torch.cat(labels, 0)   # (N,)
        self.class_counts = np.bincount(self.labels.numpy(), minlength=3)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.frames[i], self.labels[i]


def preprocess_batch(frames: torch.Tensor, device) -> torch.Tensor:
    """uint8 (B,224,224,3) → float (B,3,224,224) 归一化。"""
    x = frames.to(device).float() / 255.0
    x = x.permute(0, 3, 1, 2)
    mean = torch.tensor(DINOV2_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(DINOV2_STD, device=device).view(1, 3, 1, 1)
    return (x - mean) / std


# ═══════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(BASE, "data", "p9_frames"))
    ap.add_argument("--out", default=os.path.join(BASE, "logs", "phase9", "e3_stage1"))
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from transformers import AutoModel

    os.makedirs(args.out, exist_ok=True)
    print(f"[1] 加载 facebook/dinov2-giant + 注入 LoRA(r={args.rank}) ...")
    model = AutoModel.from_pretrained("facebook/dinov2-giant").to(args.device)
    inject_lora(model, r=args.rank)

    # 冻结主干（LoRALinear 里已冻结原权重）
    for p in model.parameters():
        p.requires_grad_(False)
    # 只开 LoRA A/B
    lora_params = [p for p in model.parameters() if p.requires_grad]
    # 头
    head = nn.Linear(1536, 3).to(args.device)

    trainable = sum(p.numel() for p in lora_params)
    print(f"  LoRA 可训练参数: {trainable/1e6:.2f}M  "
          f"({trainable/sum(p.numel() for p in model.parameters())*100:.2f}%)")

    # 数据集
    print(f"[2] 加载帧缓存: {args.data}")
    ds = FrameCacheDataset(args.data)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    cc = ds.class_counts
    print(f"  帧数: {len(ds)}  类别分布 fall/fallen/normal: {cc.tolist()}")

    # 类权重（逆频率归一化）
    w = cc.sum() / (3 * cc.astype(np.float32) + 1e-9)
    w = w / w.sum() * 3
    print(f"  类权重: {[round(x, 3) for x in w]}")
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(w, device=args.device))

    # 优化器：LoRA + 头
    optimizer = torch.optim.AdamW(list(lora_params) + list(head.parameters()),
                                  lr=args.lr, weight_decay=0.01)
    model.train(); head.train()

    print(f"[3] 训练 {args.epochs} epochs ...")
    for ep in range(args.epochs):
        total_loss, n_correct, n_total = 0.0, 0, 0
        for frames, labels in loader:
            x = preprocess_batch(frames, args.device)
            y = labels.to(args.device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(dtype=torch.float16):
                cls = model(x).last_hidden_state[:, 0, :]   # (B,1536)
                logits = head(cls.float())
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            optimizer.step()
            total_loss += loss.item() * len(y)
            n_correct += (logits.argmax(1) == y).sum().item()
            n_total += len(y)
        print(f"  epoch {ep+1}: loss={total_loss/n_total:.4f} acc={n_correct/n_total:.4f}")

    # 保存：LoRA + 头（小 checkpoint），特征重抽时重建模型
    ckpt = {
        "lora_state": {k: v for k, v in model.state_dict().items()
                       if "lora_" in k},
        "head_state": head.state_dict(),
        "rank": args.rank,
        "args": vars(args),
    }
    torch.save(ckpt, os.path.join(args.out, "finetuned_lora.pt"))
    print(f"[4] 已保存: {os.path.join(args.out, 'finetuned_lora.pt')}")
    print("下一步: python experiments/extract_p9_features.py")


if __name__ == "__main__":
    main()
