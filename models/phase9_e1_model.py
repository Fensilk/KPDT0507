"""E1 端到端模型：ViT-g(LoRA) 逐帧 CLS → Δdiff(3072d) → Phase6 时序头。

架构（Phase 9 E1 设计定稿，docs/0902实验第九阶段总结.md §E1）:
    窗口 T 帧 (B,T,3,H,W) fp32（已预处理，非 uint8）
      → ViT-g(facebook/dinov2-giant) 冻结主干 + LoRA(r=16, q/k/v/o) —— 逐帧 fp16
      → 每帧 CLS token (B*T, 1536)
      → .float() → reshape (B,T,1536)
      → Δdiff 帧间差分（首帧=0）→ concat → (B,T,3072)
      → Phase6TernaryModel(input_dim=3072, transformer, causal=False/bridge) → (B,T,3)

对照基线 Phase7 p7d_delta（两段式抽特征 + Δdiff + Phase6 bridge 头）：E1 唯一不同
是端到端反传 LoRA。因此时序头配置必须与基线一致: causal=False(bridge), heads=6,
transformer, dropout=0.3, max_len=100, num_classes=3（默认 1 层）。input_proj 显式
fp32，LoRA A/B fp32，vit 主干 fp16 —— 训练在 fp16 autocast 下进行，权重保持 fp32。

紧凑 checkpoint 重建（镜像 E3 31MB finetuned_lora.pt 配方，避免 2.3GB 全量）:
    ckpt = {
        "lora_state": {k: v for k, v in model.state_dict().items()
                       if "lora_" in k},      # 键前缀 "vit.…lora_A/B"（fp32）
        "temporal_state": model.temporal.state_dict(),
        "temporal_args": <Phase6TernaryModel 训练时 kwargs>,
        "lora_rank": 16,
        "args": <train args>,
    }
    model = E1Model.from_checkpoint(path, device="cuda")

注意:
    - 输入须已是预处理后的 float 帧（DINOv2 mean/std）；uint8 归一化由 trainer/eval
      负责，forward 内不做。
    - LoRA 注入前必须冻结全部主干参数（本模块 from_pretrained 已做）:
      freeze-all → inject_lora → 可训练 ≈ 7.86M(LoRA) + 时序头。
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.lora import inject_lora               # noqa: E402
from models.phase6_model import Phase6TernaryModel     # noqa: E402

CLS_DIM = 1536  # ViT-g CLS 维度

# Phase6 时序头默认配置 = p7d_delta 基线（bridge）。num_layers 走构造函数默认 1。
_DEFAULT_HEAD_ARGS = dict(
    input_dim=CLS_DIM * 2,   # CLS + Δdiff
    hidden_dim=384,
    decoder_type="transformer",
    causal=False,            # bridge：双向注意力（与 Phase7 基线一致）
    num_heads=6,
    dropout=0.3,
    max_len=100,
    num_classes=3,
)


class E1Model(nn.Module):
    """窗口 (B,T,3,H,W) 帧 → 逐帧 ViT-g(LoRA) CLS → Δdiff → Phase6 头 → (B,T,3)。

    vit 主干 fp16，时序头 fp32。训练/评估入口:
        E1Model.from_pretrained()        冷启动（加载 dinov2-giant + 注入 LoRA）
        E1Model.from_checkpoint(path)    从紧凑 checkpoint 重建
    """

    def __init__(self, vit_model, temporal=None, lora_rank=16):
        super().__init__()
        self.vit = vit_model                      # facebook/dinov2-giant（主干已 .half() + 冻结）
        self.lora_rank = lora_rank
        # temporal 传入时使用之（其 input_proj 由调用方保证 fp32）；否则按默认基线配置新建
        self.temporal = temporal or Phase6TernaryModel(**_DEFAULT_HEAD_ARGS)
        self.temporal.input_proj.to(torch.float32)

    # ------------------------------------------------------------------ forward
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """预处理后的帧窗口 → 帧级三分 logits。

        Args:
            frames: (B, T, 3, 224, 224) fp32，DINOv2 mean/std 归一化后的像素。
        Returns:
            logits: (B, T, 3)。
        """
        B, T = frames.shape[0], frames.shape[1]
        x = frames.reshape(B * T, *frames.shape[2:])                 # (B*T, 3, 224, 224)
        with torch.amp.autocast("cuda", dtype=torch.float16):        # vit 主干 fp16
            cls = self.vit(x).last_hidden_state[:, 0, :]             # (B*T, 1536) fp16
        cls = cls.reshape(B, T, CLS_DIM).float()                     # (B, T, 1536) fp32
        diff = torch.zeros_like(cls)
        diff[:, 1:] = cls[:, 1:] - cls[:, :-1]                       # Δdiff，diff[:, 0] = 0
        feats = torch.cat([cls, diff], dim=-1)                       # (B, T, 3072)
        return self.temporal(feats)                                  # (B, T, 3)

    # ---------------------------------------------------------- construction
    @classmethod
    def from_pretrained(cls, model_name="facebook/dinov2-giant", lora_rank=16,
                        device="cuda", temporal_args=None):
        """冷启动：加载 dinov2-giant → 冻结全部主干 → 注入 LoRA(r=16 q/k/v/o)。

        vit 主干转 fp16；LoRA A/B 与时序头保持 fp32（fp16 autocast 下前向计算、
        fp32 主权重更新）。temporal_args 给定时以其重建时序头（from_checkpoint 用），
        否则用默认基线配置。返回 .to(device) 的 E1Model。
        """
        from transformers import AutoModel

        vit = AutoModel.from_pretrained(model_name).half()
        # 必须先整体冻结主干，再注入 LoRA（LoRA 只包裹 q/k/v/o，不动其余 MLP/norm/embed）
        for p in vit.parameters():
            p.requires_grad_(False)
        inject_lora(vit, r=lora_rank)
        temporal = None
        if temporal_args is not None:
            if temporal_args.get("input_dim") != CLS_DIM * 2:
                raise ValueError(
                    f"temporal_args input_dim={temporal_args.get('input_dim')} 与 "
                    f"CLS+Δdiff 维度 {CLS_DIM * 2} 不符")
            temporal = Phase6TernaryModel(**temporal_args)
        return cls(vit, temporal=temporal, lora_rank=lora_rank).to(device)

    # ---------------------------------------------------------- reload
    @classmethod
    def from_checkpoint(cls, path, device="cuda", model_name="facebook/dinov2-giant"):
        """从紧凑 checkpoint 重建 E1Model（含时序头重建 + LoRA/头权重恢复）。

        checkpoint 键（Task 4 保存，见模块 docstring）: lora_state /
        temporal_state / temporal_args / lora_rank / args。temporal_args 缺省时
        使用默认基线头配置（与 E1 训练时一致）。加载后 model.eval()，返回在
        `device` 上。
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        rank = ckpt.get("lora_rank", 16)
        temporal_args = ckpt.get("temporal_args")
        model = cls.from_pretrained(model_name=model_name, lora_rank=rank,
                                    temporal_args=temporal_args, device="cpu")
        # LoRA：仅 vit.…lora_A/B（strict=False，缺失键=冻结主干，属预期）
        missing, unexpected = model.load_state_dict(ckpt["lora_state"], strict=False)
        if unexpected:
            raise ValueError(
                f"[E1Model.from_checkpoint] LoRA 状态含 {len(unexpected)} 个意外键: "
                f"{list(unexpected)[:5]}...（紧凑 checkpoint 不应含冻结主干键）")
        model.temporal.load_state_dict(ckpt["temporal_state"])
        n_lora = sum(v.numel() for v in ckpt["lora_state"].values())
        model.eval()
        model = model.to(device)
        print(
            f"[E1Model.from_checkpoint] 已从 {os.path.basename(path)} 恢复 "
            f"{n_lora / 1e6:.2f}M LoRA 参数 (rank={rank}) + 时序头；"
            f"缺失键 {len(missing)} 个 = 冻结主干（预期）"
        )
        return model
