"""
Cross-attention fusion model: ViT-g 主特征通过 cross-attention 查询姿态辅助特征。

（Phase 8 "公平实验"：给被冗余探针证明为**非冗余**的姿态信号一次公平的集成机会。）

此前三种融合都已试过且失败，但各有"不公平"之处：
  - concat（E2/E2b）：8/99 维姿态被压进小投影，与 384d 主投影拼接后被稀释。
  - 门控（E5）：逐维 gate，但融合发生在共享 decoder 之前、aux 无时序建模，gate 无法按内容逐帧选择。
  - 后期融合（E6，仅光流）：aux 有独立时序 decoder，但标量 α 无法逐帧加权，且姿态版从未试过。

本模型的"公平"体现在：
  1. aux 用**原始 99d 姿态**（不做 8 维手工压缩），并在模型内显式追加前向 Δ姿态（→198d），
     把探针证明"ViT-g 缺的"姿态速度直接送进去。
  2. 融合用 **cross-attention**：主特征 query、姿态 key/value，按内容、逐帧、带时序地查询姿态。
  3. **残差结构**：若姿态无用，attn 输出→0，等价 baseline（安全网，不会更差）。

架构:
    main (ViT-g + Δframe) → Linear(main_dim→384) → LayerNorm → main_h            (B,T,384)
    aux  (姿态99d + Δ姿态99d = 198d) → Linear(198→384) → LayerNorm → +PE → aux_h (B,T,384)
    cross-attn: main_h = LayerNorm( main_h + Dropout( MHA(q=main_h, k=aux_h, v=aux_h) ) )
    → +PE → Dropout → CausalTransformerDecoder(双向) → Linear(384→3)

复用 phase6_model.py 的 PositionalEncoding 和 CausalTransformerDecoder。
"""
import torch
import torch.nn as nn

from models.phase6_model import PositionalEncoding, CausalTransformerDecoder


class CrossAttentionFusionModel(nn.Module):
    def __init__(
        self,
        main_dim: int,
        aux_dim: int,
        hidden_dim: int = 384,
        num_heads: int = 6,
        dropout: float = 0.3,
        mlp_ratio: float = 4.0,
        causal: bool = False,
        max_len: int = 100,
        num_classes: int = 3,
    ):
        super().__init__()
        self.main_dim = main_dim
        self.aux_dim = aux_dim
        self.hidden_dim = hidden_dim

        self.main_proj = nn.Linear(main_dim, hidden_dim)
        self.main_norm = nn.LayerNorm(hidden_dim)

        # aux = 姿态 + Δ姿态（显式速度），维度翻倍
        self.aux_proj = nn.Linear(aux_dim * 2, hidden_dim)
        self.aux_norm = nn.LayerNorm(hidden_dim)
        self.aux_pos = PositionalEncoding(hidden_dim, max_len=max_len)

        # cross-attention: main query → pose key/value
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.cross_dropout = nn.Dropout(dropout)

        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=max_len)
        self.emb_dropout = nn.Dropout(dropout)

        self.decoder = CausalTransformerDecoder(
            d_model=hidden_dim, num_heads=num_heads, dropout=dropout,
            mlp_ratio=mlp_ratio, causal=causal,
        )
        self.head = nn.Linear(hidden_dim, num_classes)

        # 诊断：cross-attn 输出的平均幅度（≈0 表示模型在忽略姿态）
        self._attn_norm = 0.0

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, main_feat: torch.Tensor, aux_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            main_feat: (B, T, main_dim) ViT-g (+ Δframe) 主特征
            aux_feat:  (B, T, aux_dim) 原始姿态（33 关键点 × xyz = 99d）
        Returns:
            logits: (B, T, num_classes)
        """
        # 主分支
        m = self.main_norm(self.main_proj(main_feat))          # (B,T,H)

        # 辅助分支：姿态 + 前向 Δ姿态（显式速度）
        d = torch.zeros_like(aux_feat)
        d[:, 1:] = aux_feat[:, 1:] - aux_feat[:, :-1]
        a = torch.cat([aux_feat, d], dim=-1)                   # (B,T,2*aux_dim)
        a = self.aux_norm(self.aux_proj(a))                    # (B,T,H)
        a = self.aux_pos(a)                                    # 位置感知的 key/value

        # cross-attention: main 查询姿态（残差，安全网）
        attn_out, _ = self.cross_attn(m, a, a, need_weights=False)
        self._attn_norm = float(attn_out.detach().abs().mean().item())
        m = self.cross_norm(m + self.cross_dropout(attn_out))

        # 时序解码
        x = self.pos_encoder(m)
        x = self.emb_dropout(x)
        x = self.decoder(x)
        return self.head(x)


def test_forward_shapes():
    B, T = 4, 64
    main_dim, aux_dim = 3072, 99
    main_feat = torch.randn(B, T, main_dim)
    aux_feat = torch.randn(B, T, aux_dim)

    model = CrossAttentionFusionModel(
        main_dim=main_dim, aux_dim=aux_dim, hidden_dim=384,
        num_heads=6, dropout=0.3, causal=False, max_len=T + 10, num_classes=3,
    )
    model.eval()
    with torch.no_grad():
        logits = model(main_feat, aux_feat)
    assert logits.shape == (B, T, 3), f"Expected ({B},{T},3), got {logits.shape}"
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[PASS] CrossAttentionFusionModel → {logits.shape} | Params: {n_params:,}")


if __name__ == "__main__":
    test_forward_shapes()
