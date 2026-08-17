"""
Gated fusion model: 门控融合 ViT-g 主特征与辅助特征（姿态/光流）。

架构:
    main (ViT-g + diff) → Linear(main_dim→384) → main_h
    aux  (pose/flow)    → Linear(aux_dim→384)  → aux_h
    gate = sigmoid(Linear(aux_dim→384)(aux))   → 逐维 [0,1] 门控
    fused = (1-gate) ⊙ main_h + gate ⊙ aux_h
    → LayerNorm → PositionalEncoding → [Decoder] → Linear(384→3)

安全网: 若 aux 冗余, 模型学得 gate→0, fused=main_h, 等价 baseline;
若 aux 有互补信号, gate 会放大它。

复用 phase6_model.py 的 PositionalEncoding 和 CausalTransformerDecoder。
"""
import torch
import torch.nn as nn

from models.phase6_model import PositionalEncoding, CausalTransformerDecoder


class GatedFusionModel(nn.Module):
    def __init__(
        self,
        main_dim: int,
        aux_dim: int,
        hidden_dim: int = 384,
        decoder_type: str = "transformer",
        causal: bool = False,
        num_layers: int = 1,
        num_heads: int = 6,
        dropout: float = 0.3,
        mlp_ratio: float = 4.0,
        max_len: int = 100,
        num_classes: int = 3,
    ):
        super().__init__()

        self.main_dim = main_dim
        self.aux_dim = aux_dim
        self.hidden_dim = hidden_dim

        # 主分支投影（ViT-g + diff）
        self.main_proj = nn.Linear(main_dim, hidden_dim)
        # 辅助分支投影（pose/flow）
        self.aux_proj = nn.Linear(aux_dim, hidden_dim)
        # 门控投影（由 aux 决定）
        self.gate_proj = nn.Linear(aux_dim, hidden_dim)

        self.input_norm = nn.LayerNorm(hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=max_len)
        self.emb_dropout = nn.Dropout(dropout)

        # 时序解码器（复用 phase6_model 的 CausalTransformerDecoder）
        if decoder_type == "transformer":
            self.decoder = CausalTransformerDecoder(
                d_model=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                mlp_ratio=mlp_ratio,
                causal=causal,
            )
        else:
            raise ValueError(
                f"Unsupported decoder_type: {decoder_type}. "
                f"Gated fusion currently supports 'transformer' only."
            )

        self.head = nn.Linear(hidden_dim, num_classes)

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
            main_feat: (B, T, main_dim) ViT-g (+ diff) 主特征
            aux_feat:  (B, T, aux_dim) 姿态/光流辅助特征
        Returns:
            logits: (B, T, num_classes)
        """
        main_h = self.main_proj(main_feat)          # (B, T, hidden)
        aux_h = self.aux_proj(aux_feat)             # (B, T, hidden)
        gate = torch.sigmoid(self.gate_proj(aux_feat))  # (B, T, hidden)

        fused = (1 - gate) * main_h + gate * aux_h  # 门控融合

        x = self.input_norm(fused)
        x = self.pos_encoder(x)
        x = self.emb_dropout(x)
        x = self.decoder(x)
        logits = self.head(x)
        return logits


def test_forward_shapes():
    """验证前向形状。"""
    B, T = 4, 64
    main_dim, aux_dim = 3072, 99
    main_feat = torch.randn(B, T, main_dim)
    aux_feat = torch.randn(B, T, aux_dim)

    model = GatedFusionModel(
        main_dim=main_dim, aux_dim=aux_dim, hidden_dim=384,
        decoder_type="transformer", causal=False, num_layers=1,
        num_heads=6, dropout=0.3, max_len=T + 10, num_classes=3,
    )
    model.eval()
    with torch.no_grad():
        logits = model(main_feat, aux_feat)
    assert logits.shape == (B, T, 3), f"Expected ({B},{T},3), got {logits.shape}"
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[PASS] GatedFusionModel → {logits.shape} | Params: {n_params:,}")


if __name__ == "__main__":
    test_forward_shapes()
