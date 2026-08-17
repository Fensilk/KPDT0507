"""
Late fusion model: 后期融合 ViT-g 主特征与辅助特征（姿态/光流）。

架构:
    main (ViT-g+diff) → proj → transformer → head → logits_main
    aux  (pose/flow)  → proj → 小 transformer → head → logits_aux
    logits = α·logits_main + (1-α)·logits_aux   (α 可学习)

两个分支完全独立（各自过 decoder），只在 logit 层加权融合。
安全网: 若 aux 冗余, 模型学得 α→1, 等价 baseline。

对比 gated fusion（特征层门控）, 这里测的是"干扰"假设——
两个模态是否因共享投影层而互相干扰。
"""
import torch
import torch.nn as nn

from models.phase6_model import PositionalEncoding, CausalTransformerDecoder


class LateFusionModel(nn.Module):
    def __init__(
        self,
        main_dim: int,
        aux_dim: int,
        hidden_dim: int = 384,
        aux_hidden: int = 128,
        dropout: float = 0.3,
        num_heads: int = 6,
        aux_num_heads: int = 4,
        num_layers: int = 1,
        max_len: int = 100,
        num_classes: int = 3,
    ):
        super().__init__()

        self.main_dim = main_dim
        self.aux_dim = aux_dim
        self.hidden_dim = hidden_dim
        self.aux_hidden = aux_hidden

        # ── 主分支（ViT-g + diff）──
        self.main_proj = nn.Linear(main_dim, hidden_dim)
        self.main_norm = nn.LayerNorm(hidden_dim)
        self.main_pos = PositionalEncoding(hidden_dim, max_len=max_len)
        self.main_dropout = nn.Dropout(dropout)
        self.main_decoder = CausalTransformerDecoder(
            d_model=hidden_dim, num_heads=num_heads, dropout=dropout,
            mlp_ratio=4.0, causal=False,
        )
        self.main_head = nn.Linear(hidden_dim, num_classes)

        # ── 辅助分支（pose/flow，小模型）──
        self.aux_proj = nn.Linear(aux_dim, aux_hidden)
        self.aux_norm = nn.LayerNorm(aux_hidden)
        self.aux_pos = PositionalEncoding(aux_hidden, max_len=max_len)
        self.aux_dropout = nn.Dropout(dropout)
        self.aux_decoder = CausalTransformerDecoder(
            d_model=aux_hidden, num_heads=aux_num_heads, dropout=dropout,
            mlp_ratio=4.0, causal=False,
        )
        self.aux_head = nn.Linear(aux_hidden, num_classes)

        # ── 可学习融合权重 α（sigmoid 约束到 [0,1]，初始 0.5）──
        self.alpha_raw = nn.Parameter(torch.tensor(0.0))

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
            main_feat: (B, T, main_dim)
            aux_feat:  (B, T, aux_dim)
        Returns:
            logits: (B, T, num_classes)
        """
        # 主分支
        m = self.main_proj(main_feat)
        m = self.main_norm(m)
        m = self.main_pos(m)
        m = self.main_dropout(m)
        m = self.main_decoder(m)
        logits_main = self.main_head(m)

        # 辅助分支
        a = self.aux_proj(aux_feat)
        a = self.aux_norm(a)
        a = self.aux_pos(a)
        a = self.aux_dropout(a)
        a = self.aux_decoder(a)
        logits_aux = self.aux_head(a)

        # 后期融合
        alpha = torch.sigmoid(self.alpha_raw)
        logits = alpha * logits_main + (1 - alpha) * logits_aux
        return logits

    @property
    def alpha(self) -> float:
        """当前融合权重（主分支占比）。"""
        return float(torch.sigmoid(self.alpha_raw).item())


def test_forward_shapes():
    B, T = 4, 64
    main_dim, aux_dim = 3072, 128
    main_feat = torch.randn(B, T, main_dim)
    aux_feat = torch.randn(B, T, aux_dim)

    model = LateFusionModel(
        main_dim=main_dim, aux_dim=aux_dim, hidden_dim=384, aux_hidden=128,
        num_heads=6, aux_num_heads=4, num_layers=1, max_len=T + 10,
    )
    model.eval()
    with torch.no_grad():
        logits = model(main_feat, aux_feat)
    assert logits.shape == (B, T, 3), f"Expected ({B},{T},3), got {logits.shape}"
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[PASS] LateFusionModel → {logits.shape} | Params: {n_params:,} | α={model.alpha:.3f}")


if __name__ == "__main__":
    test_forward_shapes()
