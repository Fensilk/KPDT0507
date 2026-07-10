"""
Multimodal Feature Transformer for fall detection.

袭用 modelX.py 的 PositionalEncoding 和 Pre-LN Transformer 架构范式。
接口对齐 TCNModel((B,T,D) → (cls,fall,fallen))，与现有训练脚本兼容。

主线配置: hidden_dim=384, num_layers=1, num_heads=6, dropout=0.3
"""

import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (same as modelX.py)."""

    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model)"""
        return x + self.pe[:, :x.size(1), :]


class MultimodalFeatureTransformer(nn.Module):
    """
    DINOv2 + optional auxiliary features → Transformer → per-frame multi-task.

    Input:  features [B, T, D_total]
    Output: (logits_cls [B,T,16], logits_fall [B,T,1], logits_fallen [B,T,1])

    Default config (conservative): hidden_dim=384, num_layers=1,
        num_heads=6, dropout=0.3, mlp_ratio=4.0.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 384,
        num_layers: int = 1,
        num_heads: int = 6,
        dropout: float = 0.3,
        mlp_ratio: float = 4.0,
        max_len: int = 100,
        num_classes: int = 16,
    ):
        """
        Args:
            input_dim: Total input feature dimension (D_total).
            hidden_dim: Transformer hidden dimension.
            num_layers: Number of Transformer encoder layers.
            num_heads: Number of attention heads.
            dropout: Dropout rate.
            mlp_ratio: FFN hidden = hidden_dim * mlp_ratio.
            max_len: Maximum sequence length for positional encoding.
            num_classes: Number of action classes (default 16).
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        # Feature projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=max_len)

        # Transformer Encoder (Pre-LN for stability)
        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=int(hidden_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN: better stability on small stacks
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers)

        # Three output heads (per-frame, no pooling)
        self.head_cls = nn.Linear(hidden_dim, num_classes)
        self.head_fall = nn.Linear(hidden_dim, 1)
        self.head_fallen = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            features: (B, T, D_total) input features.

        Returns:
            logits_cls:    (B, T, 16) action classification logits.
            logits_fall:   (B, T, 1)  fall binary logits.
            logits_fallen: (B, T, 1)  fallen binary logits.
        """
        # Project and normalize
        x = self.input_proj(features)        # (B, T, hidden_dim)
        x = self.input_norm(x)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer
        x = self.transformer(x)              # (B, T, hidden_dim)

        # Three heads, per-frame predictions
        logits_cls = self.head_cls(x)         # (B, T, 16)
        logits_fall = self.head_fall(x)       # (B, T, 1)
        logits_fallen = self.head_fallen(x)   # (B, T, 1)

        return logits_cls, logits_fall, logits_fallen


def test_forward_shapes():
    """Verify forward pass shapes and parameter count."""
    # Default conservative config
    model = MultimodalFeatureTransformer(
        input_dim=1024,
        hidden_dim=384,
        num_layers=1,
        num_heads=6,
        dropout=0.3,
    )
    model.eval()

    B, T = 4, 32
    x = torch.randn(B, T, 1024)

    with torch.no_grad():
        cls, fall, fallen = model(x)

    assert cls.shape == (B, T, 16), f"cls: expected ({B},{T},16), got {cls.shape}"
    assert fall.shape == (B, T, 1), f"fall: expected ({B},{T},1), got {fall.shape}"
    assert fallen.shape == (B, T, 1), f"fallen: expected ({B},{T},1), got {fallen.shape}"

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[PASS] All shapes correct | Params: {n_params:,}")

    # Test with diff+pose config
    model2 = MultimodalFeatureTransformer(input_dim=2147)
    x2 = torch.randn(4, 32, 2147)
    with torch.no_grad():
        c2, f2, fn2 = model2(x2)
    assert c2.shape == (4, 32, 16)
    print(f"[PASS] Large input dim (2147) OK | Params: {sum(p.numel() for p in model2.parameters()):,}")

    return True


if __name__ == "__main__":
    test_forward_shapes()
