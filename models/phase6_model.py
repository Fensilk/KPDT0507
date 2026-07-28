"""
Phase 6: Unified ternary fall detection model with swappable temporal decoder.

Architecture:
    DINOv2 ViT-g (1536d) → Linear(1536→384) → LayerNorm → SinusoidalPE
        → [Temporal Decoder] → Linear(384→3) → (B,T,3)

Three decoder variants, same interface (B,T,384) → (B,T,384):
    1. CausalTransformerDecoder — 1L, causal mask, Pre-LN
    2. LSTMDecoder — 2L, batch_first
    3. MambaDecoder — 2L, SSM blocks

Config per the Phase 6 plan:
    hidden_dim=384, dropout=0.3, PE=sinusoidal (all three models).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


# ============================================================
# Positional Encoding
# ============================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (same as models/transformer_model.py)."""

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


# ============================================================
# 1. Causal Transformer Decoder
# ============================================================

class CausalTransformerDecoder(nn.Module):
    """
    1-layer Transformer encoder with optional causal masking.

    When causal=True: uses upper-triangular -inf mask so frame t
    can only attend to frames <= t.
    When causal=False: bidirectional (bridge experiment).
    """

    def __init__(
        self,
        d_model: int = 384,
        num_heads: int = 6,
        dropout: float = 0.3,
        mlp_ratio: float = 4.0,
        causal: bool = True,
    ):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=int(d_model * mlp_ratio),
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN for stability
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=1)
        self.causal = causal

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            (B, T, d_model)
        """
        if self.causal:
            T = x.size(1)
            mask = torch.triu(
                torch.full((T, T), float("-inf"), device=x.device),
                diagonal=1,
            )
            return self.transformer(x, mask=mask)
        else:
            return self.transformer(x)


# ============================================================
# 2. LSTM Decoder
# ============================================================

class LSTMDecoder(nn.Module):
    """2-layer LSTM decoder with dropout between layers."""

    def __init__(
        self,
        d_model: int = 384,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            (B, T, d_model)
        """
        out, _ = self.lstm(x)
        return out


# ============================================================
# 3a. Simplified Mamba Block (pure PyTorch fallback)
# ============================================================

class SimplifiedMambaBlock(nn.Module):
    """
    Simplified Mamba selective SSM block (pure PyTorch).

    Implements the core Mamba property: input-dependent state transitions
    with long-range dependencies, without CUDA-optimized selective scan.

    Architecture (simplified from Gu & Dao 2023):
        x → LayerNorm → Linear(expand) → 1D Conv → SiLU
          → SSM (selective scan) → Linear(contract) → residual

    The selective scan uses an input-dependent delta, B, C parametrization
    implemented as a sequential recurrence over time (slower than CUDA
    scan, but functionally correct for comparing decoder architectures).
    """

    def __init__(
        self,
        d_model: int = 384,
        expand_factor: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
    ):
        super().__init__()
        d_inner = int(d_model * expand_factor)

        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_inner * 2)  # x and z branches
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=d_inner,  # depthwise
        )

        # SSM parameters
        # A: (d_inner, d_state) — diagonal state matrix (log space)
        self.A_log = nn.Parameter(
            torch.log(torch.linspace(0.5, 8, d_state).repeat(d_inner, 1))
        )
        self.D = nn.Parameter(torch.ones(d_inner))

        # delta projection: input → per-channel step size
        self.delta_proj = nn.Linear(d_inner, d_inner)

        # B and C projections: input → state input/output matrices
        self.B_proj = nn.Linear(d_inner, d_state, bias=False)
        self.C_proj = nn.Linear(d_inner, d_state, bias=False)

        self.out_proj = nn.Linear(d_inner, d_model)
        self.dropout = nn.Dropout(0.1)

    def _selective_scan(self, u, delta, A, B, C, D):
        """
        Simplified selective scan over time dimension.

        B and C are shared across all channels (per Mamba paper).
        Recurrence at each channel d:
            h[d,n](t) = A_bar[d,n](t) * h[d,n](t-1) + B_bar[n](t) * u[d](t)
            y[d](t) = sum_n(h[d,n](t) * C[n](t)) + D[d] * u[d](t)

        Args:
            u: (B, L, D_inner) input sequence
            delta: (B, L, D_inner) step sizes
            A: (D_inner, N) state matrix
            B: (B, L, N) input projection (shared across D_inner)
            C: (B, L, N) output projection (shared across D_inner)
            D: (D_inner,) skip connection
        Returns:
            y: (B, L, D_inner)
        """
        B_l, L, D_inner, N = u.shape[0], u.shape[1], A.shape[0], A.shape[1]

        # Discretize A: (B, L, D_inner, N)
        A_bar = torch.exp(
            delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
        )

        # B_bar[t, n] = delta[t, d] * B[t, n]
        # B is (B, L, N) shared across all D_inner;
        # expand to (B, L, D_inner, N) by broadcasting
        B_bar = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, D_inner, N)

        # Sequential recurrence over time
        h = torch.zeros(B_l, D_inner, N, device=u.device, dtype=u.dtype)
        outputs = []
        for t in range(L):
            # h: (B, D_inner, N); A_bar[:,t]: (B, D_inner, N)
            h = A_bar[:, t] * h + B_bar[:, t] * u[:, t].unsqueeze(-1)
            # C[:, t]: (B, N) -> (B, 1, N); h: (B, D_inner, N)
            # sum over N: (B, D_inner)
            y_t = (h * C[:, t].unsqueeze(1)).sum(dim=-1)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # (B, L, D_inner)
        y = y + u * D.unsqueeze(0).unsqueeze(0)  # skip connection
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            (B, T, d_model)
        """
        residual = x
        x = self.norm(x)

        # Project and split
        proj = self.in_proj(x)  # (B, T, 2 * d_inner)
        x_proj, z = proj.chunk(2, dim=-1)  # each (B, T, d_inner)

        # 1D depthwise convolution (causal: only past context)
        x_conv = x_proj.transpose(1, 2)  # (B, d_inner, T)
        x_conv = self.conv1d(x_conv)  # (B, d_inner, T + pad)
        x_conv = x_conv[:, :, :x.size(1)]  # remove padding, (B, d_inner, T)
        x_conv = x_conv.transpose(1, 2)  # (B, T, d_inner)

        # SiLU activation
        x_act = F.silu(x_conv)

        # Selective SSM
        delta = F.softplus(self.delta_proj(x_act))  # (B, T, d_inner)
        B = self.B_proj(x_act)  # (B, T, d_state)
        C = self.C_proj(x_act)  # (B, T, d_state)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        y = self._selective_scan(
            x_act, delta, A, B, C, self.D
        )  # (B, T, d_inner)

        # Gating with z branch
        y = y * F.silu(z)

        # Output projection
        y = self.out_proj(y)  # (B, T, d_model)
        y = self.dropout(y)

        return residual + y


# ============================================================
# 3b. Mamba Decoder — uses mamba_ssm if available, else fallback
# ============================================================

class MambaDecoder(nn.Module):
    """
    Mamba SSM decoder with Pre-LN residual.

    Uses SimplifiedMambaBlock — a pure PyTorch implementation of the
    selective state-space model (Gu & Dao 2023).

    Config: 2 layers, d_model=384, expand=2.
    """

    def __init__(
        self,
        d_model: int = 384,
        num_layers: int = 2,
        expand_factor: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_layers = num_layers

        self.layers = nn.ModuleList([
            SimplifiedMambaBlock(
                d_model=d_model,
                expand_factor=expand_factor,
                d_state=16,
                d_conv=4,
            )
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            (B, T, d_model)
        """
        for layer, norm in zip(self.layers, self.norms):
            residual = x
            x = norm(x)
            x = layer(x)
            x = self.dropout(x)
            x = residual + x
        return x


# ============================================================
# Unified Ternary Model
# ============================================================

class Phase6TernaryModel(nn.Module):
    """
    Unified ternary fall detection model for Phase 6.

    Architecture:
        Input (B, T, 1536) → Linear(1536→384) → LayerNorm → PositionalEncoding
            → [Decoder] → Linear(384→3) → logits (B, T, 3)

    Classes: 0=fall, 1=fallen, 2=normal.

    Args:
        input_dim: Input feature dimension (1536 for ViT-g).
        hidden_dim: Hidden dimension for all internal representations.
        decoder_type: "transformer" | "lstm" | "mamba".
        causal: Causal masking (True for Phase 6, False for bridge).
        num_heads: Attention heads (transformer only).
        dropout: Dropout rate for all modules.
        mlp_ratio: FFN expansion ratio (transformer only).
        max_len: Max sequence length for positional encoding.
    """

    def __init__(
        self,
        input_dim: int = 1536,
        hidden_dim: int = 384,
        decoder_type: str = "transformer",
        causal: bool = True,
        num_layers: int = 1,
        num_heads: int = 6,
        dropout: float = 0.3,
        mlp_ratio: float = 4.0,
        max_len: int = 100,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.decoder_type = decoder_type
        self.causal = causal

        # Shared input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=max_len)
        self.emb_dropout = nn.Dropout(dropout)

        # Temporal decoder
        if decoder_type == "transformer":
            self.decoder = CausalTransformerDecoder(
                d_model=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                mlp_ratio=mlp_ratio,
                causal=causal,
            )
        elif decoder_type == "lstm":
            self.decoder = LSTMDecoder(
                d_model=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
            )
        elif decoder_type == "mamba":
            self.decoder = MambaDecoder(
                d_model=hidden_dim,
                num_layers=num_layers,
                expand_factor=2,
                dropout=dropout,
            )
        else:
            raise ValueError(
                f"Unknown decoder_type: {decoder_type}. "
                f"Choose 'transformer', 'lstm', or 'mamba'."
            )

        # Output head
        self.head = nn.Linear(hidden_dim, 3)  # fall, fallen, normal

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform init for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, T, D) input features.
        Returns:
            logits: (B, T, 3) ternary classification logits.
        """
        x = self.input_proj(features)       # (B, T, hidden_dim)
        x = self.input_norm(x)
        x = self.pos_encoder(x)
        x = self.emb_dropout(x)
        x = self.decoder(x)                  # (B, T, hidden_dim)
        logits = self.head(x)                # (B, T, 3)
        return logits


# ============================================================
# Self-test
# ============================================================

def test_forward_shapes():
    """Verify forward pass shapes for all three decoder types."""
    B, T, D = 4, 64, 1536
    x = torch.randn(B, T, D)

    configs = [
        ("transformer", True, 1),
        ("transformer", False, 1),  # bridge
        ("lstm", True, 2),
        ("mamba", True, 2),
    ]

    for decoder_type, causal, num_layers in configs:
        model = Phase6TernaryModel(
            input_dim=D,
            hidden_dim=384,
            decoder_type=decoder_type,
            causal=causal,
            num_layers=num_layers,
            dropout=0.3,
            max_len=T + 10,
        )
        model.eval()

        with torch.no_grad():
            logits = model(x)

        assert logits.shape == (B, T, 3), (
            f"[{decoder_type}] Expected ({B},{T},3), got {logits.shape}"
        )
        n_params = sum(p.numel() for p in model.parameters())
        causal_str = "causal" if causal else "bidir"
        print(
            f"[PASS] {decoder_type:>12s} ({causal_str:>6s}, "
            f"{num_layers}L) → {logits.shape} | Params: {n_params:,}"
        )

    print("\n[PASS] All Phase6TernaryModel shape tests passed!")


if __name__ == "__main__":
    test_forward_shapes()
