"""
Phase 7b: Event-gated ternary fall detection model.

Extends Phase6TernaryModel with a binary event gate that soft-modulates
the fallen logit — suppressing fallen predictions when no GT fall precedent
exists within the temporal context window.

Architecture:
    DINOv2 ViT-g (1536d) → Linear(1536→384) → LayerNorm → SinusoidalPE
        → Bidirectional Transformer (T=64)
        → ├─ Head_ternary: Linear(384→3)
          └─ Head_event:  Linear(384→1)  [NEW]

Inference:
    logits, event_score = model(features)
    logits[:,:,1] *= event_score  # suppress fallen without fall precedent

Event label (per-frame):
    event_label[f] = 1 if GT fall exists in [f-64, f] else 0
"""

import torch
import torch.nn as nn

from models.phase6_model import Phase6TernaryModel


class P7EventGateModel(Phase6TernaryModel):
    """
    Phase 7b model: Bridge backbone + binary event gate head.

    Identical to Phase6TernaryModel (bidirectional Transformer, T=64)
    except for an additional Linear(384→1) event head that predicts
    whether a fall event occurred within the temporal context window.

    Training:
        L = CE(ternary_mod, gt_ternary) + lambda * BCE(event_score, event_label)
        where ternary_mod[:,:,1] *= event_score (soft modulation)

    Args:
        event_lambda: Weight of BCE event loss relative to CE ternary loss.
        All other args identical to Phase6TernaryModel.
    """

    def __init__(
        self,
        input_dim: int = 1536,
        hidden_dim: int = 384,
        decoder_type: str = "transformer",
        causal: bool = False,  # P7b uses bidirectional (bridge mode)
        num_layers: int = 1,
        num_heads: int = 6,
        dropout: float = 0.3,
        mlp_ratio: float = 4.0,
        max_len: int = 100,
        event_lambda: float = 0.5,
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            decoder_type=decoder_type,
            causal=causal,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            mlp_ratio=mlp_ratio,
            max_len=max_len,
        )

        # Event gate head — shares the transformer output features
        self.event_head = nn.Linear(hidden_dim, 1)
        self.event_lambda = event_lambda

        # Re-init to cover the new head
        self._init_weights()

    def forward(self, features: torch.Tensor):
        """
        Args:
            features: (B, T, D) input features
        Returns:
            logits: (B, T, 3) ternary classification logits
            event_score: (B, T) event gate scores in [0,1] (post-sigmoid)
            event_logits: (B, T, 1) raw logits for BCEWithLogits loss
        """
        x = self.input_proj(features)
        x = self.input_norm(x)
        x = self.pos_encoder(x)
        x = self.emb_dropout(x)
        x = self.decoder(x)

        logits = self.head(x)                   # (B, T, 3)
        event_logits = self.event_head(x)       # (B, T, 1)
        event_score = torch.sigmoid(event_logits).squeeze(-1)  # (B, T)

        return logits, event_score, event_logits

    def forward_modulated(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with event modulation applied (inference only).

        Returns:
            logits_mod: (B, T, 3) with fallen channel gated by event_score
        """
        logits, event_score, _ = self.forward(features)
        logits_mod = logits.clone()
        logits_mod[:, :, 1] = logits_mod[:, :, 1] * event_score  # gate fallen
        return logits_mod


def build_event_labels(fall_labels: torch.Tensor, window_size: int = 64) -> torch.Tensor:
    """
    Build per-frame event labels from binary GT fall labels.

    event_label[f] = 1 if GT fall exists in [max(0, f-window_size), f]
    For efficiency, uses cumulative max over the time dimension.

    Args:
        fall_labels: (B, T) binary tensor, 1 = GT fall frame
        window_size: lookback window in frames (default 64)
    Returns:
        event_labels: (B, T) binary tensor
    """
    B, T = fall_labels.shape
    device = fall_labels.device

    # Expand: check for each frame t, does any frame in [max(0,t-W+1), t] have fall=1?
    # Use 1D max pooling over time with kernel_size=W, stride=1, pad left=W-1
    # maxpool over padded sequence: for position t, the pool covers [t-W+1, t]
    pad = window_size - 1
    # Pad left with zeros
    padded = torch.nn.functional.pad(
        fall_labels.float().unsqueeze(1),  # (B, 1, T)
        (pad, 0),  # pad left only
        value=0.0,
    )  # (B, 1, T+pad)

    # Max pool with kernel_size=W, stride=1
    pooled = torch.nn.functional.max_pool1d(
        padded, kernel_size=window_size, stride=1
    )  # (B, 1, T)

    event_labels = (pooled.squeeze(1) > 0.5).float()  # (B, T)
    return event_labels


# ============================================================
# Self-test
# ============================================================

def test_p7b():
    B, T, D = 4, 64, 1536
    x = torch.randn(B, T, D)
    fall_labels = torch.zeros(B, T)
    fall_labels[:, 20:30] = 1  # fall at frames 20-29
    fall_labels[2, 45:55] = 1  # another fall

    # Test event label construction
    event_labels = build_event_labels(fall_labels, window_size=64)

    # After the fall (frame 30+), event should be 1 for the next 64 frames
    assert event_labels[0, 29] == 1, f"Frame 29 should see fall at 20-29, got {event_labels[0,29]}"
    assert event_labels[0, 30] == 1, f"Frame 30 should see fall at 20-29 within 64-frame window"
    assert event_labels[0, 19] == 0, f"Frame 19 should NOT see fall (fall starts at 20)"

    # Test model forward
    model = P7EventGateModel(
        input_dim=D, hidden_dim=384, causal=False,
        num_layers=1, num_heads=6, dropout=0.3,
        max_len=T + 10, event_lambda=0.5,
    )
    model.eval()
    with torch.no_grad():
        logits, event_score, event_logits = model(x)
        logits_mod = model.forward_modulated(x)

    assert logits.shape == (B, T, 3), f"logits shape: {logits.shape}"
    assert event_score.shape == (B, T), f"event_score shape: {event_score.shape}"
    assert event_logits.shape == (B, T, 1), f"event_logits shape: {event_logits.shape}"
    assert logits_mod.shape == (B, T, 3), f"logits_mod shape: {logits_mod.shape}"

    # Modulated fallen logit should be <= original (soft gating)
    assert (logits_mod[:, :, 1] <= logits[:, :, 1] + 1e-6).all(), \
        "Fallen logit should be gated (<= original)"

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[PASS] P7EventGateModel: logits={logits.shape}, "
          f"event_score={event_score.shape}, params={n_params:,}")
    print(f"[PASS] Event labels: min={event_labels.min()}, max={event_labels.max()}, "
          f"mean={event_labels.float().mean():.3f}")
    print(f"[PASS] Event gate test passed!")


if __name__ == "__main__":
    test_p7b()
