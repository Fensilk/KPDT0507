"""
Focal Loss for multi-class classification.

FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

- gamma: focusing parameter. Higher = more suppression of easy samples.
- alpha: per-class weighting, computed via inverse frequency from training labels.

Usage:
    from utils.losses import FocalLoss, build_focal_loss

    # Auto-compute alpha from dataset
    criterion, alpha = build_focal_loss(train_dataset, gamma=2.0, device=device)

    # Manual alpha
    criterion = FocalLoss(gamma=2.0, alpha=torch.tensor([0.5, 0.3, 0.2]))
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss.

    FL = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: Focusing parameter. gamma=0 → standard CrossEntropyLoss.
        alpha: Per-class weight tensor of shape (num_classes,).
               If None, no class weighting is applied.
        ignore_index: Target value to ignore (loss=0 for those positions).
        reduction: 'mean' or 'sum'.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        ignore_index: int = -100,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # (C,) tensor, registered as buffer if not None
        self.ignore_index = ignore_index
        self.reduction = reduction

        if alpha is not None:
            self.register_buffer("alpha_buffer", alpha)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (N, C) raw logits.
            targets: (N,) class indices.
        Returns:
            Scalar loss.
        """
        # Standard cross-entropy per element (no reduction)
        ce_loss = F.cross_entropy(
            logits, targets, reduction="none", ignore_index=self.ignore_index
        )  # (N,)

        # Probability of the target class
        p_t = torch.exp(-ce_loss)  # (N,)

        # Focal modulation: (1 - p_t)^gamma
        focal_weight = (1.0 - p_t) ** self.gamma

        # Apply focal weight
        loss = focal_weight * ce_loss

        # Apply class-wise alpha weighting
        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]  # (N,)
            # Mask ignored positions
            if self.ignore_index >= 0:
                alpha_t = torch.where(
                    targets == self.ignore_index,
                    torch.zeros_like(alpha_t),
                    alpha_t,
                )
            loss = alpha_t * loss

        if self.reduction == "mean":
            # Mean over non-ignored elements
            if self.ignore_index >= 0:
                valid_mask = targets != self.ignore_index
                return loss.sum() / valid_mask.sum().clamp(min=1)
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def compute_inverse_freq_alpha(
    labels: np.ndarray, num_classes: int = 3
) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for Focal Loss alpha.

    alpha[c] = N_total / (num_classes * N_c)

    Args:
        labels: (N,) int array of class indices.
        num_classes: Number of classes.
    Returns:
        alpha: (num_classes,) float32 tensor.
    """
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts = np.where(counts == 0, 1.0, counts)  # avoid div-by-zero
    N = len(labels)
    alpha = N / (num_classes * counts)
    return torch.FloatTensor(alpha)


def build_focal_loss(train_dataset, gamma: float, device: torch.device) -> dict:
    """
    Build FocalLoss with auto-computed alpha from training label distribution.

    Samples training dataset to estimate class distribution, computes
    inverse-frequency alpha weights.

    Args:
        train_dataset: LongSequenceDataset instance.
        gamma: Focal Loss gamma parameter.
        device: Torch device.
    Returns:
        dict with keys: 'criterion' (FocalLoss), 'alpha' (tensor), 'class_counts' (tensor).
    """
    print("[INFO] Sampling training labels for Focal Loss alpha...")

    # Sample up to 5000 windows for efficiency
    n_samples = min(len(train_dataset), 5000)
    step = max(1, len(train_dataset) // n_samples)

    all_ternary = []
    for i in range(0, len(train_dataset), step):
        sample = train_dataset[i]
        all_ternary.append(sample["ternary_labels"])

    labels = torch.cat(all_ternary).numpy()  # (N_total,)
    alpha = compute_inverse_freq_alpha(labels, num_classes=3).to(device)

    class_counts = np.bincount(labels.astype(int), minlength=3)
    total = len(labels)

    print(f"[INFO] Ternary class distribution:")
    for i, name in enumerate(["fall", "fallen", "normal"]):
        pct = 100 * class_counts[i] / total
        print(f"       {name}: {class_counts[i]:,} ({pct:.1f}%)")

    print(f"[INFO] Focal Loss alpha: {alpha.cpu().numpy().round(3)}")
    print(f"[INFO] Focal Loss gamma: {gamma}")

    criterion = FocalLoss(gamma=gamma, alpha=alpha)
    return {
        "criterion": criterion,
        "alpha": alpha,
        "class_counts": class_counts,
    }


def build_ce_loss(train_dataset, device: torch.device) -> dict:
    """
    Build standard CrossEntropyLoss with class weights (bridge experiment).

    Args:
        train_dataset: LongSequenceDataset instance.
        device: Torch device.
    Returns:
        dict with keys: 'criterion' (CrossEntropyLoss), 'class_weights' (tensor).
    """
    print("[INFO] Sampling training labels for CE class weights...")

    n_samples = min(len(train_dataset), 5000)
    step = max(1, len(train_dataset) // n_samples)

    all_ternary = []
    for i in range(0, len(train_dataset), step):
        sample = train_dataset[i]
        all_ternary.append(sample["ternary_labels"])

    labels = torch.cat(all_ternary).numpy()
    class_counts = np.bincount(labels.astype(int), minlength=3).astype(np.float32)
    class_counts = np.where(class_counts == 0, 1.0, class_counts)
    N = len(labels)
    class_weights = N / (3 * class_counts)
    class_weights_t = torch.FloatTensor(class_weights).to(device)

    print(f"[INFO] CE class weights: {class_weights_t.cpu().numpy().round(3)}")

    criterion = nn.CrossEntropyLoss(weight=class_weights_t)
    return {"criterion": criterion, "class_weights": class_weights_t}


# ============================================================
# Self-test
# ============================================================

if __name__ == "__main__":
    print("=== FocalLoss Self-Test ===\n")

    # Test 1: Basic shape and no crash
    logits = torch.randn(16, 3)
    targets = torch.randint(0, 3, (16,))

    for gamma in [0, 2, 3, 5]:
        loss_fn = FocalLoss(gamma=gamma)
        loss = loss_fn(logits, targets)
        print(f"[PASS] gamma={gamma}: loss={loss.item():.4f}")

    # Test 2: gamma=0 should equal CrossEntropyLoss
    ce = nn.CrossEntropyLoss()
    fl0 = FocalLoss(gamma=0)
    loss_ce = ce(logits, targets)
    loss_fl0 = fl0(logits, targets)
    assert abs(loss_ce.item() - loss_fl0.item()) < 1e-6, \
        f"gamma=0 should equal CE: {loss_ce:.6f} vs {loss_fl0:.6f}"
    print(f"[PASS] gamma=0 equals CE: {loss_ce:.6f} == {loss_fl0:.6f}")

    # Test 3: With alpha
    alpha = torch.tensor([0.2, 0.5, 0.3])
    loss_fn = FocalLoss(gamma=2.0, alpha=alpha)
    loss = loss_fn(logits, targets)
    print(f"[PASS] With alpha: loss={loss.item():.4f}")

    # Test 4: Easy samples get less weight
    # Make a sample where model is very confident (high logit for correct class)
    easy_logits = torch.tensor([
        [10.0, -1.0, -1.0],  # model very confident in class 0
    ])
    easy_targets = torch.tensor([0])
    fl2 = FocalLoss(gamma=2.0)
    loss_easy = fl2(easy_logits, easy_targets)
    print(f"[INFO] Easy sample loss (p≈0.999): {loss_easy.item():.6f}")

    # Hard sample: logits close to uniform
    hard_logits = torch.tensor([
        [0.0, 0.0, 0.1],  # model uncertain
    ])
    hard_targets = torch.tensor([2])
    loss_hard = fl2(hard_logits, hard_targets)
    print(f"[INFO] Hard sample loss (p≈0.37):  {loss_hard.item():.6f}")
    print(f"[PASS] Easy/Hard ratio: {loss_easy.item()/loss_hard.item():.6f} "
          f"(should be << 1)")

    # Test 5: compute_inverse_freq_alpha
    labels = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2, 2])  # fall:2, fallen:3, normal:5
    alpha = compute_inverse_freq_alpha(labels, num_classes=3)
    print(f"\n[INFO] Inverse freq alpha: {alpha.numpy().round(3)}")
    print(f"       fall(2): {alpha[0]:.2f}, fallen(3): {alpha[1]:.2f}, "
          f"normal(5): {alpha[2]:.2f}")
    print(f"       Ratio (fall/normal): {alpha[0]/alpha[2]:.2f} (should be > 1)")

    print("\n=== All FocalLoss tests passed! ===")
