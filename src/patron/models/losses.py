"""PATRON composite loss.

L = 1.0 * barrier CE (masked for ambiguous bars)
  + 0.5 * pinball loss over 6 steps x 3 fields x 3 quantiles
  + 0.2 * momentum CE (class-weighted)
  + 0.15 * masked BCE over 12 pattern tags
Weights come from the config and are never hard-coded at call sites.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

N_QUANTILES = 3


def pinball_loss(
    quantile_predictions: torch.Tensor,  # (B, H, F, Q)
    targets: torch.Tensor,  # (B, H, F)
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
) -> torch.Tensor:
    """Mean pinball loss; NaN targets (never in training windows) are ignored."""
    q = torch.tensor(quantiles, dtype=quantile_predictions.dtype, device=quantile_predictions.device)
    error = targets.unsqueeze(-1) - quantile_predictions
    loss = torch.maximum(q * error, (q - 1.0) * error)
    valid = torch.isfinite(targets).unsqueeze(-1)
    loss = loss * valid
    denom = valid.expand_as(loss).sum()
    return loss.sum() / denom.clamp_min(1.0)


def masked_cross_entropy(
    logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor | None = None
) -> torch.Tensor:
    """CE with a float mask (0 = excluded sample)."""
    per_sample = F.cross_entropy(logits, target, reduction="none", weight=weight)
    return (per_sample * mask).sum() / mask.sum().clamp_min(1.0)


def masked_bce_with_logits(
    logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    per_element = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (per_element * mask).sum() / mask.sum().clamp_min(1.0)


def pattern_metrics(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict:
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        valid = mask > 0
        correct = ((probs >= 0.5).float() == target) & valid
        return {
            "element_accuracy": float(correct.sum() / valid.sum().clamp_min(1.0)),
        }


def total_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    loss_weights: dict[str, float],
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    barrier_weight: torch.Tensor | None = None,
    momentum_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return (total, per-term scalars) for logging."""
    barrier = masked_cross_entropy(
        outputs["barrier"], batch["barrier"], batch["barrier_mask"], barrier_weight
    )
    quantile = pinball_loss(outputs["quantile"], batch["range"], quantiles)
    momentum = F.cross_entropy(outputs["momentum"], batch["momentum"], weight=momentum_weight)
    pattern = masked_bce_with_logits(outputs["pattern"], batch["pattern"], batch["pattern_mask"])

    total = (
        loss_weights["barrier"] * barrier
        + loss_weights["quantile"] * quantile
        + loss_weights["momentum"] * momentum
        + loss_weights["pattern"] * pattern
    )
    return total, {
        "loss": float(total.detach()),
        "barrier_loss": float(barrier.detach()),
        "quantile_loss": float(quantile.detach()),
        "momentum_loss": float(momentum.detach()),
        "pattern_loss": float(pattern.detach()),
    }
