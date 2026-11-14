"""Post-hoc temperature scaling for the barrier head.

Temperature is fitted on the validation split only (never the test split) and
the value is stored with the evaluation report and the checkpoint.
"""

from __future__ import annotations

import numpy as np
import torch


def _nll(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
    scaled = logits / temperature.clamp_min(1e-3)
    per_sample = torch.nn.functional.cross_entropy(scaled, labels, reduction="none")
    return (per_sample * mask).sum() / mask.sum().clamp_min(1.0)


def fit_temperature(
    logits: np.ndarray, labels: np.ndarray, valid: np.ndarray
) -> float:
    """Return the temperature minimising masked validation NLL."""
    logit_t = torch.tensor(logits, dtype=torch.float32)
    label_t = torch.tensor(labels, dtype=torch.long)
    mask_t = torch.tensor(valid.astype(np.float32))
    temperature = torch.nn.Parameter(torch.tensor([1.2]))
    optimizer = torch.optim.LBFGS([temperature], lr=0.1, max_iter=100)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = _nll(logit_t, label_t, mask_t, temperature)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(max(0.05, temperature.detach().item()))


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = logits / max(temperature, 1e-3)
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)
