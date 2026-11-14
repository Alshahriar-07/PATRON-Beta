"""Five-class momentum labels from the future 6-candle close return.

The return is normalised by the ATR% at the window end (a volatility-normalised
move), then binned with the spec's initial boundaries:
``[-1.0, -0.35, +0.35, +1.0]``.
"""

from __future__ import annotations

import numpy as np

MOMENTUM_CLASSES: tuple[str, ...] = (
    "strong_bullish",
    "bullish",
    "sideways",
    "bearish",
    "strong_bearish",
)

MOMENTUM_BOUNDARIES: tuple[float, ...] = (-1.0, -0.35, 0.35, 1.0)


def momentum_labels(
    close: np.ndarray,
    atr: np.ndarray,
    horizon: int = 6,
    boundaries: tuple[float, ...] | list[float] = MOMENTUM_BOUNDARIES,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (class int8 with -1 for unavailable, normalized return)."""
    close = np.asarray(close, dtype=np.float64)
    atr = np.asarray(atr, dtype=np.float64)
    if len(boundaries) != 4:
        raise ValueError("expected 4 momentum boundaries")
    n = len(close)
    cls = np.full(n, -1, dtype=np.int8)
    norm_ret = np.full(n, np.nan, dtype=np.float64)

    valid = np.arange(n) + horizon < n
    future_close = np.full(n, np.nan)
    future_idx = np.minimum(np.arange(n) + horizon, n - 1)
    future_close[valid] = close[future_idx][valid]
    raw = future_close / close - 1.0
    atr_pct = atr / close
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = raw / atr_pct
    usable = valid & np.isfinite(normalized) & (atr_pct > 0)
    norm_ret[usable] = normalized[usable]

    b = list(boundaries)
    cls[usable & (normalized >= b[3])] = 0  # strong bullish
    cls[usable & (normalized >= b[2]) & (normalized < b[3])] = 1  # bullish
    cls[usable & (normalized > b[1]) & (normalized < b[2])] = 2  # sideways
    cls[usable & (normalized > b[0]) & (normalized <= b[1])] = 3  # bearish
    cls[usable & (normalized <= b[0])] = 4  # strong bearish
    return cls, norm_ret


def momentum_class_weights(labels: np.ndarray, valid: np.ndarray, n_classes: int = 5) -> np.ndarray:
    counts = np.bincount(labels[valid], minlength=n_classes).astype(np.float64)
    weights = np.zeros(n_classes, dtype=np.float64)
    non_empty = counts > 0
    weights[non_empty] = counts.sum() / (non_empty.sum() * counts[non_empty])
    return weights.astype(np.float32)
