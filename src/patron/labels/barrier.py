"""Triple-barrier labels for the next N candles.

Rules (from the ZIP spec):
* reference = close at the end of the input window;
* volatility scale = ATR(14) at the same bar;
* upper = reference + 1.5 * ATR, lower = reference - 1.0 * ATR;
* scan the next 6 closed candles, first barrier touched wins;
* neither -> timeout;
* both barriers inside the same candle -> intrabar order unknown -> ambiguous.

Ambiguous rows are masked out of the primary barrier loss (documented rule) and
can be analysed separately with the conservative SL-first sensitivity rule.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TP_FIRST = 0
SL_FIRST = 1
TIMEOUT = 2
AMBIGUOUS = -1

BARRIER_CLASSES: tuple[str, ...] = ("tp_first", "sl_first", "timeout")


@dataclass
class BarrierLabelResult:
    label: np.ndarray  # int8, -1 = ambiguous/unavailable
    mask: np.ndarray  # bool, False = excluded from primary loss
    tp_step: np.ndarray  # int8, 0 = not hit
    sl_step: np.ndarray
    ambiguous: np.ndarray  # bool
    upper_barrier: np.ndarray
    lower_barrier: np.ndarray

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "barrier_label": self.label,
                "barrier_mask": self.mask.astype("int8"),
                "barrier_tp_step": self.tp_step,
                "barrier_sl_step": self.sl_step,
                "barrier_ambiguous": self.ambiguous.astype("int8"),
                "barrier_upper": self.upper_barrier,
                "barrier_lower": self.lower_barrier,
            }
        )


def triple_barrier_labels(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    atr: np.ndarray,
    horizon: int = 6,
    tp_mult: float = 1.5,
    sl_mult: float = 1.0,
    conservative_ambiguous: bool = False,
) -> BarrierLabelResult:
    """Compute TP-first / SL-first / timeout labels for every bar."""
    close = np.asarray(close, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    atr = np.asarray(atr, dtype=np.float64)
    n = len(close)
    if not (len(high) == len(low) == len(atr) == n):
        raise ValueError("close/high/low/atr length mismatch")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    upper = close + tp_mult * atr
    lower = close - sl_mult * atr
    label = np.full(n, AMBIGUOUS, dtype=np.int8)
    mask = np.zeros(n, dtype=bool)
    tp_step = np.zeros(n, dtype=np.int8)
    sl_step = np.zeros(n, dtype=np.int8)
    ambiguous = np.zeros(n, dtype=bool)

    has_future = np.arange(n) + horizon < n
    unresolved = np.zeros(n, dtype=bool)

    for step in range(1, horizon + 1):
        idx = np.arange(n)
        fut = np.minimum(idx + step, n - 1)
        fut_high = high[fut]
        fut_low = low[fut]
        hit_tp = fut_high >= upper
        hit_sl = fut_low <= lower

        active = has_future & ~unresolved
        both = active & hit_tp & hit_sl
        if both.any():
            ambiguous |= both
            if conservative_ambiguous:
                label[both] = SL_FIRST
                sl_step[both] = step
                mask[both] = True
            unresolved |= both

        tp_only = active & hit_tp & ~hit_sl
        label[tp_only] = TP_FIRST
        tp_step[tp_only] = step
        mask[tp_only] = True
        unresolved |= tp_only

        sl_only = active & hit_sl & ~hit_tp
        label[sl_only] = SL_FIRST
        sl_step[sl_only] = step
        mask[sl_only] = True
        unresolved |= sl_only

    timeout = has_future & ~unresolved
    label[timeout] = TIMEOUT
    mask[timeout] = True

    # Rows without enough future candles or with unusable ATR never get labels.
    bad = ~np.isfinite(upper) | ~np.isfinite(lower) | ~has_future
    label[bad] = AMBIGUOUS
    mask[bad] = False

    return BarrierLabelResult(
        label=label,
        mask=mask,
        tp_step=tp_step,
        sl_step=sl_step,
        ambiguous=ambiguous,
        upper_barrier=upper,
        lower_barrier=lower,
    )


def barrier_class_weights(labels: np.ndarray, mask: np.ndarray, n_classes: int = 3) -> np.ndarray:
    """Inverse-frequency class weights from the training partition only."""
    valid = labels[mask]
    counts = np.bincount(valid[(valid >= 0) & (valid < n_classes)], minlength=n_classes)
    weights = np.zeros(n_classes, dtype=np.float64)
    non_empty = counts > 0
    weights[non_empty] = counts.sum() / (non_empty.sum() * counts[non_empty])
    return weights.astype(np.float32)
