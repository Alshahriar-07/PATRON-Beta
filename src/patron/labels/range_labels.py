"""Forward High/Low/Close targets for each of the next N candles.

Targets are relative to the last close of the input window (log-return space via
simple relative returns), never absolute BTC prices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RANGE_FIELDS: tuple[str, ...] = ("high", "low", "close")


def forward_range_targets(
    close: np.ndarray, high: np.ndarray, low: np.ndarray, horizon: int = 6
) -> np.ndarray:
    """Return `[N, horizon, 3]` relative targets in (high, low, close) order."""
    close = np.asarray(close, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    n = len(close)
    out = np.full((n, horizon, len(RANGE_FIELDS)), np.nan, dtype=np.float64)
    for step in range(1, horizon + 1):
        valid = np.arange(n) + step < n
        fut = np.minimum(np.arange(n) + step, n - 1)
        out[valid, step - 1, 0] = high[fut][valid] / close[valid] - 1.0
        out[valid, step - 1, 1] = low[fut][valid] / close[valid] - 1.0
        out[valid, step - 1, 2] = close[fut][valid] / close[valid] - 1.0
    return out


def range_target_frame(targets: np.ndarray, horizon: int = 6) -> pd.DataFrame:
    """Flatten `[N, H, 3]` targets into `tgt_{field}_{step}` columns."""
    n = targets.shape[0]
    columns = {}
    for field_idx, field in enumerate(RANGE_FIELDS):
        for step in range(1, horizon + 1):
            columns[f"tgt_{field}_{step}"] = targets[:, step - 1, field_idx]
    return pd.DataFrame(columns, index=pd.RangeIndex(n))


def frame_to_targets(frame: pd.DataFrame, horizon: int = 6) -> np.ndarray:
    """Inverse of :func:`range_target_frame` -> `[N, H, 3]`."""
    n = len(frame)
    out = np.empty((n, horizon, len(RANGE_FIELDS)), dtype=np.float64)
    for field_idx, field in enumerate(RANGE_FIELDS):
        for step in range(1, horizon + 1):
            out[:, step - 1, field_idx] = frame[f"tgt_{field}_{step}"].to_numpy()
    return out
