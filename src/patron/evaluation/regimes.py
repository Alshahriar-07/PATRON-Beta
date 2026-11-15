"""Ex-ante market-regime labelling and regime-sliced metrics.

Regime labels are computed with *expanding* statistics shifted by one bar, so a
bar's regime is knowable at its close: the thresholds use only earlier bars.
This matters because regimes are used for evaluation slices and pattern
outcome analysis — a regime label that peeked ahead would launder look-ahead
bias into the results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TREND_REGIMES = ("bull", "sideways", "bear")
VOL_REGIMES = ("low_vol", "normal_vol", "high_vol")

TREND_LOOKBACK = 96  # one trading day of 15m bars
VOL_LOOKBACK = 96
MIN_PERIODS = 96


def add_regimes(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach ``regime_trend`` and ``regime_vol`` columns (causal)."""
    out = frame.copy()
    close = out["close"].astype("float64")
    ret = close.pct_change(TREND_LOOKBACK)

    # Expanding z-score of the rolling return: today's regime uses only history.
    expanding_mean = ret.expanding(min_periods=MIN_PERIODS).mean().shift(1)
    expanding_std = ret.expanding(min_periods=MIN_PERIODS).std().shift(1)
    z = (ret - expanding_mean) / expanding_std.replace(0.0, np.nan)

    trend = np.where(z > 0.5, "bull", np.where(z < -0.5, "bear", "sideways"))
    out["regime_trend"] = trend
    out.loc[z.isna(), "regime_trend"] = "unknown"

    log_ret = np.log(close).diff()
    realized = log_ret.rolling(VOL_LOOKBACK, min_periods=VOL_LOOKBACK).std()
    low_q = realized.expanding(min_periods=MIN_PERIODS).quantile(0.30).shift(1)
    high_q = realized.expanding(min_periods=MIN_PERIODS).quantile(0.70).shift(1)
    vol = np.where(realized > high_q, "high_vol", np.where(realized < low_q, "low_vol", "normal_vol"))
    out["regime_vol"] = vol
    out.loc[realized.isna() | low_q.isna(), "regime_vol"] = "unknown"
    return out


def regime_breakdown(
    labels: np.ndarray,
    predictions: np.ndarray,
    regimes: pd.Series,
    metrics_fn,
) -> dict:
    """Compute ``metrics_fn`` within each regime slice plus coverage counts."""
    regimes = pd.Series(regimes).reset_index(drop=True)
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    result: dict = {}
    for name, group in regimes.groupby(regimes):
        idx = group.index.to_numpy(dtype=np.int64)
        if len(idx) < 20:
            result[str(name)] = {"n": int(len(idx)), "skipped": "fewer than 20 samples"}
            continue
        result[str(name)] = {"n": int(len(idx)), **metrics_fn(labels[idx], predictions[idx])}
    return result


def summarize_regime_coverage(frame: pd.DataFrame) -> dict:
    """Share of rows in each regime (for the report tables)."""
    if "regime_trend" not in frame or "regime_vol" not in frame:
        raise ValueError("call add_regimes first")
    total = len(frame)
    return {
        "trend": {k: round(float(v) / total, 4) for k, v in frame["regime_trend"].value_counts().items()},
        "vol": {k: round(float(v) / total, 4) for k, v in frame["regime_vol"].value_counts().items()},
    }
