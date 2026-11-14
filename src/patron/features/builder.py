"""The exact 12 PATRON features, computed causally.

Every rolling/window operation looks backwards only. The higher-timeframe
component is aligned on the HTF candle *close* timestamp, so a partially formed
1h/4h candle can never contribute.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from patron.data.htf import htf_trend_frame
from patron.features.indicators import atr_wilder, ema, rsi_wilder
from patron.features.schema import FEATURE_ORDER, FEATURE_WARMUP_BARS

LOGGER = logging.getLogger(__name__)

EPS = 1e-12

#: Auxiliary (non-model) columns carried alongside features for labelling and
#: gap-aware window filtering.
AUX_COLUMNS = ("timestamp_open", "timestamp_close", "close", "atr14", "is_gap", "feature_valid")


def build_features(
    df: pd.DataFrame,
    atr_period: int = 14,
    volume_baseline: int = FEATURE_WARMUP_BARS,
    feature_order: tuple[str, ...] | list[str] = FEATURE_ORDER,
) -> pd.DataFrame:
    """Compute the PATRON feature frame from canonical 15m candles.

    Args:
        df: canonical candles (timestamp_open, timestamp_close, OHLCV), sorted or not.
        atr_period: ATR/RSI period (spec: 14).
        volume_baseline: rolling volume baseline length (spec: 96).
        feature_order: fixed feature order; validated against the stored schema.

    Returns:
        DataFrame with timestamps, `close`, `atr14`, `is_gap`, `feature_valid`
        and the 12 feature columns in the exact configured order.
    """
    if list(feature_order) != list(FEATURE_ORDER):
        raise ValueError(
            "feature order mismatch: inference/training must use the fixed "
            f"12-feature order, got {list(feature_order)}"
        )

    frame = df.copy()
    frame["timestamp_open"] = pd.to_datetime(frame["timestamp_open"], utc=True)
    frame["timestamp_close"] = pd.to_datetime(frame["timestamp_close"], utc=True)
    frame = frame.sort_values("timestamp_open").reset_index(drop=True)
    if frame.empty:
        return pd.DataFrame(columns=[*AUX_COLUMNS, *FEATURE_ORDER])

    prev_close = frame["close"].shift(1)

    open_rel = frame["open"] / prev_close - 1.0
    high_rel = frame["high"] / prev_close - 1.0
    low_rel = frame["low"] / prev_close - 1.0
    close_rel = frame["close"] / prev_close - 1.0

    log_volume = np.log1p(frame["volume"].clip(lower=0.0))
    volume_baseline_mean = log_volume.rolling(volume_baseline, min_periods=volume_baseline).mean()
    volume_log_rel = log_volume - volume_baseline_mean

    body_to_range = (frame["close"] - frame["open"]) / (
        (frame["high"] - frame["low"]) + EPS
    )

    atr14 = atr_wilder(frame, atr_period)
    atr14_pct = atr14 / frame["close"]

    rsi14 = rsi_wilder(frame, atr_period)
    rsi14_scaled = (rsi14 - 50.0) / 50.0

    ema9 = ema(frame["close"], 9)
    ema21 = ema(frame["close"], 21)
    ema9_gap = frame["close"] / ema9 - 1.0
    ema21_gap = frame["close"] / ema21 - 1.0

    rolling_median = frame["volume"].rolling(20, min_periods=20).median().replace(0.0, np.nan)
    volume_spike = frame["volume"] / rolling_median

    htf = htf_trend_frame(frame)
    htf_trend = htf["htf_trend"].to_numpy()

    features = pd.DataFrame(
        {
            "open_rel": open_rel,
            "high_rel": high_rel,
            "low_rel": low_rel,
            "close_rel": close_rel,
            "volume_log_rel": volume_log_rel,
            "body_to_range": body_to_range,
            "atr14_pct": atr14_pct,
            "rsi14_scaled": rsi14_scaled,
            "ema9_gap": ema9_gap,
            "ema21_gap": ema21_gap,
            "volume_spike": volume_spike,
            "htf_trend": htf_trend,
        }
    )[list(FEATURE_ORDER)]

    # Clip extreme ratios to a sane numeric range (no winsorising of the
    # distribution the model learns, only protection against inf/overflow).
    features = features.replace([np.inf, -np.inf], np.nan).clip(-50.0, 50.0)

    out = pd.DataFrame(
        {
            "timestamp_open": frame["timestamp_open"],
            "timestamp_close": frame["timestamp_close"],
            "close": frame["close"],
            "atr14": atr14,
            "is_gap": frame.get("is_gap", pd.Series(0, index=frame.index)).astype("int8"),
        }
    )
    out = pd.concat([out, features], axis=1)
    out["feature_valid"] = np.isfinite(out[list(FEATURE_ORDER)].to_numpy()).all(axis=1)
    return out


def feature_matrix(frame: pd.DataFrame, feature_order: tuple[str, ...] | list[str] = FEATURE_ORDER) -> np.ndarray:
    """Return the `[N, 12]` float32 feature matrix in the fixed order."""
    return frame[list(feature_order)].to_numpy(dtype=np.float32, copy=True)


def save_feature_schema(path, feature_order: tuple[str, ...] | list[str] = FEATURE_ORDER) -> None:
    """Persist the feature schema/version next to the scaler."""
    from patron.features.schema import FEATURE_SCHEMA_VERSION, FEATURE_VERSION
    from patron.utils.io import write_json

    definitions = {
        "open_rel": "open_t / close_{t-1} - 1",
        "high_rel": "high_t / close_{t-1} - 1",
        "low_rel": "low_t / close_{t-1} - 1",
        "close_rel": "close_t / close_{t-1} - 1",
        "volume_log_rel": "log1p(volume_t) - mean(log1p(volume), 96)",
        "body_to_range": "(close_t - open_t) / (high_t - low_t + 1e-12)",
        "atr14_pct": "ATR(14)_t / close_t",
        "rsi14_scaled": "RSI(14)_t mapped from [0,100] to [-1,1]",
        "ema9_gap": "close_t / EMA9_t - 1",
        "ema21_gap": "close_t / EMA21_t - 1",
        "volume_spike": "volume_t / rolling_median(volume, 20)",
        "htf_trend": "mean(tanh((close/EMA9 - 1)/ATR14_pct)) over fully closed 1h and 4h candles",
    }
    write_json(
        {
            "feature_version": FEATURE_VERSION,
            "schema_version": FEATURE_SCHEMA_VERSION,
            "feature_order": list(feature_order),
            "feature_count": len(feature_order),
            "dtype": "float32",
            "definitions": definitions,
        },
        path,
    )
