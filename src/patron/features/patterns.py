"""Transparent candlestick pattern annotation (auxiliary labels).

Rules are pure candle geometry with configurable thresholds, evaluated on closed
candles only. Multi-candle patterns read at most three completed candles and
look strictly backwards. These labels are noisy auxiliary supervision - not
ground truth about future price.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

PATTERN_TAGS: tuple[str, ...] = (
    "doji",
    "hammer",
    "inverted_hammer",
    "hanging_man",
    "shooting_star",
    "bullish_engulfing",
    "bearish_engulfing",
    "piercing_line",
    "dark_cloud_cover",
    "morning_star",
    "evening_star",
    "three_candle_continuation_reversal",
)


@dataclass
class PatternThresholds:
    doji_body_to_range: float = 0.10
    shadow_to_body: float = 2.0
    small_body_to_range: float = 0.35
    opposite_shadow_to_range: float = 0.15
    trend_lookback: int = 5
    trend_ema: int = 21
    star_small_body_ratio: float = 0.35
    star_min_body_ratio: float = 0.50
    reversal_body_ratio: float = 1.0
    min_history: int = 3

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class PatternConfig:
    thresholds: PatternThresholds = field(default_factory=PatternThresholds)


def annotate_patterns(
    df: pd.DataFrame, config: PatternConfig | None = None
) -> pd.DataFrame:
    """Return an `[N, 12]` int8 pattern-label frame aligned with ``df``.

    Multi-candle rules produce NaN for rows without enough history so the
    masked BCE loss can ignore them.
    """
    cfg = (config or PatternConfig()).thresholds
    frame = df.reset_index(drop=True)
    o = frame["open"].to_numpy(dtype=np.float64)
    h = frame["high"].to_numpy(dtype=np.float64)
    l = frame["low"].to_numpy(dtype=np.float64)
    c = frame["close"].to_numpy(dtype=np.float64)
    n = len(frame)

    body = np.abs(c - o)
    rng = np.maximum(h - l, 1e-12)
    upper_shadow = h - np.maximum(o, c)
    lower_shadow = np.minimum(o, c) - l
    bullish = c > o
    bearish = c < o
    body_to_range = body / rng

    avg_body = pd.Series(body).rolling(14, min_periods=14).mean().to_numpy()

    close_series = pd.Series(c)
    ema_trend = close_series.ewm(span=cfg.trend_ema, adjust=False, min_periods=cfg.trend_ema).mean()
    prior_return = close_series / close_series.shift(cfg.trend_lookback) - 1.0
    uptrend = (close_series > ema_trend) & (prior_return > 0)
    downtrend = (close_series < ema_trend) & (prior_return < 0)

    out = pd.DataFrame(index=frame.index)

    out["doji"] = body_to_range <= cfg.doji_body_to_range

    hammer_shape = (
        (lower_shadow >= cfg.shadow_to_body * body)
        & (upper_shadow <= cfg.opposite_shadow_to_range * rng)
        & (body_to_range <= cfg.small_body_to_range)
    )
    out["hammer"] = hammer_shape & downtrend.to_numpy()
    inverted_shape = (
        (upper_shadow >= cfg.shadow_to_body * body)
        & (lower_shadow <= cfg.opposite_shadow_to_range * rng)
        & (body_to_range <= cfg.small_body_to_range)
    )
    out["inverted_hammer"] = inverted_shape & downtrend.to_numpy()
    out["hanging_man"] = hammer_shape & uptrend.to_numpy()
    out["shooting_star"] = inverted_shape & uptrend.to_numpy()

    prev_o, prev_c = np.roll(o, 1), np.roll(c, 1)
    prev_body = np.roll(body, 1)
    prev_mid = (prev_o + prev_c) / 2.0
    prev_bearish = np.roll(bearish, 1)
    prev_bullish = np.roll(bullish, 1)

    out["bullish_engulfing"] = prev_bearish & bullish & (o <= prev_c) & (c >= prev_o)
    out["bearish_engulfing"] = prev_bullish & bearish & (o >= prev_c) & (c <= prev_o)

    out["piercing_line"] = (
        prev_bearish & bullish & (o < prev_c) & (c > prev_mid) & (c < prev_o)
    )
    out["dark_cloud_cover"] = (
        prev_bullish & bearish & (o > prev_c) & (c < prev_mid) & (c > prev_o)
    )

    # Morning/evening star: strong candle 1, small candle 2, strong reversal 3.
    c2_o, c2_c = np.roll(o, 2), np.roll(c, 2)
    body2_abs = np.roll(body, 2)
    strong_first = body2_abs >= cfg.star_min_body_ratio * avg_body
    small_second = body <= cfg.star_small_body_ratio * body2_abs
    first_bearish = np.roll(bearish, 2)
    first_bullish = np.roll(bullish, 2)
    mid2 = (c2_o + c2_c) / 2.0
    out["morning_star"] = (
        first_bearish & strong_first & small_second & bullish & (c > mid2) & (c2_c < c2_o)
    )
    out["evening_star"] = (
        first_bullish & strong_first & small_second & bearish & (c < mid2) & (c2_c > c2_o)
    )

    # Three-candle continuation (three rising/falling closes) or a decisive
    # three-candle reversal (candle 3 closes beyond candle 1's open).
    prev_open_2 = np.roll(o, 2)
    bullish3 = bullish & np.roll(bullish, 1) & np.roll(bullish, 2)
    bearish3 = bearish & np.roll(bearish, 1) & np.roll(bearish, 2)
    rising3 = (c > np.roll(c, 1)) & (np.roll(c, 1) > np.roll(c, 2))
    falling3 = (c < np.roll(c, 1)) & (np.roll(c, 1) < np.roll(c, 2))
    continuation = (bullish3 & rising3) | (bearish3 & falling3)
    reversal = (
        (np.roll(bullish, 2) & bearish & (c < prev_open_2) & (body >= cfg.reversal_body_ratio * avg_body))
        | (np.roll(bearish, 2) & bullish & (c > prev_open_2) & (body >= cfg.reversal_body_ratio * avg_body))
    )
    out["three_candle_continuation_reversal"] = continuation | reversal

    labels = out[list(PATTERN_TAGS)].astype("float32")

    # Rows without the required history (or a valid avg body) are unknown.
    warmup = np.zeros(n, dtype=bool)
    warmup[: max(cfg.min_history, 14)] = True
    warmup |= ~np.isfinite(avg_body)
    labels.loc[warmup, :] = np.nan
    return labels


def pattern_config_from_dict(payload: dict | None) -> PatternConfig:
    return PatternConfig(PatternThresholds(**(payload or {})))
