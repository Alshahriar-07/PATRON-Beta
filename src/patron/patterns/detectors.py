"""Candle primitives and the shared detection context.

Every array is indexed by bar and computed from bars ``<= t`` only: rolling
statistics use ``shift``-ed history and gap checks compare against the previous
bar. No detector may look at ``t+1`` — the pattern-dataset builder depends on
this guarantee, and ``tests/test_pattern_registry.py`` verifies it by truncation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

EPS = 1e-12


def shift(arr: np.ndarray, k: int, fill: float = np.nan) -> np.ndarray:
    """Shift an array *forward* in time by ``k`` bars (no future values)."""
    if k <= 0:
        raise ValueError("shift k must be positive")
    out = np.full_like(arr, fill, dtype=np.float64)
    if k < len(arr):
        out[k:] = arr[:-k]
    return out


@dataclass(frozen=True)
class PatternThresholds:
    """Documented thresholds, matching the candlestick literature's conventions."""

    doji_body_ratio: float = 0.10
    small_body_ratio: float = 0.30
    long_body_ratio: float = 0.60
    shadow_to_body: float = 2.0
    opposite_shadow_ratio: float = 0.15
    marubozu_shadow_ratio: float = 0.05
    spinning_body_ratio: float = 0.30
    trend_ema: int = 21
    trend_lookback: int = 5
    star_small_body: float = 0.35
    star_long_body: float = 0.55
    equal_tolerance: float = 0.0008  # 8 bps for "matching" highs/lows
    gap_tolerance: float = 0.0
    avg_body_period: int = 14
    structure_lookback: int = 20
    structure_swing: int = 3

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class CandleContext:
    """All arrays a detector needs, precomputed once per dataframe."""

    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    v: np.ndarray
    body: np.ndarray
    rng: np.ndarray
    upper: np.ndarray
    lower: np.ndarray
    body_ratio: np.ndarray
    upper_ratio: np.ndarray
    lower_ratio: np.ndarray
    bull: np.ndarray
    bear: np.ndarray
    avg_body: np.ndarray
    atr: np.ndarray
    trend_up: np.ndarray
    trend_down: np.ndarray
    gap_up: np.ndarray
    gap_down: np.ndarray
    volume_ratio: np.ndarray
    thresholds: PatternThresholds

    @property
    def n(self) -> int:
        return len(self.c)

    # convenience, all backward-looking
    def prev(self, arr: np.ndarray, k: int = 1) -> np.ndarray:
        return shift(np.asarray(arr, dtype=np.float64), k)

    def prev_mask(self, mask: np.ndarray, k: int = 1) -> np.ndarray:
        out = np.zeros_like(mask, dtype=bool)
        if k < len(mask):
            out[k:] = np.asarray(mask, dtype=bool)[:-k]
        return out


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(arr).rolling(window, min_periods=window).mean().to_numpy()


def _ema(arr: np.ndarray, span: int) -> np.ndarray:
    return pd.Series(arr).ewm(span=span, adjust=False, min_periods=span).mean().to_numpy()


def _wilder_atr(h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int = 14) -> np.ndarray:
    prev_close = shift(c, 1)
    tr = np.maximum.reduce([h - l, np.abs(h - prev_close), np.abs(l - prev_close)])
    out = np.full_like(tr, np.nan)
    if len(tr) <= period:
        return out
    seed = np.nanmean(tr[1 : period + 1])
    out[period] = seed
    for i in range(period + 1, len(tr)):
        out[i] = (out[i - 1] * (period - 1) + tr[i]) / period
    return out


def build_context(
    df: pd.DataFrame, thresholds: PatternThresholds | None = None
) -> CandleContext:
    """Compute the shared causal context from canonical candles."""
    cfg = thresholds or PatternThresholds()
    frame = df.reset_index(drop=True)
    o = frame["open"].to_numpy(dtype=np.float64)
    h = frame["high"].to_numpy(dtype=np.float64)
    l = frame["low"].to_numpy(dtype=np.float64)
    c = frame["close"].to_numpy(dtype=np.float64)
    v = frame["volume"].to_numpy(dtype=np.float64) if "volume" in frame else np.ones_like(c)

    body = np.abs(c - o)
    rng = np.maximum(h - l, EPS)
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l
    bull = c > o
    bear = c < o
    avg_body = _rolling_mean(body, cfg.avg_body_period)
    atr = _wilder_atr(h, l, c, 14)
    ema = _ema(c, cfg.trend_ema)
    close_series = pd.Series(c)
    prior_return = (close_series / close_series.shift(cfg.trend_lookback) - 1.0).to_numpy()
    trend_up = (c > ema) & (prior_return > 0)
    trend_down = (c < ema) & (prior_return < 0)
    prev_h = shift(h, 1)
    prev_l = shift(l, 1)
    gap_up = l > prev_h * (1.0 + cfg.gap_tolerance)
    gap_down = h < prev_l * (1.0 - cfg.gap_tolerance)
    vol_base = _rolling_mean(v, cfg.avg_body_period)
    volume_ratio = np.where(np.isfinite(vol_base) & (vol_base > 0), v / vol_base, np.nan)

    return CandleContext(
        o=o,
        h=h,
        l=l,
        c=c,
        v=v,
        body=body,
        rng=rng,
        upper=upper,
        lower=lower,
        body_ratio=body / rng,
        upper_ratio=upper / rng,
        lower_ratio=lower / rng,
        bull=bull,
        bear=bear,
        avg_body=avg_body,
        atr=atr,
        trend_up=trend_up,
        trend_down=trend_down,
        gap_up=gap_up,
        gap_down=gap_down,
        volume_ratio=volume_ratio,
        thresholds=cfg,
    )


# --- reusable primitives ----------------------------------------------------


def doji(ctx: CandleContext, max_ratio: float | None = None) -> np.ndarray:
    return ctx.body_ratio <= (ctx.thresholds.doji_body_ratio if max_ratio is None else max_ratio)


def small_body(ctx: CandleContext) -> np.ndarray:
    return ctx.body_ratio <= ctx.thresholds.small_body_ratio


def long_body(ctx: CandleContext) -> np.ndarray:
    return ctx.body_ratio >= ctx.thresholds.long_body_ratio


def strong_body(ctx: CandleContext) -> np.ndarray:
    """Body at least as large as the recent average body."""
    return ctx.body >= ctx.avg_body


def hammer_shape(ctx: CandleContext) -> np.ndarray:
    cfg = ctx.thresholds
    return (
        (ctx.lower >= cfg.shadow_to_body * np.maximum(ctx.body, EPS))
        & (ctx.upper <= cfg.opposite_shadow_ratio * ctx.rng)
        & (ctx.body_ratio <= cfg.small_body_ratio)
    )


def shooting_shape(ctx: CandleContext) -> np.ndarray:
    cfg = ctx.thresholds
    return (
        (ctx.upper >= cfg.shadow_to_body * np.maximum(ctx.body, EPS))
        & (ctx.lower <= cfg.opposite_shadow_ratio * ctx.rng)
        & (ctx.body_ratio <= cfg.small_body_ratio)
    )


def engulfing(ctx: CandleContext, direction: str) -> np.ndarray:
    prev_o, prev_c = ctx.prev(ctx.o), ctx.prev(ctx.c)
    prev_body = ctx.prev(ctx.body)
    prev_bull = ctx.prev_mask(ctx.bull)
    prev_bear = ctx.prev_mask(ctx.bear)
    if direction == "bullish":
        return prev_bear & ctx.bull & (ctx.o <= prev_c) & (ctx.c >= prev_o) & (
            ctx.body >= prev_body
        )
    return prev_bull & ctx.bear & (ctx.o >= prev_c) & (ctx.c <= prev_o) & (
        ctx.body >= prev_body
    )


def harami(ctx: CandleContext, direction: str) -> np.ndarray:
    prev_o, prev_c = ctx.prev(ctx.o), ctx.prev(ctx.c)
    prev_top = np.maximum(prev_o, prev_c)
    prev_bottom = np.minimum(prev_o, prev_c)
    inside = (np.maximum(ctx.o, ctx.c) <= prev_top) & (np.minimum(ctx.o, ctx.c) >= prev_bottom)
    prev_long = ctx.prev(ctx.body) >= ctx.prev(np.nan_to_num(ctx.avg_body, nan=np.inf))
    if direction == "bullish":
        return ctx.prev_mask(ctx.bear) & ctx.bull & inside & prev_long
    return ctx.prev_mask(ctx.bull) & ctx.bear & inside & prev_long


def gap(ctx: CandleContext, direction: str) -> np.ndarray:
    return ctx.gap_up if direction == "up" else ctx.gap_down


def body_above_mid(ctx: CandleContext, prev_index: int = 1) -> np.ndarray:
    prev_o, prev_c = ctx.prev(ctx.o, prev_index), ctx.prev(ctx.c, prev_index)
    mid = (prev_o + prev_c) / 2.0
    return ctx.c > mid


def body_below_mid(ctx: CandleContext, prev_index: int = 1) -> np.ndarray:
    prev_o, prev_c = ctx.prev(ctx.o, prev_index), ctx.prev(ctx.c, prev_index)
    mid = (prev_o + prev_c) / 2.0
    return ctx.c < mid


def consecutive(ctx: CandleContext, mask: np.ndarray, count: int) -> np.ndarray:
    """True when ``mask`` holds for the current bar and the ``count-1`` before it."""
    out = np.asarray(mask, dtype=bool).copy()
    for k in range(1, count):
        out &= ctx.prev_mask(mask, k)
    return out


def higher_highs_lows(ctx: CandleContext, count: int) -> np.ndarray:
    out = np.ones(ctx.n, dtype=bool)
    for k in range(1, count):
        out &= ctx.h > ctx.prev(ctx.h, k)
        out &= ctx.l > ctx.prev(ctx.l, k)
    return out


def lower_highs_lows(ctx: CandleContext, count: int) -> np.ndarray:
    out = np.ones(ctx.n, dtype=bool)
    for k in range(1, count):
        out &= ctx.h < ctx.prev(ctx.h, k)
        out &= ctx.l < ctx.prev(ctx.l, k)
    return out


def inside_bar(ctx: CandleContext) -> np.ndarray:
    return (ctx.h <= ctx.prev(ctx.h)) & (ctx.l >= ctx.prev(ctx.l))


def outside_bar(ctx: CandleContext) -> np.ndarray:
    return (ctx.h >= ctx.prev(ctx.h)) & (ctx.l <= ctx.prev(ctx.l))
