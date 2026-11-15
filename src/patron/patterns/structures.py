"""Chart structures (a separate category from candlestick patterns).

These are geometric approximations of classical structures — support/resistance,
double tops/bottoms, head-and-shoulders, triangles, flags, pennants, wedges and
channels. They are computed on trailing windows only: a pivot is confirmed by
bars that already closed, never by the bars that follow it. Detection is
deliberately conservative and scale-free (thresholds in ATR units), and the
structures are *labels*, not trading signals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from patron.patterns.detectors import CandleContext
from patron.patterns.registry import PatternDefinition

LEFT_WINDOW = 5
"""Bars used to confirm a trailing pivot (pivot = extreme of the last N bars)."""


def _roll_max(arr: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(arr).rolling(window, min_periods=window).max().to_numpy()


def _roll_min(arr: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(arr).rolling(window, min_periods=window).min().to_numpy()


def _slope(arr: np.ndarray, lookback: int) -> np.ndarray:
    """Least-squares slope over the trailing ``lookback`` bars, in price units/bar."""
    series = pd.Series(arr)
    x = np.arange(lookback, dtype=float)
    x_centered = x - x.mean()
    denom = float((x_centered**2).sum())
    slopes = series.rolling(lookback, min_periods=lookback).apply(
        lambda y: float(np.dot(x_centered, y - y.mean()) / denom), raw=True
    )
    return slopes.to_numpy()


def _swing_high(ctx: CandleContext, window: int = LEFT_WINDOW) -> np.ndarray:
    return ctx.h >= _roll_max(ctx.h, window)


def _swing_low(ctx: CandleContext, window: int = LEFT_WINDOW) -> np.ndarray:
    return ctx.l <= _roll_min(ctx.l, window)


def _atr(ctx: CandleContext) -> np.ndarray:
    return np.where(np.isfinite(ctx.atr), ctx.atr, np.maximum(ctx.rng, 1e-9))


def support_resistance(ctx: CandleContext) -> np.ndarray:
    """Price within 0.5 ATR of the trailing 50-bar extreme."""
    atr = _atr(ctx)
    near_high = np.abs(ctx.c - _roll_max(ctx.h, 50)) <= 0.5 * atr
    near_low = np.abs(ctx.c - _roll_min(ctx.l, 50)) <= 0.5 * atr
    return near_high | near_low


def double_top(ctx: CandleContext) -> np.ndarray:
    atr = _atr(ctx)
    prior_high = pd.Series(ctx.h).shift(LEFT_WINDOW).rolling(35, min_periods=35).max().to_numpy()
    valley = pd.Series(ctx.l).shift(1).rolling(30, min_periods=30).min().to_numpy()
    equal_peaks = np.abs(ctx.h - prior_high) <= 0.25 * atr
    has_valley = valley <= prior_high - 0.8 * atr
    return _swing_high(ctx) & equal_peaks & has_valley


def double_bottom(ctx: CandleContext) -> np.ndarray:
    atr = _atr(ctx)
    prior_low = pd.Series(ctx.l).shift(LEFT_WINDOW).rolling(35, min_periods=35).min().to_numpy()
    peak = pd.Series(ctx.h).shift(1).rolling(30, min_periods=30).max().to_numpy()
    equal_troughs = np.abs(ctx.l - prior_low) <= 0.25 * atr
    has_peak = peak >= prior_low + 0.8 * atr
    return _swing_low(ctx) & equal_troughs & has_peak


def _triple_peak(ctx: CandleContext, inverse: bool) -> np.ndarray:
    atr = _atr(ctx)
    series = ctx.l if inverse else ctx.h
    out = np.zeros(ctx.n, dtype=bool)
    for first in (10, 16, 22, 28):
        for second in (10, 16, 22, 28):
            head = pd.Series(series).shift(first).to_numpy()
            shoulder = pd.Series(series).shift(first + second).to_numpy()
            shoulders_match = np.abs(series - shoulder) <= 0.35 * atr
            head_extends = (
                (head <= series - 0.6 * atr) if inverse else (head >= series + 0.6 * atr)
            )
            ok = np.isfinite(head) & np.isfinite(shoulder) & shoulders_match & head_extends
            out |= ok
    if inverse:
        out &= _swing_low(ctx)
    else:
        out &= _swing_high(ctx)
    return out


def head_and_shoulders(ctx: CandleContext) -> np.ndarray:
    return _triple_peak(ctx, inverse=False)


def inverse_head_and_shoulders(ctx: CandleContext) -> np.ndarray:
    return _triple_peak(ctx, inverse=True)


def _triangle(ctx: CandleContext, kind: str) -> np.ndarray:
    atr = _atr(ctx)
    lookback = 20
    slope_h = _slope(ctx.h, lookback)
    slope_l = _slope(ctx.l, lookback)
    flat = 0.02 * atr  # price units per bar considered "flat"
    rising_h = slope_h > flat
    falling_h = slope_h < -flat
    rising_l = slope_l > flat
    falling_l = slope_l < -flat
    if kind == "ascending":
        return rising_l & ~rising_h & ~falling_h
    if kind == "descending":
        return falling_h & ~rising_l & ~falling_l
    if kind == "symmetrical":
        return falling_h & rising_l
    raise ValueError(kind)


def ascending_triangle(ctx: CandleContext) -> np.ndarray:
    return _triangle(ctx, "ascending")


def descending_triangle(ctx: CandleContext) -> np.ndarray:
    return _triangle(ctx, "descending")


def symmetrical_triangle(ctx: CandleContext) -> np.ndarray:
    return _triangle(ctx, "symmetrical")


def _wedge(ctx: CandleContext, rising: bool) -> np.ndarray:
    atr = _atr(ctx)
    lookback = 20
    slope_h = _slope(ctx.h, lookback)
    slope_l = _slope(ctx.l, lookback)
    converging = (ctx.h - ctx.l) < (pd.Series(ctx.rng).shift(lookback).to_numpy())
    if rising:
        return (slope_h > 0) & (slope_l > slope_h) & converging & (slope_l > 0.02 * atr)
    return (slope_h < 0) & (slope_l < slope_h) & converging & (slope_h < -0.02 * atr)


def rising_wedge(ctx: CandleContext) -> np.ndarray:
    return _wedge(ctx, True)


def falling_wedge(ctx: CandleContext) -> np.ndarray:
    return _wedge(ctx, False)


def _channel(ctx: CandleContext, kind: str) -> np.ndarray:
    atr = _atr(ctx)
    lookback = 20
    slope_h = _slope(ctx.h, lookback)
    slope_l = _slope(ctx.l, lookback)
    flat = 0.02 * atr
    parallel = np.abs(slope_h - slope_l) <= 0.5 * atr
    if kind == "ascending":
        return parallel & (slope_h > flat) & (slope_l > flat)
    if kind == "descending":
        return parallel & (slope_h < -flat) & (slope_l < -flat)
    return parallel & (np.abs(slope_h) <= flat) & (np.abs(slope_l) <= flat)


def ascending_channel(ctx: CandleContext) -> np.ndarray:
    return _channel(ctx, "ascending")


def descending_channel(ctx: CandleContext) -> np.ndarray:
    return _channel(ctx, "descending")


def horizontal_channel(ctx: CandleContext) -> np.ndarray:
    return _channel(ctx, "horizontal")


def _flag(ctx: CandleContext, bullish: bool) -> np.ndarray:
    atr = _atr(ctx)
    pole = (ctx.c - pd.Series(ctx.c).shift(8).to_numpy()) / np.maximum(atr, 1e-9)
    recent = ctx.c - pd.Series(ctx.c).shift(4).to_numpy()
    shallow = np.abs(recent) <= 0.6 * atr
    if bullish:
        return (pole >= 2.0) & shallow & (recent <= 0.2 * atr)
    return (pole <= -2.0) & shallow & (recent >= -0.2 * atr)


def bull_flag(ctx: CandleContext) -> np.ndarray:
    return _flag(ctx, True)


def bear_flag(ctx: CandleContext) -> np.ndarray:
    return _flag(ctx, False)


def pennant(ctx: CandleContext) -> np.ndarray:
    atr = _atr(ctx)
    pole = np.abs((ctx.c - pd.Series(ctx.c).shift(8).to_numpy()) / np.maximum(atr, 1e-9))
    rng_now = pd.Series(ctx.rng).rolling(4, min_periods=4).mean().to_numpy()
    rng_before = pd.Series(ctx.rng).rolling(4, min_periods=4).mean().shift(6).to_numpy()
    return (pole >= 2.0) & (rng_now <= 0.5 * rng_before)


STRUCTURE_DEFINITIONS: tuple[PatternDefinition, ...] = (
    PatternDefinition(
        key="support_resistance",
        name="Support / Resistance Test",
        direction="neutral",
        candles=50,
        formula="|C_t - max(H_{t-49..t})| <= 0.5*ATR or |C_t - min(L_{t-49..t})| <= 0.5*ATR",
        conditions="price retests the trailing 50-bar extreme within half an ATR",
        context_requirement="trailing 50 bars",
        base_confidence=0.35,
        reference="Edwards & Magee (1948), Technical Analysis of Stock Trends",
        detect=support_resistance,
        category="chart_structure",
    ),
    PatternDefinition(
        key="double_top",
        name="Double Top",
        direction="bearish",
        candles=40,
        formula="two swing highs within 0.25*ATR separated by a valley >= 0.8*ATR deep",
        conditions="trailing pivot high matching a prior 35-bar high",
        context_requirement="swing high confirmed by 5 trailing bars",
        base_confidence=0.55,
        reference="Edwards & Magee (1948)",
        detect=double_top,
        category="chart_structure",
    ),
    PatternDefinition(
        key="double_bottom",
        name="Double Bottom",
        direction="bullish",
        candles=40,
        formula="two swing lows within 0.25*ATR separated by a peak >= 0.8*ATR tall",
        conditions="trailing pivot low matching a prior 35-bar low",
        context_requirement="swing low confirmed by 5 trailing bars",
        base_confidence=0.55,
        reference="Edwards & Magee (1948)",
        detect=double_bottom,
        category="chart_structure",
    ),
    PatternDefinition(
        key="head_and_shoulders",
        name="Head and Shoulders",
        direction="bearish",
        candles=60,
        formula="peaks at t, t-a, t-a-b with |shoulder_1 - shoulder_2| <= 0.35*ATR and head >= shoulder + 0.6*ATR",
        conditions="three-peak fit over lags (10,16,22,28)",
        context_requirement="swing high confirmed by 5 trailing bars",
        base_confidence=0.6,
        reference="Edwards & Magee (1948)",
        detect=head_and_shoulders,
        category="chart_structure",
    ),
    PatternDefinition(
        key="inverse_head_and_shoulders",
        name="Inverse Head and Shoulders",
        direction="bullish",
        candles=60,
        formula="troughs at t, t-a, t-a-b with shoulders within 0.35*ATR and head >= 0.6*ATR lower",
        conditions="three-trough fit over lags (10,16,22,28)",
        context_requirement="swing low confirmed by 5 trailing bars",
        base_confidence=0.6,
        reference="Edwards & Magee (1948)",
        detect=inverse_head_and_shoulders,
        category="chart_structure",
    ),
    PatternDefinition(
        key="ascending_triangle",
        name="Ascending Triangle",
        direction="bullish",
        candles=20,
        formula="slope(highs,20) ~ 0 and slope(lows,20) > 0",
        conditions="flat resistance with rising support",
        context_requirement="20-bar regression",
        base_confidence=0.5,
        reference="Edwards & Magee (1948)",
        detect=ascending_triangle,
        category="chart_structure",
    ),
    PatternDefinition(
        key="descending_triangle",
        name="Descending Triangle",
        direction="bearish",
        candles=20,
        formula="slope(lows,20) ~ 0 and slope(highs,20) < 0",
        conditions="flat support with falling resistance",
        context_requirement="20-bar regression",
        base_confidence=0.5,
        reference="Edwards & Magee (1948)",
        detect=descending_triangle,
        category="chart_structure",
    ),
    PatternDefinition(
        key="symmetrical_triangle",
        name="Symmetrical Triangle",
        direction="neutral",
        candles=20,
        formula="slope(highs,20) < 0 and slope(lows,20) > 0",
        conditions="converging boundaries",
        context_requirement="20-bar regression",
        base_confidence=0.45,
        reference="Edwards & Magee (1948)",
        detect=symmetrical_triangle,
        category="chart_structure",
    ),
    PatternDefinition(
        key="bull_flag",
        name="Bull Flag",
        direction="bullish",
        candles=8,
        formula="8-bar return >= 2*ATR followed by a 4-bar drift within +-0.6*ATR",
        conditions="impulse then shallow pause",
        context_requirement="8-bar pole",
        base_confidence=0.5,
        reference="Edwards & Magee (1948)",
        detect=bull_flag,
        category="chart_structure",
    ),
    PatternDefinition(
        key="bear_flag",
        name="Bear Flag",
        direction="bearish",
        candles=8,
        formula="8-bar return <= -2*ATR followed by a 4-bar drift within +-0.6*ATR",
        conditions="down-impulse then shallow pause",
        context_requirement="8-bar pole",
        base_confidence=0.5,
        reference="Edwards & Magee (1948)",
        detect=bear_flag,
        category="chart_structure",
    ),
    PatternDefinition(
        key="pennant",
        name="Pennant",
        direction="neutral",
        candles=8,
        formula="|8-bar return| >= 2*ATR and mean range of last 4 bars <= 50% of the prior 4",
        conditions="sharp move then contraction",
        context_requirement="8-bar pole",
        base_confidence=0.45,
        reference="Edwards & Magee (1948)",
        detect=pennant,
        category="chart_structure",
    ),
    PatternDefinition(
        key="rising_wedge",
        name="Rising Wedge",
        direction="bearish",
        candles=20,
        formula="slope(highs) > 0, slope(lows) > slope(highs), range contracting",
        conditions="converging upward boundaries",
        context_requirement="20-bar regression",
        base_confidence=0.45,
        reference="Edwards & Magee (1948)",
        detect=rising_wedge,
        category="chart_structure",
    ),
    PatternDefinition(
        key="falling_wedge",
        name="Falling Wedge",
        direction="bullish",
        candles=20,
        formula="slope(highs) < 0, slope(lows) < slope(highs), range contracting",
        conditions="converging downward boundaries",
        context_requirement="20-bar regression",
        base_confidence=0.45,
        reference="Edwards & Magee (1948)",
        detect=falling_wedge,
        category="chart_structure",
    ),
    PatternDefinition(
        key="ascending_channel",
        name="Ascending Channel",
        direction="bullish",
        candles=20,
        formula="parallel slopes, both > 0",
        conditions="parallel rising boundaries",
        context_requirement="20-bar regression",
        base_confidence=0.4,
        reference="Edwards & Magee (1948)",
        detect=ascending_channel,
        category="chart_structure",
    ),
    PatternDefinition(
        key="descending_channel",
        name="Descending Channel",
        direction="bearish",
        candles=20,
        formula="parallel slopes, both < 0",
        conditions="parallel falling boundaries",
        context_requirement="20-bar regression",
        base_confidence=0.4,
        reference="Edwards & Magee (1948)",
        detect=descending_channel,
        category="chart_structure",
    ),
    PatternDefinition(
        key="horizontal_channel",
        name="Horizontal Channel",
        direction="neutral",
        candles=20,
        formula="|slope(highs)| ~ 0 and |slope(lows)| ~ 0",
        conditions="range-bound tape",
        context_requirement="20-bar regression",
        base_confidence=0.35,
        reference="Edwards & Magee (1948)",
        detect=horizontal_channel,
        category="chart_structure",
    ),
)

STRUCTURE_BY_KEY: dict[str, PatternDefinition] = {p.key: p for p in STRUCTURE_DEFINITIONS}


def all_definitions() -> tuple[PatternDefinition, ...]:
    from patron.patterns.registry import CANDLESTICK_PATTERNS

    return tuple(CANDLESTICK_PATTERNS) + STRUCTURE_DEFINITIONS
