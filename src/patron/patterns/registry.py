"""Formal candlestick pattern registry (100+ patterns).

Each entry carries a name, direction, required candle count, the mathematical
definition, the detection conditions actually executed, an optional trend/context
requirement, a static base confidence (a documented heuristic prior — *not* a
fitted probability) and a literature reference.

Detection is causal: the detector receives only bars up to and including the
signal bar. Confidence is refined per occurrence by the pattern-dataset builder
using relative body size and trend alignment.

References (definitions cross-checked against at least two of):
  * Nison, S. (1991) *Japanese Candlestick Charting Techniques*.
  * Bulkowski, T. (2005) *Encyclopedia of Candlestick Charts*.
  * Caginalp & Laurent (1998), "The predictive power of price patterns".
  * A. K. (2019), "A formal approach to candlestick pattern classification in
    financial time series" — used for the pattern-classification framing.
  * TA-Lib / TA-Lib-compatible `CDL*` function documentation for CDL semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from patron.patterns.detectors import (
    CandleContext,
    consecutive,
    doji,
    engulfing,
    gap,
    hammer_shape,
    harami,
    higher_highs_lows,
    inside_bar,
    long_body,
    lower_highs_lows,
    outside_bar,
    shooting_shape,
    small_body,
    strong_body,
)

Detector = Callable[[CandleContext], np.ndarray]


@dataclass(frozen=True)
class PatternDefinition:
    key: str
    name: str
    direction: str  # bullish | bearish | neutral
    candles: int
    formula: str
    conditions: str
    context_requirement: str
    base_confidence: float
    reference: str
    detect: Detector = field(compare=False, repr=False)
    category: str = "candlestick"

    def __post_init__(self) -> None:
        if self.direction not in ("bullish", "bearish", "neutral"):
            raise ValueError(f"{self.key}: bad direction {self.direction!r}")
        if self.candles < 1:
            raise ValueError(f"{self.key}: candles must be >= 1")


def _empty(ctx: CandleContext) -> np.ndarray:
    return np.zeros(ctx.n, dtype=bool)


# --- shared building blocks -------------------------------------------------


def _marubozu(ctx: CandleContext, bullish: bool) -> np.ndarray:
    t = ctx.thresholds.marubozu_shadow_ratio
    side = ctx.bull if bullish else ctx.bear
    return side & (ctx.upper_ratio <= t) & (ctx.lower_ratio <= t) & (ctx.body_ratio >= 0.9)


def _spinning_top(ctx: CandleContext) -> np.ndarray:
    t = ctx.thresholds
    return (
        (ctx.body_ratio <= t.spinning_body_ratio)
        & (ctx.upper >= ctx.body)
        & (ctx.lower >= ctx.body)
    )


def _high_wave(ctx: CandleContext) -> np.ndarray:
    t = ctx.thresholds
    return (ctx.rng >= 2.0 * np.nan_to_num(ctx.avg_body, nan=np.inf)) & (
        ctx.body_ratio <= 0.35
    )


def _belt_hold(ctx: CandleContext, bullish: bool) -> np.ndarray:
    side = ctx.bull if bullish else ctx.bear
    if bullish:
        return side & long_body(ctx) & (ctx.o <= ctx.prev(ctx.l) * 1.002)
    return side & long_body(ctx) & (ctx.o >= ctx.prev(ctx.h) * 0.998)


def _doji_variants(ctx: CandleContext, kind: str) -> np.ndarray:
    base = doji(ctx)
    body_eps = ctx.body <= 1e-9
    if kind == "long_legged":
        return base & (ctx.upper_ratio >= 0.3) & (ctx.lower_ratio >= 0.3)
    if kind == "dragonfly":
        return base & (ctx.upper_ratio <= 0.05) & (ctx.lower_ratio >= 0.6)
    if kind == "gravestone":
        return base & (ctx.lower_ratio <= 0.05) & (ctx.upper_ratio >= 0.6)
    if kind == "four_price":
        return base & body_eps & (ctx.upper_ratio <= 0.02) & (ctx.lower_ratio <= 0.02)
    if kind == "rickshaw_man":
        return doji(ctx, 0.05) & (ctx.upper_ratio >= 0.4) & (ctx.lower_ratio >= 0.4)
    raise ValueError(kind)


def _shaven(ctx: CandleContext, bullish: bool, head: bool) -> np.ndarray:
    side = ctx.bull if bullish else ctx.bear
    if head:  # no upper shadow
        return side & (ctx.upper <= 0.01 * ctx.rng) & long_body(ctx)
    return side & (ctx.lower <= 0.01 * ctx.rng) & long_body(ctx)


def _tweezer(ctx: CandleContext, top: bool) -> np.ndarray:
    tol = ctx.thresholds.equal_tolerance
    if top:
        return (
            ctx.prev_mask(ctx.bull)
            & ctx.bear
            & (np.abs(ctx.h / np.maximum(ctx.prev(ctx.h), 1e-12) - 1.0) <= tol)
        )
    return (
        ctx.prev_mask(ctx.bear)
        & ctx.bull
        & (np.abs(ctx.l / np.maximum(ctx.prev(ctx.l), 1e-12) - 1.0) <= tol)
    )


def _kicking(ctx: CandleContext, bullish: bool) -> np.ndarray:
    prev_maru = _marubozu(ctx, not bullish)
    prev_maru = ctx.prev_mask(prev_maru)
    gap_up = ctx.gap_up
    gap_down = ctx.gap_down
    if bullish:
        return prev_maru & _marubozu(ctx, True) & gap_up
    return prev_maru & _marubozu(ctx, False) & gap_down


def _matching(ctx: CandleContext, low: bool) -> np.ndarray:
    tol = ctx.thresholds.equal_tolerance
    if low:
        return (
            ctx.prev_mask(ctx.bear)
            & ctx.bear
            & (np.abs(ctx.l / np.maximum(ctx.prev(ctx.l), 1e-12) - 1.0) <= tol)
        )
    return (
        ctx.prev_mask(ctx.bull)
        & ctx.bull
        & (np.abs(ctx.h / np.maximum(ctx.prev(ctx.h), 1e-12) - 1.0) <= tol)
    )


def _neck(ctx: CandleContext, kind: str) -> np.ndarray:
    prev_c = ctx.prev(ctx.c)
    prev_l = ctx.prev(ctx.l)
    prev_o = ctx.prev(ctx.o)
    # Tolerance in ATR units: a body-relative band is too tight on 15m candles.
    near_low = np.abs(ctx.c - prev_l) <= 0.1 * np.nan_to_num(ctx.atr, nan=np.inf)
    if kind == "on_neck":
        return ctx.prev_mask(ctx.bear) & ctx.bull & near_low
    if kind == "in_neck":
        return ctx.prev_mask(ctx.bear) & ctx.bull & (ctx.c > prev_l) & (ctx.c < prev_c)
    if kind == "thrusting":
        return (
            ctx.prev_mask(ctx.bear)
            & ctx.bull
            & (ctx.c > prev_c)
            & (ctx.c < (prev_o + prev_c) / 2.0)
        )
    raise ValueError(kind)


def _separating(ctx: CandleContext, bullish: bool) -> np.ndarray:
    tol = ctx.thresholds.equal_tolerance
    same_open = np.abs(ctx.o / np.maximum(ctx.prev(ctx.o), 1e-12) - 1.0) <= tol
    if bullish:
        return ctx.prev_mask(ctx.bear) & ctx.bull & same_open & (ctx.c > ctx.prev(ctx.o))
    return ctx.prev_mask(ctx.bull) & ctx.bear & same_open & (ctx.c < ctx.prev(ctx.o))


def _last_engulfing(ctx: CandleContext, top: bool) -> np.ndarray:
    """Three trend candles followed by an engulfing candle (4 candles total)."""
    if top:
        return (
            ctx.prev_mask(ctx.bull, 3)
            & ctx.prev_mask(ctx.bull, 2)
            & ctx.prev_mask(ctx.bull, 1)
            & ctx.prev_mask(engulfing(ctx, "bearish"), 1)
        )
    return (
        ctx.prev_mask(ctx.bear, 3)
        & ctx.prev_mask(ctx.bear, 2)
        & ctx.prev_mask(ctx.bear, 1)
        & ctx.prev_mask(engulfing(ctx, "bullish"), 1)
    )


def _counterattack(ctx: CandleContext, bullish: bool) -> np.ndarray:
    tol = 0.002
    if bullish:
        return (
            ctx.prev_mask(ctx.bear)
            & ctx.bull
            & long_body(ctx)
            & (np.abs(ctx.c / np.maximum(ctx.prev(ctx.c), 1e-12) - 1.0) <= tol)
        )
    return (
        ctx.prev_mask(ctx.bull)
        & ctx.bear
        & long_body(ctx)
        & (np.abs(ctx.c / np.maximum(ctx.prev(ctx.c), 1e-12) - 1.0) <= tol)
    )


def _tasuki(ctx: CandleContext, bullish: bool) -> np.ndarray:
    if bullish:
        return (
            ctx.prev_mask(ctx.bear)
            & ctx.bull
            & gap(ctx, "up")
            & ctx.prev_mask(ctx.bull, 2)
            & (ctx.c > ctx.prev(ctx.c, 2))
        )
    return (
        ctx.prev_mask(ctx.bull)
        & ctx.bear
        & gap(ctx, "down")
        & ctx.prev_mask(ctx.bear, 2)
        & (ctx.c < ctx.prev(ctx.c, 2))
    )


def _star(ctx: CandleContext, bullish: bool, doji_middle: bool) -> np.ndarray:
    t = ctx.thresholds
    first_body = ctx.prev(ctx.body, 2)
    middle_small = ctx.prev(ctx.body) <= t.star_small_body * np.maximum(first_body, 1e-12)
    if doji_middle:
        middle_small = middle_small & ctx.prev_mask(doji(ctx))
    if bullish:
        return (
            ctx.prev_mask(ctx.bear, 2)
            & middle_small
            & ctx.bull
            & (ctx.prev(ctx.body, 2) >= t.star_long_body * ctx.prev(ctx.body, 2))
            & (ctx.c > (ctx.prev(ctx.o, 2) + ctx.prev(ctx.c, 2)) / 2.0)
        )
    return (
        ctx.prev_mask(ctx.bull, 2)
        & middle_small
        & ctx.bear
        & (ctx.c < (ctx.prev(ctx.o, 2) + ctx.prev(ctx.c, 2)) / 2.0)
    )


def _three_soldiers(ctx: CandleContext, bullish: bool) -> np.ndarray:
    side = ctx.bull if bullish else ctx.bear
    three = consecutive(ctx, side, 3) & consecutive(ctx, long_body(ctx), 3)
    higher = higher_highs_lows(ctx, 3) if bullish else lower_highs_lows(ctx, 3)
    opens_inside = (
        (ctx.o >= ctx.prev(ctx.o)) & (ctx.o <= ctx.prev(ctx.c)) if bullish
        else (ctx.o <= ctx.prev(ctx.o)) & (ctx.o >= ctx.prev(ctx.c))
    )
    return three & higher & opens_inside


def _three_inside(ctx: CandleContext, bullish: bool) -> np.ndarray:
    direction = "bullish" if bullish else "bearish"
    inside_previous = ctx.prev_mask(harami(ctx, direction))
    confirmation = ctx.c > ctx.prev(ctx.h) if bullish else ctx.c < ctx.prev(ctx.l)
    return inside_previous & confirmation


def _three_outside(ctx: CandleContext, bullish: bool) -> np.ndarray:
    direction = "bullish" if bullish else "bearish"
    engulf_previous = ctx.prev_mask(engulfing(ctx, direction))
    confirmation = ctx.c > ctx.prev(ctx.h) if bullish else ctx.c < ctx.prev(ctx.l)
    return engulf_previous & confirmation


def _three_line_strike(ctx: CandleContext, bullish: bool) -> np.ndarray:
    """Three trending candles, then a long candle closing beyond all three."""
    four = np.zeros(ctx.n, dtype=bool)
    four[3:] = True
    if bullish:
        return (
            four
            & consecutive(ctx, ctx.bull, 3)
            & ctx.prev_mask(long_body(ctx), 1)
            & ctx.bull
            & long_body(ctx)
            & (ctx.c > ctx.prev(ctx.h, 3))
        )
    return (
        four
        & consecutive(ctx, ctx.bear, 3)
        & ctx.prev_mask(long_body(ctx), 1)
        & ctx.bear
        & long_body(ctx)
        & (ctx.c < ctx.prev(ctx.l, 3))
    )


def _abandoned_baby(ctx: CandleContext, bullish: bool) -> np.ndarray:
    if bullish:
        return (
            ctx.prev_mask(ctx.bear, 2)
            & ctx.prev_mask(doji(ctx))
            & ctx.bull
            & ctx.prev_mask(ctx.gap_down)
            & ctx.gap_up
        )
    return (
        ctx.prev_mask(ctx.bull, 2)
        & ctx.prev_mask(doji(ctx))
        & ctx.bear
        & ctx.prev_mask(ctx.gap_up)
        & ctx.gap_down
    )


def _tri_star(ctx: CandleContext, bullish: bool) -> np.ndarray:
    three_doji = consecutive(ctx, doji(ctx), 3)
    if bullish:
        return three_doji & (ctx.gap_down | ctx.prev_mask(ctx.gap_down))
    return three_doji & (ctx.gap_up | ctx.prev_mask(ctx.gap_up))


def _three_methods(ctx: CandleContext, rising: bool, gap_variant: bool = False) -> np.ndarray:
    side = ctx.bull if rising else ctx.bear
    first_long = ctx.prev_mask(long_body(ctx), 4)
    five = np.zeros(ctx.n, dtype=bool)
    five[4:] = True
    small_middle = (
        ctx.prev_mask(small_body(ctx), 1)
        & ctx.prev_mask(small_body(ctx), 2)
        & ctx.prev_mask(small_body(ctx), 3)
    )
    if gap_variant:
        gap_cond = ctx.prev_mask(ctx.gap_up, 4) if rising else ctx.prev_mask(ctx.gap_down, 4)
    else:
        gap_cond = np.ones(ctx.n, dtype=bool)
    close_beyond = (
        ctx.c > ctx.prev(ctx.h, 4) if rising else ctx.c < ctx.prev(ctx.l, 4)
    )
    return five & first_long & small_middle & side & close_beyond & gap_cond


def _concealing_baby_swallow(ctx: CandleContext) -> np.ndarray:
    t = ctx.thresholds
    return (
        ctx.prev_mask(ctx.bear, 3)
        & ctx.prev_mask(long_body(ctx), 3)
        & ctx.prev_mask(ctx.gap_down, 3)
        & ctx.prev_mask(ctx.bear, 1)
        & ctx.bull
        & small_body(ctx)
        & (ctx.c > ctx.prev(ctx.o))
        & (ctx.prev(ctx.body) <= t.star_small_body * ctx.prev(ctx.body, 3))
    )


def _breakaway(ctx: CandleContext, bullish: bool) -> np.ndarray:
    five = np.zeros(ctx.n, dtype=bool)
    five[4:] = True
    if bullish:
        return (
            five
            & ctx.prev_mask(long_body(ctx), 4)
            & ctx.prev_mask(ctx.gap_down, 4)
            & consecutive(ctx, ctx.bear, 3)
            & ctx.bull
            & long_body(ctx)
            & (ctx.c > ctx.prev(ctx.o, 4))
        )
    return (
        five
        & ctx.prev_mask(long_body(ctx), 4)
        & ctx.prev_mask(ctx.gap_up, 4)
        & consecutive(ctx, ctx.bull, 3)
        & ctx.bear
        & long_body(ctx)
        & (ctx.c < ctx.prev(ctx.o, 4))
    )


def _ladder(ctx: CandleContext, bottom: bool) -> np.ndarray:
    five = np.zeros(ctx.n, dtype=bool)
    five[4:] = True
    if bottom:
        return (
            five
            & ctx.prev_mask(ctx.bear, 4)
            & ctx.prev_mask(ctx.bear, 3)
            & ctx.prev_mask(ctx.bear, 2)
            & ctx.prev_mask(ctx.bear, 1)
            & ctx.bull
            & (ctx.h > ctx.prev(ctx.h))
            & (ctx.l > ctx.prev(ctx.l))
            & (ctx.c > ctx.prev(ctx.o))
        )
    return (
        five
        & ctx.prev_mask(ctx.bull, 4)
        & ctx.prev_mask(ctx.bull, 3)
        & ctx.prev_mask(ctx.bull, 2)
        & ctx.prev_mask(ctx.bull, 1)
        & ctx.bear
        & (ctx.l < ctx.prev(ctx.l))
        & (ctx.h < ctx.prev(ctx.h))
        & (ctx.c < ctx.prev(ctx.o))
    )


def _mat_hold(ctx: CandleContext) -> np.ndarray:
    five = np.zeros(ctx.n, dtype=bool)
    five[4:] = True
    return (
        five
        & ctx.prev_mask(ctx.bear, 4)
        & ctx.prev_mask(long_body(ctx), 4)
        & ctx.prev_mask(ctx.bear, 3)
        & ctx.prev_mask(small_body(ctx), 3)
        & ctx.prev_mask(ctx.bull, 2)
        & ctx.prev_mask(small_body(ctx), 2)
        & ctx.prev_mask(ctx.bear, 1)
        & ctx.prev_mask(small_body(ctx), 1)
        & ctx.bull
        & long_body(ctx)
        & (ctx.c > ctx.prev(ctx.o, 3))
    )


def _tower(ctx: CandleContext, top: bool) -> np.ndarray:
    five = np.zeros(ctx.n, dtype=bool)
    five[4:] = True
    if top:
        return (
            five
            & ctx.prev_mask(ctx.bull, 4)
            & ctx.prev_mask(long_body(ctx), 4)
            & ctx.prev_mask(ctx.bull, 3)
            & ctx.prev_mask(small_body(ctx), 3)
            & ctx.prev_mask(small_body(ctx), 2)
            & ctx.prev_mask(small_body(ctx), 1)
            & ctx.bear
            & (ctx.c < ctx.prev(ctx.o, 4))
        )
    return (
        five
        & ctx.prev_mask(ctx.bear, 4)
        & ctx.prev_mask(long_body(ctx), 4)
        & ctx.prev_mask(ctx.bear, 3)
        & ctx.prev_mask(small_body(ctx), 3)
        & ctx.prev_mask(small_body(ctx), 2)
        & ctx.prev_mask(ctx.bull, 1)
        & ctx.bull
        & (ctx.c > ctx.prev(ctx.o, 4))
    )


def _three_stars_south(ctx: CandleContext) -> np.ndarray:
    four = np.zeros(ctx.n, dtype=bool)
    four[3:] = True
    return (
        four
        & ctx.prev_mask(ctx.bear, 3)
        & ctx.prev_mask(long_body(ctx), 3)
        & ctx.prev_mask(ctx.bear, 2)
        & ctx.prev_mask(ctx.bear, 1)
        & ctx.bear
        & (ctx.prev(ctx.l, 2) < ctx.prev(ctx.l, 3))
        & (ctx.prev(ctx.l, 1) < ctx.prev(ctx.l, 2))
        & (ctx.l < ctx.prev(ctx.l, 1))
        & small_body(ctx)
    )


def _eight_lines(ctx: CandleContext, up: bool) -> np.ndarray:
    rows = np.ones(ctx.n, dtype=bool)
    for k in range(1, 8):
        rows &= ctx.c > ctx.prev(ctx.c, k) if up else ctx.c < ctx.prev(ctx.c, k)
    return consecutive(ctx, ctx.bull if up else ctx.bear, 8) & rows


def _three_gaps(ctx: CandleContext, up: bool) -> np.ndarray:
    if up:
        return consecutive(ctx, ctx.gap_up, 3) & consecutive(ctx, ctx.bull, 3)
    return consecutive(ctx, ctx.gap_down, 3) & consecutive(ctx, ctx.bear, 3)


def _island_reversal(ctx: CandleContext, bullish: bool) -> np.ndarray:
    # gap in the reversal direction, one isolated bar, gap back in the trend direction
    if bullish:
        return (
            ctx.prev_mask(ctx.gap_down, 2)
            & ctx.prev_mask(ctx.bull, 1)
            & ctx.gap_down
            & (ctx.c < ctx.prev(ctx.c))
        )
    return (
        ctx.prev_mask(ctx.gap_up, 2)
        & ctx.prev_mask(ctx.bear, 1)
        & ctx.gap_up
        & (ctx.c > ctx.prev(ctx.c))
    )


def _unique_three_river(ctx: CandleContext) -> np.ndarray:
    return (
        ctx.prev_mask(ctx.bear, 2)
        & ctx.prev_mask(long_body(ctx), 2)
        & ctx.prev_mask(ctx.bear, 1)
        & (ctx.prev(ctx.l, 1) < ctx.prev(ctx.l, 2))
        & ctx.bull
        & small_body(ctx)
        & (ctx.c < ctx.prev(ctx.c, 2))
        & (ctx.h < ctx.prev(ctx.h, 1))
    )


def _stick_sandwich(ctx: CandleContext) -> np.ndarray:
    tol = ctx.thresholds.equal_tolerance
    return (
        ctx.prev_mask(ctx.bear, 2)
        & ctx.prev_mask(ctx.bull, 1)
        & ctx.bear
        & (np.abs(ctx.c / np.maximum(ctx.prev(ctx.c, 2), 1e-12) - 1.0) <= tol)
    )


def _deliberation(ctx: CandleContext) -> np.ndarray:
    return (
        ctx.prev_mask(ctx.bull, 2)
        & ctx.prev_mask(long_body(ctx), 2)
        & ctx.prev_mask(ctx.bull, 1)
        & ctx.prev_mask(long_body(ctx), 1)
        & ctx.bull
        & small_body(ctx)
        & (ctx.c > ctx.prev(ctx.c, 1))
    )


def _two_crows_gap(ctx: CandleContext) -> np.ndarray:
    return (
        ctx.prev_mask(ctx.bull, 2)
        & ctx.prev_mask(long_body(ctx), 2)
        & ctx.prev_mask(ctx.bear, 1)
        & ctx.prev_mask(ctx.gap_up, 1)
        & ctx.bear
        & (ctx.o > ctx.prev(ctx.o))
        & (ctx.c < ctx.prev(ctx.c))
        & (ctx.c > ctx.prev(ctx.c, 2))
    )


def _stalled(ctx: CandleContext) -> np.ndarray:
    return (
        ctx.prev_mask(ctx.bull, 2)
        & ctx.prev_mask(long_body(ctx), 2)
        & ctx.prev_mask(ctx.bull, 1)
        & ctx.bull
        & small_body(ctx)
        & (ctx.h <= ctx.prev(ctx.h))
    )


def _side_by_side(ctx: CandleContext, bullish: bool) -> np.ndarray:
    side = ctx.bull if bullish else ctx.bear
    return (
        ctx.prev_mask(side, 1)
        & side
        & (np.abs(ctx.o / np.maximum(ctx.prev(ctx.o), 1e-12) - 1.0) <= 0.003)
        & (np.abs(ctx.c / np.maximum(ctx.prev(ctx.c), 1e-12) - 1.0) <= 0.003)
    )


def _advance_decline_block(ctx: CandleContext, up: bool) -> np.ndarray:
    if up:
        return (
            ctx.prev_mask(ctx.bull, 2)
            & ctx.prev_mask(long_body(ctx), 2)
            & ctx.prev_mask(ctx.bull, 1)
            & ctx.bull
            & small_body(ctx)
            & (ctx.c > ctx.prev(ctx.c, 1))
        )
    return (
        ctx.prev_mask(ctx.bear, 2)
        & ctx.prev_mask(long_body(ctx), 2)
        & ctx.prev_mask(ctx.bear, 1)
        & ctx.bear
        & small_body(ctx)
        & (ctx.c < ctx.prev(ctx.c, 1))
    )


def _gapping_doji_star(ctx: CandleContext, bullish: bool) -> np.ndarray:
    if bullish:
        return (
            ctx.prev_mask(ctx.bear, 2)
            & ctx.prev_mask(ctx.gap_down, 1)
            & ctx.prev_mask(doji(ctx), 1)
            & ctx.bull
            & (ctx.c > ctx.prev(ctx.c))
        )
    return (
        ctx.prev_mask(ctx.bull, 2)
        & ctx.prev_mask(ctx.gap_up, 1)
        & ctx.prev_mask(doji(ctx), 1)
        & ctx.bear
        & (ctx.c < ctx.prev(ctx.c))
    )


def _long_shadow(ctx: CandleContext, lower: bool) -> np.ndarray:
    if lower:
        return (ctx.lower_ratio >= 0.6) & (ctx.body_ratio <= 0.3)
    return (ctx.upper_ratio >= 0.6) & (ctx.body_ratio <= 0.3)


def _pattern_table() -> list[PatternDefinition]:
    P: list[PatternDefinition] = []

    def add(
        key: str,
        name: str,
        direction: str,
        candles: int,
        detect: Detector,
        formula: str,
        conditions: str,
        context: str = "none",
        confidence: float = 0.5,
        reference: str = "Nison (1991)",
    ) -> None:
        P.append(
            PatternDefinition(
                key=key,
                name=name,
                direction=direction,  # type: ignore[arg-type]
                candles=candles,
                formula=formula,
                conditions=conditions,
                context_requirement=context,
                base_confidence=confidence,
                reference=reference,
                detect=detect,
            )
        )

    # ---- single candle -----------------------------------------------------
    add("doji", "Doji", "neutral", 1, lambda c: doji(c),
        "|C-O| <= 0.10 * (H-L)", "body/range <= 0.10", confidence=0.35)
    add("long_legged_doji", "Long-Legged Doji", "neutral", 1,
        lambda c: _doji_variants(c, "long_legged"),
        "doji and upper/range >= 0.3 and lower/range >= 0.3",
        "doji with both shadows >= 30% of range", confidence=0.4)
    add("dragonfly_doji", "Dragonfly Doji", "bullish", 1,
        lambda c: _doji_variants(c, "dragonfly"),
        "doji, upper/range <= 0.05, lower/range >= 0.6",
        "long lower shadow, no upper shadow", confidence=0.55)
    add("gravestone_doji", "Gravestone Doji", "bearish", 1,
        lambda c: _doji_variants(c, "gravestone"),
        "doji, lower/range <= 0.05, upper/range >= 0.6",
        "long upper shadow, no lower shadow", confidence=0.55)
    add("four_price_doji", "Four-Price Doji", "neutral", 1,
        lambda c: _doji_variants(c, "four_price"),
        "O=H=L=C (within 2% of range)", "body and both shadows ~0", confidence=0.3,
        reference="Bulkowski (2005)")
    add("rickshaw_man", "Rickshaw Man", "neutral", 1,
        lambda c: _doji_variants(c, "rickshaw_man"),
        "doji with body/range <= 0.05 and both shadows >= 40%",
        "mid-range doji in sideways tape", confidence=0.3)
    add("takuri", "Takuri (Dragonfly in downtrend)", "bullish", 1,
        lambda c: _doji_variants(c, "dragonfly") & c.trend_down,
        "dragonfly doji after a decline", "dragonfly + close<EMA21 + 5-bar return<0",
        "downtrend", 0.55)
    add("hammer", "Hammer", "bullish", 1, lambda c: hammer_shape(c) & c.trend_down,
        "lower >= 2*body, upper <= 0.15*range, body/range <= 0.30",
        "hammer shape in a downtrend", "downtrend", 0.6)
    add("inverted_hammer", "Inverted Hammer", "bullish", 1,
        lambda c: shooting_shape(c) & c.trend_down,
        "upper >= 2*body, lower <= 0.15*range, body/range <= 0.30",
        "inverted hammer after a decline", "downtrend", 0.5)
    add("hanging_man", "Hanging Man", "bearish", 1, lambda c: hammer_shape(c) & c.trend_up,
        "hammer shape in an uptrend", "same geometry as hammer, in uptrend",
        "uptrend", 0.45)
    add("shooting_star", "Shooting Star", "bearish", 1,
        lambda c: shooting_shape(c) & c.trend_up,
        "upper >= 2*body, lower <= 0.15*range, in uptrend", "uptrend", "uptrend", 0.6)
    add("bullish_marubozu", "Bullish Marubozu", "bullish", 1,
        lambda c: _marubozu(c, True),
        "C>O, upper/range<=0.05, lower/range<=0.05, body/range>=0.9",
        "no shadows, full body", confidence=0.6)
    add("bearish_marubozu", "Bearish Marubozu", "bearish", 1,
        lambda c: _marubozu(c, False),
        "C<O, upper/range<=0.05, lower/range<=0.05, body/range>=0.9",
        "no shadows, full body", confidence=0.6)
    add("bullish_opening_marubozu", "Bullish Opening Marubozu", "bullish", 1,
        lambda c: _shaven(c, True, head=True),
        "O=L, C=H, body/range >= 0.60", "shaven head bullish candle", confidence=0.45,
        reference="Bulkowski (2005)")
    add("bearish_opening_marubozu", "Bearish Opening Marubozu", "bearish", 1,
        lambda c: _shaven(c, False, head=False),
        "O=H, C=L, body/range >= 0.60", "shaven bottom bearish candle", confidence=0.45,
        reference="Bulkowski (2005)")
    add("spinning_top", "Spinning Top", "neutral", 1, _spinning_top,
        "body/range <= 0.30 and upper >= body and lower >= body",
        "small body with two shadows", confidence=0.3)
    add("high_wave", "High-Wave Candle", "neutral", 1, _high_wave,
        "range >= 2*avg_body and body/range <= 0.35", "unusually wide range, small body",
        confidence=0.3, reference="Bulkowski (2005)")
    add("bullish_belt_hold", "Bullish Belt Hold", "bullish", 1,
        lambda c: _belt_hold(c, True),
        "long white candle opening at/below prior low", "long body, open near prior low",
        confidence=0.45)
    add("bearish_belt_hold", "Bearish Belt Hold", "bearish", 1,
        lambda c: _belt_hold(c, False),
        "long black candle opening at/above prior high", "long body, open near prior high",
        confidence=0.45)
    add("long_lower_shadow", "Long Lower Shadow", "bullish", 1,
        lambda c: _long_shadow(c, True),
        "lower/range >= 0.6 and body/range <= 0.3", "demand tail", confidence=0.4)
    add("long_upper_shadow", "Long Upper Shadow", "bearish", 1,
        lambda c: _long_shadow(c, False),
        "upper/range >= 0.6 and body/range <= 0.3", "supply tail", confidence=0.4)

    # ---- double candle -----------------------------------------------------
    add("bullish_engulfing", "Bullish Engulfing", "bullish", 2,
        lambda c: engulfing(c, "bullish"),
        "prior bearish; O<=prior C; C>=prior O; body >= prior body",
        "engulfing body after a black candle", "downtrend preferred", 0.65)
    add("bearish_engulfing", "Bearish Engulfing", "bearish", 2,
        lambda c: engulfing(c, "bearish"),
        "prior bullish; O>=prior C; C<=prior O; body >= prior body",
        "engulfing body after a white candle", "uptrend preferred", 0.65)
    add("piercing_line", "Piercing Line", "bullish", 2,
        lambda c: c.prev_mask(c.bear)
        & c.bull
        & (c.o <= c.prev(c.c) * (1.0 + c.thresholds.equal_tolerance))
        & (c.c > (c.prev(c.o) + c.prev(c.c)) / 2.0)
        & (c.c < c.prev(c.o)),
        "open below prior close, close above prior midpoint but below prior open",
        "bullish penetration > 50% of prior body", "downtrend", 0.55)
    add("dark_cloud_cover", "Dark Cloud Cover", "bearish", 2,
        lambda c: c.prev_mask(c.bull)
        & c.bear
        & (c.o >= c.prev(c.c) * (1.0 - c.thresholds.equal_tolerance))
        & (c.c < (c.prev(c.o) + c.prev(c.c)) / 2.0)
        & (c.c > c.prev(c.o)),
        "open above prior close, close below prior midpoint but above prior open",
        "bearish penetration > 50% of prior body", "uptrend", 0.55)
    add("bullish_harami", "Bullish Harami", "bullish", 2, lambda c: harami(c, "bullish"),
        "prior long black candle; current body inside prior body; current white",
        "inside body after decline", "downtrend", 0.5)
    add("bearish_harami", "Bearish Harami", "bearish", 2, lambda c: harami(c, "bearish"),
        "prior long white candle; current body inside prior body; current black",
        "inside body after advance", "uptrend", 0.5)
    add("bullish_harami_cross", "Bullish Harami Cross", "bullish", 2,
        lambda c: harami(c, "bullish") & doji(c),
        "harami where the second candle is a doji", "inside doji after decline",
        "downtrend", 0.55)
    add("bearish_harami_cross", "Bearish Harami Cross", "bearish", 2,
        lambda c: harami(c, "bearish") & doji(c),
        "harami where the second candle is a doji", "inside doji after advance",
        "uptrend", 0.55)
    add("tweezer_top", "Tweezer Top", "bearish", 2, lambda c: _tweezer(c, True),
        "two candles with matching highs (within 8 bps); second bearish",
        "equal highs at resistance", "uptrend", 0.45)
    add("tweezer_bottom", "Tweezer Bottom", "bullish", 2, lambda c: _tweezer(c, False),
        "two candles with matching lows (within 8 bps); second bullish",
        "equal lows at support", "downtrend", 0.45)
    add("kicking_bullish", "Bullish Kicking", "bullish", 2, lambda c: _kicking(c, True),
        "bearish marubozu then gap up into a bullish marubozu", "two opposite marubozus with a gap",
        confidence=0.6, reference="Bulkowski (2005)")
    add("kicking_bearish", "Bearish Kicking", "bearish", 2, lambda c: _kicking(c, False),
        "bullish marubozu then gap down into a bearish marubozu",
        "two opposite marubozus with a gap", confidence=0.6,
        reference="Bulkowski (2005)")
    add("matching_low", "Matching Low", "bullish", 2, lambda c: _matching(c, True),
        "two bearish candles whose lows match", "double bottom at one bar",
        "downtrend", 0.4)
    add("matching_high", "Matching High", "bearish", 2, lambda c: _matching(c, False),
        "two bullish candles whose highs match", "double top at one bar",
        "uptrend", 0.4)
    add("on_neck", "On-Neck Line", "bearish", 2, lambda c: _neck(c, "on_neck"),
        "bullish candle closing at the prior low", "weak bounce", "downtrend", 0.35)
    add("in_neck", "In-Neck Line", "bearish", 2, lambda c: _neck(c, "in_neck"),
        "bullish candle closing slightly above the prior close", "weak bounce",
        "downtrend", 0.35)
    add("thrusting", "Thrusting Line", "bearish", 2, lambda c: _neck(c, "thrusting"),
        "bullish close above prior close but below prior midpoint", "weak bounce",
        "downtrend", 0.35)
    add("separating_lines_bullish", "Bullish Separating Lines", "bullish", 2,
        lambda c: _separating(c, True), "same open, opposite colours, bullish close",
        "open equals prior open", "downtrend", 0.4)
    add("separating_lines_bearish", "Bearish Separating Lines", "bearish", 2,
        lambda c: _separating(c, False), "same open, opposite colours, bearish close",
        "open equals prior open", "uptrend", 0.4)
    add("last_engulfing_top", "Last Engulfing Top", "bearish", 4,
        lambda c: _last_engulfing(c, True),
        "three bullish candles then a bearish bar engulfing the third",
        "engulfing top after three up closes", "uptrend", 0.4,
        reference="Bulkowski (2005)")
    add("last_engulfing_bottom", "Last Engulfing Bottom", "bullish", 4,
        lambda c: _last_engulfing(c, False),
        "three bearish candles then a bullish bar engulfing the third",
        "engulfing bottom after three down closes", "downtrend", 0.4,
        reference="Bulkowski (2005)")
    add("homing_pigeon", "Homing Pigeon", "bullish", 2,
        lambda c: c.prev_mask(c.bear) & c.bear & (c.o < c.prev(c.o)) & (c.c > c.prev(c.c))
        & (np.maximum(c.o, c.c) <= c.prev(c.o)),
        "two bearish candles, second inside first and above prior close",
        "bearish harami variant", "downtrend", 0.4)
    add("descending_hawk", "Descending Hawk", "bearish", 2,
        lambda c: c.prev_mask(c.bull) & c.bull & (c.o > c.prev(c.o)) & (c.c < c.prev(c.c))
        & (np.minimum(c.o, c.c) >= c.prev(c.c)),
        "two bullish candles, second inside first and below prior close",
        "bullish harami variant", "uptrend", 0.4)
    add("rising_window", "Rising Window", "bullish", 2, lambda c: c.gap_up,
        "low(i) > high(i-1)", "upward gap (window)", "uptrend", 0.5)
    add("falling_window", "Falling Window", "bearish", 2, lambda c: c.gap_down,
        "high(i) < low(i-1)", "downward gap (window)", "downtrend", 0.5)
    add("counterattack_bullish", "Bullish Counterattack", "bullish", 2,
        lambda c: _counterattack(c, True),
        "long black then long white closing at the prior close", "equal closes",
        "downtrend", 0.4)
    add("counterattack_bearish", "Bearish Counterattack", "bearish", 2,
        lambda c: _counterattack(c, False),
        "long white then long black closing at the prior close", "equal closes",
        "uptrend", 0.4)
    add("bullish_tasuki_gap", "Bullish Tasuki Gap", "bullish", 3,
        lambda c: _tasuki(c, True),
        "bullish gap then a black candle that does not close the gap",
        "gap continuation", "uptrend", 0.45)
    add("bearish_tasuki_gap", "Bearish Tasuki Gap", "bearish", 3,
        lambda c: _tasuki(c, False),
        "bearish gap then a white candle that does not close the gap",
        "gap continuation", "downtrend", 0.45)
    add("side_by_side_white_bullish", "Bullish Side-by-Side White Lines", "bullish", 2,
        lambda c: _side_by_side(c, True),
        "two white candles opening and closing at the same levels",
        "equal bodies after a gap", "uptrend", 0.4, reference="Bulkowski (2005)")
    add("side_by_side_white_bearish", "Bearish Side-by-Side White Lines", "bearish", 2,
        lambda c: _side_by_side(c, False),
        "two black candles opening and closing at the same levels",
        "equal bodies after a gap", "downtrend", 0.4, reference="Bulkowski (2005)")

    # ---- triple candle -----------------------------------------------------
    add("morning_star", "Morning Star", "bullish", 3, lambda c: _star(c, True, False),
        "long black, small body, long white closing above the first midpoint",
        "three-candle bottom reversal", "downtrend", 0.7)
    add("evening_star", "Evening Star", "bearish", 3, lambda c: _star(c, False, False),
        "long white, small body, long black closing below the first midpoint",
        "three-candle top reversal", "uptrend", 0.7)
    add("morning_doji_star", "Morning Doji Star", "bullish", 3,
        lambda c: _star(c, True, True),
        "morning star whose middle candle is a doji", "doji star", "downtrend", 0.75)
    add("evening_doji_star", "Evening Doji Star", "bearish", 3,
        lambda c: _star(c, False, True),
        "evening star whose middle candle is a doji", "doji star", "uptrend", 0.75)
    add("three_white_soldiers", "Three White Soldiers", "bullish", 3,
        lambda c: _three_soldiers(c, True),
        "three consecutive long white candles with higher highs/lows",
        "each open inside the prior body", "downtrend", 0.65)
    add("three_black_crows", "Three Black Crows", "bearish", 3,
        lambda c: _three_soldiers(c, False),
        "three consecutive long black candles with lower highs/lows",
        "each open inside the prior body", "uptrend", 0.65)
    add("three_inside_up", "Three Inside Up", "bullish", 3,
        lambda c: _three_inside(c, True),
        "bullish harami followed by a close above the prior high", "confirmation candle",
        "downtrend", 0.6)
    add("three_inside_down", "Three Inside Down", "bearish", 3,
        lambda c: _three_inside(c, False),
        "bearish harami followed by a close below the prior low", "confirmation candle",
        "uptrend", 0.6)
    add("three_outside_up", "Three Outside Up", "bullish", 3,
        lambda c: _three_outside(c, True),
        "bullish engulfing followed by a higher close", "confirmation candle",
        "downtrend", 0.65)
    add("three_outside_down", "Three Outside Down", "bearish", 3,
        lambda c: _three_outside(c, False),
        "bearish engulfing followed by a lower close", "confirmation candle",
        "uptrend", 0.65)
    add("three_line_strike_bullish", "Bullish Three-Line Strike", "bullish", 4,
        lambda c: _three_line_strike(c, True),
        "three rising closes then a long white candle closing above them",
        "trend continuation", "uptrend", 0.5)
    add("three_line_strike_bearish", "Bearish Three-Line Strike", "bearish", 4,
        lambda c: _three_line_strike(c, False),
        "three falling closes then a long black candle closing below them",
        "trend continuation", "downtrend", 0.5)
    add("abandoned_baby_bullish", "Bullish Abandoned Baby", "bullish", 3,
        lambda c: _abandoned_baby(c, True),
        "doji gapped below/above both neighbouring bodies, then reversal",
        "island doji", "downtrend", 0.75)
    add("abandoned_baby_bearish", "Bearish Abandoned Baby", "bearish", 3,
        lambda c: _abandoned_baby(c, False),
        "island doji at a top", "gap-isolated doji", "uptrend", 0.75)
    add("tri_star_bullish", "Bullish Tri-Star", "bullish", 3,
        lambda c: _tri_star(c, True),
        "three consecutive dojis with a downward gap", "rare three-doji bottom",
        "downtrend", 0.55, reference="Bulkowski (2005)")
    add("tri_star_bearish", "Bearish Tri-Star", "bearish", 3,
        lambda c: _tri_star(c, False),
        "three consecutive dojis with an upward gap", "rare three-doji top",
        "uptrend", 0.55, reference="Bulkowski (2005)")
    add("unique_three_river_bottom", "Unique Three-River Bottom", "bullish", 3,
        _unique_three_river,
        "long black, lower black with lower low, small white inside the second body",
        "three-river bottom", "downtrend", 0.4, reference="Bulkowski (2005)")
    add("stick_sandwich", "Stick Sandwich", "bullish", 3,
        _stick_sandwich,
        "black candle, white candle, black candle closing at the first close",
        "equal closes sandwich", "downtrend", 0.4)
    add("deliberation", "Deliberation", "bearish", 3, _deliberation,
        "two long white candles then a small white candle at the top",
        "stalling rally", "uptrend", 0.4)
    def _opens_at_prior_close(c: CandleContext) -> np.ndarray:
        mask = c.o >= c.prev(c.c) * 0.999
        mask[0] = False
        return mask

    add("identical_three_crows", "Identical Three Crows", "bearish", 3,
        lambda c: consecutive(c, c.bear, 3) & _opens_at_prior_close(c),
        "three black candles each opening at the prior close", "equal opens",
        "uptrend", 0.45, reference="Bulkowski (2005)")
    add("two_crows", "Two Crows", "bearish", 2,
        lambda c: c.prev_mask(c.bull) & c.bear & c.gap_up,
        "gap up into a bearish candle after a white candle", "gap then reversal",
        "uptrend", 0.4, reference="Bulkowski (2005)")
    add("inside_bar", "Inside Bar", "neutral", 2, inside_bar,
        "H <= prior H and L >= prior L", "contained range", confidence=0.25,
        reference="Bulkowski (2005)")
    add("outside_bar", "Outside Bar", "neutral", 2, outside_bar,
        "H >= prior H and L <= prior L", "engulfing range", confidence=0.25,
        reference="Bulkowski (2005)")
    add("upside_gap_two_crows", "Upside Gap Two Crows", "bearish", 3,
        _two_crows_gap,
        "white candle, gapped-up black candle, second black candle above the first close",
        "failed gap", "uptrend", 0.45)
    add("three_advance_block", "Advance Block", "bearish", 3,
        lambda c: _advance_decline_block(c, True),
        "three white candles with progressively smaller bodies", "weakening rally",
        "uptrend", 0.4)
    add("three_decline_block", "Decline Block", "bullish", 3,
        lambda c: _advance_decline_block(c, False),
        "three black candles with progressively smaller bodies", "weakening decline",
        "downtrend", 0.4)
    add("upside_gap_side_by_side_white", "Upside Gap Side-by-Side White Lines", "bullish", 3,
        lambda c: c.prev_mask(c.gap_up, 1) & _side_by_side(c, True),
        "gap up followed by two equal white bodies", "continuation after gap",
        "uptrend", 0.45, reference="Bulkowski (2005)")
    add("downside_gap_side_by_side_white", "Downside Gap Side-by-Side White Lines", "bearish", 3,
        lambda c: c.prev_mask(c.gap_down, 1) & _side_by_side(c, True),
        "gap down followed by two equal white bodies", "failed continuation",
        "downtrend", 0.45, reference="Bulkowski (2005)")
    add("stalled_pattern", "Stalled Pattern", "bearish", 3, _stalled,
        "two rising long white candles then a small white candle", "loss of momentum",
        "uptrend", 0.35, reference="Bulkowski (2005)")
    add("gapping_doji_star_bullish", "Bullish Gapping Doji Star", "bullish", 3,
        lambda c: _gapping_doji_star(c, True),
        "black candle, gapped-down doji, closing white candle", "island-ish doji star",
        "downtrend", 0.5, reference="Bulkowski (2005)")
    add("gapping_doji_star_bearish", "Bearish Gapping Doji Star", "bearish", 3,
        lambda c: _gapping_doji_star(c, False),
        "white candle, gapped-up doji, closing black candle", "island-ish doji star",
        "uptrend", 0.5, reference="Bulkowski (2005)")

    # ---- four to eight candles --------------------------------------------
    add("rising_three_methods", "Rising Three Methods", "bullish", 5,
        lambda c: _three_methods(c, True),
        "long white, three small pullback candles, long white closing above the first high",
        "five-candle continuation", "uptrend", 0.55)
    add("falling_three_methods", "Falling Three Methods", "bearish", 5,
        lambda c: _three_methods(c, False),
        "long black, three small bounce candles, long black closing below the first low",
        "five-candle continuation", "downtrend", 0.55)
    add("upside_gap_three_methods", "Upside Gap Three Methods", "bullish", 5,
        lambda c: _three_methods(c, True, gap_variant=True),
        "rising three methods preceded by an upward gap", "gap continuation",
        "uptrend", 0.55)
    add("downside_gap_three_methods", "Downside Gap Three Methods", "bearish", 5,
        lambda c: _three_methods(c, False, gap_variant=True),
        "falling three methods preceded by a downward gap", "gap continuation",
        "downtrend", 0.55)
    add("concealing_baby_swallow", "Concealing Baby Swallow", "bullish", 4,
        _concealing_baby_swallow,
        "two black marubozu-like candles, a black candle inside, then a bullish bar",
        "rare four-candle bottom", "downtrend", 0.45, reference="Bulkowski (2005)")
    add("breakaway_bullish", "Bullish Breakaway", "bullish", 5,
        lambda c: _breakaway(c, True),
        "long black with gap down, three declining candles, long white closing above the gap",
        "five-candle reversal", "downtrend", 0.5)
    add("breakaway_bearish", "Bearish Breakaway", "bearish", 5,
        lambda c: _breakaway(c, False),
        "long white with gap up, three rising candles, long black closing below the gap",
        "five-candle reversal", "uptrend", 0.5)
    add("ladder_bottom", "Ladder Bottom", "bullish", 5, lambda c: _ladder(c, True),
        "four black candles with lower lows, then a bullish bar above the prior open",
        "five-candle bottom", "downtrend", 0.45, reference="Bulkowski (2005)")
    add("ladder_top", "Ladder Top", "bearish", 5, lambda c: _ladder(c, False),
        "four white candles with higher highs, then a bearish bar", "five-candle top",
        "uptrend", 0.45, reference="Bulkowski (2005)")
    add("mat_hold", "Mat Hold", "bullish", 5, _mat_hold,
        "long white, small pullback cluster staying above its open, then a higher close",
        "five-candle continuation", "uptrend", 0.5)
    add("tower_top", "Tower Top", "bearish", 5, lambda c: _tower(c, True),
        "long white, small bodies, then a long black candle", "five-candle top",
        "uptrend", 0.5)
    add("tower_bottom", "Tower Bottom", "bullish", 5, lambda c: _tower(c, False),
        "long black, small bodies, then a long white candle", "five-candle bottom",
        "downtrend", 0.5)
    add("three_stars_in_the_south", "Three Stars in the South", "bullish", 4,
        _three_stars_south,
        "three black candles with shrinking ranges and lower lows, then a small body",
        "exhaustion bottom", "downtrend", 0.4, reference="Bulkowski (2005)")
    add("eight_new_price_lines", "Eight New Price Lines", "bullish", 8,
        lambda c: _eight_lines(c, True),
        "eight consecutive higher closes and white candles", "strong trend",
        "uptrend", 0.4, reference="Bulkowski (2005)")
    add("eight_new_price_lines_down", "Eight New Price Lines (down)", "bearish", 8,
        lambda c: _eight_lines(c, False),
        "eight consecutive lower closes and black candles", "strong downtrend",
        "downtrend", 0.4, reference="Bulkowski (2005)")
    add("three_gaps_up", "Three Gaps Up", "bearish", 3, lambda c: _three_gaps(c, True),
        "three consecutive upward gaps", "exhaustion gap sequence", "uptrend", 0.4)
    add("three_gaps_down", "Three Gaps Down", "bullish", 3, lambda c: _three_gaps(c, False),
        "three consecutive downward gaps", "exhaustion gap sequence", "downtrend", 0.4)
    add("island_reversal_bullish", "Bullish Island Reversal", "bullish", 3,
        lambda c: _island_reversal(c, True),
        "gap down, isolated bar, gap down back above", "island bottom", "downtrend", 0.45)
    add("island_reversal_bearish", "Bearish Island Reversal", "bearish", 3,
        lambda c: _island_reversal(c, False),
        "gap up, isolated bar, gap up back below", "island top", "uptrend", 0.45)
    add("three_candle_continuation_reversal", "Three-Candle Continuation/Reversal",
        "neutral", 3,
        lambda c: (
            (consecutive(c, c.bull, 3) | consecutive(c, c.bear, 3))
            | (c.prev_mask(c.bull, 2) & c.bear & (c.c < c.prev(c.o, 2)))
            | (c.prev_mask(c.bear, 2) & c.bull & (c.c > c.prev(c.o, 2)))
        ),
        "three same-direction candles or a decisive third-candle reversal",
        "v1-compatible aggregate tag", confidence=0.3)

    return P


CANDLESTICK_PATTERNS: tuple[PatternDefinition, ...] = tuple(_pattern_table())
PATTERN_BY_KEY: dict[str, PatternDefinition] = {p.key: p for p in CANDLESTICK_PATTERNS}


def detect_pattern(
    pattern: PatternDefinition, ctx: CandleContext
) -> np.ndarray:
    """Run one detector and enforce causality/shape invariants."""
    mask = np.asarray(pattern.detect(ctx), dtype=bool)
    if mask.shape != (ctx.n,):
        raise ValueError(f"{pattern.key}: detector returned shape {mask.shape}, expected ({ctx.n},)")
    if pattern.candles > 1:
        mask[: pattern.candles - 1] = False
    return mask


def detect_all(
    ctx: CandleContext, patterns: tuple[PatternDefinition, ...] | None = None
) -> dict[str, np.ndarray]:
    return {p.key: detect_pattern(p, ctx) for p in (patterns or CANDLESTICK_PATTERNS)}


def registry_summary() -> dict:
    by_direction: dict[str, int] = {}
    for p in CANDLESTICK_PATTERNS:
        by_direction[p.direction] = by_direction.get(p.direction, 0) + 1
    return {
        "patterns": len(CANDLESTICK_PATTERNS),
        "by_direction": by_direction,
        "by_candles": {
            str(k): sum(1 for p in CANDLESTICK_PATTERNS if p.candles == k)
            for k in sorted({p.candles for p in CANDLESTICK_PATTERNS})
        },
    }
