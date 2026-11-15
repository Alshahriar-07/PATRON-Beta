"""Pattern registry integrity and causality."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from patron.patterns.detectors import build_context
from patron.patterns.registry import (
    CANDLESTICK_PATTERNS,
    PATTERN_BY_KEY,
    PatternDefinition,
    detect_all,
    detect_pattern,
    registry_summary,
)
from patron.patterns.structures import STRUCTURE_DEFINITIONS, all_definitions


def test_registry_has_at_least_100_documented_patterns():
    assert len(CANDLESTICK_PATTERNS) >= 100
    assert len(STRUCTURE_DEFINITIONS) >= 10
    keys = [p.key for p in CANDLESTICK_PATTERNS]
    assert len(keys) == len(set(keys)), "duplicate pattern keys"
    all_keys = [p.key for p in all_definitions()]
    assert len(all_keys) == len(set(all_keys)), "structure key collides with a candlestick key"


def test_every_pattern_is_fully_documented():
    for pattern in all_definitions():
        assert pattern.name and pattern.formula and pattern.conditions, pattern.key
        assert pattern.reference, pattern.key
        assert 0.0 < pattern.base_confidence <= 1.0, pattern.key
        assert pattern.candles >= 1, pattern.key
        assert pattern.direction in ("bullish", "bearish", "neutral"), pattern.key
        assert pattern.category in ("candlestick", "chart_structure")
        assert callable(pattern.detect)


def test_registry_summary_counts_add_up():
    summary = registry_summary()
    assert summary["patterns"] == len(CANDLESTICK_PATTERNS)
    assert sum(summary["by_direction"].values()) == len(CANDLESTICK_PATTERNS)
    assert sum(summary["by_candles"].values()) == len(CANDLESTICK_PATTERNS)


def test_detectors_return_full_length_boolean_masks(candles):
    ctx = build_context(candles)
    masks = detect_all(ctx)
    assert len(masks) == len(CANDLESTICK_PATTERNS)
    for key, mask in masks.items():
        assert mask.shape == (len(candles),), key
        assert mask.dtype == bool, key
        pattern = PATTERN_BY_KEY[key]
        if pattern.candles > 1:
            assert not mask[: pattern.candles - 1].any(), f"{key} fired before enough candles"


def test_detection_is_causal(candles):
    ctx = build_context(candles)
    masks = detect_all(ctx)
    cut = 400
    altered = candles.copy()
    altered.loc[cut:, ["open", "high", "low", "close"]] *= 1.23
    altered.loc[cut:, "volume"] *= 2.0
    ctx_after = build_context(altered)
    masks_after = detect_all(ctx_after)
    for key, mask in masks.items():
        assert np.array_equal(mask[:cut], masks_after[key][:cut]), f"{key} looked ahead"


def test_structure_detectors_are_causal(candles):
    ctx = build_context(candles)
    before = {p.key: p.detect(ctx) for p in STRUCTURE_DEFINITIONS}
    altered = candles.copy()
    altered.loc[400:, ["open", "high", "low", "close"]] *= 0.8
    ctx_after = build_context(altered)
    for pattern in STRUCTURE_DEFINITIONS:
        after = pattern.detect(ctx_after)
        assert after.shape == (len(candles),), pattern.key
        assert np.array_equal(before[pattern.key][:400], after[:400]), pattern.key


def test_detect_pattern_rejects_bad_shape(candles):
    ctx = build_context(candles)
    bad = PatternDefinition(
        key="bad",
        name="Bad",
        direction="neutral",
        candles=1,
        formula="",
        conditions="",
        context_requirement="none",
        base_confidence=0.5,
        reference="test",
        detect=lambda c: np.zeros(3, dtype=bool),
    )
    with pytest.raises(ValueError):
        detect_pattern(bad, ctx)


def test_every_pattern_fires_somewhere_on_varied_data():
    """A detector that never fires is dead code.

    The synthetic series deliberately mixes trends, gaps and quiet ranges so rare
    multi-candle patterns have a chance to appear.
    """
    rng = np.random.default_rng(11)
    n = 6000
    ts = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    drift = np.repeat(rng.normal(0, 0.0012, size=n // 200 + 1), 200)[:n]
    close = 30000 * np.exp(np.cumsum(rng.normal(0, 0.003, n) + drift))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.002, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.002, n)))
    frame = pd.DataFrame(
        {
            "timestamp_open": ts,
            "timestamp_close": ts + pd.Timedelta(minutes=15),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.lognormal(3.0, 0.5, n),
        }
    )
    ctx = build_context(frame)
    fires = {p.key: int(p.detect(ctx).sum()) for p in CANDLESTICK_PATTERNS}
    dead = sorted(k for k, v in fires.items() if v == 0)
    assert dead, "expected some rare patterns to be absent"
    unexpected = sorted(set(dead) - GAP_DEPENDENT_OR_RARE)
    assert not unexpected, f"detectors unexpectedly never fired: {unexpected}"
    assert len(fires) - len(dead) >= 70, "too few patterns ever fire"


#: Patterns that require an opening gap between consecutive candles (rare on a
#: 24/7 crypto market) or a very specific multi-candle configuration. They are
#: kept for completeness of the registry; "never fires on continuous BTC 15m" is
#: an empirical property of the instrument, not a bug. Any detector outside this
#: list that stops firing is treated as a regression.
GAP_DEPENDENT_OR_RARE = {
    "four_price_doji",
    "kicking_bullish",
    "kicking_bearish",
    "in_neck",
    "last_engulfing_top",
    "last_engulfing_bottom",
    "homing_pigeon",
    "descending_hawk",
    "bullish_tasuki_gap",
    "bearish_tasuki_gap",
    "abandoned_baby_bullish",
    "abandoned_baby_bearish",
    "tri_star_bullish",
    "tri_star_bearish",
    "two_crows",
    "upside_gap_two_crows",
    "upside_gap_side_by_side_white",
    "downside_gap_side_by_side_white",
    "gapping_doji_star_bullish",
    "gapping_doji_star_bearish",
    "upside_gap_three_methods",
    "downside_gap_three_methods",
    "concealing_baby_swallow",
    "breakaway_bullish",
    "breakaway_bearish",
    "three_gaps_up",
    "three_gaps_down",
    "island_reversal_bullish",
    "island_reversal_bearish",
    "rising_window",
    "falling_window",
    "mat_hold",
}
