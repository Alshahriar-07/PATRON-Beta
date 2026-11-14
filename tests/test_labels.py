"""Label horizon, first-touch and leakage tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from patron.labels.barrier import AMBIGUOUS, SL_FIRST, TIMEOUT, TP_FIRST, triple_barrier_labels
from patron.labels.momentum import momentum_labels
from patron.labels.range_labels import forward_range_targets


def _flat(close=100.0, n=20, high=None, low=None):
    close = np.full(n, close, dtype=float)
    high = np.full(n, close + 0.8) if high is None else high
    low = np.full(n, close - 0.8) if low is None else low
    return close, high, low


def test_tp_first_detected_on_the_correct_step():
    close, high, low = _flat(n=20)
    high = high.copy()
    high[11] = 103.0  # row 5's upper barrier (100 + 1.5*1)
    atr = np.ones(20)
    res = triple_barrier_labels(close, high, low, atr, horizon=6)
    assert res.label[5] == TP_FIRST
    assert res.tp_step[5] == 6
    assert res.mask[5]


def test_sl_first_detected():
    close, high, low = _flat(n=20)
    low = low.copy()
    low[9] = 98.0  # row 5's lower barrier (100 - 1.0*1)
    res = triple_barrier_labels(close, high, low, np.ones(20), horizon=6)
    assert res.label[5] == SL_FIRST
    assert res.sl_step[5] == 4


def test_timeout_when_no_barrier_touched():
    close, high, low = _flat(n=20)
    res = triple_barrier_labels(close, high, low, np.ones(20), horizon=6)
    assert res.label[0] == TIMEOUT
    assert res.mask[0]


def test_same_candle_double_touch_is_ambiguous_and_masked():
    close, high, low = _flat(n=20)
    high = high.copy()
    low = low.copy()
    high[8] = 110.0
    low[8] = 90.0
    res = triple_barrier_labels(close, high, low, np.ones(20), horizon=6)
    assert res.ambiguous[3]
    assert res.label[3] == AMBIGUOUS
    assert not res.mask[3]


def test_conservative_rule_labels_ambiguous_as_sl_first():
    close, high, low = _flat(n=20)
    high[8] = 110.0
    low[8] = 90.0
    res = triple_barrier_labels(close, high, low, np.ones(20), horizon=6, conservative_ambiguous=True)
    assert res.label[3] == SL_FIRST
    assert res.mask[3]


def test_tail_rows_have_no_label():
    close, high, low = _flat(n=20)
    res = triple_barrier_labels(close, high, low, np.ones(20), horizon=6)
    assert not res.mask[-6:].any()
    assert (res.label[-6:] == AMBIGUOUS).all()


def test_labels_ignore_candles_beyond_the_horizon():
    close, high, low = _flat(n=20)
    rng = np.random.default_rng(0)
    base = triple_barrier_labels(close, high, low, np.ones(20), horizon=6)
    far = high.copy()
    far[:] = high
    far[15:] = 500.0  # only rows >= i+10 for i < 5
    altered = triple_barrier_labels(close, far, low, np.ones(20), horizon=6)
    assert np.array_equal(base.label[:9], altered.label[:9])
    assert np.array_equal(base.mask[:9], altered.mask[:9])


def test_barrier_uses_atr_scaling():
    close, high, low = _flat(n=20)
    atr = np.full(20, 2.0)
    res = triple_barrier_labels(close, high, low, atr, horizon=6)
    assert res.upper_barrier[0] == pytest.approx(103.0)
    assert res.lower_barrier[0] == pytest.approx(98.0)


def test_range_targets_are_relative_and_forward_only():
    close = np.arange(1, 21, dtype=float)
    high = close + 0.5
    low = close - 0.5
    targets = forward_range_targets(close, high, low, horizon=6)
    assert targets.shape == (20, 6, 3)
    assert targets[0, 0, 2] == pytest.approx(close[1] / close[0] - 1)
    assert targets[0, 5, 2] == pytest.approx(close[6] / close[0] - 1)
    assert np.isnan(targets[-6:, -1, 2]).all()
    assert targets[0, 0, 0] >= targets[0, 0, 2]


def test_momentum_classes_match_sign_and_magnitude():
    close = np.full(20, 100.0)
    atr = np.full(20, 1.0)  # atr_pct = 1%
    close[6] = 104.0  # +4% over 6 bars -> +4 ATR -> strong bullish
    close[7] = 97.0  # -3% -> strong bearish for row 1
    cls, norm = momentum_labels(close, atr, horizon=6)
    assert cls[0] == 0
    assert norm[0] == pytest.approx(4.0)
    assert cls[1] == 4
    assert cls[-6] == -1
    assert np.isnan(norm[-6])
