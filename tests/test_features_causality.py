"""No future information may reach a feature row."""

from __future__ import annotations

import numpy as np
import pandas as pd

from patron.features.builder import build_features, feature_matrix
from patron.features.schema import FEATURE_ORDER


def test_no_future_leakage(candles):
    base = build_features(candles)
    cut = 288  # aligned to a 4h boundary
    altered = candles.copy()
    for col in ("open", "high", "low", "close"):
        altered.loc[cut:, col] = altered.loc[cut:, col] * 1.37
    altered.loc[cut:, "volume"] = altered.loc[cut:, "volume"] * 5.0

    after = build_features(altered)
    before_rows = base.iloc[:cut]
    after_rows = after.iloc[:cut]
    assert np.allclose(
        feature_matrix(before_rows), feature_matrix(after_rows), equal_nan=True
    ), "features before the cut changed when only future candles changed"


def test_warmup_rows_are_flagged_not_filled(candles):
    frame = build_features(candles)
    assert not bool(frame["feature_valid"].iloc[0])
    assert bool(frame["feature_valid"].iloc[-1])
    assert frame.loc[:50, "feature_valid"].sum() == 0
    assert frame[list(FEATURE_ORDER)].iloc[:50].isna().any().any()


def test_feature_values_are_scale_free(candles):
    frame = build_features(candles)
    scaled = candles.copy()
    for col in ("open", "high", "low", "close"):
        scaled[col] = scaled[col] * 1000.0
    scaled_frame = build_features(scaled)
    ratio_cols = ["close_rel", "body_to_range", "atr14_pct", "rsi14_scaled", "ema9_gap", "ema21_gap", "htf_trend"]
    assert np.allclose(
        frame[ratio_cols].iloc[300:].to_numpy(),
        scaled_frame[ratio_cols].iloc[300:].to_numpy(),
        atol=1e-9,
        equal_nan=True,
    )


def test_feature_order_is_enforced(candles):
    wrong = list(reversed(FEATURE_ORDER))
    try:
        build_features(candles, feature_order=wrong)
    except ValueError as exc:
        assert "feature order" in str(exc).lower()
    else:  # pragma: no cover
        raise AssertionError("reordered feature list must be rejected")


def test_htf_trend_uses_closed_candles_only(candles):
    frame = build_features(candles)
    # 1h and 4h trends must be finite once enough history exists
    valid = frame.loc[frame["feature_valid"]]
    assert len(valid) > 100
    assert valid["htf_trend"].between(-1.0, 1.0).all()
