"""Advanced feature groups: causality, documentation and scale invariance."""

from __future__ import annotations

import numpy as np
import pandas as pd

from patron.features.advanced import (
    ALL_FEATURES,
    FEATURE_FORMULAS,
    FEATURE_GROUPS,
    build_advanced_features,
    group_feature_columns,
)


def test_every_feature_has_a_documented_formula():
    assert set(ALL_FEATURES) == set(FEATURE_FORMULAS), "feature/formula mismatch"
    for group, columns in FEATURE_GROUPS.items():
        assert columns, f"empty group {group}"
        assert len(set(columns)) == len(columns), f"duplicate feature in {group}"


def test_features_are_causal(candles):
    base = build_advanced_features(candles)
    cut = 300
    altered = candles.copy()
    altered.loc[cut:, ["open", "high", "low", "close"]] *= 1.4
    altered.loc[cut:, "volume"] *= 3.0
    after = build_advanced_features(altered)
    cols = [c for c in base.columns if c not in ("timestamp_open",)]
    pd.testing.assert_frame_equal(
        base.loc[: cut - 1, cols].reset_index(drop=True),
        after.loc[: cut - 1, cols].reset_index(drop=True),
        check_dtype=False,
    )


def test_warmup_rows_flagged_not_filled(candles):
    frame = build_advanced_features(candles)
    assert not bool(frame["feature_valid"].iloc[0])
    assert bool(frame["feature_valid"].iloc[-1])
    assert frame.loc[frame["feature_valid"], ALL_FEATURES].isna().sum().sum() == 0


def test_group_subset_only_returns_requested_columns(candles):
    frame = build_advanced_features(candles, groups=("price", "volume"))
    expected = set(group_feature_columns(("price", "volume")))
    present = set(frame.columns) - {"timestamp_open", "feature_valid"}
    assert present == expected


def test_unknown_group_rejected(candles):
    try:
        build_advanced_features(candles, groups=("price", "nope"))
    except KeyError as exc:
        assert "nope" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("unknown group must raise")


def test_features_are_scale_free(candles):
    base = build_advanced_features(candles)
    scaled = candles.copy()
    for col in ("open", "high", "low", "close"):
        scaled[col] *= 100.0
    scaled["quote_volume"] = scaled["volume"] * scaled["close"]
    after = build_advanced_features(scaled)
    ratio_features = [
        f
        for f in ALL_FEATURES
        if f
        not in {
            "volume_z_96",
            "rel_volume_96",
            "volume_mom_12",
            "consec_bull",
            "consec_bear",
            "candle_dir",
            "breakout_up_20",
            "breakout_down_20",
            "trend_regime",
            "vol_regime",
            "pattern_conf_bull",
            "pattern_conf_bear",
        }
    ]
    assert np.allclose(
        base[ratio_features].iloc[400:].to_numpy(dtype=float),
        after[ratio_features].iloc[400:].to_numpy(dtype=float),
        atol=1e-8,
        equal_nan=True,
    )
