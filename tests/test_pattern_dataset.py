"""Pattern occurrence dataset: causality, outcome isolation, no contamination."""

from __future__ import annotations

import numpy as np
import pandas as pd

from patron.patterns.dataset import DEFAULT_HORIZONS, build_pattern_dataset, dataset_stats


def test_dataset_schema_and_outcomes(candles):
    dataset = build_pattern_dataset(
        candles, symbol="BTC/USDT", exchange="binance", timeframe="15m", min_candles=200
    )
    for column in (
        "pattern",
        "timestamp",
        "symbol",
        "exchange",
        "timeframe",
        "direction",
        "strength",
        "regime_trend",
        "regime_vol",
        "direction_hit_6",
    ):
        assert column in dataset.columns, column
    for horizon in DEFAULT_HORIZONS:
        assert f"ret_{horizon}" in dataset.columns
        assert f"atr_ret_{horizon}" in dataset.columns
    assert dataset["strength"].between(0, 1).all()
    assert dataset["direction_hit_6"].dropna().isin([0.0, 1.0]).all()
    assert dataset["timestamp"].max() <= candles["timestamp_open"].max()
    assert dataset["pattern"].nunique() > 10


def test_outcomes_are_stored_separately(candles):
    dataset = build_pattern_dataset(
        candles, symbol="BTC/USDT", exchange="binance", timeframe="15m", min_candles=200
    )
    occurrence_columns = set(dataset.columns) - {
        c for c in dataset.columns if c.startswith(("ret_", "atr_ret_", "max_", "direction_hit"))
    }
    assert "ret_6" not in occurrence_columns
    assert "direction_hit_6" not in occurrence_columns


def test_no_lookahead_in_pattern_detection(candles):
    """Occurrences before a cut must not change when future candles change."""
    cut = 400
    altered = candles.copy()
    altered.loc[cut:, ["open", "high", "low", "close"]] *= 1.3
    altered.loc[cut:, "volume"] *= 4.0
    base = build_pattern_dataset(
        candles, symbol="BTC/USDT", exchange="binance", timeframe="15m", min_candles=200
    )
    after = build_pattern_dataset(
        altered, symbol="BTC/USDT", exchange="binance", timeframe="15m", min_candles=200
    )
    # Outcomes look forward by design, so only occurrences whose full 24-candle
    # outcome window ends before the cut are required to be identical.
    max_horizon = max(DEFAULT_HORIZONS)
    cutoff_ts = candles["timestamp_open"].iloc[cut - max_horizon]
    base_prefix = base[base["timestamp"] < cutoff_ts].reset_index(drop=True)
    after_prefix = after[after["timestamp"] < cutoff_ts].reset_index(drop=True)
    pd.testing.assert_frame_equal(base_prefix, after_prefix, check_dtype=False)


def test_outcomes_match_manual_computation(candles):
    dataset = build_pattern_dataset(
        candles, symbol="BTC/USDT", exchange="binance", timeframe="15m", min_candles=200
    )
    row = dataset[dataset["pattern"] == "doji"].iloc[0]
    idx = int(candles.index[candles["timestamp_open"] == row["timestamp"]][0])
    expected = candles["close"].iloc[idx + 6] / candles["close"].iloc[idx] - 1.0
    assert np.isclose(row["ret_6"], expected)


def test_dataset_stats_are_consistent(candles):
    dataset = build_pattern_dataset(
        candles, symbol="BTC/USDT", exchange="binance", timeframe="15m", min_candles=200
    )
    stats = dataset_stats(dataset)
    assert stats.rows == len(dataset)
    assert stats.patterns == dataset["pattern"].nunique()
    assert stats.per_pattern
