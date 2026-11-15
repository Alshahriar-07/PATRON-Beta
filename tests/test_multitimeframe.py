"""Multi-timeframe resampling and HTF alignment must be causal."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from patron.data.timeframes import (
    align_htf_causal,
    resample_ohlcv,
    timeframe_alignment_check,
    timeframe_ratio,
)


def test_timeframe_ratio_validation():
    assert timeframe_ratio("1m", "1d") == 1440
    assert timeframe_ratio("15m", "1h") == 4
    with pytest.raises(ValueError):
        timeframe_ratio("1h", "15m")
    with pytest.raises(KeyError):
        timeframe_ratio("15m", "7m")


def test_resample_matches_manual_aggregation(candles):
    hourly = resample_ohlcv(candles, "15m", "1h")
    assert len(hourly) > 0
    assert set(["timestamp_open", "timestamp_close", "open", "high", "low", "close", "volume"]).issubset(hourly.columns)
    first_bucket = candles.iloc[:4]
    row = hourly.iloc[0]
    assert row["open"] == pytest.approx(first_bucket["open"].iloc[0])
    assert row["close"] == pytest.approx(first_bucket["close"].iloc[-1])
    assert row["high"] == pytest.approx(first_bucket["high"].max())
    assert row["low"] == pytest.approx(first_bucket["low"].min())
    assert row["volume"] == pytest.approx(first_bucket["volume"].sum())
    assert (hourly["timestamp_close"] - hourly["timestamp_open"]).eq(pd.Timedelta(hours=1)).all()


def test_incomplete_bucket_is_dropped(candles):
    truncated = candles.iloc[:97]  # 96 complete 1h buckets + 1 stray 15m bar
    hourly = resample_ohlcv(truncated, "15m", "1h")
    assert len(hourly) == 24
    assert hourly["timestamp_close"].iloc[-1] == truncated["timestamp_close"].iloc[-1] - pd.Timedelta(minutes=15)


def test_resample_has_no_lookahead(candles):
    cut = 192  # two complete 1h buckets
    altered = candles.copy()
    altered.loc[cut:, ["open", "high", "low", "close"]] *= 1.5
    altered.loc[cut:, "volume"] *= 3.0
    base = resample_ohlcv(candles, "15m", "1h")
    after = resample_ohlcv(altered, "15m", "1h")
    pd.testing.assert_frame_equal(
        base.iloc[: cut // 4].reset_index(drop=True),
        after.iloc[: cut // 4].reset_index(drop=True),
        check_dtype=False,
    )


def test_htf_alignment_uses_only_closed_candles(candles):
    hourly = resample_ohlcv(candles, "15m", "1h")
    base = candles.iloc[:40].copy()
    merged = align_htf_causal(base, hourly, ["close"])
    # The first base bars close before the first 1h candle completes.
    assert np.isnan(merged["close_htf"].iloc[0])
    first_valid = merged["close_htf"].first_valid_index()
    assert first_valid is not None
    assert merged.loc[first_valid, "timestamp_close"] >= hourly["timestamp_close"].iloc[0]
    # Once aligned, the HTF value must equal the last completed hourly close.
    row = merged.loc[first_valid]
    matching = hourly[hourly["timestamp_close"] <= row["timestamp_close"]]
    assert row["close_htf"] == pytest.approx(matching["close"].iloc[-1])


def test_alignment_check_detects_identical_resample(candles):
    hourly = resample_ohlcv(candles, "15m", "1h")
    check = timeframe_alignment_check(candles, hourly, "15m", "1h")
    assert check["identical"] is True
    assert check["matched"] == len(hourly)


def test_resample_to_daily(candles):
    daily = resample_ohlcv(candles, "15m", "1d")
    assert len(daily) == len(candles) // 96
    assert daily["volume"].iloc[0] == pytest.approx(candles.iloc[:96]["volume"].sum())
