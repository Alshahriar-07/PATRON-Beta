from __future__ import annotations

import pandas as pd
import pytest

from patron.data.validate import clean_candles, validate_candles
from tests.conftest import make_candles


def _validate(df):
    return validate_candles(df, "BTC/USDT", "15m")


def test_valid_frame_reports_no_issues(candles):
    report = _validate(candles)
    assert report.valid
    assert report.gap_count == 0
    assert report.invalid_ohlc == 0
    assert report.rows == len(candles)


def test_gap_is_detected_and_flagged_not_filled(candles):
    with_gap = candles.drop(index=[100, 101]).reset_index(drop=True)
    report = _validate(with_gap)
    assert report.missing_bars == 2
    cleaned, _ = clean_candles(with_gap, report)
    assert len(cleaned) == len(with_gap)  # no synthetic candles inserted
    assert int(cleaned["is_gap"].sum()) >= 1
    assert (cleaned.loc[cleaned["is_gap"] == 1, "bars_since_prev"] == 3).any()


def test_invalid_ohlc_rows_are_rejected(candles):
    bad = candles.copy()
    bad.loc[10, "high"] = bad.loc[10, "low"] * 0.5
    report = _validate(bad)
    assert not report.valid
    assert report.invalid_ohlc >= 1
    cleaned, _ = clean_candles(bad, report)
    assert len(cleaned) == len(candles) - 1
    assert (cleaned["high"] > 0).all()


def test_duplicates_are_removed(candles):
    duplicated = pd.concat([candles, candles.iloc[[5]]], ignore_index=True)
    report = _validate(duplicated)
    assert report.duplicate_timestamps == 1
    cleaned, _ = clean_candles(duplicated, report)
    assert cleaned["timestamp_open"].is_unique
    assert len(cleaned) == len(candles)


def test_unsorted_input_is_sorted(candles):
    shuffled = candles.sample(frac=1.0, random_state=3).reset_index(drop=True)
    cleaned, _ = clean_candles(shuffled, _validate(shuffled))
    assert cleaned["timestamp_open"].is_monotonic_increasing


def test_missing_expected_bars_counted():
    ts = pd.date_range("2026-01-01", periods=10, freq="15min", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp_open": ts,
            "timestamp_close": ts + pd.Timedelta(minutes=15),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1.0,
            "quote_volume": 100.5,
            "trade_count": 1,
        }
    ).drop(index=[4, 5])
    report = _validate(frame)
    assert report.missing_bars == 2
    assert report.gap_count == 1
    assert report.rows == 8


def test_unclosed_candle_rejected():
    frame = make_candles(50, start="2026-01-01T00:00:00Z")
    now = frame["timestamp_close"].iloc[10].to_pydatetime()
    report = validate_candles(frame, "BTC/USDT", "15m", now=now)
    assert report.unclosed_candles == 39
    assert not report.valid
