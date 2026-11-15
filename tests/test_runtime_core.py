"""Unit tests for the runtime building blocks: store, validation, window, horizons."""

from __future__ import annotations

import json
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from patron.runtime.collector import SourceError, StaticSource
from patron.runtime.config import RuntimeConfig, load_runtime_config
from patron.runtime.maturation import evaluate_barrier, pinball_loss
from patron.runtime.store import RuntimeStore, utc_iso
from patron.runtime.validation import validate_candles
from patron.runtime.window import build_window, target_timestamps

CANDLE_COLUMNS = ["timestamp_open", "timestamp_close", "open", "high", "low", "close", "volume"]


@pytest.fixture()
def runtime_cfg(tmp_path):
    cfg = load_runtime_config()
    cfg.storage.database = str(tmp_path / "patron.db")
    cfg.storage.output_dir = str(tmp_path / "live")
    cfg.storage.forecast_ledger = str(tmp_path / "forecasts")
    cfg.source.confirm_delay_seconds = 5
    cfg.learning.enabled = False
    return cfg


@pytest.fixture()
def store(runtime_cfg):
    return RuntimeStore(runtime_cfg.storage.database)


def _insert(store, frame, symbol="BTC/USDT", timeframe="15m"):
    return store.upsert_candles(frame[list(CANDLE_COLUMNS)].copy(), symbol, timeframe, source="test")


# --------------------------------------------------------------------- storage
def test_store_keeps_one_candle_per_open_and_records_revisions(store, candles):
    frame = candles[list(CANDLE_COLUMNS)].tail(10).reset_index(drop=True)
    first = _insert(store, frame)
    assert first == {"inserted": 10, "revised": 0, "unchanged": 0}

    revised = frame.copy()
    revised.loc[5, "close"] = revised.loc[5, "close"] * 1.001
    result = _insert(store, revised)
    assert result == {"inserted": 0, "revised": 1, "unchanged": 9}
    assert store.candle_count("BTC/USDT", "15m") == 10
    assert store.events(kind="candle_revised")


def test_forecast_insert_is_unique_per_candle(tmp_path, store, runtime_cfg):
    record = {
        "forecast_id": "f-1",
        "created_at": utc_iso(pd.Timestamp("2026-01-01T00:00:00Z")),
        "as_of": "2026-01-01T00:15:00Z",
        "symbol": "BTC/USDT",
        "timeframe": "15m",
        "candle_ts": "2026-01-01T00:00:00Z",
        "input_start": "2026-01-01T00:00:00Z",
        "input_end": "2026-01-01T00:15:00Z",
        "input_hash": "a" * 64,
        "model_name": "patron",
        "model_version": "v-test",
        "checkpoint": "x",
        "checkpoint_sha": "b" * 64,
        "feature_version": "features-1.0.0",
        "calibration_version": None,
        "reference_close": 100.0,
        "reference_atr": 1.0,
        "source": "test",
        "payload": json.dumps({"targets": []}),
        "model_output": "{}",
    }
    assert store.insert_forecast(record) is True
    duplicate = dict(record, forecast_id="f-2")
    assert store.insert_forecast(duplicate) is False
    assert len(store.recent_forecasts(10)) == 1
    assert store.forecast("f-1")["candle_ts"] == "2026-01-01T00:00:00Z"


def test_evaluation_insert_is_idempotent(store):
    target = "2026-01-01T00:15:00Z"
    assert store.record_evaluation("f-1", 1, target, "EVALUATED", {"close": 1.0}, {"pinball_mean": 0.1})
    assert not store.record_evaluation("f-1", 1, target, "EVALUATED", {"close": 2.0}, {"pinball_mean": 0.2})
    rows = store.evaluations("f-1")
    assert len(rows) == 1 and json.loads(rows[0]["metrics"])["pinball_mean"] == 0.1


def test_champion_swap_is_atomic_and_rollback_restores(store):
    store.register_model("v1", "patron", "p1", "s1", "champion", {"val_loss": 1.5})
    store.register_model("v2", "patron", "p2", "s2", "candidate", {"val_loss": 1.4})
    assert store.set_champion("v2") == {"version": "v2", "previous": "v1"}
    assert store.champion()["version"] == "v2"
    assert store.model("v1")["role"] == "retired"
    assert store.rollback()["version"] == "v1"
    assert store.champion()["version"] == "v1"


# ------------------------------------------------------------------ validation
def test_forming_candle_is_dropped(candles):
    frame = candles[list(CANDLE_COLUMNS)].tail(5).reset_index(drop=True)
    now = frame["timestamp_close"].iloc[-1] - pd.Timedelta(minutes=1)
    result = validate_candles(frame, "15m", now=now, confirm_delay_seconds=0)
    assert result.unclosed == 1
    assert len(result.frame) == 4
    assert result.frame["timestamp_close"].iloc[-1] <= now


def test_confirmation_delay_is_respected(candles):
    frame = candles[list(CANDLE_COLUMNS)].tail(3).reset_index(drop=True)
    close = frame["timestamp_close"].iloc[-1]
    inside = validate_candles(frame, "15m", now=close + pd.Timedelta(seconds=2), confirm_delay_seconds=20)
    assert inside.unclosed == 1
    after = validate_candles(frame, "15m", now=close + pd.Timedelta(seconds=21), confirm_delay_seconds=20)
    assert after.unclosed == 0


def test_duplicate_and_invalid_rows_are_handled_not_filled(candles):
    frame = candles[list(CANDLE_COLUMNS)].tail(6).reset_index(drop=True)
    frame = pd.concat([frame, frame.tail(1)], ignore_index=True)
    frame.loc[2, "low"] = frame.loc[2, "high"] * 2  # low above high
    now = frame["timestamp_close"].max() + pd.Timedelta(minutes=1)
    result = validate_candles(frame, "15m", now=now)
    assert result.duplicates == 1
    assert result.invalid_rows == 1
    assert result.status == "OK"  # newest candle is fine
    assert len(result.frame) == 5


def test_invalid_newest_candle_blocks_the_cycle(candles):
    frame = candles[list(CANDLE_COLUMNS)].tail(4).reset_index(drop=True)
    frame.loc[len(frame) - 1, "high"] = frame.loc[len(frame) - 1, "low"] * 0.5
    now = frame["timestamp_close"].max() + pd.Timedelta(minutes=1)
    result = validate_candles(frame, "15m", now=now)
    assert result.status == "INVALID_INPUT"
    assert not result.ok


def test_empty_source_batch_is_stale():
    result = validate_candles(pd.DataFrame(columns=CANDLE_COLUMNS), "15m")
    assert result.status == "STALE_DATA"


# ---------------------------------------------------------------------- window
def test_window_needs_history_then_builds_96_contiguous(store, candles):
    frame = candles[list(CANDLE_COLUMNS)].copy()
    assert build_window(store, "BTC/USDT", "15m", 96, 384).status == "INSUFFICIENT_HISTORY"
    _insert(store, frame.tail(500))
    window = build_window(store, "BTC/USDT", "15m", 96, 384)
    assert window.status == "OK"
    assert len(window.frame) == 96
    assert len(window.warmup) >= 192
    assert window.input_hash and len(window.input_hash) == 64
    # The warm-up ends exactly one bar before the model window starts.
    assert window.warmup["timestamp_close"].iloc[-1] == window.frame["timestamp_open"].iloc[0]


def test_window_rejects_gap_inside_sequence(store, candles):
    frame = candles[list(CANDLE_COLUMNS)].tail(500).reset_index(drop=True)
    missing = frame.drop(index=len(frame) - 10)
    _insert(store, missing)
    window = build_window(store, "BTC/USDT", "15m", 96, 384)
    assert window.status == "DATA_GAP"
    assert "gap" in (window.reason or "")


def test_target_timestamps_are_exact(candles):
    candle_open = pd.Timestamp(candles["timestamp_open"].iloc[100])
    targets = target_timestamps(candle_open, "15m", 6)
    assert [t["minutes_ahead"] for t in targets] == [15, 30, 45, 60, 75, 90]
    assert targets[0]["target_ts"] == utc_iso(candle_open + timedelta(minutes=15))
    assert targets[-1]["target_ts"] == utc_iso(candle_open + timedelta(minutes=90))


# ------------------------------------------------------------------- maturation
def test_pinball_loss_matches_definition():
    assert pinball_loss(0.02, 0.0, 0.9) == pytest.approx(0.018)
    assert pinball_loss(-0.02, 0.0, 0.1) == pytest.approx(0.018)
    assert pinball_loss(0.01, 0.01, 0.5) == pytest.approx(0.0)


def test_barrier_needs_all_six_future_candles(cfg):
    future = [
        {"high": 101, "low": 99, "close": 100} for _ in range(5)
    ]
    result = evaluate_barrier(100.0, 1.0, future, cfg)
    assert result["label"] is None and result["reason"] == "immature"

    rising = [{"high": 103, "low": 99.5, "close": 102}] + [
        {"high": 103, "low": 101, "close": 102} for _ in range(5)
    ]
    result = evaluate_barrier(100.0, 1.0, rising, cfg)
    assert result["label"] == 0  # tp_first with 1.5*ATR = 101.5


def test_static_source_failure_is_explicit(candles):
    source = StaticSource(frame=candles[list(CANDLE_COLUMNS)], fail_times=1)
    with pytest.raises(SourceError):
        source.fetch("BTC/USDT", "15m", pd.Timestamp("2026-01-01T00:00:00Z"),
                     pd.Timestamp("2026-01-02T00:00:00Z"))


def test_runtime_config_rejects_inverted_coverage(tmp_path):
    payload = load_runtime_config().to_dict()
    payload["monitoring"]["coverage_bounds"] = [0.99, 0.6]
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        RuntimeConfig.model_validate(payload)


def test_store_round_trip_survives_reopen(runtime_cfg, store, candles):
    _insert(store, candles[list(CANDLE_COLUMNS)].tail(20))
    store.mark_cycle("k", "2026-01-01T00:00:00Z", "OK", "done", 1.0)
    reopened = RuntimeStore(runtime_cfg.storage.database)
    assert reopened.candle_count("BTC/USDT", "15m") == 20
    assert reopened.last_cycle()["status"] == "OK"
