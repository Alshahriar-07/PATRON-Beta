"""Multi-venue registry, adapters, validation and alignment tests (offline)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from patron.data.align import align_venue_panel, assert_panel_integrity
from patron.data.markets import CANONICAL_COLUMNS, VENUES, get_venue
from patron.data.quality import frame_checksum, validate_venue_frame
from patron.data.venues import canonicalise_external


def _rows(start: str, n: int, step_minutes: int = 15, price: float = 30000.0) -> list[dict]:
    ts = pd.date_range(pd.Timestamp(start), periods=n, freq=f"{step_minutes}min", tz="UTC")
    return [
        {
            "timestamp_open": t,
            "timestamp_close": t + pd.Timedelta(minutes=step_minutes),
            "open": price + i,
            "high": price + i + 50,
            "low": price + i - 50,
            "close": price + i + 10,
            "volume": 10.0 + i,
            "quote_volume": None,
            "trade_count": None,
        }
        for i, t in enumerate(ts)
    ]


def test_venue_symbol_mapping_is_explicit():
    assert get_venue("bybit").venue_symbol("BTC") == "BTCUSDT"
    assert get_venue("okx").venue_symbol("BTC") == "BTC-USDT"
    assert get_venue("coinbase").venue_symbol("BTC") == "BTC-USD"
    assert get_venue("kraken").venue_symbol("BTC") == "XBTUSD"
    assert get_venue("coinbase").quote == "USD"
    assert get_venue("binance").quote == "USDT"
    with pytest.raises(KeyError):
        get_venue("kraken").venue_symbol("BNB")


def test_venue_timeframe_support_is_enforced():
    assert get_venue("binance").supports("1d")
    assert not get_venue("coinbase").supports("4h")
    with pytest.raises(KeyError):
        get_venue("coinbase").interval("4h")


def test_canonicalise_external_normalises_and_dedupes():
    rows = _rows("2026-01-01T00:00:00Z", 4)
    rows.append(dict(rows[-1]))  # duplicate timestamp
    frame = canonicalise_external(rows, get_venue("coinbase"), "BTC", "15m")
    assert list(frame.columns[:9]) == list(CANONICAL_COLUMNS)
    assert len(frame) == 4
    assert frame["timestamp_open"].dt.tz is not None
    assert frame["timestamp_open"].is_monotonic_increasing
    assert frame["exchange"].iloc[0] == "coinbase"
    assert frame["symbol"].iloc[0] == "BTC-USD"
    assert frame["trade_count"].isna().all()  # coinbase does not publish it


def test_quality_report_flags_defects_without_fixing_them():
    frame = canonicalise_external(_rows("2026-01-01T00:00:00Z", 40), get_venue("binance"), "BTC", "15m")
    frame.loc[5, "high"] = frame.loc[5, "low"] - 1.0  # invalid OHLC
    broken = pd.concat([frame, frame.iloc[[10]]], ignore_index=True).sort_values("timestamp_open")
    report = validate_venue_frame(broken, get_venue("binance"), "BTC", "15m")
    assert report.invalid_ohlc >= 1
    assert report.duplicate_timestamps >= 1
    assert not report.valid
    assert report.rows == len(broken)


def test_quality_detects_gaps_and_outliers():
    rows = _rows("2026-01-01T00:00:00Z", 60)
    # Drop a contiguous block to create a genuine 8-bar hole (never filled).
    rows = [r for i, r in enumerate(rows) if not 20 <= i < 28]
    rows[35]["high"] = rows[35]["close"] * 3.0
    frame = canonicalise_external(rows, get_venue("bybit"), "BTC", "15m")
    report = validate_venue_frame(frame, get_venue("bybit"), "BTC", "15m")
    assert report.gap_count >= 1
    assert report.missing_bars >= 7
    assert report.outlier_count >= 1
    assert report.coverage_ratio < 1.0


def test_checksum_is_stable_and_order_insensitive():
    frame = canonicalise_external(_rows("2026-01-01T00:00:00Z", 10), get_venue("okx"), "BTC", "15m")
    first = frame_checksum(frame)
    assert first == frame_checksum(frame.sample(frac=1.0, random_state=1))
    changed = frame.copy()
    changed.loc[3, "close"] += 1.0
    assert frame_checksum(changed) != first


def test_align_venue_panel_keeps_venues_separate():
    a = canonicalise_external(_rows("2026-01-01T00:00:00Z", 200), get_venue("binance"), "BTC", "15m")
    b = canonicalise_external(_rows("2026-01-01T00:30:00Z", 190), get_venue("bybit"), "BTC", "15m")
    panel, stats = align_venue_panel({"binance": a, "bybit": b}, timeframe="15m")
    assert {"close_binance", "close_bybit"}.issubset(panel.columns)
    assert stats["duplicate_timestamps"] == 0
    assert stats["venues"]["binance"]["bars"] == len(a)
    assert stats["venues"]["bybit"]["bars"] == len(b)
    key = "binance|bybit"
    assert stats["overlap"][key]["n"] > 100
    assert stats["overlap"][key]["return_corr"] > 0.99
    assert stats["all_venues_present_share"] < 1.0
    assert_panel_integrity(panel)


def test_align_panel_integrity_rejects_off_grid_timestamps():
    frame = canonicalise_external(_rows("2026-01-01T00:00:00Z", 5), get_venue("binance"), "BTC", "15m")
    frame.loc[2, "timestamp_open"] = frame.loc[2, "timestamp_open"] + pd.Timedelta(minutes=7)
    with pytest.raises(ValueError):
        assert_panel_integrity(frame)


def test_all_venues_have_required_fields():
    for slug, venue in VENUES.items():
        assert venue.symbols, slug
        assert venue.licence_note, slug
        assert venue.market_type == "spot"
        assert venue.quote in ("USDT", "USD")


def test_manifest_merge_prefers_ok_rows_and_keeps_history():
    from patron.data.collect import _merge_results

    old = [
        {"exchange": "binance", "asset": "DOGE", "timeframe": "15m", "status": "failed", "rows": 0},
        {"exchange": "bybit", "asset": "DOGE", "timeframe": "15m", "status": "ok", "rows": 10},
    ]
    new = [
        {"exchange": "binance", "asset": "DOGE", "timeframe": "15m", "status": "ok", "rows": 20},
        {"exchange": "binance", "asset": "ETH", "timeframe": "15m", "status": "ok", "rows": 30},
    ]
    merged = {(r["exchange"], r["asset"]): r for r in _merge_results(old, new)}
    assert merged[("binance", "DOGE")]["status"] == "ok"
    assert merged[("binance", "DOGE")]["rows"] == 20
    assert merged[("bybit", "DOGE")]["rows"] == 10  # untouched history survives
    assert merged[("binance", "ETH")]["rows"] == 30


def test_retry_drops_poisoned_connection_pool(monkeypatch):
    """A reset keep-alive connection is closed before the retry, so one outage
    does not exhaust every attempt (the live 2026-11-15 failure mode)."""
    import requests

    from patron.data import venues

    class Response:
        status_code = 200

        def json(self):
            return {"ok": True}

    class Session:
        def __init__(self):
            self.calls = 0
            self.closed = 0

        def get(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise requests.ConnectionError("connection reset")
            return Response()

        def close(self):
            self.closed += 1

    monkeypatch.setattr(venues.time, "sleep", lambda *_: None)
    session = Session()
    assert venues._get_json(session, "https://example.invalid", {}, 0.0) == {"ok": True}
    assert session.calls == 2
    assert session.closed == 1
