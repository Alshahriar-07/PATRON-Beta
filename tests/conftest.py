"""Shared fixtures: deterministic synthetic candles and the real config."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from patron.config import load_config


def make_candles(n: int = 600, seed: int = 7, start: str = "2026-01-01T00:00:00Z") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range(pd.Timestamp(start), periods=n, freq="15min", tz="UTC")
    steps = rng.normal(0.0, 0.0025, size=n).cumsum()
    close = 30000.0 * np.exp(steps)
    open_ = np.concatenate([[close[0] * 0.999], close[:-1]])
    spread = np.abs(rng.normal(0.0, 0.0015, size=n)) + 0.0005
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)
    volume = rng.lognormal(mean=3.0, sigma=0.4, size=n).cumsum() % 500 + 10
    frame = pd.DataFrame(
        {
            "timestamp_open": ts,
            "timestamp_close": ts + pd.Timedelta(minutes=15),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "quote_volume": volume * close,
            "trade_count": np.full(n, 100),
            "is_gap": np.zeros(n, dtype=np.int8),
            "bars_since_prev": np.ones(n, dtype=np.int32),
            "timeframe": "15m",
        }
    )
    return frame


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def candles():
    return make_candles()
