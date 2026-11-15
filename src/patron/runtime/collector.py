"""Market-data source adapters for the live runtime.

The runtime only needs *recent closed candles*: the exchange adapter is asked for a
bounded lookback window each cycle, which doubles as reconciliation of delayed or
revised bars. Production uses the same paging/retry code as the collection scripts
(:mod:`patron.data.venues`); tests inject :class:`StaticSource` so the whole
lifecycle can be driven candle by candle without a network.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Protocol

import pandas as pd

from patron.data.markets import TIMEFRAME_MINUTES, get_venue
from patron.data.venues import fetch_candles

LOGGER = logging.getLogger(__name__)


class SourceError(RuntimeError):
    """The source could not deliver usable candles for this cycle."""


class CandleSource(Protocol):
    name: str

    def fetch(self, symbol: str, timeframe: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        ...


@dataclass
class VenueSource:
    """Public REST adapter for one venue (default: Binance spot)."""

    venue: str = "binance"
    market_type: str = "spot"
    pause: float = 0.15
    name: str = field(init=False)

    def __post_init__(self) -> None:
        self.name = f"{self.venue}:{self.market_type}"

    def fetch(self, symbol: str, timeframe: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        venue = get_venue(self.venue)
        base_asset = symbol.split("/")[0]
        try:
            frame = fetch_candles(venue, base_asset, timeframe, start, end, pause=self.pause)
        except Exception as exc:  # network, HTTP, venue rejection
            raise SourceError(f"{self.name} fetch failed: {exc}") from exc
        if frame.empty:
            raise SourceError(f"{self.name} returned no candles for {start} -> {end}")
        LOGGER.info("%s: %d candles %s..%s", self.name, len(frame),
                    frame["timestamp_open"].iloc[0], frame["timestamp_open"].iloc[-1])
        return frame


@dataclass
class StaticSource:
    """Deterministic in-memory source used by tests and offline replays.

    ``fail_times`` makes the next N fetches raise, which exercises the retry /
    ``RECOVERING`` path without a network. ``revise`` can rewrite already-issued
    candles to test reconciliation.
    """

    frame: pd.DataFrame
    name: str = "static"
    fail_times: int = 0
    calls: list[tuple[pd.Timestamp, pd.Timestamp]] = field(default_factory=list)

    def fetch(self, symbol: str, timeframe: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        self.calls.append((pd.Timestamp(start), pd.Timestamp(end)))
        if self.fail_times > 0:
            self.fail_times -= 1
            raise SourceError(f"{self.name}: simulated outage ({self.fail_times} left)")
        window = self.frame[
            (self.frame["timestamp_open"] >= pd.Timestamp(start))
            & (self.frame["timestamp_open"] < pd.Timestamp(end))
        ]
        if window.empty:
            raise SourceError(f"{self.name}: no candles in requested window")
        return window.reset_index(drop=True)


def default_window(now: datetime | None = None, lookback_hours: int = 48) -> tuple[pd.Timestamp, pd.Timestamp]:
    now = now or datetime.now(timezone.utc)
    end = pd.Timestamp(now).tz_convert("UTC") if pd.Timestamp(now).tzinfo else pd.Timestamp(now, tz="UTC")
    return end - timedelta(hours=lookback_hours), end


def timeframe_delta(timeframe: str) -> pd.Timedelta:
    return pd.Timedelta(minutes=TIMEFRAME_MINUTES[timeframe])
