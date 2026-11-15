"""Rolling 96-candle model window, rebuilt from durable state.

The window is not kept as in-memory state that could drift after a restart: it is
reconstructed from the candle store on every cycle, and the append/drop rule is
enforced by checking that the newest 96 stored, closed candles are contiguous.
That makes "append a new candle, drop the oldest" and "restart and resume" the
same operation.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from patron.data.markets import TIMEFRAME_MINUTES
from patron.runtime.store import RuntimeStore, parse_ts, utc_iso

WINDOW_COLUMNS = ("timestamp_open", "timestamp_close", "open", "high", "low", "close", "volume")


@dataclass
class WindowState:
    status: str
    frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    warmup: pd.DataFrame = field(default_factory=pd.DataFrame)
    input_hash: str | None = None
    reason: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "OK" and not self.frame.empty

    @property
    def last_close(self) -> pd.Timestamp | None:
        if self.frame.empty:
            return None
        return pd.Timestamp(self.frame["timestamp_close"].iloc[-1])


def window_hash(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for row in frame[list(WINDOW_COLUMNS)].itertuples(index=False):
        digest.update(
            f"{utc_iso(row.timestamp_open)}|{row.open:.10f}|{row.high:.10f}|{row.low:.10f}|"
            f"{row.close:.10f}|{row.volume:.10f}".encode("utf-8")
        )
    return digest.hexdigest()


def build_window(
    store: RuntimeStore,
    symbol: str,
    timeframe: str,
    sequence_length: int = 96,
    warmup_bars: int = 384,
) -> WindowState:
    tf = pd.Timedelta(minutes=TIMEFRAME_MINUTES[timeframe])
    needed = sequence_length + warmup_bars
    candles = store.latest_candles(symbol, timeframe, needed)
    if len(candles) < sequence_length:
        return WindowState(
            status="INSUFFICIENT_HISTORY",
            reason=f"{len(candles)} closed candles stored, need {sequence_length}",
        )

    window = candles.tail(sequence_length)
    diffs = window["timestamp_open"].diff().dropna().to_numpy()
    irregular = np.flatnonzero(diffs != tf.to_timedelta64())
    if irregular.size:
        gap_at = window["timestamp_open"].iloc[int(irregular[0]) + 1]
        return WindowState(
            status="DATA_GAP",
            reason=f"gap inside the {sequence_length}-candle window at {utc_iso(gap_at)}",
        )

    start = pd.Timestamp(window["timestamp_open"].iloc[0])
    earlier = candles[candles["timestamp_open"] < start].sort_values("timestamp_open")
    warmup = earlier.tail(warmup_bars)
    if not warmup.empty:
        # Keep only the contiguous tail that ends exactly one bar before the window.
        contiguous = [warmup["timestamp_open"].iloc[-1]]
        times = list(warmup["timestamp_open"])
        for ts in reversed(times[:-1]):
            if contiguous[-1] - ts == tf:
                contiguous.append(ts)
            else:
                break
        warmup = warmup[warmup["timestamp_open"].isin(contiguous)].sort_values("timestamp_open")

    warnings: list[str] = []
    minimum_warmup = max(1, warmup_bars // 2)
    if len(warmup) < minimum_warmup:
        return WindowState(
            status="INSUFFICIENT_HISTORY",
            frame=window.reset_index(drop=True),
            warmup=warmup.reset_index(drop=True),
            reason=(
                f"warm-up has {len(warmup)} contiguous candles immediately before the window, "
                f"need at least {minimum_warmup} for ATR/volume baselines"
            ),
        )
    if len(warmup) < warmup_bars:
        warnings.append(f"warm-up truncated to {len(warmup)} contiguous candles")

    clean_window = window[list(WINDOW_COLUMNS)].reset_index(drop=True)
    return WindowState(
        status="OK",
        frame=clean_window,
        warmup=warmup[list(WINDOW_COLUMNS)].reset_index(drop=True),
        input_hash=window_hash(clean_window),
        warnings=warnings,
    )


def candles_payload(frame: pd.DataFrame) -> list[dict]:
    """Canonical candle list for the inference input contract."""
    return [
        {
            "timestamp": utc_iso(row.timestamp_open),
            "open": float(row.open),
            "high": float(row.high),
            "low": float(row.low),
            "close": float(row.close),
            "volume": float(row.volume),
        }
        for row in frame.itertuples(index=False)
    ]


def target_timestamps(candle_open: pd.Timestamp, timeframe: str, horizon: int) -> list[dict]:
    tf = pd.Timedelta(minutes=TIMEFRAME_MINUTES[timeframe])
    candle_open = parse_ts(utc_iso(candle_open))
    return [
        {
            "step": step,
            "minutes_ahead": step * TIMEFRAME_MINUTES[timeframe],
            "target_ts": utc_iso(candle_open + step * tf),
        }
        for step in range(1, horizon + 1)
    ]
