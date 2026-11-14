"""Candle validation and data-quality reporting.

Validation never fabricates prices. Invalid rows are rejected explicitly and
counted; gaps are reported, not filled. Downstream dataset code excludes windows
that touch a gap so no sample silently spans missing market data.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from patron.data.downloader import TIMEFRAME_MINUTES, timeframe_delta

LOGGER = logging.getLogger(__name__)

PRICE_COLUMNS = ["open", "high", "low", "close"]
NUMERIC_COLUMNS = PRICE_COLUMNS + ["volume", "quote_volume", "trade_count"]


@dataclass
class DataQualityReport:
    symbol: str
    timeframe: str
    rows: int = 0
    start: str | None = None
    end: str | None = None
    duplicate_timestamps: int = 0
    non_monotonic: bool = False
    unclosed_candles: int = 0
    nan_or_inf: int = 0
    invalid_ohlc: int = 0
    non_positive_prices: int = 0
    negative_volume: int = 0
    missing_bars: int = 0
    gap_count: int = 0
    largest_gap_bars: int = 0
    suspicious_jumps: int = 0
    rows_rejected: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    gap_examples: list[dict] = field(default_factory=list)
    jump_examples: list[dict] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["valid"] = self.valid
        return payload

    def summary(self) -> str:
        return (
            f"{self.symbol} {self.timeframe}: {self.rows} rows "
            f"{self.start} -> {self.end} | gaps={self.gap_count} "
            f"missing_bars={self.missing_bars} invalid_ohlc={self.invalid_ohlc} "
            f"dupes={self.duplicate_timestamps} rejected={self.rows_rejected} "
            f"valid={self.valid}"
        )


def expected_bars(start: pd.Timestamp, end: pd.Timestamp, timeframe: str) -> int:
    minutes = TIMEFRAME_MINUTES[timeframe]
    return int((end - start).total_seconds() // 60 // minutes) + 1


def validate_candles(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    now: datetime | None = None,
    jump_threshold: float = 0.20,
    max_gap_examples: int = 20,
) -> DataQualityReport:
    """Run every Phase 1 validation rule and return a structured report."""
    report = DataQualityReport(symbol=symbol, timeframe=timeframe, rows=int(len(df)))
    if df.empty:
        report.errors.append("empty candle frame")
        return report

    now_ts = pd.Timestamp(now or datetime.now(timezone.utc))
    if now_ts.tzinfo is None:
        now_ts = now_ts.tz_localize("UTC")

    required = {"timestamp_open", "timestamp_close", *NUMERIC_COLUMNS}
    missing_cols = required - set(df.columns)
    if missing_cols:
        report.errors.append(f"missing required columns: {sorted(missing_cols)}")
        return report

    ts_open = pd.to_datetime(df["timestamp_open"], utc=True)
    ts_close = pd.to_datetime(df["timestamp_close"], utc=True)
    report.start = ts_open.iloc[0].isoformat()
    report.end = ts_open.iloc[-1].isoformat()

    # duplicates / ordering
    report.duplicate_timestamps = int(ts_open.duplicated().sum())
    if report.duplicate_timestamps:
        report.errors.append(f"{report.duplicate_timestamps} duplicate timestamps")
    if not ts_open.is_monotonic_increasing:
        report.non_monotonic = True
        report.errors.append("timestamps are not strictly ascending")

    # closed candles only
    report.unclosed_candles = int((ts_close > now_ts).sum())
    if report.unclosed_candles:
        report.errors.append(f"{report.unclosed_candles} unclosed candles present")

    # numeric integrity
    numeric = df[NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")
    report.nan_or_inf = int((~np.isfinite(numeric.to_numpy(dtype="float64"))).sum())
    if report.nan_or_inf:
        report.errors.append(f"{report.nan_or_inf} NaN/inf values")
    report.non_positive_prices = int((numeric[PRICE_COLUMNS] <= 0).any(axis=1).sum())
    if report.non_positive_prices:
        report.errors.append(f"{report.non_positive_prices} non-positive prices")
    report.negative_volume = int((numeric[["volume", "quote_volume"]] < 0).any(axis=1).sum())
    if report.negative_volume:
        report.errors.append(f"{report.negative_volume} negative volumes")

    hi_ok = numeric["high"] >= numeric[["open", "close", "low"]].max(axis=1)
    lo_ok = numeric["low"] <= numeric[["open", "close", "high"]].min(axis=1)
    report.invalid_ohlc = int((~(hi_ok & lo_ok)).sum())
    if report.invalid_ohlc:
        report.errors.append(f"{report.invalid_ohlc} OHLC-inconsistent rows")

    # gaps
    delta = timeframe_delta(timeframe)
    diffs = ts_open.diff()
    gap_mask = diffs > delta
    report.gap_count = int(gap_mask.sum())
    if report.gap_count:
        gap_rows = df.loc[gap_mask, ["timestamp_open"]].copy()
        gap_rows["prev_open"] = ts_open.shift(1)[gap_mask].values
        gap_rows["missing_bars"] = (diffs[gap_mask] // delta - 1).astype(int).values
        report.missing_bars = int(gap_rows["missing_bars"].sum())
        report.largest_gap_bars = int(gap_rows["missing_bars"].max())
        report.gap_examples = [
            {
                "prev_open": str(r.prev_open),
                "next_open": str(r.timestamp_open),
                "missing_bars": int(r.missing_bars),
            }
            for r in gap_rows.head(max_gap_examples).itertuples()
        ]
        report.warnings.append(
            f"{report.gap_count} gaps covering {report.missing_bars} missing bars "
            "(not filled; affected windows are excluded downstream)"
        )

    # suspicious jumps
    close = numeric["close"]
    jumps = close.pct_change().abs()
    jump_mask = jumps > jump_threshold
    report.suspicious_jumps = int(jump_mask.sum())
    if report.suspicious_jumps:
        idx = np.flatnonzero(jump_mask.to_numpy())[:max_gap_examples]
        report.jump_examples = [
            {"timestamp": str(ts_open.iloc[int(i)]), "pct_change": float(jumps.iloc[int(i)])}
            for i in idx
        ]
        report.warnings.append(f"{report.suspicious_jumps} close-to-close jumps > {jump_threshold:.0%}")

    report.rows_rejected = int(
        (
            ~np.isfinite(numeric.to_numpy(dtype="float64")).all(axis=1)
            | (numeric[PRICE_COLUMNS] <= 0).any(axis=1).to_numpy()
            | (~(hi_ok & lo_ok)).to_numpy()
        ).sum()
    )

    expected = expected_bars(ts_open.iloc[0], ts_open.iloc[-1], timeframe)
    LOGGER.info(
        "%s expected=%d actual=%d missing=%d", timeframe, expected, len(df), expected - len(df)
    )
    return report


def clean_candles(
    df: pd.DataFrame, report: DataQualityReport | None = None
) -> tuple[pd.DataFrame, DataQualityReport]:
    """Sort, deduplicate and reject invalid rows; never interpolate."""
    symbol = report.symbol if report else "BTC/USDT"
    timeframe = report.timeframe if report else "15m"
    if report is None:
        report = validate_candles(df, symbol, timeframe)

    frame = df.copy()
    frame["timestamp_open"] = pd.to_datetime(frame["timestamp_open"], utc=True)
    frame["timestamp_close"] = pd.to_datetime(frame["timestamp_close"], utc=True)
    frame = frame.sort_values("timestamp_open").drop_duplicates(
        subset=["timestamp_open"], keep="last"
    )

    numeric = frame[NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")
    hi_ok = numeric["high"] >= numeric[["open", "close", "low"]].max(axis=1)
    lo_ok = numeric["low"] <= numeric[["open", "close", "high"]].min(axis=1)
    keep = (
        np.isfinite(numeric.to_numpy(dtype="float64")).all(axis=1)
        & (numeric[PRICE_COLUMNS] > 0).all(axis=1).to_numpy()
        & (numeric[["volume", "quote_volume"]] >= 0).all(axis=1).to_numpy()
        & (hi_ok & lo_ok).to_numpy()
    )
    removed = int((~keep).sum())
    frame = frame.loc[keep].reset_index(drop=True)

    delta = timeframe_delta(timeframe)
    gaps = frame["timestamp_open"].diff() > delta
    frame["is_gap"] = gaps.fillna(False).astype("int8")
    frame["bars_since_prev"] = (
        (frame["timestamp_open"].diff() // delta).fillna(1).astype("int32")
    )
    if removed:
        LOGGER.warning("rejected %d invalid rows from %s", removed, timeframe)
    return frame, report
