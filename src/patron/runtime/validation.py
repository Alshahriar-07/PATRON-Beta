"""Candle validation for the live path.

The rules mirror the inference contract but are *operational*: a batch of freshly
fetched candles is accepted only after duplicates are collapsed, forming bars are
dropped, and OHLCV sanity holds. Nothing is filled, interpolated or fabricated —
rejected bars are reported and the caller decides whether a forecast is possible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from patron.data.markets import TIMEFRAME_MINUTES

PRICE_COLUMNS = ("open", "high", "low", "close")
REQUIRED = ("timestamp_open", "timestamp_close", *PRICE_COLUMNS, "volume")


@dataclass
class CandleValidation:
    status: str
    frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    duplicates: int = 0
    unclosed: int = 0
    invalid_rows: int = 0
    invalid_reasons: list[str] = field(default_factory=list)
    gaps: int = 0
    zero_volume: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "OK" and not self.frame.empty


def _empty(reason: str) -> pd.DataFrame:
    return pd.DataFrame(columns=list(REQUIRED))


def validate_candles(
    frame: pd.DataFrame,
    timeframe: str,
    now: datetime | None = None,
    confirm_delay_seconds: float = 0.0,
) -> CandleValidation:
    """Return closed, valid, deduplicated candles plus a precise status."""
    now_ts = pd.Timestamp(now or datetime.now(timezone.utc))
    now_ts = now_ts.tz_convert("UTC") if now_ts.tzinfo else now_ts.tz_localize("UTC")
    tf = pd.Timedelta(minutes=TIMEFRAME_MINUTES[timeframe])
    delay = pd.Timedelta(seconds=float(confirm_delay_seconds))

    if frame is None or len(frame) == 0:
        return CandleValidation(status="STALE_DATA", frame=_empty("no candles returned"),
                                warnings=["source returned no candles"])

    missing = [col for col in REQUIRED if col not in frame.columns]
    if missing:
        return CandleValidation(
            status="INVALID_INPUT",
            frame=_empty("missing columns"),
            invalid_reasons=[f"missing columns: {', '.join(missing)}"],
        )

    work = frame.copy()
    work["timestamp_open"] = pd.to_datetime(work["timestamp_open"], utc=True, errors="coerce")
    work["timestamp_close"] = pd.to_datetime(work["timestamp_close"], utc=True, errors="coerce")
    if work["timestamp_open"].isna().any():
        return CandleValidation(status="INVALID_INPUT", frame=_empty("bad timestamps"),
                                invalid_reasons=["unparseable timestamps"])

    duplicates = int(work["timestamp_open"].duplicated().sum())
    work = work.sort_values("timestamp_open").drop_duplicates("timestamp_open", keep="last")

    # A candle is usable only once its close plus the configured safety delay passed.
    unclosed_mask = work["timestamp_close"] + delay > now_ts
    unclosed = int(unclosed_mask.sum())
    work = work[~unclosed_mask].copy()

    if work.empty:
        return CandleValidation(
            status="STALE_DATA",
            frame=_empty("no confirmed closed candle"),
            duplicates=duplicates,
            unclosed=unclosed,
            warnings=["latest candle is still forming or inside the confirmation delay"],
        )

    for col in (*PRICE_COLUMNS, "volume"):
        work[col] = pd.to_numeric(work[col], errors="coerce")
    numeric_ok = work[list((*PRICE_COLUMNS, "volume"))].notna().all(axis=1)
    positive = (work[list(PRICE_COLUMNS)] > 0).all(axis=1)
    high_ok = work["high"] >= work[list(PRICE_COLUMNS)].max(axis=1) - 1e-9
    low_ok = work["low"] <= work[list(PRICE_COLUMNS)].min(axis=1) + 1e-9
    volume_ok = work["volume"] >= 0
    valid_mask = numeric_ok & positive & high_ok & low_ok & volume_ok
    invalid_rows = int((~valid_mask).sum())
    # An unusable newest bar means this cycle has no trustworthy fresh candle.
    latest_invalid = bool(len(valid_mask) and not bool(valid_mask.iloc[-1]))
    reasons: list[str] = []
    if not numeric_ok.all():
        reasons.append("non-numeric OHLCV values")
    if not positive.all():
        reasons.append("non-positive prices")
    if not high_ok.all():
        reasons.append("high below open/close/low")
    if not low_ok.all():
        reasons.append("low above open/close/high")
    if not volume_ok.all():
        reasons.append("negative volume")
    work = work[valid_mask].copy()

    if work.empty:
        return CandleValidation(
            status="INVALID_INPUT",
            frame=_empty("all rows invalid"),
            duplicates=duplicates,
            unclosed=unclosed,
            invalid_rows=invalid_rows,
            invalid_reasons=reasons,
        )

    # Timestamps must sit on the timeframe grid and close one bar later.
    off_grid = (work["timestamp_open"].dt.floor(tf) != work["timestamp_open"]).sum()
    if off_grid:
        return CandleValidation(
            status="INVALID_INPUT",
            frame=_empty("off-grid timestamps"),
            duplicates=duplicates,
            unclosed=unclosed,
            invalid_reasons=[f"{int(off_grid)} candles not aligned to {timeframe}"],
        )
    bad_close = (work["timestamp_close"] != work["timestamp_open"] + tf).sum()
    if bad_close:
        return CandleValidation(
            status="INVALID_INPUT",
            frame=_empty("close timestamp mismatch"),
            duplicates=duplicates,
            unclosed=unclosed,
            invalid_reasons=[f"{int(bad_close)} candles with inconsistent close timestamps"],
        )

    gaps = int((work["timestamp_open"].diff().dropna() != tf).sum())
    zero_volume = int((work["volume"] == 0).sum())
    warnings: list[str] = []
    if duplicates:
        warnings.append(f"{duplicates} duplicate candle(s) collapsed (kept newest)")
    if unclosed:
        warnings.append(f"{unclosed} forming candle(s) ignored")
    if invalid_rows:
        warnings.append(f"{invalid_rows} invalid candle(s) rejected: {', '.join(reasons)}")
    if zero_volume:
        warnings.append(f"{zero_volume} genuine zero-volume candle(s) kept")
    if gaps:
        warnings.append(f"{gaps} irregular gap(s) inside the fetched window")

    status = "INVALID_INPUT" if latest_invalid else "OK"
    return CandleValidation(
        status=status,
        frame=work.reset_index(drop=True),
        duplicates=duplicates,
        unclosed=unclosed,
        invalid_rows=invalid_rows,
        invalid_reasons=reasons,
        gaps=gaps,
        zero_volume=zero_volume,
        warnings=warnings,
    )


def check_contiguity(frame: pd.DataFrame, timeframe: str, expected: int) -> str | None:
    """Return an error string when the last ``expected`` bars are not contiguous."""
    tf = pd.Timedelta(minutes=TIMEFRAME_MINUTES[timeframe])
    if len(frame) < expected:
        return f"need {expected} candles, have {len(frame)}"
    tail = frame.tail(expected)
    diffs = tail["timestamp_open"].diff().dropna()
    if (diffs != tf).any():
        return f"{int((diffs != tf).sum())} gap(s) inside the model window"
    if not np.isfinite(tail[["open", "high", "low", "close", "volume"]].to_numpy()).all():
        return "non-finite values inside the model window"
    return None
