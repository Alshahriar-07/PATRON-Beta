"""Venue-aware candle validation and provenance.

The original :mod:`patron.data.validate` module enforces the Binance schema
(every auxiliary column present). Expanded venues do not all publish quote
volume or trade counts, so this module validates what a venue *does* publish and
records the rest as unavailable. It never fills, interpolates or drops silently:
every defect is counted and kept in the report.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from patron.data.markets import CANONICAL_COLUMNS, TIMEFRAME_MINUTES, Venue

PRICE_COLUMNS = ("open", "high", "low", "close")

#: Return magnitude above which a bar is flagged as an outlier candidate.
DEFAULT_JUMP_THRESHOLD = 0.25
#: Robust z-score above which a bar is flagged as an outlier candidate.
DEFAULT_MAD_Z = 12.0


@dataclass
class VenueQualityReport:
    exchange: str
    market_type: str
    symbol: str
    timeframe: str
    rows: int = 0
    start: str | None = None
    end: str | None = None
    duplicate_timestamps: int = 0
    non_monotonic: bool = False
    unclosed_candles: int = 0
    missing_bars: int = 0
    gap_count: int = 0
    coverage_ratio: float = 0.0
    invalid_ohlc: int = 0
    non_positive_prices: int = 0
    negative_volume: int = 0
    zero_volume_rows: int = 0
    nan_volume_rows: int = 0
    quote_volume_available: bool = True
    trade_count_available: bool = True
    outlier_count: int = 0
    outlier_examples: list[dict] = field(default_factory=list)
    gap_examples: list[dict] = field(default_factory=list)
    stale_minutes: float | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["valid"] = self.valid
        return payload

    def summary(self) -> str:
        return (
            f"{self.exchange}:{self.symbol} {self.timeframe} rows={self.rows} "
            f"coverage={self.coverage_ratio:.3f} gaps={self.gap_count} "
            f"missing={self.missing_bars} outliers={self.outlier_count} "
            f"valid={self.valid}"
        )


def frame_checksum(frame: pd.DataFrame) -> str:
    """Deterministic sha256 over the canonical candle columns."""
    if frame.empty:
        return hashlib.sha256(b"").hexdigest()
    columns = [c for c in CANONICAL_COLUMNS if c in frame.columns]
    ordered = frame.sort_values("timestamp_open")[columns]
    payload = ordered.to_csv(index=False, float_format="%.10g").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_venue_frame(
    frame: pd.DataFrame,
    venue: Venue,
    base_asset: str,
    timeframe: str,
    now: datetime | None = None,
    jump_threshold: float = DEFAULT_JUMP_THRESHOLD,
    mad_z: float = DEFAULT_MAD_Z,
    max_examples: int = 10,
) -> VenueQualityReport:
    """Validate one venue/asset/timeframe candle frame."""
    asset = base_asset.upper()
    report = VenueQualityReport(
        exchange=venue.slug,
        market_type=venue.market_type,
        symbol=f"{asset}/{venue.quote}",
        timeframe=timeframe,
        quote_volume_available=venue.provides_quote_volume,
        trade_count_available=venue.provides_trade_count,
        rows=int(len(frame)),
    )
    if frame.empty:
        report.errors.append("empty frame (venue returned no candles)")
        return report

    missing = set(CANONICAL_COLUMNS) - set(frame.columns)
    if missing:
        report.errors.append(f"missing canonical columns: {sorted(missing)}")
        return report

    ts_open = pd.to_datetime(frame["timestamp_open"], utc=True)
    ts_close = pd.to_datetime(frame["timestamp_close"], utc=True)
    report.start = ts_open.iloc[0].isoformat()
    report.end = ts_open.iloc[-1].isoformat()

    if ts_open.duplicated().any():
        report.duplicate_timestamps = int(ts_open.duplicated().sum())
        report.errors.append(f"{report.duplicate_timestamps} duplicate timestamps")
    if not ts_open.is_monotonic_increasing:
        report.non_monotonic = True
        report.errors.append("timestamps not strictly ascending")

    now_ts = pd.Timestamp(now or datetime.now(timezone.utc))
    if now_ts.tzinfo is None:
        now_ts = now_ts.tz_localize("UTC")
    report.unclosed_candles = int((ts_close > now_ts).sum())
    if report.unclosed_candles:
        report.errors.append(f"{report.unclosed_candles} unclosed candles")
    report.stale_minutes = float((now_ts - ts_close.iloc[-1]).total_seconds() / 60.0)

    numeric = frame[list(PRICE_COLUMNS) + ["volume"]].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(numeric.to_numpy(dtype="float64"))
    if not finite.all():
        report.errors.append(f"{int((~finite).sum())} NaN/inf price/volume values")
    report.non_positive_prices = int((numeric[list(PRICE_COLUMNS)] <= 0).any(axis=1).sum())
    if report.non_positive_prices:
        report.errors.append(f"{report.non_positive_prices} non-positive prices")
    report.negative_volume = int((numeric["volume"] < 0).sum())
    if report.negative_volume:
        report.errors.append(f"{report.negative_volume} negative volumes")
    report.zero_volume_rows = int((numeric["volume"] == 0).sum())
    report.nan_volume_rows = int(numeric["volume"].isna().sum())

    hi_ok = numeric["high"] >= numeric[["open", "close", "low"]].max(axis=1)
    lo_ok = numeric["low"] <= numeric[["open", "close", "high"]].min(axis=1)
    report.invalid_ohlc = int((~(hi_ok & lo_ok)).sum())
    if report.invalid_ohlc:
        report.errors.append(f"{report.invalid_ohlc} OHLC-inconsistent rows")

    # Continuity
    tf = pd.Timedelta(minutes=TIMEFRAME_MINUTES[timeframe])
    diffs = ts_open.diff()
    gap_mask = diffs > tf
    report.gap_count = int(gap_mask.sum())
    if report.gap_count:
        gap_rows = pd.DataFrame(
            {
                "prev_open": ts_open.shift(1)[gap_mask].astype(str),
                "next_open": ts_open[gap_mask].astype(str),
                "missing_bars": (diffs[gap_mask] // tf - 1).astype(int),
            }
        )
        report.missing_bars = int(gap_rows["missing_bars"].sum())
        report.gap_examples = gap_rows.head(max_examples).to_dict("records")
        report.warnings.append(f"{report.gap_count} gaps covering {report.missing_bars} bars")
    expected = int((ts_open.iloc[-1] - ts_open.iloc[0]) / tf) + 1
    report.coverage_ratio = float(len(frame) / expected) if expected > 0 else 0.0
    if report.coverage_ratio < 0.95:
        report.warnings.append(f"coverage {report.coverage_ratio:.1%} below 95%")

    # Outliers on close-to-close log returns.
    close = numeric["close"].to_numpy(dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = np.diff(np.log(close))
    if len(returns) > 10:
        median = float(np.median(returns))
        mad = float(np.median(np.abs(returns - median)))
        scale = 1.4826 * mad if mad > 0 else float(np.std(returns) or 1e-12)
        z = np.abs(returns - median) / max(scale, 1e-12)
        jump = np.abs(returns) > jump_threshold
        flags = np.flatnonzero(jump | (z > mad_z))
        report.outlier_count = int(len(flags))
        if report.outlier_count:
            report.outlier_examples = [
                {
                    "timestamp": str(ts_open.iloc[int(i) + 1]),
                    "log_return": float(returns[int(i)]),
                    "robust_z": float(z[int(i)]),
                }
                for i in flags[:max_examples]
            ]
            report.warnings.append(
                f"{report.outlier_count} return outliers flagged for review (kept, not removed)"
            )
    return report


def coverage_summary(reports: list[VenueQualityReport]) -> dict:
    """Aggregate a list of venue reports (used in the data-expansion report)."""
    if not reports:
        return {"venues": 0, "valid": 0}
    return {
        "venues": len(reports),
        "valid": sum(1 for r in reports if r.valid),
        "rows": int(sum(r.rows for r in reports)),
        "start": min((r.start for r in reports if r.start), default=None),
        "end": max((r.end for r in reports if r.end), default=None),
        "total_gaps": int(sum(r.gap_count for r in reports)),
        "total_outliers": int(sum(r.outlier_count for r in reports)),
    }


def assert_no_cross_venue_duplicates(panel: pd.DataFrame) -> None:
    """A (timestamp, exchange, symbol) triple must be unique in any stored frame."""
    keys = ["timestamp_open", "exchange", "symbol"]
    present = [k for k in keys if k in panel.columns]
    if len(present) < len(keys):
        raise ValueError(f"panel missing key columns: {set(keys) - set(present)}")
    duplicates = panel.duplicated(subset=keys).sum()
    if duplicates:
        raise ValueError(f"{duplicates} duplicate (timestamp, exchange, symbol) observations")
