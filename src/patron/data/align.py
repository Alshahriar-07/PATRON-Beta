"""Multi-venue alignment and comparison without blind merging.

Venues are aligned on UTC ``timestamp_open``. Nothing is concatenated row-wise:
the panel keeps one column group per venue, and every comparison reports what is
missing instead of filling it. Venues that quote in USD (Coinbase, Kraken) are
labelled with their real quote currency and compared by *returns*, never by raw
price level.

Known exchange-specific caveats that matter for alignment:
* Coinbase and Kraken quote BTC in USD, not USDT: raw closes differ by the
  USDT/USD basis, so only returns and relative measures are comparable.
* Kraken uses the legacy ``XBT`` ticker; the venue registry handles the mapping.
* Venue outages and maintenance windows produce genuine missing bars; those are
  counted per venue and never imputed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from patron.data.markets import TIMEFRAME_MINUTES, Venue


def align_venue_panel(
    frames: dict[str, pd.DataFrame],
    timeframe: str = "15m",
    tolerance_bars: int = 0,
) -> tuple[pd.DataFrame, dict]:
    """Outer-join venue candles on the UTC grid.

    Args:
        frames: ``venue_slug -> canonical candles`` (must include ``timestamp_open``).
        timeframe: canonical timeframe of all frames.
        tolerance_bars: allowed offset in bars for a venue row to count as present;
            default 0 = exact timestamp match.

    Returns:
        ``(panel, stats)`` where panel has ``timestamp_open`` plus per-venue
        ``open_<venue>``, ``high_<venue>``, ``low_<venue>``, ``close_<venue>``,
        ``volume_<venue>`` columns. Stats describe overlap and basis.
    """
    if not frames:
        raise ValueError("no venue frames to align")

    step = pd.Timedelta(minutes=TIMEFRAME_MINUTES[timeframe])
    panel = pd.DataFrame()
    for slug, frame in frames.items():
        if frame.empty:
            continue
        df = frame.copy()
        df["timestamp_open"] = pd.to_datetime(df["timestamp_open"], utc=True)
        if tolerance_bars:
            df["timestamp_open"] = df["timestamp_open"].dt.round(step)
        df = df.sort_values("timestamp_open").drop_duplicates("timestamp_open", keep="last")
        renamed = {
            col: f"{col}_{slug}" for col in ("open", "high", "low", "close", "volume") if col in df.columns
        }
        df = df[["timestamp_open", *renamed]].rename(columns=renamed)
        panel = df if panel.empty else panel.merge(df, on="timestamp_open", how="outer")

    panel = panel.sort_values("timestamp_open").reset_index(drop=True)
    close_columns = [c for c in panel.columns if c.startswith("close_")]
    stats: dict = {
        "timeframe": timeframe,
        "rows": int(len(panel)),
        "venues": {},
        "overlap": {},
    }
    for col in close_columns:
        slug = col.removeprefix("close_")
        stats["venues"][slug] = {
            "bars": int(panel[col].notna().sum()),
            "coverage_of_panel": round(float(panel[col].notna().mean()), 4),
            "first": str(panel["timestamp_open"].loc[panel[col].first_valid_index()])
            if panel[col].notna().any()
            else None,
            "last": str(panel["timestamp_open"].loc[panel[col].last_valid_index()])
            if panel[col].notna().any()
            else None,
        }

    slugs = list(stats["venues"])
    for i, a in enumerate(slugs):
        for b in slugs[i + 1 :]:
            both = panel[[f"close_{a}", f"close_{b}"]].dropna()
            if len(both) < 10:
                stats["overlap"][f"{a}|{b}"] = {"n": int(len(both))}
                continue
            ret_a = np.log(both[f"close_{a}"]).diff().dropna()
            ret_b = np.log(both[f"close_{b}"]).diff().dropna()
            stats["overlap"][f"{a}|{b}"] = {
                "n": int(len(both)),
                "return_corr": round(float(ret_a.corr(ret_b)), 6),
                "mean_abs_log_basis": round(float(np.abs(np.log(both[f"close_{a}"] / both[f"close_{b}"])).mean()), 6),
            }

    # Panel-wide completeness: share of timestamps where every venue is present.
    if close_columns:
        stats["all_venues_present_share"] = round(float(panel[close_columns].notna().all(axis=1).mean()), 4)
        stats["duplicate_timestamps"] = int(panel["timestamp_open"].duplicated().sum())
    return panel, stats


def assert_panel_integrity(panel: pd.DataFrame, expected_step_minutes: int = 15) -> None:
    """Fail loudly on duplicate or mis-gridded timestamps."""
    ts = pd.to_datetime(panel["timestamp_open"], utc=True)
    if ts.duplicated().any():
        raise ValueError(f"{int(ts.duplicated().sum())} duplicate panel timestamps")
    if not ts.is_monotonic_increasing:
        raise ValueError("panel timestamps are not ascending")
    deltas = ts.diff().dropna().dt.total_seconds() / 60
    if not np.all(np.isclose(deltas % expected_step_minutes, 0)):
        raise ValueError("panel contains timestamps off the timeframe grid")


def venue_metadata(venue: Venue, frame: pd.DataFrame) -> dict:
    """Exchange-specific metadata written next to every partition."""
    return {
        "exchange": venue.slug,
        "market_type": venue.market_type,
        "quote_currency": venue.quote,
        "venue_symbol": frame["symbol"].iloc[0] if len(frame) and "symbol" in frame else None,
        "provides_quote_volume": venue.provides_quote_volume,
        "provides_trade_count": venue.provides_trade_count,
        "licence_note": venue.licence_note,
        "rows": int(len(frame)),
    }
