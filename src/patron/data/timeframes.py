"""Causal multi-timeframe resampling and higher-timeframe alignment.

Rules:
* A higher-timeframe candle is only emitted when every source bar inside it has
  closed — a partially formed bucket is dropped, not forward-filled.
* Alignment onto a base timeframe uses ``merge_asof(direction="backward")`` on
  the HTF **close** timestamp, so a base bar can only see HTF candles that
  finished at or before its own close.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from patron.data.markets import TIMEFRAME_MINUTES

AGGREGATION = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
    "quote_volume": "sum",
    "trade_count": "sum",
}


def timeframe_ratio(source: str, target: str) -> int:
    """How many source bars make one target bar (must divide evenly)."""
    if source not in TIMEFRAME_MINUTES or target not in TIMEFRAME_MINUTES:
        raise KeyError(f"unsupported timeframe: {source} -> {target}")
    src = TIMEFRAME_MINUTES[source]
    tgt = TIMEFRAME_MINUTES[target]
    if tgt < src or tgt % src != 0:
        raise ValueError(f"cannot resample {source} -> {target} (ratio must be a positive integer)")
    return tgt // src


def resample_ohlcv(
    frame: pd.DataFrame,
    source: str,
    target: str,
    require_complete: bool = True,
) -> pd.DataFrame:
    """Resample canonical candles to a higher timeframe, causally."""
    ratio = timeframe_ratio(source, target)
    if source == target:
        return frame.copy()

    df = frame.copy()
    df["timestamp_open"] = pd.to_datetime(df["timestamp_open"], utc=True)
    df = df.sort_values("timestamp_open").reset_index(drop=True)
    available = {k: v for k, v in AGGREGATION.items() if k in df.columns}
    source_delta = pd.Timedelta(minutes=TIMEFRAME_MINUTES[source])

    grouped = df.set_index("timestamp_open").resample(
        f"{TIMEFRAME_MINUTES[target]}min", label="left", closed="left"
    )
    out = grouped.agg(available)
    counts = grouped.size()
    # Expected source bars per target bucket, from the bucket's own position.
    out = out[counts >= ratio] if require_complete else out[counts > 0]
    out = out.reset_index()

    # Verify the bucket's last source bar closes exactly when the target closes.
    last_source_close = grouped["timestamp_close"].max().reindex(out["timestamp_open"]).to_numpy()
    expected_close = out["timestamp_open"] + pd.Timedelta(minutes=TIMEFRAME_MINUTES[target])
    complete = pd.to_datetime(last_source_close, utc=True) == pd.to_datetime(expected_close, utc=True)
    out = out.loc[np.asarray(complete, dtype=bool)].reset_index(drop=True)

    out["timestamp_close"] = out["timestamp_open"] + pd.Timedelta(minutes=TIMEFRAME_MINUTES[target])
    out["is_gap"] = out["timestamp_open"].diff().gt(pd.Timedelta(minutes=TIMEFRAME_MINUTES[target])).astype("int8").fillna(0)
    out["bars_since_prev"] = (
        (out["timestamp_open"].diff() // pd.Timedelta(minutes=TIMEFRAME_MINUTES[target])).fillna(1).astype("int32")
    )
    out["timeframe"] = target
    ordered = ["timestamp_open", "timestamp_close", *available.keys(), "is_gap", "bars_since_prev", "timeframe"]
    return out[[c for c in ordered if c in out.columns]]


def align_htf_causal(
    base: pd.DataFrame,
    htf: pd.DataFrame,
    columns: list[str],
    suffix: str = "_htf",
) -> pd.DataFrame:
    """Attach HTF columns to a base frame without look-ahead.

    ``htf`` must contain ``timestamp_close``; a row is joined when the HTF candle
    closed at or before the base bar's close. HTF rows are never interpolated
    forward across gaps — the join is time-based and can legitimately produce
    NaN when no HTF candle exists yet, which downstream feature code must handle.
    """
    base_frame = base.copy()
    htf_frame = htf.copy()
    base_frame["timestamp_close"] = pd.to_datetime(base_frame["timestamp_close"], utc=True)
    htf_frame["timestamp_close"] = pd.to_datetime(htf_frame["timestamp_close"], utc=True)
    missing = [c for c in columns if c not in htf_frame.columns]
    if missing:
        raise KeyError(f"HTF frame missing columns: {missing}")

    htf_frame = htf_frame.sort_values("timestamp_close")[
        ["timestamp_close", *columns]
    ].rename(columns={c: f"{c}{suffix}" for c in columns})

    merged = pd.merge_asof(
        base_frame.sort_values("timestamp_close"),
        htf_frame,
        on="timestamp_close",
        direction="backward",
        allow_exact_matches=True,
    )
    return merged.sort_values("timestamp_open").reset_index(drop=True)


def timeframe_alignment_check(
    base: pd.DataFrame, htf: pd.DataFrame, source: str, target: str
) -> dict:
    """Sanity-check that an HTF frame really is the resampled base."""
    reference = resample_ohlcv(base, source, target)
    merged = reference.merge(
        htf[["timestamp_open", "close"]],
        on="timestamp_open",
        how="inner",
        suffixes=("_ref", "_actual"),
    )
    if merged.empty:
        return {"matched": 0, "max_abs_close_diff": None}
    diff = float(np.max(np.abs(merged["close_ref"] - merged["close_actual"])))
    return {
        "matched": int(len(merged)),
        "reference_bars": int(len(reference)),
        "max_abs_close_diff": diff,
        "identical": bool(diff <= 1e-12),
    }
