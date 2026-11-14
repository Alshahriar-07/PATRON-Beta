"""Higher-timeframe candle construction.

The 1h/4h context features are derived from the 15m series itself so that the
inference path (which only receives 15m candles) is bit-identical to training.
Downloaded 1h/4h archives are used as an independent cross-check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HTF_RULES: dict[str, str] = {"1h": "1h", "4h": "4h"}
BARS_PER_HTF: dict[str, int] = {"1h": 4, "4h": 16}


def resample_closed(df_15m: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Aggregate 15m candles into fully closed higher-timeframe candles.

    Incomplete buckets are dropped: a partially formed 1h/4h candle must never
    influence a feature (future information).
    """
    frame = df_15m.set_index("timestamp_open")
    agg = frame.resample(rule, label="left", closed="left").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        rows=("close", "size"),
    )
    expected = BARS_PER_HTF[rule]
    agg = agg[agg["rows"] == expected].copy()
    agg["timestamp_close"] = agg.index + pd.Timedelta(rule)
    return agg.reset_index()


def htf_trend_frame(df_15m: pd.DataFrame) -> pd.DataFrame:
    """Compute `htf_trend` aligned to 15m candle close times.

    Definition (documented refinement of the spec's "normalized 1h/4h trend"):

        trend_htf = tanh( (close_htf / EMA9_htf - 1) / atr14_pct_htf )
        htf_trend = mean(trend_1h, trend_4h)

    Both components use only fully closed HTF candles; they are aligned with
    ``merge_asof`` on the HTF close timestamp so a candle that closes later than
    the 15m decision time is never used.
    """
    from patron.features.indicators import atr_wilder

    out = pd.DataFrame(
        {"timestamp_close": pd.to_datetime(df_15m["timestamp_close"], utc=True)}
    )
    components = []
    for key, rule in HTF_RULES.items():
        htf = resample_closed(df_15m, rule)
        if htf.empty:
            out[f"trend_{key}"] = np.nan
            continue
        ema9 = htf["close"].ewm(span=9, adjust=False, min_periods=9).mean()
        atr14 = atr_wilder(htf, 14)
        atr_pct = (atr14 / htf["close"]).replace(0.0, np.nan)
        trend = np.tanh(((htf["close"] / ema9) - 1.0) / atr_pct)
        comp = pd.DataFrame(
            {
                "htf_close": htf["timestamp_close"].to_numpy(),
                f"trend_{key}": trend.to_numpy(),
            }
        ).sort_values("htf_close")
        left = out[["timestamp_close"]].sort_values("timestamp_close")
        merged = pd.merge_asof(
            left,
            comp,
            left_on="timestamp_close",
            right_on="htf_close",
            direction="backward",
            allow_exact_matches=True,
        )
        out[f"trend_{key}"] = merged[f"trend_{key}"].to_numpy()
        components.append(f"trend_{key}")

    if len(components) == 2:
        out["htf_trend"] = out[components].mean(axis=1, skipna=False)
    else:
        out["htf_trend"] = np.nan
    return out[["timestamp_close", "htf_trend"]]
