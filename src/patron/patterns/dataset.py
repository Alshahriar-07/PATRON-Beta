"""Reproducible pattern-occurrence dataset.

One row per detected pattern occurrence at bar ``t``. Detection uses only bars
``<= t``; outcomes are stored in separate columns and are never fed back into
features. Outcomes are ATR-normalised so they are comparable across price
levels, and raw returns are kept for auditing.

Regime labels come from :func:`patron.evaluation.regimes.add_regimes` (causal,
expanding statistics) so a pattern's regime is knowable at detection time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from patron.evaluation.regimes import add_regimes
from patron.patterns.detectors import build_context
from patron.patterns.registry import CANDLESTICK_PATTERNS, PatternDefinition
from patron.patterns.structures import STRUCTURE_DEFINITIONS

LOGGER = logging.getLogger(__name__)

DEFAULT_HORIZONS: tuple[int, ...] = (1, 2, 3, 6, 12, 24)

OCCURRENCE_COLUMNS = (
    "pattern",
    "category",
    "timestamp",
    "symbol",
    "exchange",
    "timeframe",
    "direction",
    "candles",
    "strength",
    "regime_trend",
    "regime_vol",
    "close",
    "atr",
)

OUTCOME_COLUMNS = tuple(
    [f"ret_{h}" for h in DEFAULT_HORIZONS]
    + [f"atr_ret_{h}" for h in DEFAULT_HORIZONS]
    + [f"max_up_{h}" for h in DEFAULT_HORIZONS]
    + [f"max_down_{h}" for h in DEFAULT_HORIZONS]
    + ["direction_hit_6", "direction_hit_12"]
)


@dataclass
class PatternDatasetStats:
    rows: int
    patterns: int
    start: str | None
    end: str | None
    per_pattern: dict

    def to_dict(self) -> dict:
        return {
            "rows": self.rows,
            "patterns": self.patterns,
            "start": self.start,
            "end": self.end,
            "per_pattern": self.per_pattern,
        }


def _strength(ctx, pattern: PatternDefinition, mask: np.ndarray) -> np.ndarray:
    """Heuristic 0..1 confidence: base prior x relative body x trend alignment."""
    body_scale = np.where(
        np.isfinite(ctx.avg_body) & (ctx.avg_body > 0), ctx.body / ctx.avg_body, np.nan
    )
    body_scale = np.clip(np.nan_to_num(body_scale, nan=1.0), 0.25, 2.0) / 2.0
    direction = pattern.direction
    aligned = np.ones(ctx.n, dtype=np.float64)
    if direction == "bullish":
        aligned = np.where(ctx.trend_down, 1.0, 0.7)
    elif direction == "bearish":
        aligned = np.where(ctx.trend_up, 1.0, 0.7)
    strength = np.clip(pattern.base_confidence * (0.5 + body_scale) * aligned, 0.0, 1.0)
    return np.where(mask, strength, np.nan)


def _forward_outcomes(
    close: np.ndarray, high: np.ndarray, low: np.ndarray, atr: np.ndarray, horizons: tuple[int, ...]
) -> dict[str, np.ndarray]:
    n = len(close)
    out: dict[str, np.ndarray] = {}
    for h in horizons:
        future_close = np.full(n, np.nan)
        if h < n:
            future_close[:-h] = close[h:]
        max_up = np.full(n, np.nan)
        max_down = np.full(n, np.nan)
        for k in range(1, h + 1):
            shifted_high = np.full(n, np.nan)
            shifted_low = np.full(n, np.nan)
            if k < n:
                shifted_high[:-k] = high[k:]
                shifted_low[:-k] = low[k:]
            max_up = np.fmax(max_up, shifted_high)
            max_down = np.fmin(max_down, shifted_low)
        safe_atr = np.where(np.isfinite(atr) & (atr > 0), atr, np.nan)
        out[f"ret_{h}"] = future_close / close - 1.0
        out[f"atr_ret_{h}"] = (future_close - close) / safe_atr
        out[f"max_up_{h}"] = (max_up - close) / safe_atr
        out[f"max_down_{h}"] = (max_down - close) / safe_atr
    return out


def build_pattern_dataset(
    candles: pd.DataFrame,
    *,
    symbol: str,
    exchange: str,
    timeframe: str,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    min_candles: int = 250,
    include_structures: bool = False,
) -> pd.DataFrame:
    """Build the occurrence table for one candle series.

    Args:
        candles: canonical candles with ``timestamp_open`` and OHLCV.
        symbol: ``BTC/USDT`` style canonical symbol.
        exchange: venue slug (``binance`` etc.).
        timeframe: canonical timeframe string.
        horizons: forward horizons in candles (default 1/2/3/6/12/24).
        min_candles: minimum history before detections are stored.
        include_structures: also emit chart structures (separate category).

    Returns:
        Long dataframe: one row per occurrence with outcomes attached.
    """
    if len(candles) < min_candles:
        raise ValueError(f"need at least {min_candles} candles, got {len(candles)}")
    frame = candles.sort_values("timestamp_open").reset_index(drop=True)
    frame = add_regimes(frame)
    ctx = build_context(frame)

    patterns = tuple(CANDLESTICK_PATTERNS)
    if include_structures:
        patterns += tuple(STRUCTURE_DEFINITIONS)

    outcomes = _forward_outcomes(
        frame["close"].to_numpy(dtype=float),
        frame["high"].to_numpy(dtype=float),
        frame["low"].to_numpy(dtype=float),
        ctx.atr,
        horizons,
    )

    base = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(frame["timestamp_open"], utc=True),
            "symbol": symbol,
            "exchange": exchange,
            "timeframe": timeframe,
            "regime_trend": frame["regime_trend"].to_numpy(),
            "regime_vol": frame["regime_vol"].to_numpy(),
            "close": frame["close"].to_numpy(dtype=float),
            "atr": ctx.atr,
        }
    )
    for key, arr in outcomes.items():
        base[key] = arr

    chunks: list[pd.DataFrame] = []
    for pattern in patterns:
        mask = pattern.detect(ctx)
        mask = np.asarray(mask, dtype=bool).copy()
        mask[: min_candles - 1] = False
        if pattern.candles > 1:
            mask[: pattern.candles - 1] = False
        # An occurrence needs a fully observed 6-candle outcome for the headline
        # hit flag; longer horizons may be NaN at the tail.
        hit_col = np.full(len(frame), np.nan)
        horizon = 6
        ret = outcomes[f"ret_{horizon}"]
        # Neutral patterns have no direction to hit, so their "hit" is the share
        # of positive 6/12-candle returns (an up-rate), reported as such.
        if pattern.direction == "bullish":
            hit_col = (ret > 0).astype(float)
        elif pattern.direction == "bearish":
            hit_col = (ret < 0).astype(float)
        else:
            hit_col = (ret > 0).astype(float)
        hit12 = np.full(len(frame), np.nan)
        ret12 = outcomes["ret_12"]
        if pattern.direction == "bearish":
            hit12 = (ret12 < 0).astype(float)
        else:
            hit12 = (ret12 > 0).astype(float)

        rows = base.loc[mask].copy()
        if rows.empty:
            continue
        idx = np.flatnonzero(mask)
        rows["pattern"] = pattern.key
        rows["category"] = pattern.category
        rows["direction"] = pattern.direction
        rows["candles"] = pattern.candles
        rows["strength"] = _strength(ctx, pattern, mask)[idx]
        rows["direction_hit_6"] = hit_col[idx]
        rows["direction_hit_12"] = hit12[idx]
        chunks.append(rows)

    if not chunks:
        raise ValueError("no pattern occurrences detected")

    result = pd.concat(chunks, ignore_index=True)
    ordered = [c for c in OCCURRENCE_COLUMNS if c in result.columns]
    rest = [c for c in result.columns if c not in ordered]
    result = result[ordered + rest]
    result = result.sort_values(["timestamp", "pattern"]).reset_index(drop=True)
    LOGGER.info(
        "pattern dataset: %d occurrences across %d patterns (%s -> %s)",
        len(result),
        result["pattern"].nunique(),
        result["timestamp"].iloc[0],
        result["timestamp"].iloc[-1],
    )
    return result


def dataset_stats(dataset: pd.DataFrame) -> PatternDatasetStats:
    per_pattern = (
        dataset.groupby("pattern")
        .agg(
            occurrences=("pattern", "size"),
            mean_strength=("strength", "mean"),
            hit_rate_6=("direction_hit_6", "mean"),
            mean_atr_ret_6=("atr_ret_6", "mean"),
        )
        .round(4)
        .to_dict(orient="index")
    )
    return PatternDatasetStats(
        rows=int(len(dataset)),
        patterns=int(dataset["pattern"].nunique()),
        start=str(dataset["timestamp"].min()),
        end=str(dataset["timestamp"].max()),
        per_pattern=per_pattern,
    )
