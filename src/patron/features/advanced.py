"""Grouped, causal advanced features with ablation switches.

Every feature is a ratio, a difference or a rolling statistic of bars up to and
including ``t``. Columns are grouped so an ablation can drop a whole family and
measure the effect, and every group has a documented formula in
``FEATURE_GROUPS`` and a causality test in ``tests/test_advanced_features.py``.

The 12 features used by the shipped v2 model remain untouched and are still what
the deployed checkpoint consumes; this module is the v3 feature candidate set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from patron.features.indicators import atr_wilder, ema, rsi_wilder
from patron.patterns.detectors import build_context
from patron.patterns.registry import CANDLESTICK_PATTERNS

EPS = 1e-12

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "price": (
        "ret_1",
        "ret_3",
        "ret_12",
        "ret_96",
        "realized_vol_96",
        "atr_pct_14",
        "dist_high_96",
        "dist_low_96",
    ),
    "candle": (
        "body_ratio",
        "upper_wick_ratio",
        "lower_wick_ratio",
        "candle_dir",
        "consec_bull",
        "consec_bear",
        "pattern_conf_bull",
        "pattern_conf_bear",
    ),
    "volume": (
        "rel_volume_96",
        "volume_z_96",
        "volume_mom_12",
    ),
    "trend": (
        "ema9_gap_atr",
        "ema21_gap_atr",
        "ema50_gap_atr",
        "sma200_gap_atr",
        "adx_14",
        "rsi_14",
        "macd_hist_atr",
        "bb_width_20",
        "vwap_gap_96",
    ),
    "structure": (
        "swing_high_dist_20",
        "swing_low_dist_20",
        "breakout_up_20",
        "breakout_down_20",
        "trend_regime",
        "vol_regime",
    ),
}

ALL_FEATURES: tuple[str, ...] = tuple(f for group in FEATURE_GROUPS.values() for f in group)

#: Docstring per feature (surfaced in the ablation report).
FEATURE_FORMULAS: dict[str, str] = {
    "ret_1": "log(C_t / C_{t-1})",
    "ret_3": "log(C_t / C_{t-3})",
    "ret_12": "log(C_t / C_{t-12})",
    "ret_96": "log(C_t / C_{t-96})",
    "realized_vol_96": "std(log returns, 96 bars)",
    "atr_pct_14": "ATR14 / C_t",
    "dist_high_96": "C_t / max(H, 96) - 1",
    "dist_low_96": "C_t / min(L, 96) - 1",
    "body_ratio": "(C-O)/(H-L)",
    "upper_wick_ratio": "(H-max(O,C))/(H-L)",
    "lower_wick_ratio": "(min(O,C)-L)/(H-L)",
    "candle_dir": "sign(C-O)",
    "consec_bull": "count of trailing consecutive bullish closes (capped at 10)",
    "consec_bear": "count of trailing consecutive bearish closes (capped at 10)",
    "pattern_conf_bull": "max base confidence over bullish patterns detected at t",
    "pattern_conf_bear": "max base confidence over bearish patterns detected at t",
    "rel_volume_96": "V_t / mean(V, 96)",
    "volume_z_96": "(V_t - mean(V,96)) / std(V,96)",
    "volume_mom_12": "mean(V,12) / mean(V,96) - 1",
    "ema9_gap_atr": "(C_t - EMA9) / ATR14",
    "ema21_gap_atr": "(C_t - EMA21) / ATR14",
    "ema50_gap_atr": "(C_t - EMA50) / ATR14",
    "sma200_gap_atr": "(C_t - SMA200) / ATR14",
    "adx_14": "Wilder ADX(14) scaled to [0,1]",
    "rsi_14": "RSI(14)/100",
    "macd_hist_atr": "MACD histogram (12,26,9) / ATR14",
    "bb_width_20": "(upper-lower)/middle of Bollinger(20,2)",
    "vwap_gap_96": "(C_t - rolling VWAP96) / ATR14",
    "swing_high_dist_20": "(C_t - max(H,20)) / ATR14",
    "swing_low_dist_20": "(C_t - min(L,20)) / ATR14",
    "breakout_up_20": "1 if C_t > max(H_{t-20..t-1})",
    "breakout_down_20": "1 if C_t < min(L_{t-20..t-1})",
    "trend_regime": "sign(C_t - SMA200) with a 0.5 ATR dead band",
    "vol_regime": "1 if realized_vol_96 > expanding 70th pct, -1 if < 30th pct",
}


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0.0, np.nan)


def _adx(h: pd.Series, l: pd.Series, c: pd.Series, period: int = 14) -> pd.Series:
    up = h.diff()
    down = -l.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_di = 100 * pd.Series(plus_dm, index=h.index).ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=h.index).ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def _consecutive(close: pd.Series, bullish: bool) -> pd.Series:
    direction = np.sign(close.diff())
    target = 1.0 if bullish else -1.0
    streak = (direction == target).astype(float)
    groups = (streak != streak.shift()).cumsum()
    counts = streak.groupby(groups).cumsum()
    return counts.clip(upper=10).fillna(0.0)


def _pattern_confidence(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    ctx = build_context(frame)
    bull = np.zeros(ctx.n, dtype=np.float32)
    bear = np.zeros(ctx.n, dtype=np.float32)
    for pattern in CANDLESTICK_PATTERNS:
        mask = pattern.detect(ctx)
        if pattern.direction == "bullish":
            bull = np.maximum(bull, np.where(mask, pattern.base_confidence, 0.0))
        elif pattern.direction == "bearish":
            bear = np.maximum(bear, np.where(mask, pattern.base_confidence, 0.0))
    return bull, bear


def build_advanced_features(
    frame: pd.DataFrame,
    groups: tuple[str, ...] | list[str] | None = None,
    include_pattern_confidence: bool = True,
) -> pd.DataFrame:
    """Return a frame with the requested feature groups + ``feature_valid``.

    Args:
        frame: canonical candles (``timestamp_open``, OHLCV).
        groups: subset of :data:`FEATURE_GROUPS`; default all.
        include_pattern_confidence: computing pattern confidence needs the
            registry context; disable for fast unit tests.
    """
    selected = tuple(groups or FEATURE_GROUPS)
    unknown = set(selected) - set(FEATURE_GROUPS)
    if unknown:
        raise KeyError(f"unknown feature groups: {sorted(unknown)}")

    df = frame.copy()
    df["timestamp_open"] = pd.to_datetime(df["timestamp_open"], utc=True)
    df = df.sort_values("timestamp_open").reset_index(drop=True)
    o, h, l, c, v = (df[col].astype("float64") for col in ("open", "high", "low", "close", "volume"))

    out = pd.DataFrame({"timestamp_open": df["timestamp_open"]})
    atr = atr_wilder(df, 14)
    atr_safe = atr.replace(0.0, np.nan)

    if "price" in selected:
        log_close = np.log(c)
        out["ret_1"] = log_close.diff()
        out["ret_3"] = log_close.diff(3)
        out["ret_12"] = log_close.diff(12)
        out["ret_96"] = log_close.diff(96)
        out["realized_vol_96"] = out["ret_1"].rolling(96, min_periods=96).std()
        out["atr_pct_14"] = _safe_div(atr, c)
        out["dist_high_96"] = _safe_div(c, h.rolling(96, min_periods=96).max()) - 1.0
        out["dist_low_96"] = _safe_div(c, l.rolling(96, min_periods=96).min()) - 1.0

    if "candle" in selected:
        rng = (h - l).replace(0.0, np.nan)
        out["body_ratio"] = (c - o) / rng
        out["upper_wick_ratio"] = (h - np.maximum(o, c)) / rng
        out["lower_wick_ratio"] = (np.minimum(o, c) - l) / rng
        out["candle_dir"] = np.sign(c - o)
        out["consec_bull"] = _consecutive(c, True)
        out["consec_bear"] = _consecutive(c, False)

    if "volume" in selected:
        vol_mean = v.rolling(96, min_periods=96).mean()
        vol_std = v.rolling(96, min_periods=96).std()
        out["rel_volume_96"] = _safe_div(v, vol_mean)
        out["volume_z_96"] = _safe_div(v - vol_mean, vol_std)
        out["volume_mom_12"] = _safe_div(v.rolling(12, min_periods=12).mean(), vol_mean) - 1.0

    if "trend" in selected:
        ema9 = ema(c, 9)
        ema21 = ema(c, 21)
        ema50 = ema(c, 50)
        sma200 = c.rolling(200, min_periods=200).mean()
        out["ema9_gap_atr"] = (c - ema9) / atr_safe
        out["ema21_gap_atr"] = (c - ema21) / atr_safe
        out["ema50_gap_atr"] = (c - ema50) / atr_safe
        out["sma200_gap_atr"] = (c - sma200) / atr_safe
        out["adx_14"] = _adx(h, l, c, 14) / 100.0
        out["rsi_14"] = rsi_wilder(df, 14) / 100.0
        macd = ema(c, 12) - ema(c, 26)
        signal = ema(macd, 9)
        out["macd_hist_atr"] = (macd - signal) / atr_safe
        mid = c.rolling(20, min_periods=20).mean()
        std = c.rolling(20, min_periods=20).std()
        out["bb_width_20"] = _safe_div(4.0 * std, mid)
        typical = (h + l + c) / 3.0
        vwap = _safe_div((typical * v).rolling(96, min_periods=96).sum(), v.rolling(96, min_periods=96).sum())
        out["vwap_gap_96"] = (c - vwap) / atr_safe

    if "structure" in selected:
        prior_high = h.rolling(20, min_periods=20).max().shift(1)
        prior_low = l.rolling(20, min_periods=20).min().shift(1)
        out["swing_high_dist_20"] = (c - prior_high) / atr_safe
        out["swing_low_dist_20"] = (c - prior_low) / atr_safe
        out["breakout_up_20"] = (c > prior_high).astype("float64")
        out["breakout_down_20"] = (c < prior_low).astype("float64")
        sma200 = c.rolling(200, min_periods=200).mean()
        gap = (c - sma200) / atr_safe
        out["trend_regime"] = np.sign(np.where(gap.abs() < 0.5, 0.0, gap))
        vol = out.get("realized_vol_96")
        if vol is None:
            vol = np.log(c).diff().rolling(96, min_periods=96).std()
        low_q = vol.expanding(min_periods=96).quantile(0.30).shift(1)
        high_q = vol.expanding(min_periods=96).quantile(0.70).shift(1)
        out["vol_regime"] = np.select([vol > high_q, vol < low_q], [1.0, -1.0], default=0.0)

    if "candle" in selected and include_pattern_confidence:
        bull, bear = _pattern_confidence(df)
        out["pattern_conf_bull"] = bull
        out["pattern_conf_bear"] = bear

    feature_columns = [f for group in FEATURE_GROUPS.values() for f in group if f in out.columns]
    out = out[["timestamp_open", *feature_columns]].copy()
    out["feature_valid"] = np.isfinite(out[feature_columns].to_numpy(dtype="float64")).all(axis=1)
    return out


def group_feature_columns(groups: tuple[str, ...] | list[str]) -> list[str]:
    return [f for group in groups for f in FEATURE_GROUPS.get(group, ())]
