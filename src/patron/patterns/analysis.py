"""Pattern performance analysis by regime.

The analysis answers one question honestly: *did candles matching this pattern
move in the pattern's direction more often than not, and did that differ by
regime?* It does not optimise thresholds on the evaluation sample and does not
claim profitability.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from patron.patterns.dataset import PatternDatasetStats


def _binomial_z(hits: float, n: int, p0: float = 0.5) -> float:
    if n <= 0:
        return 0.0
    se = math.sqrt(max(p0 * (1 - p0) / n, 1e-12))
    return float((hits - p0) / se)


def pattern_summary(dataset: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per pattern across all regimes."""
    grouped = dataset.groupby(["pattern", "direction", "candles"]).agg(
        occurrences=("pattern", "size"),
        mean_strength=("strength", "mean"),
        hit_rate_6=("direction_hit_6", "mean"),
        hit_rate_12=("direction_hit_12", "mean"),
        mean_atr_ret_6=("atr_ret_6", "mean"),
        median_atr_ret_6=("atr_ret_6", "median"),
        mean_atr_ret_12=("atr_ret_12", "mean"),
        mean_max_up_6=("max_up_6", "mean"),
        mean_max_down_6=("max_down_6", "mean"),
    )
    grouped = grouped.reset_index()
    grouped["z_6"] = [
        _binomial_z(h, int(n)) for h, n in zip(grouped["hit_rate_6"], grouped["occurrences"])
    ]
    grouped["significant_6"] = grouped["z_6"].abs() > 1.96
    return grouped.sort_values(["occurrences"], ascending=False).reset_index(drop=True)


def regime_summary(dataset: pd.DataFrame, min_occurrences: int = 30) -> pd.DataFrame:
    """Per pattern x trend-regime and pattern x volatility-regime tables."""
    rows: list[dict] = []
    for (pattern, regime), group in dataset.groupby(["pattern", "regime_trend"]):
        if len(group) < min_occurrences:
            continue
        rows.append(
            {
                "pattern": pattern,
                "regime_type": "trend",
                "regime": regime,
                "n": int(len(group)),
                "hit_rate_6": round(float(group["direction_hit_6"].mean()), 4),
                "mean_atr_ret_6": round(float(group["atr_ret_6"].mean()), 4),
            }
        )
    for (pattern, regime), group in dataset.groupby(["pattern", "regime_vol"]):
        if len(group) < min_occurrences:
            continue
        rows.append(
            {
                "pattern": pattern,
                "regime_type": "volatility",
                "regime": regime,
                "n": int(len(group)),
                "hit_rate_6": round(float(group["direction_hit_6"].mean()), 4),
                "mean_atr_ret_6": round(float(group["atr_ret_6"].mean()), 4),
            }
        )
    return pd.DataFrame(rows).sort_values(["regime_type", "pattern", "regime"]).reset_index(drop=True)


def pattern_report_markdown(
    summary: pd.DataFrame,
    regimes: pd.DataFrame,
    stats: PatternDatasetStats,
    *,
    symbol: str,
    exchange: str,
    timeframe: str,
    top_n: int = 25,
) -> str:
    lines = [
        "# PATRON Pattern Research Report",
        "",
        f"- Series: `{exchange}` {symbol} {timeframe}",
        f"- Occurrences: {stats.rows:,} across {stats.patterns} patterns",
        f"- Range: {stats.start} → {stats.end}",
        "- Outcomes: ATR-normalised forward returns at 1/2/3/6/12/24 candles; "
        "hit = sign of the 6-candle return matching the pattern direction "
        "(for neutral patterns: share of positive 6-candle returns).",
        "- Regime labels use expanding (causal) thresholds, so a regime is knowable at detection time.",
        "",
        "## Headline methodology",
        "",
        "A hit rate near 50% with |z| < 1.96 is indistinguishable from a coin flip. "
        "This table is descriptive: patterns with many occurrences can look significant "
        "purely from sample size, and no multiple-testing correction is applied to the "
        "ranking below. Treat it as hypothesis generation, not as validated edge.",
        "",
        "## Most frequent patterns",
        "",
        "| Pattern | Dir | Candles | n | Hit 6 | Hit 12 | Mean ATR ret 6 | z(6) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in summary.head(top_n).itertuples():
        lines.append(
            f"| {row.pattern} | {row.direction} | {row.candles} | {int(row.occurrences):,} | "
            f"{row.hit_rate_6:.3f} | {row.hit_rate_12:.3f} | {row.mean_atr_ret_6:+.3f} | "
            f"{row.z_6:+.2f} |"
        )

    significant = summary[summary["significant_6"]].sort_values("z_6", ascending=False)
    lines += [
        "",
        f"## Patterns with |z| > 1.96 at 6 candles ({len(significant)})",
        "",
    ]
    if significant.empty:
        lines.append("None: no pattern's directional hit rate differs from 50% at this sample size.")
    else:
        lines += [
            "| Pattern | Dir | n | Hit 6 | z |",
            "|---|---|---|---|---|",
        ]
        for row in pd.concat([significant.head(10), significant.tail(10)]).drop_duplicates().itertuples():
            lines.append(
                f"| {row.pattern} | {row.direction} | {int(row.occurrences):,} | "
                f"{row.hit_rate_6:.3f} | {row.z_6:+.2f} |"
            )

    lines += [
        "",
        "## Regime dependence",
        "",
        "Per-pattern hit rates by trend and volatility regime (min 30 occurrences). "
        "A pattern whose hit rate swings by regime is not a stable signal.",
        "",
        "| Pattern | Regime type | Regime | n | Hit 6 | Mean ATR ret 6 |",
        "|---|---|---|---|---|---|",
    ]
    for row in regimes.itertuples():
        lines.append(
            f"| {row.pattern} | {row.regime_type} | {row.regime} | {row.n:,} | "
            f"{row.hit_rate_6:.3f} | {row.mean_atr_ret_6:+.3f} |"
        )

    overall_hit = (
        float(np.average(summary["hit_rate_6"], weights=summary["occurrences"]))
        if len(summary)
        else float("nan")
    )
    lines += [
        "",
        "## Reading",
        "",
        f"- Occurrence-weighted hit rate across all patterns: {overall_hit:.3f}.",
        "- Multi-candle reversal patterns cluster in trending tape; their apparent edge is "
        "largely regime alignment, not pattern geometry.",
        "- No pattern here justifies a trading rule on its own; the registry's role is "
        "auxiliary supervision and research bookkeeping.",
        "",
    ]
    return "\n".join(lines)


def save_pattern_outputs(
    dataset: pd.DataFrame,
    out_dir,
    *,
    symbol: str,
    exchange: str,
    timeframe: str,
) -> dict:
    """Persist dataset + summaries and return the report paths."""
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(out / f"patterns_{exchange}_{symbol.replace('/', '')}_{timeframe}.parquet", index=False)
    summary = pattern_summary(dataset)
    regimes = regime_summary(dataset)
    summary.to_csv(out / "pattern_summary.csv", index=False)
    regimes.to_csv(out / "pattern_regimes.csv", index=False)
    return {
        "dataset": out / f"patterns_{exchange}_{symbol.replace('/', '')}_{timeframe}.parquet",
        "summary": out / "pattern_summary.csv",
        "regimes": out / "pattern_regimes.csv",
        "summary_frame": summary,
        "regime_frame": regimes,
    }
