"""Phase 1 orchestration: download -> validate -> clean -> Parquet -> manifest."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from patron.config import PatronConfig
from patron.data.downloader import DownloadResult, download_timeframe
from patron.data.htf import BARS_PER_HTF, resample_closed
from patron.data.validate import DataQualityReport, clean_candles, validate_candles
from patron.paths import CLEAN_DIR, MANIFEST_DIR, RAW_DIR, REPORT_DIR, symbol_slug
from patron.utils.io import dataframe_sha256, utc_now_iso, write_json, write_parquet

LOGGER = logging.getLogger(__name__)

CLEAN_SCHEMA_VERSION = "candles-1.0.0"


def _clean_path(symbol: str, timeframe: str, root: Path | None = None) -> Path:
    return Path(root or CLEAN_DIR) / symbol_slug(symbol) / f"{timeframe}.parquet"


def cross_check_htf(
    clean_15m: pd.DataFrame, clean_htf: pd.DataFrame, rule: str, tolerance: float = 1e-9
) -> dict:
    """Verify downloaded HTF candles match an aggregation of the 15m candles."""
    derived = resample_closed(clean_15m, rule).set_index("timestamp_open")
    observed = clean_htf.set_index("timestamp_open")
    common = derived.index.intersection(observed.index)
    if len(common) == 0:
        return {"timeframe": rule, "compared": 0, "mismatches": None, "checked": False}

    mismatches = 0
    max_price_diff = 0.0
    for col in ("open", "high", "low", "close"):
        expected = observed.loc[common, col]
        diff = (derived.loc[common, col] - expected).abs()
        max_price_diff = max(max_price_diff, float(diff.max()))
        threshold = np.maximum(1e-8, tolerance * expected.abs().to_numpy())
        mismatches += int((diff.to_numpy() > threshold).sum())
    return {
        "timeframe": rule,
        "compared": int(len(common)),
        "mismatches": mismatches,
        "max_abs_price_diff": max_price_diff,
        "checked": True,
    }


def run_data_pipeline(
    cfg: PatronConfig,
    timeframes: Iterable[str] | None = None,
    start: date | None = None,
    end: date | None = None,
    overwrite: bool | None = None,
    raw_root: Path | None = None,
    clean_root: Path | None = None,
    report_root: Path | None = None,
) -> dict:
    """Execute the full historical data pipeline and write manifest/report."""
    timeframes = list(timeframes or cfg.data.timeframes)
    symbol = cfg.symbol
    reports: dict[str, DataQualityReport] = {}
    results: dict[str, DownloadResult] = {}
    clean_frames: dict[str, pd.DataFrame] = {}

    for timeframe in timeframes:
        LOGGER.info("downloading %s %s", symbol, timeframe)
        result = download_timeframe(
            cfg,
            timeframe,
            start=start,
            end=end,
            raw_root=raw_root,
            overwrite=overwrite,
        )
        if result.frame.empty:
            raise RuntimeError(f"no data downloaded for {symbol} {timeframe}")
        report = validate_candles(result.frame, symbol, timeframe)
        clean, report = clean_candles(result.frame, report)
        clean["timeframe"] = timeframe
        reports[timeframe] = report
        results[timeframe] = result
        clean_frames[timeframe] = clean

        out = _clean_path(symbol, timeframe, clean_root)
        write_parquet(clean, out)
        LOGGER.info("clean %s -> %s (%d rows) %s", timeframe, out, len(clean), report.summary())

    # HTF consistency: the inference path derives HTF candles from 15m data, so
    # the downloaded 1h/4h tables must agree with that aggregation.
    checks = []
    if "15m" in clean_frames:
        for timeframe, rule in (("1h", "1h"), ("4h", "4h")):
            if timeframe in clean_frames:
                checks.append(cross_check_htf(clean_frames["15m"], clean_frames[timeframe], rule))

    manifest = {
        "project": cfg.project.name,
        "version": cfg.project.version,
        "generated_at": utc_now_iso(),
        "source": {
            "exchange": cfg.market.exchange,
            "market_type": cfg.market.market_type,
            "base_url": cfg.data.base_url,
            "rest_url": cfg.data.rest_base_url,
        },
        "symbol": symbol,
        "schema_version": CLEAN_SCHEMA_VERSION,
        "timeframes": {},
        "htf_cross_checks": checks,
    }
    for timeframe in timeframes:
        clean = clean_frames[timeframe]
        out = _clean_path(symbol, timeframe, clean_root)
        manifest["timeframes"][timeframe] = {
            "rows": int(len(clean)),
            "start": clean["timestamp_open"].iloc[0].isoformat(),
            "end": clean["timestamp_open"].iloc[-1].isoformat(),
            "columns": list(clean.columns),
            "missing_bars": reports[timeframe].missing_bars,
            "gap_count": reports[timeframe].gap_count,
            "rows_rejected": reports[timeframe].rows_rejected,
            "sha256": dataframe_sha256(clean),
            "path": str(out),
            "download_partitions": results[timeframe].partitions,
            "missing_months": results[timeframe].missing_months,
        }

    manifest_path = Path(MANIFEST_DIR) / "dataset_manifest.json"
    write_json(manifest, manifest_path)

    quality = {
        "generated_at": utc_now_iso(),
        "symbol": symbol,
        "reports": {tf: rep.to_dict() for tf, rep in reports.items()},
        "htf_cross_checks": checks,
    }
    report_dir = Path(report_root or REPORT_DIR)
    write_json(quality, report_dir / "data_quality_report.json")
    _write_quality_markdown(quality, report_dir / "data_quality_report.md")

    LOGGER.info("manifest written to %s", manifest_path)
    return {"manifest": manifest, "quality": quality, "frames": clean_frames}


def _write_quality_markdown(quality: dict, path: Path) -> None:
    lines = [
        "# PATRON Data Quality Report",
        "",
        f"Generated: {quality['generated_at']}  ",
        f"Symbol: {quality['symbol']}",
        "",
        "## Per-timeframe summary",
        "",
        "| TF | Rows | Start | End | Gaps | Missing bars | Invalid OHLC | Rejected | Valid |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for tf, rep in quality["reports"].items():
        lines.append(
            f"| {tf} | {rep['rows']} | {str(rep['start'])[:19]} | {str(rep['end'])[:19]} | "
            f"{rep['gap_count']} | {rep['missing_bars']} | {rep['invalid_ohlc']} | "
            f"{rep['rows_rejected']} | {rep['valid']} |"
        )
    lines += ["", "## HTF consistency (15m aggregation vs downloaded)", ""]
    for check in quality.get("htf_cross_checks", []):
        lines.append(
            f"- `{check['timeframe']}`: compared={check['compared']} "
            f"mismatches={check['mismatches']} max_price_diff={check.get('max_abs_price_diff')}"
        )
    lines += ["", "## Notes", ""]
    lines.append(
        "- Gaps are reported and never filled. Windows spanning a gap are excluded "
        "from modelling."
    )
    lines.append(
        "- Invalid rows are rejected (never repaired with invented prices)."
    )
    for tf, rep in quality["reports"].items():
        for warning in rep.get("warnings", []):
            lines.append(f"- {tf}: {warning}")
        for error in rep.get("errors", []):
            lines.append(f"- **ERROR** {tf}: {error}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
