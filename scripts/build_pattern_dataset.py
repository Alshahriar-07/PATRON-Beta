"""Build the pattern-occurrence dataset and the pattern research report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from patron.logging_utils import setup_logging
from patron.paths import MANIFEST_DIR, PATTERN_DIR, REPORT_DIR, ensure_dirs
from patron.patterns.analysis import pattern_report_markdown, save_pattern_outputs
from patron.patterns.dataset import build_pattern_dataset, dataset_stats
from patron.patterns.registry import registry_summary
from patron.patterns.structures import STRUCTURE_DEFINITIONS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PATRON pattern dataset builder")
    parser.add_argument("--candles", default="data/clean/BTCUSDT/15m.parquet")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--timeframe", default="15m")
    parser.add_argument("--include-structures", action="store_true", default=True)
    parser.add_argument("--no-structures", dest="include_structures", action="store_false")
    parser.add_argument("--out", default=str(PATTERN_DIR / "v1"))
    args = parser.parse_args(argv)

    ensure_dirs()
    setup_logging()
    candles = pd.read_parquet(args.candles)
    dataset = build_pattern_dataset(
        candles,
        symbol=args.symbol,
        exchange=args.exchange,
        timeframe=args.timeframe,
        include_structures=args.include_structures,
    )
    outputs = save_pattern_outputs(
        dataset,
        Path(args.out),
        symbol=args.symbol,
        exchange=args.exchange,
        timeframe=args.timeframe,
    )
    stats = dataset_stats(dataset)
    report = pattern_report_markdown(
        outputs["summary_frame"],
        outputs["regime_frame"],
        stats,
        symbol=args.symbol,
        exchange=args.exchange,
        timeframe=args.timeframe,
    )
    report_path = REPORT_DIR / "pattern_research_report.md"
    report_path.write_text(report, encoding="utf-8")

    summary = {
        "dataset": str(outputs["dataset"]),
        "report": str(report_path),
        "registry": registry_summary(),
        "structures": len(STRUCTURE_DEFINITIONS),
        "stats": stats.to_dict(),
        "regimes": {
            "trend": dataset["regime_trend"].value_counts().to_dict(),
            "vol": dataset["regime_vol"].value_counts().to_dict(),
        },
    }
    manifest_path = MANIFEST_DIR / "pattern_dataset_v1.json"
    manifest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["stats"], indent=2)[:800])
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
