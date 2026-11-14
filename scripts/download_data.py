"""CLI: download and validate PATRON historical data."""

from __future__ import annotations

import argparse
import sys
from datetime import date

from patron.config import load_config
from patron.data.pipeline import run_data_pipeline
from patron.logging_utils import get_logger

LOGGER = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download + validate BTC/USDT OHLCV")
    parser.add_argument("--config", default=None)
    parser.add_argument("--start", default=None, help="YYYY-MM-DD (default: config data.start)")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD (default: today UTC)")
    parser.add_argument("--timeframes", nargs="+", default=None)
    parser.add_argument("--force", action="store_true", help="re-download existing months")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    result = run_data_pipeline(
        cfg,
        timeframes=args.timeframes,
        start=date.fromisoformat(args.start) if args.start else None,
        end=date.fromisoformat(args.end) if args.end else None,
        overwrite=True if args.force else None,
    )
    for tf, meta in result["manifest"]["timeframes"].items():
        LOGGER.info("%s: %d rows (%s -> %s)", tf, meta["rows"], meta["start"], meta["end"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
