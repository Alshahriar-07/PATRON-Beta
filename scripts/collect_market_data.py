"""Collect multi-venue / multi-asset OHLCV into per-venue parquet partitions."""

from __future__ import annotations

import argparse
import json
import sys

from patron.data.collect import collect_many, rebuild_manifest
from patron.data.markets import AUXILIARY_ASSETS, VENUE_ORDER, all_timeframes
from patron.logging_utils import setup_logging
from patron.paths import MANIFEST_DIR, ensure_dirs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PATRON multi-venue market data collection")
    parser.add_argument("--exchanges", nargs="+", default=list(VENUE_ORDER))
    parser.add_argument("--assets", nargs="+", default=["BTC"])
    parser.add_argument("--all-assets", action="store_true", help="use the 10 tracked assets")
    parser.add_argument("--timeframe", default="15m", choices=list(all_timeframes()))
    parser.add_argument("--start", default="2023-11-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--pause", type=float, default=None, help="per-request pause override")
    parser.add_argument("--manifest-name", default=None)
    parser.add_argument("--refresh", action="store_true", help="refetch even if the partition exists")
    parser.add_argument(
        "--rebuild-manifest",
        action="store_true",
        help="rewrite the named manifest from on-disk partitions only (no network)",
    )
    args = parser.parse_args(argv)

    ensure_dirs()
    setup_logging()
    if args.rebuild_manifest:
        if not args.manifest_name:
            parser.error("--rebuild-manifest requires --manifest-name")
        wanted = list(AUXILIARY_ASSETS) if args.all_assets else [a.upper() for a in args.assets]
        rebuilt = rebuild_manifest(args.manifest_name, args.timeframe, wanted)
        print(json.dumps(rebuilt["summary"], indent=2))
        print(f"manifest: {MANIFEST_DIR / args.manifest_name}")
        return 0
    assets = list(AUXILIARY_ASSETS) if args.all_assets else [a.upper() for a in args.assets]
    manifest = collect_many(
        exchanges=args.exchanges,
        assets=assets,
        timeframe=args.timeframe,
        start=args.start,
        end=args.end,
        pause=args.pause,
        manifest_name=args.manifest_name,
        refresh=args.refresh,
    )
    print(json.dumps(manifest["summary"], indent=2))
    print(f"manifest: {MANIFEST_DIR}")
    for result in manifest["results"]:
        if result["status"] in ("failed", "empty"):
            print(f"  {result['exchange']} {result['asset']} {result['timeframe']}: "
                  f"{result['status']} — {result['error']}", file=sys.stderr)
    return 0 if manifest["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
