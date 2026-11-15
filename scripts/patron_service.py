"""PATRON autonomous service CLI.

    python scripts/patron_service.py once            # one idempotent candle cycle
    python scripts/patron_service.py serve           # run continuously
    python scripts/patron_service.py status          # health + latest published files
    python scripts/patron_service.py mature          # mature pending horizons only
    python scripts/patron_service.py rollback        # restore the previous champion
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from patron.logging_utils import setup_logging
from patron.paths import ensure_dirs
from patron.runtime.collector import StaticSource, VenueSource
from patron.runtime.config import load_runtime_config
from patron.runtime.governance import rollback as rollback_champion
from patron.runtime.maturation import mature_forecasts
from patron.runtime.service import PatronRuntime
from patron.runtime.store import RuntimeStore


def _runtime(args) -> PatronRuntime:
    store = RuntimeStore(args.db) if args.db else None
    source = None
    if args.offline_source:
        frame = pd.read_parquet(args.offline_source)
        frame["timestamp_open"] = pd.to_datetime(frame["timestamp_open"], utc=True)
        source = StaticSource(frame=frame, name="offline")
    return PatronRuntime(
        runtime_cfg_path=args.runtime_config,
        model_cfg_path=args.config,
        store=store,
        source=source,
        allow_training=not args.no_train,
    )


def _now(args) -> datetime | None:
    if args.now:
        return pd.Timestamp(args.now, tz="UTC").to_pydatetime()
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PATRON autonomous runtime")
    parser.add_argument("command", choices=["once", "serve", "status", "mature", "rollback", "health"])
    parser.add_argument("--config", default="config/model_config.yaml")
    parser.add_argument("--runtime-config", default=None)
    parser.add_argument("--db", default=None)
    parser.add_argument("--offline-source", default=None, help="parquet replay source for tests/offline")
    parser.add_argument("--now", default=None, help="ISO timestamp override (replay)")
    parser.add_argument("--max-cycles", type=int, default=None)
    parser.add_argument("--no-train", action="store_true")
    parser.add_argument("--version", default=None, help="model version for rollback")
    args = parser.parse_args(argv)

    ensure_dirs()
    setup_logging()

    if args.command == "status":
        cfg = load_runtime_config(args.runtime_config)
        out = Path(cfg.storage.output_dir)
        payload = {"runtime": cfg.to_dict()}
        for name in ("status.json", "performance.json", "health.json", "models.json",
                     "drift.json", "training.json", "latest_forecast.json"):
            path = out / name
            payload[name] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        print(json.dumps(payload, indent=2, default=str))
        return 0

    runtime = _runtime(args)

    if args.command == "mature":
        result = mature_forecasts(runtime.store, runtime.cfg, runtime.model_cfg)
        print(json.dumps({"matured": result["count"], "awaiting": result["awaiting_candles"]}, indent=2))
        return 0

    if args.command == "health":
        print(json.dumps(runtime.health(), indent=2, default=str))
        return 0 if runtime.health()["healthy"] else 1

    if args.command == "rollback":
        try:
            print(json.dumps(rollback_champion(runtime.store, args.version), indent=2))
        except RuntimeError as exc:
            print(f"rollback refused: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.command == "once":
        result = runtime.cycle(_now(args))
        print(json.dumps(result.to_dict(), indent=2))
        return 0 if result.status in {"OK", "NO_OUTPUT"} else 1

    results = runtime.run_forever(max_cycles=args.max_cycles)
    print(json.dumps([result.to_dict() for result in results], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
