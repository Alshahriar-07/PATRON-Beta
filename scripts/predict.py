"""Run one PATRON inference and emit a schema-valid JSON forecast."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from patron.inference.pipeline import (
    InputValidationError,
    load_predictor,
    predict,
    validate_output,
)
from patron.logging_utils import setup_logging
from patron.paths import REPORT_DIR, ensure_dirs


def _latest_checkpoint() -> Path | None:
    candidates = sorted(Path("experiments").glob("*/best.pt"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PATRON prediction from 96 closed 15m candles")
    parser.add_argument("--input", required=True, help="input JSON (schemas/input.schema.json)")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--warmup", default=None, help="parquet with prior 15m candles")
    parser.add_argument("--output", default=str(REPORT_DIR / "predictions" / "latest.json"))
    args = parser.parse_args(argv)

    ensure_dirs()
    setup_logging()

    checkpoint = Path(args.checkpoint) if args.checkpoint else _latest_checkpoint()
    if checkpoint is None or not checkpoint.exists():
        print("no checkpoint found; train first or pass --checkpoint", file=sys.stderr)
        return 2

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    predictor = load_predictor(checkpoint, warmup_path=args.warmup)
    try:
        result = predict(payload, predictor)
    except InputValidationError as exc:
        print(f"input validation error: {exc}", file=sys.stderr)
        return 3

    validate_output(result)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\nwritten: {out_path}")
    return 0 if result["status"] == "OK" else 4


if __name__ == "__main__":
    raise SystemExit(main())
