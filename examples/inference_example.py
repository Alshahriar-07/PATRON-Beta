"""Run one inference and print the forecast envelope.

    python examples/inference_example.py \
        --checkpoint experiments/patron_v2_20261114-203859/best.pt \
        --input examples/sample_input.json \
        --warmup data/clean/BTCUSDT/15m.parquet

``--warmup`` is optional but recommended: two features look further back than the
96-candle window, so without warm-up the predictor returns ``NO_OUTPUT`` with an
explicit reason instead of inventing feature values.
"""

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", default="examples/sample_input.json")
    parser.add_argument("--warmup", default=None)
    args = parser.parse_args(argv)

    if not Path(args.checkpoint).exists():
        print(f"checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 2

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    predictor = load_predictor(args.checkpoint, warmup_path=args.warmup)
    try:
        result = predict(payload, predictor)
    except InputValidationError as exc:
        print(f"invalid input: {exc}", file=sys.stderr)
        return 3

    if result["status"] != "OK":
        print(json.dumps(result, indent=2))
        print("\nno forecast published: the payload does not satisfy the contract")
        return 4

    validate_output(result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
