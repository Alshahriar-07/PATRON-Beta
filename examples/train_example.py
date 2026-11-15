"""Train a small candidate run and report where its artifacts landed.

    python examples/train_example.py --epochs 1 --train-stride 8
    python examples/train_example.py --dry-run          # validate data/model only

This is a smoke-scale run, not a usable model. Never overwrites the champion:
each run gets its own ``experiments/<name>_<timestamp>/`` directory.
"""

from __future__ import annotations

import argparse
import json
import sys

from patron.config import load_config
from patron.datasets.loader import prepare_data
from patron.logging_utils import setup_logging
from patron.models.patron_net import build_model, count_parameters
from patron.training.trainer import train


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/model_config.yaml")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train-stride", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    setup_logging()
    cfg = load_config(args.config)
    try:
        data = prepare_data(cfg, train_stride=args.train_stride)
    except FileNotFoundError as exc:
        print(f"{exc}\nrun: python scripts/download_data.py && python scripts/build_dataset.py", file=sys.stderr)
        return 2

    model = build_model(cfg)
    print(f"windows: {len(data.bundles.train)} train / {len(data.bundles.val)} val")
    print(f"parameters: {count_parameters(model):,}")
    if args.dry_run:
        print("dry run complete; no training performed")
        return 0

    history = train(
        model,
        data,
        cfg,
        run_name="example",
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print(json.dumps({k: v for k, v in history.to_dict().items() if k != "history"}, indent=2))
    print(f"checkpoint: {history.run_dir / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
