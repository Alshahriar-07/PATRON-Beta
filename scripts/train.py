"""Train PATRON and write checkpoints + training history."""

from __future__ import annotations

import argparse
import json
import sys

from patron.config import load_config
from patron.datasets.loader import prepare_data
from patron.logging_utils import setup_logging
from patron.models.patron_net import build_model, count_parameters
from patron.paths import ensure_dirs
from patron.training.trainer import train


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the PATRON model")
    parser.add_argument("--config", default="config/model_config.yaml")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=int, default=None, help="override max epochs (debug)")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1, help="subsample every split (legacy)")
    parser.add_argument(
        "--train-stride",
        type=int,
        default=None,
        help="subsample training windows only; validation/test keep full size",
    )
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args(argv)

    ensure_dirs()
    setup_logging()
    cfg = load_config(args.config)

    data = prepare_data(cfg, args.dataset, stride=args.stride, train_stride=args.train_stride)
    model = build_model(cfg)
    print(f"model parameters: {count_parameters(model):,}")

    history = train(
        model,
        data,
        cfg,
        run_name=args.run_name,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        num_threads=args.threads,
    )
    print(json.dumps({k: v for k, v in history.to_dict().items() if k != "history"}, indent=2))
    return 0 if history.best_epoch > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
