"""Evaluate a checkpoint on the held-out split and print a compact summary.

    python examples/evaluate_example.py --checkpoint experiments/patron_v2_20261114-203859/best.pt

Requires the built dataset (`python scripts/build_dataset.py`). Baselines are
skipped by default because they are slow; pass ``--with-baselines`` to include them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from patron.config import load_config
from patron.datasets.loader import prepare_data
from patron.evaluation.evaluator import run_evaluation
from patron.training.trainer import build_model_from_checkpoint, load_checkpoint


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="config/model_config.yaml")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--with-baselines", action="store_true")
    args = parser.parse_args(argv)

    if not Path(args.checkpoint).exists():
        print(f"checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 2

    cfg = load_config(args.config)
    try:
        data = prepare_data(cfg, args.dataset)
    except FileNotFoundError as exc:
        print(f"{exc}\nrun: python scripts/download_data.py && python scripts/build_dataset.py", file=sys.stderr)
        return 2

    payload = load_checkpoint(args.checkpoint)
    model = build_model_from_checkpoint(payload)
    report = run_evaluation(
        model, data, cfg, Path(args.checkpoint), with_baselines=args.with_baselines
    )
    test = report["test"]
    print(f"checkpoint      {args.checkpoint}")
    print(f"temperature     {report['temperature']:.4f}")
    print(f"test windows    {report['dataset']['test_windows']:,}")
    print(f"barrier         accuracy {test['barrier']['raw']['accuracy']:.4f}  "
          f"macro-F1 {test['barrier']['raw']['macro_f1']:.4f}  "
          f"log loss {test['barrier']['raw']['log_loss']:.4f}")
    print(f"momentum        accuracy {test['momentum']['accuracy']:.4f}  "
          f"macro-F1 {test['momentum']['macro_f1']:.4f}  "
          f"log loss {test['momentum']['log_loss']:.4f}")
    print(f"quantile        pinball {test['quantile']['pinball_mean']:.6f}  "
          f"close coverage {test['quantile']['per_field']['close']['coverage_q10_q90']:.3f}")
    print(f"pattern         mAP {test['pattern']['macro_average_precision']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
