"""Evaluate a PATRON checkpoint on the held-out test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from patron.config import load_config
from patron.datasets.loader import prepare_data
from patron.evaluation.evaluator import report_to_markdown, run_evaluation
from patron.logging_utils import setup_logging
from patron.paths import REPORT_DIR, ensure_dirs
from patron.training.trainer import build_model_from_checkpoint, load_checkpoint
from patron.utils.io import write_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a PATRON checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="config/model_config.yaml")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--out", default=str(REPORT_DIR / "evaluation_report.json"))
    parser.add_argument("--no-baselines", action="store_true")
    args = parser.parse_args(argv)

    ensure_dirs()
    setup_logging()
    cfg = load_config(args.config)

    payload = load_checkpoint(args.checkpoint)
    model = build_model_from_checkpoint(payload)
    data = prepare_data(cfg, args.dataset)

    report = run_evaluation(
        model, data, cfg, Path(args.checkpoint), with_baselines=not args.no_baselines
    )
    out_path = Path(args.out)
    write_json(report, out_path)
    md_path = out_path.with_suffix(".md")
    md_path.write_text(report_to_markdown(report), encoding="utf-8")
    print(json.dumps(
        {
            "test_barrier_accuracy": report["test_raw"]["barrier"]["raw"]["accuracy"],
            "test_barrier_macro_f1": report["test_raw"]["barrier"]["raw"]["macro_f1"],
            "test_momentum_accuracy": report["test_raw"]["momentum"]["accuracy"],
            "test_pinball": report["test_raw"]["quantile"]["pinball_mean"],
            "temperature": report["temperature"],
        },
        indent=2,
    ))
    print(f"report: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
