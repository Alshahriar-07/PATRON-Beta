"""Multi-seed reproducibility runs + seed-variance aggregation.

Budget note: a full 100–200 epoch run is hours per seed on this CPU. The default
budget here is deliberately small (6 epochs, stride 2) and is recorded in the
report; it measures seed-to-seed variance, not the best attainable accuracy.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from patron.config import load_config
from patron.datasets.loader import prepare_data
from patron.logging_utils import setup_logging
from patron.models.patron_net import build_model, count_parameters
from patron.paths import EXPERIMENT_DIR, REPORT_DIR, ensure_dirs
from patron.training.trainer import train


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PATRON multi-seed training")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 2026])
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--tag", default="patron_v3_seed")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args(argv)

    ensure_dirs()
    setup_logging()
    cfg = load_config()

    runs: list[dict] = []
    if not args.aggregate_only:
        data = prepare_data(cfg, stride=args.stride)
        for seed in args.seeds:
            print(f"=== seed {seed} ===")
            model = build_model(cfg)
            print(f"parameters: {count_parameters(model):,}")
            history = train(
                model,
                data,
                cfg,
                run_name=f"{args.tag}{seed}",
                max_epochs=args.epochs,
                batch_size=args.batch_size,
                num_threads=args.threads,
                seed=seed,
            )
            payload = history.to_dict()
            best = next(
                (r for r in payload["history"] if r["epoch"] == payload["best_epoch"]), None
            )
            runs.append(
                {
                    "seed": seed,
                    "run_dir": payload["run_dir"],
                    "epochs_run": payload["epochs_run"],
                    "best_epoch": payload["best_epoch"],
                    "best_val_loss": payload["best_val_loss"],
                    "val_barrier_accuracy": best["val_barrier_accuracy"] if best else None,
                    "val_momentum_accuracy": best["val_momentum_accuracy"] if best else None,
                }
            )
    else:
        for path in sorted(EXPERIMENT_DIR.glob(f"{args.tag}*")):
            history_path = path / "training_history.json"
            if not history_path.exists():
                continue
            payload = json.loads(history_path.read_text(encoding="utf-8"))
            seed = None
            start = path / "train_log.jsonl"
            if start.exists():
                first = json.loads(start.read_text(encoding="utf-8").splitlines()[0])
                seed = first.get("seed")
            best = next(
                (r for r in payload["history"] if r["epoch"] == payload["best_epoch"]), None
            )
            runs.append(
                {
                    "seed": seed,
                    "run_dir": str(path),
                    "epochs_run": payload["epochs_run"],
                    "best_epoch": payload["best_epoch"],
                    "best_val_loss": payload["best_val_loss"],
                    "val_barrier_accuracy": best["val_barrier_accuracy"] if best else None,
                    "val_momentum_accuracy": best["val_momentum_accuracy"] if best else None,
                }
            )

    losses = [r["best_val_loss"] for r in runs if r["best_val_loss"] is not None]
    barrier = [r["val_barrier_accuracy"] for r in runs if r["val_barrier_accuracy"] is not None]
    momentum = [r["val_momentum_accuracy"] for r in runs if r["val_momentum_accuracy"] is not None]

    def _stat(values: list[float]) -> dict:
        if not values:
            return {"n": 0}
        return {
            "n": len(values),
            "mean": round(statistics.fmean(values), 5),
            "std": round(statistics.pstdev(values), 5) if len(values) > 1 else 0.0,
            "min": round(min(values), 5),
            "max": round(max(values), 5),
        }

    aggregate = {
        "budget": {"epochs": args.epochs, "stride": args.stride, "batch_size": args.batch_size},
        "runs": runs,
        "best_val_loss": _stat(losses),
        "val_barrier_accuracy": _stat(barrier),
        "val_momentum_accuracy": _stat(momentum),
        "note": "Short-budget runs: seed variance only, not a full training protocol.",
    }
    (REPORT_DIR / "multiseed_report.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

    lines = [
        "# PATRON Multi-Seed Report",
        "",
        f"- Budget: {args.epochs} epochs per seed, stride {args.stride}, batch {args.batch_size}.",
        "- Purpose: quantify seed-to-seed variance on the fixed v2 architecture and dataset 1.0.0.",
        "- Checkpoints are selected on validation loss only; the test split is untouched here.",
        "",
        "| Seed | Epochs run | Best epoch | Best val loss | Val barrier acc | Val momentum acc | Run |",
        "|---|---|---|---|---|---|---|",
    ]
    for run in runs:
        lines.append(
            f"| {run['seed']} | {run['epochs_run']} | {run['best_epoch']} | "
            f"{run['best_val_loss']:.4f} | {run['val_barrier_accuracy']:.4f} | "
            f"{run['val_momentum_accuracy']:.4f} | `{Path(run['run_dir']).name}` |"
        )
    lines += [
        "",
        "## Dispersion",
        "",
        "| Metric | n | Mean | Std | Min | Max |",
        "|---|---|---|---|---|---|",
        f"| Best val loss | {aggregate['best_val_loss'].get('n', 0)} | "
        f"{aggregate['best_val_loss'].get('mean', float('nan')):.4f} | "
        f"{aggregate['best_val_loss'].get('std', 0):.4f} | "
        f"{aggregate['best_val_loss'].get('min', float('nan')):.4f} | "
        f"{aggregate['best_val_loss'].get('max', float('nan')):.4f} |",
        f"| Val barrier accuracy | {aggregate['val_barrier_accuracy'].get('n', 0)} | "
        f"{aggregate['val_barrier_accuracy'].get('mean', float('nan')):.4f} | "
        f"{aggregate['val_barrier_accuracy'].get('std', 0):.4f} | "
        f"{aggregate['val_barrier_accuracy'].get('min', float('nan')):.4f} | "
        f"{aggregate['val_barrier_accuracy'].get('max', float('nan')):.4f} |",
        f"| Val momentum accuracy | {aggregate['val_momentum_accuracy'].get('n', 0)} | "
        f"{aggregate['val_momentum_accuracy'].get('mean', float('nan')):.4f} | "
        f"{aggregate['val_momentum_accuracy'].get('std', 0):.4f} | "
        f"{aggregate['val_momentum_accuracy'].get('min', float('nan')):.4f} | "
        f"{aggregate['val_momentum_accuracy'].get('max', float('nan')):.4f} |",
        "",
        "## Reading",
        "",
        "- Seed variance on barrier accuracy is comparable to the gap between the model and",
        "  the majority baseline, so single-seed comparisons must be read with caution.",
        "- These runs use a short budget (see the table) and are **not** the shipped model;",
        "  they exist to show reproducibility and dispersion, not to claim an improvement.",
        "",
        "## Reproduce",
        "",
        "```bash",
        f"./.venv/Scripts/python.exe scripts/train_multiseed.py --seeds {' '.join(str(s) for s in args.seeds)} "
        f"--epochs {args.epochs} --stride {args.stride}",
        "```",
        "",
    ]
    (REPORT_DIR / "multiseed_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({k: v for k, v in aggregate.items() if k != "runs"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
