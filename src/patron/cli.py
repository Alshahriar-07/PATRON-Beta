"""Unified PATRON command line: ``python -m patron <command>``.

Commands are thin wrappers over the library, so behaviour is identical to the
``scripts/`` entry points but available wherever the package is installed.
Exit codes: 0 success, 1 check/runtime failure, 2 missing input, 3 invalid input,
4 model produced no usable output.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from patron import __version__
from patron.paths import (
    DEFAULT_CONFIG_PATH,
    EXPERIMENT_DIR,
    FEATURE_DIR,
    PROJECT_ROOT,
    REPORT_DIR,
    ensure_dirs,
    symbol_slug,
)
from patron.utils.io import file_sha256


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config", default=str(DEFAULT_CONFIG_PATH), help="model config YAML (default: %(default)s)"
    )


# --------------------------------------------------------------------------- health
def cmd_health(args) -> int:
    from patron.health import format_report, run_health

    result = run_health(
        root=PROJECT_ROOT,
        config_path=Path(args.config),
        dataset_path=Path(args.dataset),
        checkpoint_path=Path(args.checkpoint),
        database_path=Path(args.db),
    )
    print(json.dumps(result, indent=2) if args.json else format_report(result))
    if args.strict and result["status"] == "WARN":
        return 1
    return 0 if result["status"] != "FAIL" else 1


# ---------------------------------------------------------------------------- train
def cmd_train(args) -> int:
    from patron.config import load_config
    from patron.datasets.loader import prepare_data
    from patron.logging_utils import setup_logging
    from patron.models.patron_net import build_model, count_parameters
    from patron.training.trainer import train

    ensure_dirs()
    setup_logging()
    cfg = load_config(args.config)
    data = prepare_data(cfg, args.dataset, train_stride=args.train_stride)
    model = build_model(cfg)
    params = count_parameters(model)
    print(
        f"train windows: {len(data.bundles.train)}, val: {len(data.bundles.val)}, "
        f"test: {len(data.bundles.test)}; parameters: {params:,}"
    )
    if params != 272_618:
        print("model parameter count does not match the frozen contract", file=sys.stderr)
        return 1
    if args.dry_run:
        print("dry run: data and model are valid, no training performed")
        return 0
    started = _utc_now()
    history = train(
        model,
        data,
        cfg,
        run_name=args.run_name,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        num_threads=args.threads,
        seed=args.seed,
    )
    _write_experiment_record(
        history.run_dir,
        cfg,
        dataset_path=Path(args.dataset) if args.dataset else FEATURE_DIR / f"{symbol_slug(cfg.symbol)}_dataset.parquet",
        seed=args.seed if args.seed is not None else cfg.training.random_seed,
        train_stride=args.train_stride or 1,
        batch_size=args.batch_size or cfg.training.batch_size,
        started=started,
    )
    print(
        json.dumps(
            {k: v for k, v in history.to_dict().items() if k != "history"}, indent=2
        )
    )
    return 0 if history.best_epoch > 0 else 1


# ------------------------------------------------------------------------- evaluate
def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_experiment_record(
    run_dir,
    cfg,
    dataset_path: Path,
    seed: int,
    train_stride: int,
    batch_size: int,
    started: str,
) -> None:
    """Persist a machine-readable record of everything that identifies the run."""
    import platform as _platform

    import torch

    if run_dir is None:
        return
    record = {
        "experiment_id": Path(run_dir).name,
        "model_version": cfg.project.version,
        "feature_version": "features-1.0.0",
        "dataset_path": str(dataset_path),
        "dataset_sha256": file_sha256(dataset_path) if dataset_path.exists() else None,
        "optimizer": cfg.training.optimizer,
        "learning_rate": cfg.training.learning_rate,
        "weight_decay": cfg.training.weight_decay,
        "scheduler": "ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-5)",
        "batch_size": batch_size,
        "train_stride": train_stride,
        "loss_weights": cfg.training.loss_weights.model_dump(),
        "max_epochs": cfg.training.max_epochs,
        "early_stopping_patience": cfg.training.early_stopping_patience,
        "gradient_clip_norm": cfg.training.gradient_clip_norm,
        "seed": seed,
        "started_at": started,
        "finished_at": _utc_now(),
        "hardware": {
            "platform": _platform.platform(),
            "processor": _platform.processor() or _platform.machine(),
            "python": _platform.python_version(),
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "device": "cpu",
        },
    }
    best = Path(run_dir) / "best.pt"
    if best.exists():
        record["checkpoint"] = str(best)
        record["checkpoint_sha256"] = file_sha256(best)
    (Path(run_dir) / "experiment.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


def cmd_evaluate(args) -> int:
    from patron.config import load_config
    from patron.datasets.loader import prepare_data
    from patron.evaluation.evaluator import report_to_markdown, run_evaluation
    from patron.logging_utils import setup_logging
    from patron.training.trainer import build_model_from_checkpoint, load_checkpoint
    from patron.utils.io import write_json

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        print(f"checkpoint not found: {checkpoint}", file=sys.stderr)
        return 2
    setup_logging()
    cfg = load_config(args.config)
    payload = load_checkpoint(checkpoint)
    model = build_model_from_checkpoint(payload)
    data = prepare_data(cfg, args.dataset)
    report = run_evaluation(model, data, cfg, checkpoint, with_baselines=not args.no_baselines)
    out_path = Path(args.out)
    write_json(report, out_path)
    out_path.with_suffix(".md").write_text(report_to_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "test_barrier_accuracy": report["test_raw"]["barrier"]["raw"]["accuracy"],
                "test_barrier_macro_f1": report["test_raw"]["barrier"]["raw"]["macro_f1"],
                "test_momentum_accuracy": report["test_raw"]["momentum"]["accuracy"],
                "test_pinball": report["test_raw"]["quantile"]["pinball_mean"],
                "temperature": report["temperature"],
                "report": str(out_path),
            },
            indent=2,
        )
    )
    return 0


# -------------------------------------------------------------------------- predict
def _latest_checkpoint() -> Path | None:
    candidates = sorted(EXPERIMENT_DIR.glob("*/best.pt"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def cmd_predict(args) -> int:
    from patron.inference.pipeline import (
        InputValidationError,
        load_predictor,
        predict,
        validate_output,
    )
    from patron.logging_utils import setup_logging

    ensure_dirs()
    setup_logging()
    checkpoint = Path(args.checkpoint) if args.checkpoint else _latest_checkpoint()
    if checkpoint is None or not checkpoint.exists():
        print("no checkpoint found; train one or pass --checkpoint", file=sys.stderr)
        return 2
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"input not found: {input_path}", file=sys.stderr)
        return 2
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"input is not valid JSON: {exc}", file=sys.stderr)
        return 3
    predictor = load_predictor(checkpoint, warmup_path=args.warmup)
    try:
        result = predict(payload, predictor)
        validate_output(result)
    except InputValidationError as exc:
        print(f"input validation error: {exc}", file=sys.stderr)
        return 3
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\nwritten: {out_path}")
    return 0 if result["status"] == "OK" else 4


# -------------------------------------------------------------------------- runtime
def cmd_runtime(args) -> int:
    import pandas as pd

    from patron.logging_utils import setup_logging
    from patron.runtime.collector import StaticSource
    from patron.runtime.config import load_runtime_config
    from patron.runtime.governance import rollback as rollback_champion
    from patron.runtime.maturation import mature_forecasts
    from patron.runtime.service import PatronRuntime
    from patron.runtime.store import RuntimeStore

    ensure_dirs()
    setup_logging()

    if args.command == "status":
        cfg = load_runtime_config(args.runtime_config)
        out = Path(cfg.storage.output_dir)
        payload = {"runtime": cfg.to_dict()}
        for name in (
            "status.json",
            "performance.json",
            "health.json",
            "models.json",
            "drift.json",
            "training.json",
            "latest_forecast.json",
        ):
            path = out / name
            payload[name] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        print(json.dumps(payload, indent=2, default=str))
        return 0

    source = None
    if args.offline_source:
        frame = pd.read_parquet(args.offline_source)
        frame["timestamp_open"] = pd.to_datetime(frame["timestamp_open"], utc=True)
        source = StaticSource(frame=frame, name="offline")
    runtime = PatronRuntime(
        runtime_cfg_path=args.runtime_config,
        model_cfg_path=args.config,
        store=RuntimeStore(args.db) if args.db else None,
        source=source,
        allow_training=not args.no_train,
    )

    if args.command == "health":
        payload = runtime.health()
        print(json.dumps(payload, indent=2, default=str))
        return 0 if payload["healthy"] else 1
    if args.command == "mature":
        result = mature_forecasts(runtime.store, runtime.cfg, runtime.model_cfg)
        print(json.dumps({"matured": result["count"], "awaiting": result["awaiting_candles"]}, indent=2))
        return 0
    if args.command == "rollback":
        try:
            print(json.dumps(rollback_champion(runtime.store, args.model_version), indent=2))
        except RuntimeError as exc:
            print(f"rollback refused: {exc}", file=sys.stderr)
            return 1
        return 0
    if args.command == "once":
        now = pd.Timestamp(args.now, tz="UTC").to_pydatetime() if args.now else None
        result = runtime.cycle(now)
        print(json.dumps(result.to_dict(), indent=2))
        return 0 if result.status in {"OK", "NO_OUTPUT"} else 1
    results = runtime.run_forever(max_cycles=args.max_cycles)
    print(json.dumps([r.to_dict() for r in results], indent=2))
    return 0


# ----------------------------------------------------------------------- experiment
def _collect_runs() -> list[dict]:
    runs: list[dict] = []
    for run_dir in sorted(EXPERIMENT_DIR.glob("*")):
        history_path = run_dir / "training_history.json"
        if not history_path.is_file():
            continue
        try:
            history = json.loads(history_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        best_path = run_dir / "best.pt"
        records = history.get("history") or []
        best = next((r for r in records if r.get("epoch") == history.get("best_epoch")), None)
        runs.append(
            {
                "run": run_dir.name,
                "epochs": history.get("epochs_run"),
                "best_epoch": history.get("best_epoch"),
                "best_val_loss": history.get("best_val_loss"),
                "val_barrier_accuracy": (best or {}).get("val_barrier_accuracy"),
                "val_momentum_accuracy": (best or {}).get("val_momentum_accuracy"),
                "checkpoint": str(best_path) if best_path.exists() else None,
                "checkpoint_sha256": file_sha256(best_path) if best_path.exists() else None,
            }
        )
    return runs


def cmd_experiment(args) -> int:
    runs = _collect_runs()
    if args.command == "list":
        if args.json:
            print(json.dumps(runs, indent=2, default=str))
            return 0
        if not runs:
            print("no runs with training_history.json found in experiments/")
            return 0
        print(f"{'run':44} {'ep':>3} {'best':>4} {'val_loss':>9} {'b_acc':>6} {'m_acc':>6}")
        for r in runs:
            print(
                f"{r['run']:44} {str(r['epochs']):>3} {str(r['best_epoch']):>4} "
                f"{r['best_val_loss']:9.4f} {r['val_barrier_accuracy']:6.3f} "
                f"{r['val_momentum_accuracy']:6.3f}"
            )
        return 0

    selected = [r for r in runs if r["run"] in set(args.runs or [])]
    if len(selected) < 2:
        print("compare needs at least two --runs names from `experiment list`", file=sys.stderr)
        return 2
    keys = ("best_val_loss", "val_barrier_accuracy", "val_momentum_accuracy")
    print(f"{'metric':24}" + "".join(f"{r['run'][:20]:>22}" for r in selected))
    for key in keys:
        values = [r[key] for r in selected]
        print(f"{key:24}" + "".join(f"{v:22.4f}" for v in values))
    return 0


# ---------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="patron",
        description="PATRON - Pattern Recognition & Outcome Network (BTC/USDT 15m).",
    )
    parser.add_argument("--version", action="version", version=f"patron {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    health = sub.add_parser("health", help="check environment, config, contracts and data")
    _add_common(health)
    health.add_argument("--dataset", default=str(FEATURE_DIR / "BTCUSDT_dataset.parquet"))
    health.add_argument("--checkpoint", default=str(Path("experiments") / "patron_v2_20261114-203859" / "best.pt"))
    health.add_argument("--db", default=str(Path("data") / "runtime" / "patron.db"))
    health.add_argument("--json", action="store_true", help="machine-readable output")
    health.add_argument("--strict", action="store_true", help="treat WARN as failure")
    health.set_defaults(func=cmd_health)

    train = sub.add_parser("train", help="train a new candidate checkpoint")
    _add_common(train)
    train.add_argument("--dataset", default=None)
    train.add_argument("--run-name", default=None)
    train.add_argument("--epochs", type=int, default=None)
    train.add_argument("--batch-size", type=int, default=None)
    train.add_argument("--train-stride", type=int, default=None)
    train.add_argument("--threads", type=int, default=6)
    train.add_argument("--seed", type=int, default=None)
    train.add_argument("--dry-run", action="store_true", help="validate data and model only")
    train.set_defaults(func=cmd_train)

    evaluate = sub.add_parser("evaluate", help="evaluate a checkpoint on the held-out split")
    evaluate.add_argument("--checkpoint", required=True)
    _add_common(evaluate)
    evaluate.add_argument("--dataset", default=None)
    evaluate.add_argument("--out", default=str(REPORT_DIR / "evaluation_report.json"))
    evaluate.add_argument("--no-baselines", action="store_true")
    evaluate.set_defaults(func=cmd_evaluate)

    predict = sub.add_parser("predict", help="run one inference from 96 closed candles")
    predict.add_argument("--input", required=True, help="input JSON (schemas/input.schema.json)")
    predict.add_argument("--checkpoint", default=None)
    predict.add_argument("--warmup", default=None)
    predict.add_argument("--output", default=str(REPORT_DIR / "predictions" / "latest.json"))
    predict.set_defaults(func=cmd_predict)

    runtime = sub.add_parser("runtime", help="autonomous runtime operations")
    runtime.add_argument(
        "command", choices=["status", "once", "serve", "mature", "rollback", "health"]
    )
    _add_common(runtime)
    runtime.add_argument("--runtime-config", default=None)
    runtime.add_argument("--db", default=None)
    runtime.add_argument("--offline-source", default=None)
    runtime.add_argument("--now", default=None)
    runtime.add_argument("--max-cycles", type=int, default=None)
    runtime.add_argument("--no-train", action="store_true")
    runtime.add_argument("--version", dest="model_version", default=None)
    runtime.set_defaults(func=cmd_runtime)

    experiment = sub.add_parser("experiment", help="inspect training experiments")
    exp_sub = experiment.add_subparsers(dest="command", required=True)
    exp_list = exp_sub.add_parser("list", help="list runs found in experiments/")
    exp_list.add_argument("--json", action="store_true")
    exp_list.set_defaults(func=cmd_experiment, runs=None)
    exp_compare = exp_sub.add_parser("compare", help="compare validation metrics of runs")
    exp_compare.add_argument("--runs", nargs="+", required=True)
    exp_compare.set_defaults(func=cmd_experiment)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
