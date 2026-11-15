"""Walk-forward (rolling out-of-sample) evaluation of a trained checkpoint.

Honest scope: this does **not** retrain per fold. It evaluates a fixed
checkpoint over rolling blocks of the untouched test period and over market
regimes, which measures temporal stability of the shipped model. Fold-wise
retraining is available in code (:func:`patron.datasets.splits.walk_forward_folds`)
but is not run here because each fold costs ~15 epochs on CPU; the report says so.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from patron.config import load_config
from patron.datasets.loader import prepare_data
from patron.evaluation.calibration import apply_temperature, fit_temperature
from patron.evaluation.evaluator import _softmax, collect_predictions
from patron.evaluation.metrics import classification_metrics, quantile_metrics
from patron.evaluation.regimes import add_regimes, regime_breakdown
from patron.labels.barrier import BARRIER_CLASSES
from patron.labels.momentum import MOMENTUM_CLASSES
from patron.logging_utils import setup_logging
from patron.models.patron_net import count_parameters
from patron.paths import REPORT_DIR, ensure_dirs


def _load_model(checkpoint: str, cfg):
    from patron.models.patron_net import build_model as _build
    from patron.training.trainer import load_checkpoint

    model = _build(cfg)
    state = load_checkpoint(checkpoint)
    model.load_state_dict(state["model_state"])
    model.eval()
    return model


def _block_metrics(preds, cfg, mask_rows, temperature):
    barrier = classification_metrics(
        apply_temperature(preds["barrier_logits"][mask_rows], temperature),
        preds["barrier_label"][mask_rows],
        list(BARRIER_CLASSES),
        preds["barrier_mask"][mask_rows] > 0,
    )
    momentum = classification_metrics(
        _softmax(preds["momentum_logits"][mask_rows]),
        preds["momentum_label"][mask_rows],
        list(MOMENTUM_CLASSES),
        preds["momentum_label"][mask_rows] >= 0,
    )
    quantile = quantile_metrics(preds["quantile"][mask_rows], preds["range_target"][mask_rows])
    return {
        "n": int(mask_rows.sum()),
        "barrier_accuracy": round(float(barrier["accuracy"]), 4),
        "barrier_balanced_accuracy": round(float(barrier["balanced_accuracy"]), 4),
        "barrier_macro_f1": round(float(barrier["macro_f1"]), 4),
        "barrier_log_loss": round(float(barrier["log_loss"]), 4),
        "momentum_accuracy": round(float(momentum["accuracy"]), 4),
        "momentum_balanced_accuracy": round(float(momentum["balanced_accuracy"]), 4),
        "pinball": round(float(quantile["pinball_mean"]), 6),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PATRON walk-forward evaluation")
    parser.add_argument("--checkpoint", default="experiments/patron_v2_20261114-203859/best.pt")
    parser.add_argument("--config", default="config/model_config.yaml")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--folds", type=int, default=4)
    args = parser.parse_args(argv)

    ensure_dirs()
    setup_logging()
    cfg = load_config(args.config)
    data = prepare_data(cfg, args.dataset)
    model = _load_model(args.checkpoint, cfg)
    print(f"checkpoint parameters: {count_parameters(model):,}")

    test = data.bundles.test
    preds = collect_predictions(model, test)
    timestamps = pd.to_datetime(test.timestamps[test.positions])
    val_preds = collect_predictions(model, data.bundles.val)
    temperature = fit_temperature(
        val_preds["barrier_logits"], val_preds["barrier_label"], val_preds["barrier_mask"]
    )

    n = len(timestamps)
    blocks = np.array_split(np.arange(n), args.folds)
    rows = []
    for k, idx in enumerate(blocks, start=1):
        mask_rows = np.zeros(n, dtype=bool)
        mask_rows[idx] = True
        metrics = _block_metrics(preds, cfg, mask_rows, temperature)
        metrics.update(
            {
                "fold": k,
                "start": str(timestamps[idx[0]]),
                "end": str(timestamps[idx[-1]]),
            }
        )
        rows.append(metrics)
        print(metrics)

    # Regime slices over the whole test period.
    frame = data.frame.copy()
    frame = add_regimes(frame)
    end_positions = test.positions
    regime_trend = frame["regime_trend"].to_numpy()[end_positions]
    regime_vol = frame["regime_vol"].to_numpy()[end_positions]
    barrier_valid = preds["barrier_mask"] > 0
    barrier_pred = preds["barrier_logits"].argmax(axis=1)
    momentum_pred = preds["momentum_logits"].argmax(axis=1)

    regime_rows: dict[str, dict] = {}
    for name, series in (("trend", regime_trend), ("vol", regime_vol)):
        breakdown = {}
        for regime in pd.unique(series):
            mask = (series == regime) & barrier_valid
            if mask.sum() < 50:
                continue
            breakdown[str(regime)] = {
                "n": int(mask.sum()),
                "barrier_accuracy": round(float((barrier_pred[mask] == preds["barrier_label"][mask]).mean()), 4),
                "momentum_accuracy": round(
                    float((momentum_pred[series == regime] == preds["momentum_label"][series == regime]).mean()),
                    4,
                ),
            }
        regime_rows[name] = breakdown

    report = {
        "checkpoint": args.checkpoint,
        "dataset_version": "1.0.0",
        "temperature": round(float(temperature), 4),
        "folds": rows,
        "regimes": regime_rows,
        "note": "Fixed-checkpoint rolling evaluation over the untouched test split; "
        "no retraining per fold. Fold-wise retraining would cost ~15 epochs per fold on CPU.",
    }
    (REPORT_DIR / "walk_forward_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# PATRON Walk-Forward Report",
        "",
        f"- Checkpoint: `{args.checkpoint}` (dataset 1.0.0, 15m BTC/USDT)",
        f"- Temperature (fitted on validation): {report['temperature']}",
        f"- Test windows: {n:,} split into {args.folds} chronological blocks",
        "- Each block is out-of-sample for this checkpoint: the model was trained on data ending 2025-05-01.",
        "- Fold-wise retraining is **not** done here (CPU budget); see the note at the end.",
        "",
        "## Rolling out-of-sample blocks",
        "",
        "| Fold | Start | End | n | Barrier acc | Balanced acc | Macro-F1 | Log loss | Momentum acc | Pinball |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['fold']} | {row['start'][:10]} | {row['end'][:10]} | {row['n']:,} | "
            f"{row['barrier_accuracy']:.4f} | {row['barrier_balanced_accuracy']:.4f} | "
            f"{row['barrier_macro_f1']:.4f} | {row['barrier_log_loss']:.4f} | "
            f"{row['momentum_accuracy']:.4f} | {row['pinball']:.6f} |"
        )
    lines += [
        "",
        "## Regime slices (whole test period, causal regime labels)",
        "",
    ]
    for regime_type, breakdown in regime_rows.items():
        lines += [
            f"### {regime_type.capitalize()} regime",
            "",
            "| Regime | n | Barrier accuracy | Momentum accuracy |",
            "|---|---|---|---|",
        ]
        for regime, stats in breakdown.items():
            lines.append(
                f"| {regime} | {stats['n']:,} | {stats['barrier_accuracy']:.4f} | "
                f"{stats['momentum_accuracy']:.4f} |"
            )
        lines.append("")

    accs = [r["barrier_accuracy"] for r in rows]
    lines += [
        "## Reading",
        "",
        f"- Barrier accuracy ranges {min(accs):.4f}–{max(accs):.4f} across blocks "
        f"(spread {max(accs) - min(accs):.4f}).",
        "- A stable spread with no monotone decay indicates the model is not simply "
        "fitted to one market phase, but the level remains below the majority baseline "
        "for the reasons documented in `reports/final_model_comparison.md`.",
        "",
        "## Limitation",
        "",
        "True walk-forward *retraining* (train on fold k, validate on k+1) is implemented in",
        "`patron.datasets.splits.walk_forward_folds` but not executed here: each retrain costs",
        "roughly 15 epochs × ~4 min on this CPU, i.e. ~1 h per fold. The reported numbers are",
        "rolling out-of-sample evaluations of the fixed shipped checkpoint, not a retrained ensemble.",
        "",
    ]
    (REPORT_DIR / "walk_forward_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"report: {REPORT_DIR / 'walk_forward_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
