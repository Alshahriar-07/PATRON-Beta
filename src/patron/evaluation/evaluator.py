"""End-to-end evaluation of a trained PATRON checkpoint."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from patron.datasets.loader import PreparedData
from patron.datasets.splits import split_dates
from patron.evaluation.baselines import run_baselines
from patron.evaluation.calibration import apply_temperature, fit_temperature
from patron.evaluation.metrics import classification_metrics, pattern_metrics, quantile_metrics
from patron.labels.barrier import BARRIER_CLASSES
from patron.labels.momentum import MOMENTUM_CLASSES
from patron.models.patron_net import PatronNet

LOGGER = logging.getLogger(__name__)


@torch.no_grad()
def collect_predictions(
    model: PatronNet, dataset, batch_size: int = 256
) -> dict[str, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    chunks: dict[str, list[np.ndarray]] = {
        "barrier_logits": [],
        "momentum_logits": [],
        "pattern_logits": [],
        "quantile": [],
        "barrier_label": [],
        "barrier_mask": [],
        "momentum_label": [],
        "pattern_target": [],
        "pattern_mask": [],
        "range_target": [],
        "index": [],
    }
    for batch in loader:
        out = model(batch["x"])
        chunks["barrier_logits"].append(out["barrier"].numpy())
        chunks["momentum_logits"].append(out["momentum"].numpy())
        chunks["pattern_logits"].append(out["pattern"].numpy())
        chunks["quantile"].append(out["quantile"].numpy())
        chunks["barrier_label"].append(batch["barrier"].numpy())
        chunks["barrier_mask"].append(batch["barrier_mask"].numpy())
        chunks["momentum_label"].append(batch["momentum"].numpy())
        chunks["pattern_target"].append(batch["pattern"].numpy())
        chunks["pattern_mask"].append(batch["pattern_mask"].numpy())
        chunks["range_target"].append(batch["range"].numpy())
        chunks["index"].append(batch["index"].numpy())
    return {k: np.concatenate(v) for k, v in chunks.items()}


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def evaluate_split(preds: dict[str, np.ndarray], cfg, temperature: float | None = None) -> dict:
    barrier_logits = preds["barrier_logits"]
    barrier_probs = (
        apply_temperature(barrier_logits, temperature) if temperature else _softmax(barrier_logits)
    )
    barrier_mask = preds["barrier_mask"] > 0
    momentum_valid = preds["momentum_label"] >= 0

    return {
        "barrier": {
            "raw": classification_metrics(_softmax(barrier_logits), preds["barrier_label"], list(BARRIER_CLASSES), barrier_mask),
            "calibrated": (
                classification_metrics(barrier_probs, preds["barrier_label"], list(BARRIER_CLASSES), barrier_mask)
                if temperature
                else None
            ),
            "ambiguous_rate": float((preds["barrier_mask"] == 0).mean()),
        },
        "momentum": classification_metrics(
            _softmax(preds["momentum_logits"]), preds["momentum_label"], list(MOMENTUM_CLASSES), momentum_valid
        ),
        "quantile": quantile_metrics(preds["quantile"], preds["range_target"], tuple(cfg.quantiles)),
        "pattern": pattern_metrics(
            torch.sigmoid(torch.tensor(preds["pattern_logits"])).numpy(),
            preds["pattern_target"],
            preds["pattern_mask"],
            list(cfg.pattern_tags),
        ),
    }


def run_evaluation(
    model: PatronNet,
    data: PreparedData,
    cfg,
    checkpoint_path: Path,
    with_baselines: bool = True,
) -> dict:
    LOGGER.info("collecting predictions on val and test")
    val_preds = collect_predictions(model, data.bundles.val, cfg.runtime.inference_batch_size)
    test_preds = collect_predictions(model, data.bundles.test, cfg.runtime.inference_batch_size)

    temperature = fit_temperature(
        val_preds["barrier_logits"], val_preds["barrier_label"], val_preds["barrier_mask"] > 0
    )
    LOGGER.info("fitted barrier temperature: %.4f", temperature)

    report = {
        "checkpoint": str(checkpoint_path),
        "symbol": cfg.symbol,
        "timeframe": cfg.timeframe,
        "model_version": cfg.project.version,
        "feature_version": "features-1.0.0",
        "splits": {
            "val": split_dates(data.frame, data.bundles.split, data.bundles.positions)["val"],
            "test": split_dates(data.frame, data.bundles.split, data.bundles.positions)["test"],
        },
        "temperature": temperature,
        "val": evaluate_split(val_preds, cfg, temperature),
        "test": evaluate_split(test_preds, cfg, temperature),
        "test_raw": evaluate_split(test_preds, cfg, None),
        "dataset": {
            "train_windows": len(data.bundles.train),
            "val_windows": len(data.bundles.val),
            "test_windows": len(data.bundles.test),
        },
    }

    if with_baselines:
        frames = data.frame
        report["baselines"] = run_baselines(
            frames,
            data.scaled_features,
            data.bundles.positions,
            data.bundles.split.train,
            data.bundles.split.test,
            horizon=cfg.horizon,
        )

    return report


def report_to_markdown(report: dict) -> str:
    test = report["test"]
    lines = [
        "# PATRON Evaluation Report",
        "",
        f"Checkpoint: `{report['checkpoint']}`  ",
        f"Test split: {report['splits']['test']['start']} → {report['splits']['test']['end']} "
        f"({report['splits']['test']['count']} windows)  ",
        f"Calibration temperature (fitted on val): {report['temperature']:.4f}",
        "",
        "## Barrier (6-candle triple barrier)",
        "",
        "| Metric | Raw | Calibrated |",
        "|---|---|---|",
    ]
    raw = test["barrier"]["raw"]
    cal = test["barrier"].get("calibrated") or {}
    for key in ("accuracy", "balanced_accuracy", "macro_f1", "log_loss", "ece", "roc_auc_ovr"):
        lines.append(
            f"| {key} | {_fmt(raw.get(key))} | {_fmt(cal.get(key))} |"
        )
    lines += [
        "",
        f"Ambiguous (excluded) share: {test['barrier']['ambiguous_rate']:.2%}",
        "",
        "Confusion matrix (rows=true, cols=pred): "
        f"`{raw.get('confusion_matrix')}` for {list(BARRIER_CLASSES)}",
        "",
        "## Momentum (5 classes)",
        "",
        f"- accuracy: {_fmt(test['momentum'].get('accuracy'))}",
        f"- balanced accuracy: {_fmt(test['momentum'].get('balanced_accuracy'))}",
        f"- macro F1: {_fmt(test['momentum'].get('macro_f1'))}",
        f"- log loss: {_fmt(test['momentum'].get('log_loss'))}",
        f"- ECE: {_fmt(test['momentum'].get('ece'))}",
        "",
        "## Range quantiles (relative to last close)",
        "",
        "| Field | Pinball | Coverage q10–q90 | Median MAE |",
        "|---|---|---|---|",
    ]
    for field, m in test["quantile"]["per_field"].items():
        lines.append(
            f"| {field} | {m['pinball_mean']:.6f} | {m['coverage_q10_q90']:.3f} | {m['median_mae']:.6f} |"
        )
    lines += [
        "",
        f"Quantile crossing violations: {test['quantile']['crossing_violations']}",
        "",
        "## Pattern tags",
        "",
        f"Macro average precision: {_fmt(test['pattern'].get('macro_average_precision'))}",
        "",
        "## Baselines (test split)",
        "",
    ]
    for b in report.get("baselines", {}).get("baselines", []):
        metric = b["metrics"].get("accuracy", b["metrics"].get("close_mae_all_steps"))
        lines.append(f"- {b['name']} / {b['task']}: {metric}")
    lines.append("")
    return "\n".join(lines)


def _fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return "nan" if value != value else f"{value:.4f}"
    return str(value)
