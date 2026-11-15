"""Metrics for all four PATRON heads."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)

N_BINS = 10


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = N_BINS) -> float:
    """Top-1 confidence ECE."""
    confidence = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if not mask.any():
            continue
        ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def classification_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    valid: np.ndarray | None = None,
) -> dict:
    if valid is not None:
        probabilities = probabilities[valid]
        labels = labels[valid]
    if len(labels) == 0:
        return {"n": 0}
    predictions = probabilities.argmax(axis=1)
    present = sorted(set(labels.tolist()) | set(predictions.tolist()))
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=list(range(len(class_names))), zero_division=0
    )
    out = {
        "n": int(len(labels)),
        "accuracy": float((predictions == labels).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "log_loss": float(log_loss(labels, probabilities, labels=list(range(len(class_names))))),
        "ece": expected_calibration_error(probabilities, labels),
        "per_class": {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(class_names)
        },
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=list(range(len(class_names)))
        ).tolist(),
        "prediction_distribution": {
            name: int((predictions == i).sum()) for i, name in enumerate(class_names)
        },
    }
    try:
        out["roc_auc_ovr"] = float(
            roc_auc_score(
                labels,
                probabilities,
                multi_class="ovr",
                average="macro",
                labels=list(range(len(class_names))),
            )
        )
    except ValueError:
        out["roc_auc_ovr"] = float("nan")
    if len(present) < len(class_names):
        out["missing_classes"] = [class_names[i] for i in range(len(class_names)) if i not in present]
    return out


def pinball(predictions: np.ndarray, targets: np.ndarray, quantiles: tuple[float, ...]) -> np.ndarray:
    """Element-wise pinball loss; predictions/targets broadcast to (..., Q)."""
    q = np.asarray(quantiles, dtype=np.float64)
    error = targets[..., None] - predictions
    return np.maximum(q * error, (q - 1.0) * error)


def quantile_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    field_names: tuple[str, ...] = ("high", "low", "close"),
) -> dict:
    """``predictions``: (N, H, F, Q); ``targets``: (N, H, F)."""
    if predictions.shape[-1] != len(quantiles):
        raise ValueError("quantile axis mismatch")
    loss = pinball(predictions, targets, quantiles)  # (N, H, F, Q)
    valid = np.isfinite(targets)
    valid_q = np.repeat(valid[..., None], len(quantiles), axis=-1)
    per_field = {}
    for f, name in enumerate(field_names):
        field_loss = loss[:, :, f, :][valid_q[:, :, f, :]]
        coverage = (
            (targets[:, :, f] >= predictions[:, :, f, 0])
            & (targets[:, :, f] <= predictions[:, :, f, -1])
        )
        per_field[name] = {
            "pinball_mean": float(field_loss.mean()),
            "coverage_q10_q90": float(coverage[valid[:, :, f]].mean()),
            "median_mae": float(
                np.abs(targets[:, :, f] - predictions[:, :, f, 1])[valid[:, :, f]].mean()
            ),
        }
    crossing = int(
        (
            (predictions[..., 0] > predictions[..., 1])
            | (predictions[..., 1] > predictions[..., 2])
        ).sum()
    )
    return {
        "n": int(valid.any(axis=(1, 2)).sum()),
        "pinball_mean": float(loss[valid_q].mean()),
        "crossing_violations": crossing,
        "per_field": per_field,
    }


def pattern_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
    tags: list[str],
    threshold: float = 0.5,
) -> dict:
    per_tag = {}
    aps = []
    for i, tag in enumerate(tags):
        valid = mask[:, i] > 0
        y = targets[valid, i].astype(int)
        p = probabilities[valid, i]
        if len(np.unique(y)) < 2:
            per_tag[tag] = {"support": int(y.sum()), "n": int(len(y)), "average_precision": None}
            continue
        ap = float(average_precision_score(y, p))
        aps.append(ap)
        per_tag[tag] = {
            "support": int(y.sum()),
            "n": int(len(y)),
            "positive_rate": float(y.mean()),
            "average_precision": ap,
            "brier": float(np.mean((p - y) ** 2)),
            "precision_at_0.5": float(((p >= threshold) & (y == 1)).sum() / max(1, (p >= threshold).sum())),
            "recall_at_0.5": float(((p >= threshold) & (y == 1)).sum() / max(1, (y == 1).sum())),
        }
    return {"macro_average_precision": float(np.mean(aps)) if aps else None, "per_tag": per_tag}
