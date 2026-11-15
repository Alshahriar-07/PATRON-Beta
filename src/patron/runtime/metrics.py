"""Rolling performance statistics for matured forecasts.

Everything is computed from evaluation rows that already exist: no metric is
estimated from a forecast that has not matured, and no metric mixes horizons. The
output is a plain dictionary so it can be published as JSON and asserted in tests.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict

import numpy as np

from patron.labels.barrier import BARRIER_CLASSES
from patron.labels.momentum import MOMENTUM_CLASSES
from patron.runtime.config import RuntimeConfig
from patron.runtime.store import RuntimeStore

RANGE_FIELDS = ("high", "low", "close")


def _parse(value):
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    return json.loads(value)


def evaluations_in_window(store: RuntimeStore, window_forecasts: int) -> list[dict]:
    rows = store.execute(
        "WITH recent AS (SELECT forecast_id, candle_ts, model_version, payload FROM forecasts "
        "                ORDER BY candle_ts DESC LIMIT ?) "
        "SELECT e.*, recent.candle_ts, recent.model_version, recent.payload "
        "FROM evaluations e JOIN recent ON recent.forecast_id = e.forecast_id "
        "ORDER BY e.target_ts",
        (max(1, int(window_forecasts)),),
    ).fetchall()
    return [dict(row) for row in rows]


def _macro_f1(pairs: list[tuple[str | None, str | None]], labels: tuple[str, ...]) -> float | None:
    valid = [(actual, pred) for actual, pred in pairs if actual in labels]
    if not valid:
        return None
    scores = []
    for label in labels:
        tp = sum(1 for actual, pred in valid if actual == label and pred == label)
        fp = sum(1 for actual, pred in valid if actual != label and pred == label)
        fn = sum(1 for actual, pred in valid if actual == label and pred != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return float(np.mean(scores))


def _ece(confidences: list[float], hits: list[bool], bins: int = 10) -> float | None:
    if not confidences:
        return None
    buckets: dict[int, list[tuple[float, bool]]] = defaultdict(list)
    for conf, hit in zip(confidences, hits):
        index = min(int(conf * bins), bins - 1)
        buckets[index].append((conf, hit))
    total = len(confidences)
    ece = 0.0
    for entries in buckets.values():
        bucket_conf = float(np.mean([c for c, _ in entries]))
        bucket_acc = float(np.mean([1.0 if h else 0.0 for _, h in entries]))
        ece += len(entries) / total * abs(bucket_acc - bucket_conf)
    return float(ece)


def compute_metrics(
    evaluation_rows: list[dict],
    model_cfg=None,
    include_regime: bool = False,
) -> dict:
    """Aggregate evaluation rows into the published performance block."""
    per_step: dict[int, dict] = {}
    step_errors: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    step_pinball: dict[int, list[float]] = defaultdict(list)
    step_coverage: dict[int, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    step_direction: dict[int, list[bool]] = defaultdict(list)

    barrier_pairs: list[tuple[str, str]] = []
    barrier_conf: list[float] = []
    barrier_hits: list[bool] = []
    momentum_pairs: list[tuple[str, str]] = []
    momentum_conf: list[float] = []
    momentum_hits: list[bool] = []
    ambiguous_barriers = 0

    for row in evaluation_rows:
        metrics = _parse(row["metrics"])
        if not metrics:
            continue
        step = int(row["step"])
        for field in RANGE_FIELDS:
            if field in metrics.get("pct_error", {}):
                step_errors[step][field].append(float(metrics["pct_error"][field]))
            if field in metrics.get("coverage", {}):
                step_coverage[step][field].append(bool(metrics["coverage"][field]))
        if "pinball_mean" in metrics:
            step_pinball[step].append(float(metrics["pinball_mean"]))
        if metrics.get("direction"):
            step_direction[step].append(bool(metrics["direction"]["correct"]))

        barrier = metrics.get("barrier")
        if barrier:
            probs = barrier.get("probs") or {}
            if barrier.get("ambiguous"):
                ambiguous_barriers += 1
            elif barrier.get("actual"):
                barrier_pairs.append((barrier["actual"], barrier["predicted"]))
                if probs:
                    confidence = max(probs.values())
                    barrier_conf.append(float(confidence))
                    barrier_hits.append(bool(barrier["correct"]))

        momentum = metrics.get("momentum")
        if momentum and momentum.get("actual"):
            momentum_pairs.append((momentum["actual"], momentum["predicted"]))
            payload = row.get("payload")
            probs = json.loads(payload) if isinstance(payload, str) else payload
            if probs:
                model_probs = probs["model_output"]["momentum"]["class_probabilities"]
                momentum_conf.append(float(max(model_probs.values())))
                momentum_hits.append(bool(momentum["correct"]))

    for step in sorted(set(step_errors) | set(step_pinball) | set(step_direction)):
        block = {
            "n": len(step_pinball.get(step, [])),
            "mape": {
                field: float(np.mean(values)) if values else None
                for field, values in step_errors.get(step, {}).items()
            },
            "pinball_mean": float(np.mean(step_pinball[step])) if step_pinball.get(step) else None,
            "coverage": {
                field: float(np.mean(values)) if values else None
                for field, values in step_coverage.get(step, {}).items()
            },
            "direction_accuracy": (
                float(np.mean(step_direction[step])) if step_direction.get(step) else None
            ),
        }
        per_step[step] = block

    barrier_accuracy = (
        sum(1 for actual, pred in barrier_pairs if actual == pred) / len(barrier_pairs)
        if barrier_pairs
        else None
    )
    barrier_counts = Counter()
    for actual, pred in barrier_pairs:
        barrier_counts[f"{actual}->{pred}"] += 1

    momentum_accuracy = (
        sum(1 for actual, pred in momentum_pairs if actual == pred) / len(momentum_pairs)
        if momentum_pairs
        else None
    )

    total = len([row for row in evaluation_rows if row.get("metrics")])
    all_coverage = [value for step in step_coverage.values() for values in step.values() for value in values]
    return {
        "forecasts_evaluated": len({row["forecast_id"] for row in evaluation_rows}),
        "horizons_evaluated": total,
        "range": {
            "per_step": per_step,
            "pinball_mean": float(np.mean([p for values in step_pinball.values() for p in values]))
            if step_pinball
            else None,
            "coverage_mean": float(np.mean(all_coverage)) if all_coverage else None,
            "direction_accuracy": (
                float(np.mean([v for values in step_direction.values() for v in values]))
                if step_direction
                else None
            ),
            "mape_mean": float(
                np.mean(
                    [value for fields in step_errors.values() for values in fields.values() for value in values]
                )
            )
            if step_errors
            else None,
        },
        "barrier": {
            "n": len(barrier_pairs),
            "accuracy": barrier_accuracy,
            "ambiguous": ambiguous_barriers,
            "ece": _ece(barrier_conf, barrier_hits),
            "confusion": dict(barrier_counts),
            "classes": list(BARRIER_CLASSES),
        },
        "momentum": {
            "n": len(momentum_pairs),
            "accuracy": momentum_accuracy,
            "macro_f1": _macro_f1(momentum_pairs, MOMENTUM_CLASSES),
            "ece": _ece(momentum_conf, momentum_hits),
            "distribution": dict(Counter(actual for actual, _ in momentum_pairs)),
        },
    }


def rolling_performance(store: RuntimeStore, cfg: RuntimeConfig, model_cfg=None) -> dict:
    rows = evaluations_in_window(store, cfg.monitoring.window_forecasts)
    metrics = compute_metrics(rows, model_cfg=model_cfg)
    metrics["window_forecasts"] = cfg.monitoring.window_forecasts
    metrics["source"] = "matured-evaluations-only"
    if rows:
        metrics["first_target_ts"] = rows[0]["target_ts"]
        metrics["last_target_ts"] = rows[-1]["target_ts"]
    return metrics


def degradation_state(store: RuntimeStore, cfg: RuntimeConfig, model_cfg=None) -> dict:
    """Apply the configured degradation rule to the rolling window."""
    rows = evaluations_in_window(store, cfg.monitoring.degradation.window_forecasts)
    metrics = compute_metrics(rows, model_cfg=model_cfg)
    metric = cfg.monitoring.degradation.metric
    value = None
    if metric == "barrier_accuracy":
        value = metrics["barrier"]["accuracy"]
        samples = metrics["barrier"]["n"]
    elif metric == "momentum_accuracy":
        value = metrics["momentum"]["accuracy"]
        samples = metrics["momentum"]["n"]
    elif metric == "direction_accuracy":
        value = metrics["range"]["direction_accuracy"]
        samples = metrics["range"].get("n", metrics["horizons_evaluated"])
    elif metric == "pinball_mean":
        value = metrics["range"]["pinball_mean"]
        samples = metrics["horizons_evaluated"]
    else:
        raise ValueError(f"unsupported degradation metric: {metric}")
    enough = samples >= cfg.monitoring.degradation.min_samples
    degraded = bool(enough and value is not None and value < cfg.monitoring.degradation.floor)
    return {
        "metric": metric,
        "value": value,
        "samples": samples,
        "min_samples": cfg.monitoring.degradation.min_samples,
        "floor": cfg.monitoring.degradation.floor,
        "enough_samples": enough,
        "degraded": degraded,
        "reason": None
        if not degraded
        else f"{metric}={value:.4f} < floor {cfg.monitoring.degradation.floor:.4f} over {samples} samples",
    }
