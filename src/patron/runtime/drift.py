"""Causal drift detection for features and matured outcomes.

Drift is a *signal*, not a verdict: crossing a threshold asks for an investigation
and makes the learning manager eligible to train, but it never promotes a model by
itself. Reference statistics come from the dataset that trained the champion, are
stored with the registry entry, and are never recomputed from live data.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from patron.features.schema import FEATURE_ORDER
from patron.labels.momentum import MOMENTUM_CLASSES
from patron.runtime.config import RuntimeConfig
from patron.runtime.store import RuntimeStore

LOGGER = logging.getLogger(__name__)

PSI_BINS = 10


def _bins(values: np.ndarray, bins: int = PSI_BINS) -> list[float]:
    edges = np.quantile(values, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return []
    edges[0] = -np.inf
    edges[-1] = np.inf
    return [float(edge) for edge in edges]


def _proportions(values: np.ndarray, edges: list[float]) -> np.ndarray:
    counts, _ = np.histogram(values, bins=np.array(edges))
    total = counts.sum()
    if total == 0:
        return np.zeros(len(counts))
    return counts / total


def population_stability_index(reference: np.ndarray, recent: np.ndarray, edges: list[float] | None = None) -> float | None:
    """PSI between a reference sample and the recent sample (same bins)."""
    reference = np.asarray(reference, dtype=np.float64)
    recent = np.asarray(recent, dtype=np.float64)
    reference = reference[np.isfinite(reference)]
    recent = recent[np.isfinite(recent)]
    if len(reference) < 10 or len(recent) < 10:
        return None
    edges = edges or _bins(reference)
    if not edges:
        return None
    expected = np.clip(_proportions(reference, edges), 1e-6, None)
    actual = np.clip(_proportions(recent, edges), 1e-6, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def build_reference(dataset_path: str | Path, feature_order: tuple[str, ...] = FEATURE_ORDER) -> dict:
    """Compute the training distribution stored with a registered champion."""
    frame = pd.read_parquet(dataset_path, columns=[*feature_order, "momentum_class"])
    values = frame[list(feature_order)].to_numpy(dtype=np.float64)
    keep = frame.get("sample_valid", pd.Series(True, index=frame.index)).to_numpy(dtype=bool)
    values = values[keep]
    reference: dict = {"samples": int(len(values)), "features": {}, "outcomes": {}}
    for index, feature in enumerate(feature_order):
        column = values[:, index]
        column = column[np.isfinite(column)]
        edges = _bins(column)
        reference["features"][feature] = {
            "mean": float(np.mean(column)),
            "std": float(np.std(column) or 1.0),
            "p05": float(np.quantile(column, 0.05)),
            "p95": float(np.quantile(column, 0.95)),
            "edges": edges,
            "proportions": _proportions(column, edges).tolist() if edges else [],
        }
    momentum = frame["momentum_class"].to_numpy()
    momentum = momentum[momentum >= 0]
    counts = np.bincount(momentum, minlength=len(MOMENTUM_CLASSES)).astype(float)
    reference["outcomes"] = {
        "momentum": {
            "classes": list(MOMENTUM_CLASSES),
            "proportions": (counts / counts.sum()).tolist() if counts.sum() else [],
        }
    }
    return reference


def recent_feature_matrix(store: RuntimeStore, limit: int) -> pd.DataFrame:
    rows = store.recent_forecasts(limit)
    records = []
    for row in rows:
        payload = json.loads(row["payload"])
        snapshot = payload.get("reference", {}).get("feature_snapshot", {})
        if snapshot:
            records.append(snapshot)
    return pd.DataFrame(records)


def detect_drift(
    store: RuntimeStore,
    cfg: RuntimeConfig,
    reference: dict | None,
    model_cfg=None,
) -> dict:
    """Compare recent live inputs and matured outcomes with the training reference."""
    result: dict = {
        "schema_version": "1.0.0",
        "evaluated_at": None,
        "reference_available": bool(reference),
        "samples": 0,
        "feature": {"drifting": False, "drifting_features": [], "z_scores": {}, "psi": {}},
        "outcome": {"drifting": False, "psi": None, "distribution": {}},
        "triggered": False,
        "reasons": [],
    }
    if not reference:
        result["reasons"].append("no reference distribution stored for the champion")
        return result

    from patron.runtime.store import utc_iso
    from datetime import datetime, timezone

    result["evaluated_at"] = utc_iso(datetime.now(timezone.utc))
    recent = recent_feature_matrix(store, cfg.drift.min_samples * 3)
    result["samples"] = int(len(recent))
    if len(recent) < cfg.drift.min_samples:
        result["reasons"].append(
            f"only {len(recent)} recent snapshots, need {cfg.drift.min_samples}"
        )
        return result

    drifting: list[str] = []
    for feature in FEATURE_ORDER:
        ref = reference["features"].get(feature)
        if ref is None or feature not in recent:
            continue
        values = recent[feature].to_numpy(dtype=np.float64)
        values = values[np.isfinite(values)]
        if len(values) < 10:
            continue
        std = ref["std"] if ref["std"] > 0 else 1.0
        z = float((np.mean(values) - ref["mean"]) / std)
        result["feature"]["z_scores"][feature] = z
        psi = None
        if ref.get("edges") and ref.get("proportions"):
            expected = np.clip(np.array(ref["proportions"], dtype=float), 1e-6, None)
            actual = np.clip(_proportions(values, ref["edges"]), 1e-6, None)
            if len(expected) == len(actual):
                psi = float(np.sum((actual - expected) * np.log(actual / expected)))
        result["feature"]["psi"][feature] = psi
        if abs(z) > cfg.drift.feature_z_threshold or (psi is not None and psi > cfg.drift.psi_threshold):
            drifting.append(feature)

    result["feature"]["drifting_features"] = drifting
    result["feature"]["drifting"] = len(drifting) >= cfg.drift.min_drifting_features
    if result["feature"]["drifting"]:
        result["reasons"].append(
            f"{len(drifting)} features drifted beyond thresholds: {', '.join(drifting)}"
        )

    rows = store.execute(
        "SELECT metrics FROM evaluations WHERE step=6 ORDER BY target_ts DESC LIMIT ?",
        (max(cfg.drift.min_samples * 3, 50),),
    ).fetchall()
    counts = np.zeros(len(MOMENTUM_CLASSES), dtype=float)
    for row in rows:
        metrics = json.loads(row["metrics"]) if row["metrics"] else {}
        actual = (metrics.get("momentum") or {}).get("actual")
        if actual in MOMENTUM_CLASSES:
            counts[MOMENTUM_CLASSES.index(actual)] += 1
    reference_props = np.array(reference["outcomes"]["momentum"]["proportions"], dtype=float)
    if counts.sum() >= cfg.drift.min_samples and reference_props.size == len(counts):
        observed = counts / counts.sum()
        psi = float(np.sum((observed - np.clip(reference_props, 1e-6, None))
                           * np.log(np.clip(observed, 1e-6, None) / np.clip(reference_props, 1e-6, None))))
        result["outcome"] = {
            "drifting": bool(psi > cfg.drift.outcome_psi_threshold),
            "psi": psi,
            "samples": int(counts.sum()),
            "distribution": {name: int(counts[i]) for i, name in enumerate(MOMENTUM_CLASSES)},
        }
        if result["outcome"]["drifting"]:
            result["reasons"].append(f"momentum outcome PSI {psi:.3f} > {cfg.drift.outcome_psi_threshold}")
    else:
        result["outcome"]["samples"] = int(counts.sum())
        result["reasons"].append(
            f"outcome drift not evaluated: {int(counts.sum())} matured horizons, need {cfg.drift.min_samples}"
        )

    result["triggered"] = bool(result["feature"]["drifting"] or result["outcome"]["drifting"])
    return result
