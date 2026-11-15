"""Champion / challenger governance.

A candidate is promoted only if it passes every gate that was configured *before*
it was evaluated. Gates compare the candidate with the active champion on the same
validation split, and nothing here touches the final test split: that split is
reserved for reporting.

A rejected candidate is kept in the registry with its report and reasons so the
decision is auditable; the champion pointer never moves on a failed gate.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from patron.config import PatronConfig
from patron.paths import PROJECT_ROOT
from patron.runtime.config import RuntimeConfig
from patron.runtime.store import RuntimeStore
from patron.utils.io import file_sha256

LOGGER = logging.getLogger(__name__)


@dataclass
class Gate:
    name: str
    passed: bool
    value: float | bool | None
    threshold: str
    detail: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _abs(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_run_metrics(run_dir: str | Path) -> dict:
    """Read training history and (if present) the evaluation report for a run."""
    run_path = _abs(run_dir)
    metrics: dict = {"run_dir": str(run_path), "available": False}
    history_path = run_path / "training_history.json"
    if not history_path.exists():
        metrics["error"] = "training_history.json missing"
        return metrics
    history = json.loads(history_path.read_text(encoding="utf-8"))
    records = history.get("history") or []
    best_epoch = int(history.get("best_epoch") or 0)
    best = next((row for row in records if int(row.get("epoch", -1)) == best_epoch), None)
    if best is None and records:
        best = min(records, key=lambda row: row.get("val_loss", float("inf")))
    metrics.update(
        {
            "available": True,
            "epochs_run": history.get("epochs_run"),
            "best_epoch": best_epoch,
            "best_val_loss": history.get("best_val_loss"),
            "val_loss": None if best is None else best.get("val_loss"),
            "val_barrier_accuracy": None if best is None else best.get("val_barrier_accuracy"),
            "val_momentum_accuracy": None if best is None else best.get("val_momentum_accuracy"),
            "val_samples": None,
        }
    )

    report_path = run_path / "evaluation_report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        val = report.get("val", {})
        quantile = val.get("quantile", {})
        coverage = {
            field: quantile.get("per_field", {}).get(field, {}).get("coverage_q10_q90")
            for field in ("high", "low", "close")
        }
        metrics["evaluation"] = {
            "path": str(report_path),
            "val_samples": int(quantile.get("n") or 0),
            "coverage": coverage,
            "pinball_mean": quantile.get("pinball_mean"),
            "barrier_accuracy": (val.get("barrier") or {}).get("accuracy"),
            "momentum_accuracy": (val.get("momentum") or {}).get("accuracy"),
            "splits": report.get("splits"),
        }
        metrics["val_samples"] = metrics["evaluation"]["val_samples"]
    else:
        metrics["evaluation"] = None
    return metrics


def champion_metrics(store: RuntimeStore) -> dict:
    champion = store.champion()
    if champion is None:
        return {}
    payload = champion.get("metrics")
    return json.loads(payload) if isinstance(payload, str) else (payload or {})


def evaluate_gates(
    candidate: dict,
    champion: dict,
    cfg: RuntimeConfig,
    validity: dict,
    seeds: int = 1,
) -> dict:
    """Apply the configured promotion policy. Returns ``{passed, gates, reasons}``."""
    gates: list[Gate] = []
    gov = cfg.governance

    gates.append(
        Gate(
            "schema_valid",
            bool(validity.get("schema_valid")),
            validity.get("schema_valid"),
            "checkpoint loads and produces a schema-valid, finite output",
            "; ".join(validity.get("problems", [])),
        )
    )
    gates.append(
        Gate(
            "probabilities_valid",
            bool(validity.get("output_valid")),
            validity.get("output_valid"),
            "probabilities sum to 1 and quantiles are ordered",
            "; ".join(validity.get("problems", [])),
        )
    )

    evaluation = candidate.get("evaluation") or {}
    samples = int(evaluation.get("val_samples") or 0)
    gates.append(
        Gate(
            "min_samples",
            samples >= gov.min_validation_samples,
            samples,
            f">= {gov.min_validation_samples} validation windows",
            "evaluation_report.json missing" if not evaluation else "",
        )
    )

    cand_loss = candidate.get("val_loss")
    champ_loss = champion.get("val_loss")
    loss_ok = (
        cand_loss is not None
        and champ_loss is not None
        and float(cand_loss) <= float(champ_loss) + gov.max_logloss_delta
    )
    gates.append(
        Gate(
            "logloss_not_worse",
            bool(loss_ok),
            cand_loss,
            f"val log loss <= champion {champ_loss} + {gov.max_logloss_delta}",
        )
    )

    cand_acc = candidate.get("val_barrier_accuracy")
    champ_acc = champion.get("val_barrier_accuracy")
    acc_ok = (
        cand_acc is not None
        and champ_acc is not None
        and float(cand_acc) >= float(champ_acc) - gov.max_accuracy_drop
    )
    gates.append(
        Gate(
            "accuracy_not_worse",
            bool(acc_ok),
            cand_acc,
            f"val barrier accuracy >= champion {champ_acc} - {gov.max_accuracy_drop}",
        )
    )

    coverage = evaluation.get("coverage") or {}
    low, high = gov.coverage_bounds
    coverage_values = [value for value in coverage.values() if isinstance(value, (int, float))]
    coverage_ok = (
        len(coverage_values) == 3
        and all(low <= float(value) <= high for value in coverage_values)
    )
    gates.append(
        Gate(
            "coverage_ok",
            bool(coverage_ok),
            min(coverage_values) if coverage_values else None,
            f"q10-q90 coverage within [{low}, {high}] for high/low/close",
            "coverage unavailable" if not coverage_values else "",
        )
    )

    gates.append(
        Gate(
            "seed_count",
            seeds >= gov.min_seeds,
            seeds,
            f">= {gov.min_seeds} seed(s)",
        )
    )

    mandatory = set(gov.mandatory_gates)
    selected = [gate for gate in gates if gate.name in mandatory or gate.name == "seed_count"]
    failed = [gate for gate in selected if not gate.passed]
    return {
        "passed": not failed,
        "gates": [gate.to_dict() for gate in selected],
        "reasons": [f"{gate.name}: {gate.detail or gate.threshold}" for gate in failed],
    }


def validate_candidate_checkpoint(
    checkpoint: str | Path,
    window,
    cfg: RuntimeConfig,
    model_cfg: PatronConfig,
) -> dict:
    """Load the candidate and run one real inference on the current window."""
    from patron.inference.pipeline import load_predictor
    from patron.runtime.engine import check_output_consistency
    from patron.runtime.window import candles_payload

    problems: list[str] = []
    predictor = None
    try:
        predictor = load_predictor(checkpoint)
    except Exception as exc:
        problems.append(f"checkpoint load failed: {exc}")
    if predictor is None:
        return {"schema_valid": False, "output_valid": False, "problems": problems}
    if not window.ok:
        problems.append(f"window unavailable: {window.status}")
        return {"schema_valid": False, "output_valid": False, "problems": problems}

    payload = {"symbol": cfg.symbol, "timeframe": cfg.timeframe, "candles": candles_payload(window.frame)}
    predictor.warmup_frame = window.warmup
    try:
        from patron.inference.pipeline import predict

        output = predict(payload, predictor)
    except Exception as exc:
        return {"schema_valid": False, "output_valid": False, "problems": [f"inference failed: {exc}"]}
    if output.get("status") != "OK":
        problems.extend(output.get("data_quality", {}).get("warnings", []))
        return {"schema_valid": False, "output_valid": False, "problems": problems or [output.get("status")]}

    problems.extend(check_output_consistency(output))
    return {"schema_valid": True, "output_valid": not problems, "problems": problems}


def register_candidate(
    store: RuntimeStore,
    run_dir: str | Path,
    metrics: dict,
    reference: dict | None = None,
    name: str = "patron",
) -> str:
    run_path = _abs(run_dir)
    checkpoint = run_path / "best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"candidate checkpoint missing: {checkpoint}")
    version = run_path.name
    store.register_model(
        version=version,
        name=name,
        path=str(checkpoint),
        sha256=file_sha256(checkpoint),
        role="candidate",
        metrics=metrics,
        notes="auto-trained candidate",
        reference=reference,
    )
    store.log_event("candidate_registered", version, {"sha256": file_sha256(checkpoint)})
    return version


def promote_candidate(store: RuntimeStore, version: str) -> dict:
    if store.model(version) is None:
        raise KeyError(f"candidate {version} is not registered")
    result = store.set_champion(version)
    store.log_event("candidate_promoted", version, result)
    LOGGER.info("promoted %s (previous %s)", version, result["previous"])
    return result


def reject_candidate(store: RuntimeStore, version: str | None, gate_result: dict) -> None:
    store.log_event(
        "candidate_rejected",
        version,
        {"gates": gate_result.get("gates"), "reasons": gate_result.get("reasons")},
    )


def rollback(store: RuntimeStore, version: str | None = None) -> dict:
    result = store.rollback(version)
    store.log_event("rollback_completed", result["version"], result)
    return result
