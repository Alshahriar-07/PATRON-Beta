"""Live prediction engine: champion model + validated window -> immutable forecast.

The engine does not invent a second inference path. It feeds the frozen
96-candle contract through :func:`patron.inference.pipeline.predict` and wraps the
schema-validated model output in an auditable envelope (identity, input hash,
target timestamps, freshness, feature snapshot). The model output itself is never
post-processed: if a forecast fails validity checks it is refused, not repaired.
"""

from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from patron.config import PatronConfig
from patron.features.builder import build_features
from patron.inference.pipeline import (
    InputValidationError,
    Predictor,
    load_predictor,
    predict,
    validate_output,
)
from patron.paths import PROJECT_ROOT
from patron.runtime.config import RuntimeConfig
from patron.runtime.store import utc_iso
from patron.runtime.window import WindowState, candles_payload, target_timestamps
from patron.utils.io import file_sha256

LOGGER = logging.getLogger(__name__)

FORECAST_SCHEMA_VERSION = "2.0.0"


class ModelUnavailable(RuntimeError):
    """No usable champion checkpoint (missing registry entry or missing file)."""


class InferenceFailure(RuntimeError):
    """The model raised; the cycle must not publish a forecast."""


@dataclass
class Champion:
    version: str
    name: str
    path: Path
    sha256: str
    predictor: Predictor


@dataclass
class ForecastResult:
    status: str
    envelope: dict | None = None
    record: dict | None = None
    output: dict | None = None
    warnings: list[str] | None = None

    @property
    def forecast_id(self) -> str | None:
        return None if self.envelope is None else self.envelope.get("forecast_id")


def resolve_checkpoint(path_value: str | Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def load_champion(store, cache: dict[str, Champion] | None = None) -> Champion:
    cache = cache if cache is not None else {}
    entry = store.champion()
    if entry is None:
        raise ModelUnavailable("no champion registered in the model registry")
    path = resolve_checkpoint(entry["path"])
    if not path.exists():
        raise ModelUnavailable(f"champion checkpoint missing on disk: {path}")
    actual_sha = file_sha256(path)
    if entry["sha256"] and actual_sha != entry["sha256"]:
        raise ModelUnavailable(
            f"champion checkpoint hash mismatch: registry {entry['sha256'][:12]} != file {actual_sha[:12]}"
        )
    if entry["version"] in cache:
        return cache[entry["version"]]
    predictor = load_predictor(path)
    champion = Champion(
        version=entry["version"],
        name=entry["name"],
        path=path,
        sha256=actual_sha,
        predictor=predictor,
    )
    cache[entry["version"]] = champion
    return champion


def reference_context(window: WindowState, atr_period: int = 14) -> tuple[dict, float, float]:
    """ATR/close at the forecast origin plus the 12-feature snapshot of that bar.

    Reuses the production feature builder over warm-up + window, so the snapshot is
    exactly what the model saw (minus scaling) and can be used for drift detection.
    """
    combined = pd.concat([window.warmup, window.frame], ignore_index=True)
    features = build_features(combined, atr_period=atr_period)
    last = features.iloc[-1]
    snapshot = {
        col: float(last[col])
        for col in features.columns
        if col not in {"timestamp_open", "timestamp_close", "close", "atr14", "is_gap", "feature_valid"}
    }
    return snapshot, float(last["atr14"]), float(last["close"])


def _finite(values: list[float]) -> bool:
    return all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in values)


def check_output_consistency(output: dict) -> list[str]:
    """Validate probabilities and quantile ordering. Returns human-readable problems."""
    problems: list[str] = []
    barrier = output.get("barrier_probabilities", {})
    values = [float(v) for v in barrier.values()]
    if not _finite(values) or any(v < 0 or v > 1 for v in values):
        problems.append("barrier probabilities not finite in [0, 1]")
    elif abs(sum(values) - 1.0) > 1e-3:
        problems.append(f"barrier probabilities sum to {sum(values):.6f}")

    momentum = output.get("momentum", {}).get("class_probabilities", {})
    values = [float(v) for v in momentum.values()]
    if not values:
        problems.append("momentum probabilities missing")
    else:
        if not _finite(values) or any(v < 0 or v > 1 for v in values):
            problems.append("momentum probabilities not finite in [0, 1]")
        elif abs(sum(values) - 1.0) > 1e-3:
            problems.append(f"momentum probabilities sum to {sum(values):.6f}")

    for step in output.get("range_quantiles", []):
        for field in ("high", "low", "close"):
            band = step.get(field, {})
            q = [band.get("q10"), band.get("q50"), band.get("q90")]
            if any(value is None for value in q):
                problems.append(f"{field} quantiles missing at step {step.get('step')}")
                continue
            values = [float(value) for value in q]
            if not _finite(values):
                problems.append(f"{field} quantiles not finite at step {step.get('step')}")
            elif not (values[0] <= values[1] <= values[2]):
                problems.append(f"{field} quantiles not ordered at step {step.get('step')}")
        if step["low"]["q50"] > step["high"]["q50"]:
            problems.append(f"predicted low above high at step {step['step']}")

    for name, value in output.get("pattern_scores", {}).items():
        if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            problems.append(f"pattern score out of range: {name}")
            break
    return problems


def build_envelope(
    cfg: RuntimeConfig,
    model_cfg: PatronConfig,
    champion: Champion,
    window: WindowState,
    output: dict,
    feature_snapshot: dict,
    atr: float,
    reference_close: float,
    now: datetime,
    forecast_id: str,
    source: str,
    extra_warnings: list[str] | None = None,
) -> dict:
    candle_open = pd.Timestamp(window.frame["timestamp_open"].iloc[-1])
    as_of = utc_iso(window.frame["timestamp_close"].iloc[-1])
    generated_at = utc_iso(now)
    targets = target_timestamps(candle_open, cfg.timeframe, model_cfg.horizon)
    age_seconds = max(0.0, (pd.Timestamp(now) - pd.Timestamp(as_of)).total_seconds())
    warnings = list(window.warnings) + list(extra_warnings or [])
    warnings += [f"model: {w}" for w in output.get("data_quality", {}).get("warnings", [])]
    return {
        "schema_version": FORECAST_SCHEMA_VERSION,
        "forecast_id": forecast_id,
        "status": output.get("status", "NO_OUTPUT"),
        "generated_at": generated_at,
        "as_of": as_of,
        "symbol": cfg.symbol,
        "timeframe": cfg.timeframe,
        "candle_open": utc_iso(candle_open),
        "model": {
            "name": champion.name,
            "version": champion.version,
            "checkpoint": str(champion.path),
            "checkpoint_sha256": champion.sha256,
            "project_version": f"{model_cfg.project.name.lower()}-{model_cfg.project.version}",
            "feature_version": output.get("feature_version"),
            "calibration_version": (
                f"temperature-{champion.predictor.temperature:.4f}"
                if champion.predictor.temperature is not None
                else None
            ),
        },
        "input_window": {
            "candles": int(len(window.frame)),
            "start": utc_iso(window.frame["timestamp_open"].iloc[0]),
            "end": as_of,
            "hash": window.input_hash,
            "warmup_candles": int(len(window.warmup)),
            "sequence_length": model_cfg.sequence_length,
            "feature_count": model_cfg.feature_count,
        },
        "targets": targets,
        "horizons": [
            {"step": target["step"], "target_ts": target["target_ts"], "status": "PENDING"}
            for target in targets
        ],
        "reference": {
            "close": reference_close,
            "atr14": atr,
            "feature_snapshot": feature_snapshot,
        },
        "freshness": {
            "as_of": as_of,
            "age_seconds": round(age_seconds, 3),
            "source": source,
            "stale": bool(age_seconds > cfg.monitoring.stale_after_seconds),
        },
        "model_output": output,
        "warnings": warnings,
    }


def generate_forecast(
    store,
    cfg: RuntimeConfig,
    model_cfg: PatronConfig,
    champion: Champion,
    window: WindowState,
    now: datetime | None = None,
    source: str = "",
    forecast_id: str | None = None,
) -> ForecastResult:
    now = now or datetime.now(timezone.utc)
    if not window.ok:
        return ForecastResult(status=window.status, warnings=[window.reason] if window.reason else [])

    payload = {"symbol": cfg.symbol, "timeframe": cfg.timeframe, "candles": candles_payload(window.frame)}
    predictor: Predictor = champion.predictor
    predictor.warmup_frame = window.warmup
    try:
        output = predict(payload, predictor, now=now)
    except InputValidationError as exc:
        return ForecastResult(status="INVALID_INPUT", warnings=[str(exc)])
    except Exception as exc:  # model/runtime failure: never publish silently
        raise InferenceFailure(f"inference failed: {exc}") from exc

    if output.get("status") != "OK":
        return ForecastResult(status=output.get("status", "NO_OUTPUT"), output=output,
                              warnings=output.get("data_quality", {}).get("warnings", []))

    try:
        validate_output(output)
    except Exception as exc:
        return ForecastResult(status="NO_OUTPUT", output=output,
                              warnings=[f"output schema violation: {exc}"])

    problems = check_output_consistency(output)
    if problems:
        return ForecastResult(status="NO_OUTPUT", output=output, warnings=problems)

    snapshot, atr, reference_close = reference_context(window, model_cfg.forecast.barrier.atr_period)
    forecast_id = forecast_id or _forecast_id(window)
    envelope = build_envelope(
        cfg, model_cfg, champion, window, output, snapshot, atr, reference_close, now, forecast_id, source
    )
    record = {
        "forecast_id": forecast_id,
        "created_at": envelope["generated_at"],
        "as_of": envelope["as_of"],
        "symbol": cfg.symbol,
        "timeframe": cfg.timeframe,
        "candle_ts": envelope["candle_open"],
        "input_start": envelope["input_window"]["start"],
        "input_end": envelope["input_window"]["end"],
        "input_hash": envelope["input_window"]["hash"],
        "model_name": champion.name,
        "model_version": champion.version,
        "checkpoint": str(champion.path),
        "checkpoint_sha": champion.sha256,
        "feature_version": output.get("feature_version", ""),
        "calibration_version": envelope["model"]["calibration_version"],
        "reference_close": reference_close,
        "reference_atr": atr,
        "source": source,
        "payload": _dump(envelope),
        "model_output": _dump(output),
    }
    return ForecastResult(status="OK", envelope=envelope, record=record, output=output)


def _dump(payload: dict) -> str:
    import json

    return json.dumps(payload, default=str)


def _forecast_id(window: WindowState) -> str:
    stamp = utc_iso(window.frame["timestamp_open"].iloc[-1]).replace("-", "").replace(":", "")
    return f"{stamp[:15]}-{uuid.uuid4().hex[:8]}"
