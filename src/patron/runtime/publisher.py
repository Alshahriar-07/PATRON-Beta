"""Versioned JSON outputs for the live service.

The forecast *ledger* is the per-forecast files plus the SQLite rows; these files are
the current view an API or dashboard reads. ``latest_forecast.json`` is a pointer,
so it carries the freshness flag and the current status next to the immutable
forecast — a stale forecast is never presented as a fresh ``OK`` one.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from patron.paths import PROJECT_ROOT
from patron.runtime.config import RuntimeConfig
from patron.runtime.store import RuntimeStore, utc_iso

SCHEMA_VERSION = "1.0.0"


def output_dir(cfg: RuntimeConfig) -> Path:
    path = Path(cfg.storage.output_dir)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def ledger_dir(cfg: RuntimeConfig) -> Path:
    path = Path(cfg.storage.forecast_ledger)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write(payload: dict, path: Path) -> Path:
    from patron.utils.io import write_json

    return write_json(payload, path)


def horizon_statuses(store: RuntimeStore, forecast_id: str, envelope: dict) -> list[dict]:
    rows = {int(row["step"]): row for row in store.evaluations(forecast_id)}
    out = []
    for target in envelope.get("targets", []):
        step = int(target["step"])
        row = rows.get(step)
        if row is None:
            out.append({**target, "status": "PENDING"})
        else:
            metrics = json.loads(row["metrics"]) if row.get("metrics") else {}
            out.append(
                {
                    **target,
                    "status": row["status"],
                    "evaluated_at": row["evaluated_at"],
                    "pinball_mean": metrics.get("pinball_mean"),
                    "direction_correct": (metrics.get("direction") or {}).get("correct"),
                    "coverage": metrics.get("coverage"),
                }
            )
    return out


def publish_forecast(store: RuntimeStore, cfg: RuntimeConfig, envelope: dict, performance: dict | None = None) -> Path:
    """Persist the immutable record copy and refresh the latest pointer."""
    forecast_id = envelope["forecast_id"]
    record_path = ledger_dir(cfg) / f"{forecast_id}.json"
    _write(envelope, record_path)

    latest = dict(envelope)
    latest["horizons"] = horizon_statuses(store, forecast_id, envelope)
    latest["performance_summary"] = performance or {}
    latest["current_status"] = envelope.get("status", "OK")
    latest["stale"] = False
    latest["published_at"] = utc_iso(datetime.now(timezone.utc))
    return _write(latest, output_dir(cfg) / "latest_forecast.json")


def refresh_latest(
    store: RuntimeStore,
    cfg: RuntimeConfig,
    current_status: str,
    performance: dict | None = None,
    now: datetime | None = None,
) -> Path | None:
    """Re-link maturation results onto the latest pointer without touching the ledger."""
    path = output_dir(cfg) / "latest_forecast.json"
    if not path.exists():
        return None
    latest = json.loads(path.read_text(encoding="utf-8"))
    forecast_id = latest.get("forecast_id")
    record = store.forecast(forecast_id) if forecast_id else None
    if record is None:
        return None
    envelope = json.loads(record["payload"])
    latest["horizons"] = horizon_statuses(store, forecast_id, envelope)
    latest["current_status"] = current_status
    latest["performance_summary"] = performance or latest.get("performance_summary", {})
    now = now or datetime.now(timezone.utc)
    age = (now - datetime.fromisoformat(envelope["as_of"].replace("Z", "+00:00"))).total_seconds()
    latest["stale"] = bool(age > cfg.monitoring.stale_after_seconds or current_status not in {"OK", "PROVISIONAL"})
    latest["published_at"] = utc_iso(datetime.now(timezone.utc))
    return _write(latest, path)


def publish_status(
    cfg: RuntimeConfig,
    status: str,
    store: RuntimeStore,
    detail: str = "",
    last_cycle: dict | None = None,
    champion: dict | None = None,
    extra: dict | None = None,
    now: datetime | None = None,
) -> Path:
    now = now or datetime.now(timezone.utc)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "as_of": utc_iso(now),
        "symbol": cfg.symbol,
        "timeframe": cfg.timeframe,
        "detail": detail,
        "last_cycle": last_cycle,
        "active_model": None
        if champion is None
        else {
            "version": champion["version"],
            "path": champion["path"],
            "sha256": champion["sha256"],
        },
        "freshness": _freshness(cfg, store, now),
    }
    if extra:
        payload.update(extra)
    return _write(payload, output_dir(cfg) / "status.json")


def _freshness(cfg: RuntimeConfig, store: RuntimeStore, now: datetime | None = None) -> dict:
    latest = store.recent_forecasts(1)
    if not latest:
        return {"has_forecast": False, "age_seconds": None, "stale": True}
    envelope = json.loads(latest[0]["payload"])
    as_of = envelope["as_of"]
    now = now or datetime.now(timezone.utc)
    age = (now - datetime.fromisoformat(as_of.replace("Z", "+00:00"))).total_seconds()
    return {
        "has_forecast": True,
        "as_of": as_of,
        "forecast_id": envelope["forecast_id"],
        "age_seconds": round(age, 1),
        "stale": age > cfg.monitoring.stale_after_seconds,
    }


def publish_performance(cfg: RuntimeConfig, performance: dict, degradation: dict) -> Path:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "as_of": utc_iso(datetime.now(timezone.utc)),
        "metrics": performance,
        "degradation": degradation,
        "source": "matured forecasts only",
    }
    return _write(payload, output_dir(cfg) / "performance.json")


def publish_drift(cfg: RuntimeConfig, drift: dict) -> Path:
    payload = {"schema_version": SCHEMA_VERSION, "as_of": utc_iso(datetime.now(timezone.utc)), **drift}
    return _write(payload, output_dir(cfg) / "drift.json")


def publish_models(cfg: RuntimeConfig, store: RuntimeStore) -> Path:
    models = []
    for row in store.models():
        models.append(
            {
                "version": row["version"],
                "role": row["role"],
                "path": row["path"],
                "sha256": row["sha256"],
                "promoted_at": row["promoted_at"],
                "metrics": json.loads(row["metrics"]) if row["metrics"] else None,
                "previous_version": row["previous_version"],
                "notes": row["notes"],
            }
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "as_of": utc_iso(datetime.now(timezone.utc)),
        "champion": (store.champion() or {}).get("version"),
        "models": models,
    }
    return _write(payload, output_dir(cfg) / "models.json")


def publish_training(cfg: RuntimeConfig, store: RuntimeStore, decision: dict | None, jobs: list[dict]) -> Path:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "as_of": utc_iso(datetime.now(timezone.utc)),
        "decision": decision,
        "jobs": [
            {
                key: job.get(key)
                for key in ("job_id", "status", "created_at", "started_at", "finished_at", "run_dir", "result")
            }
            for job in jobs
        ],
    }
    return _write(payload, output_dir(cfg) / "training.json")


def publish_health(cfg: RuntimeConfig, health: dict) -> Path:
    payload = {"schema_version": SCHEMA_VERSION, "as_of": utc_iso(datetime.now(timezone.utc)), **health}
    return _write(payload, output_dir(cfg) / "health.json")


def append_history(cfg: RuntimeConfig, envelope: dict) -> None:
    path = output_dir(cfg) / "forecast_history.jsonl"
    line = json.dumps(
        {
            "schema_version": SCHEMA_VERSION,
            "forecast_id": envelope["forecast_id"],
            "as_of": envelope["as_of"],
            "status": envelope["status"],
            "model_version": envelope["model"]["version"],
        },
        default=str,
    )
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
