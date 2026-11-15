"""The autonomous runtime: one idempotent, logged cycle per closed 15m candle.

    FETCH/CONFIRM -> VALIDATE -> MATURE FORECASTS -> METRICS -> ROLLING WINDOW
        -> NEW FORECAST -> PERSIST -> PUBLISH -> (TRIGGER / TRAIN IN BACKGROUND)

Every stage is separately logged. Reprocessing the same candle writes no duplicate
forecast and no duplicate evaluation because both are keyed in the store. A failed
fetch or an invalid window publishes an explicit non-OK status; it never emits a
forecast built from missing or stale candles.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Event

import pandas as pd

from patron.config import PatronConfig, load_config
from patron.data.markets import TIMEFRAME_MINUTES
from patron.paths import DATA_DIR, EXPERIMENT_DIR, PROJECT_ROOT, ensure_dirs
from patron.runtime.collector import CandleSource, SourceError, VenueSource, default_window
from patron.runtime.config import RuntimeConfig, load_runtime_config
from patron.runtime.drift import build_reference, detect_drift
from patron.runtime.engine import (
    InferenceFailure,
    ModelUnavailable,
    generate_forecast,
    load_champion,
    resolve_checkpoint,
)
from patron.runtime.jobs import TrainingJobManager
from patron.runtime.learning import TriggerDecision, matured_horizon_count
from patron.runtime.maturation import mature_forecasts
from patron.runtime.metrics import degradation_state, rolling_performance
from patron.runtime.publisher import (
    append_history,
    refresh_latest,
    publish_drift,
    publish_forecast,
    publish_health,
    publish_models,
    publish_performance,
    publish_status,
    publish_training,
)
from patron.runtime.store import RuntimeStore, parse_ts, utc_iso
from patron.runtime.validation import validate_candles
from patron.runtime.window import build_window
from patron.utils.io import file_sha256, read_json, write_json

LOGGER = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = EXPERIMENT_DIR / "patron_v2_20261114-203859" / "best.pt"


@dataclass
class CycleResult:
    status: str
    candle_ts: str | None = None
    forecast_id: str | None = None
    matured: int = 0
    detail: str = ""
    warnings: list[str] = field(default_factory=list)
    duration_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "candle_ts": self.candle_ts,
            "forecast_id": self.forecast_id,
            "matured": self.matured,
            "detail": self.detail,
            "warnings": self.warnings,
            "duration_ms": round(self.duration_ms, 1),
        }


def best_available_checkpoint() -> Path | None:
    if DEFAULT_CHECKPOINT.exists():
        return DEFAULT_CHECKPOINT
    candidates = sorted(EXPERIMENT_DIR.glob("*/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]
    fallback = PROJECT_ROOT / "checkpoints" / "best.pt"
    return fallback if fallback.exists() else None


def champion_metrics_from_run(run_dir: Path) -> dict:
    history_path = run_dir / "training_history.json"
    metrics: dict = {}
    if history_path.exists():
        history = read_json(history_path)
        records = history.get("history") or []
        best_epoch = int(history.get("best_epoch") or 0)
        best = next((row for row in records if int(row.get("epoch", -1)) == best_epoch), None)
        if best is None and records:
            best = min(records, key=lambda row: row.get("val_loss", float("inf")))
        if best:
            metrics.update(
                {
                    "val_loss": best.get("val_loss"),
                    "val_barrier_accuracy": best.get("val_barrier_accuracy"),
                    "val_momentum_accuracy": best.get("val_momentum_accuracy"),
                    "best_epoch": best_epoch,
                }
            )
    report_path = PROJECT_ROOT / "reports" / "evaluation_report.json"
    if report_path.exists():
        report = read_json(report_path)
        val = report.get("val", {})
        metrics["evaluation"] = {
            "val_samples": (val.get("quantile") or {}).get("n"),
            "coverage": {
                field: (val.get("quantile") or {}).get("per_field", {}).get(field, {}).get("coverage_q10_q90")
                for field in ("high", "low", "close")
            },
            "barrier_accuracy": (val.get("barrier") or {}).get("accuracy"),
        }
    return metrics


class PatronRuntime:
    def __init__(
        self,
        runtime_cfg: RuntimeConfig | None = None,
        runtime_cfg_path: str | Path | None = None,
        model_cfg_path: str | Path | None = None,
        store: RuntimeStore | None = None,
        source: CandleSource | None = None,
        runner=None,
        clock=None,
        allow_training: bool = True,
        model_cfg: PatronConfig | None = None,
    ) -> None:
        ensure_dirs()
        self.cfg = runtime_cfg or load_runtime_config(runtime_cfg_path)
        self.model_cfg = model_cfg or load_config(model_cfg_path)
        self.store = store or RuntimeStore(self._resolve(self.cfg.storage.database))
        self.source = source or VenueSource(
            venue=self.cfg.source.venue,
            market_type=self.cfg.source.market_type,
            pause=self.cfg.source.request_pause_seconds,
        )
        self.manager = TrainingJobManager(self.store, self.cfg, self.model_cfg, runner=runner)
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.allow_training = allow_training
        self._champion_cache: dict = {}
        self.last_decision: TriggerDecision | None = None
        self.started_at = self.clock()
        self.ensure_champion()

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _resolve(value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else PROJECT_ROOT / path

    def ensure_champion(self) -> dict | None:
        """Register the shipped checkpoint once so the service has a champion."""
        existing = self.store.champion()
        if existing is not None:
            return existing
        checkpoint = best_available_checkpoint()
        if checkpoint is None:
            LOGGER.warning("no checkpoint found; service will publish MODEL_UNAVAILABLE")
            return None
        run_dir = checkpoint.parent
        metrics = champion_metrics_from_run(run_dir)
        reference_path = self._resolve("data/runtime/reference.json")
        reference = None
        if reference_path.exists():
            reference = read_json(reference_path)
        else:
            replay = self._resolve(self.cfg.learning.replay_dataset)
            if replay.exists():
                reference = build_reference(replay, self.model_cfg.feature_order)
                write_json(reference, reference_path)
        self.store.register_model(
            version=run_dir.name,
            name=self.model_cfg.project.name.lower(),
            path=str(checkpoint),
            sha256=file_sha256(checkpoint),
            role="champion",
            metrics=metrics,
            notes="bootstrapped from the shipped v2 checkpoint",
            reference=reference,
        )
        self.store.log_event("champion_bootstrapped", run_dir.name, {"path": str(checkpoint)})
        return self.store.champion()

    def current_champion(self):
        champion = self.store.champion()
        return None if champion is None else load_champion(self.store, self._champion_cache)

    # -------------------------------------------------------------------- cycle
    def cycle(self, now: datetime | None = None) -> CycleResult:
        started = time.perf_counter()
        now = now or self.clock()
        now_ts = pd.Timestamp(now).tz_convert("UTC") if pd.Timestamp(now).tzinfo else pd.Timestamp(now, tz="UTC")
        result = CycleResult(status="OK")

        # 1. FETCH / CONFIRM -----------------------------------------------------
        start, end = default_window(now, self.cfg.source.lookback_hours)
        try:
            raw = self.source.fetch(self.cfg.symbol, self.cfg.timeframe, start, end)
        except SourceError as exc:
            result.status = "STALE_DATA"
            result.detail = str(exc)
            self.store.log_event("cycle_fetch_failed", None, {"error": str(exc)})
            self._mark_failed_cycle(now, result, started)
            self._publish(result, now)
            result.duration_ms = (time.perf_counter() - started) * 1000
            return result

        # 2. VALIDATE -----------------------------------------------------------
        validation = validate_candles(
            raw, self.cfg.timeframe, now=now,
            confirm_delay_seconds=self.cfg.source.confirm_delay_seconds,
        )
        result.warnings = list(validation.warnings)
        if not validation.ok:
            result.status = validation.status
            result.detail = "; ".join(validation.invalid_reasons) or "no confirmed closed candle"
            self.store.log_event(
                "cycle_validation_failed", None,
                {"status": validation.status, "reasons": validation.invalid_reasons,
                 "warnings": validation.warnings},
            )
            self._mark_failed_cycle(now, result, started)
            self._publish(result, now)
            result.duration_ms = (time.perf_counter() - started) * 1000
            return result

        counts = self.store.upsert_candles(
            validation.frame, self.cfg.symbol, self.cfg.timeframe, source=self.source.name, fetched_at=now
        )
        self.store.log_event("candles_ingested", utc_iso(now), counts)

        latest = self.store.latest_candles(self.cfg.symbol, self.cfg.timeframe, 1)
        if latest.empty:
            result.status = "INSUFFICIENT_HISTORY"
            result.detail = "no candles stored after ingestion"
            self._mark_failed_cycle(now, result, started)
            self._publish(result, now)
            result.duration_ms = (time.perf_counter() - started) * 1000
            return result
        candle_ts = utc_iso(latest["timestamp_open"].iloc[0])
        result.candle_ts = candle_ts
        cycle_key = f"{self.cfg.symbol}:{self.cfg.timeframe}:{candle_ts}"

        # 3. MATURE EVERY HORIZON WHOSE CANDLE IS NOW STORED --------------------
        maturation = mature_forecasts(self.store, self.cfg, self.model_cfg, now=now)
        result.matured = int(maturation["count"])

        # 4. METRICS / DRIFT / DEGRADATION -------------------------------------
        performance = rolling_performance(self.store, self.cfg, self.model_cfg)
        champion_row = self.store.champion()
        reference = None
        if champion_row and champion_row.get("reference"):
            reference = json.loads(champion_row["reference"])
        drift = detect_drift(self.store, self.cfg, reference, self.model_cfg)
        degradation = degradation_state(self.store, self.cfg, self.model_cfg)

        # 5. ROLLING WINDOW + 6. NEW FORECAST ----------------------------------
        existing = self.store.forecast_for_candle(self.cfg.symbol, self.cfg.timeframe, candle_ts)
        if existing is not None:
            result.status = "OK"
            result.forecast_id = existing["forecast_id"]
            result.detail = "candle already has an immutable forecast"
        else:
            window = build_window(
                self.store, self.cfg.symbol, self.cfg.timeframe,
                self.cfg.history.sequence_length, self.cfg.history.warmup_bars,
            )
            result.warnings.extend(window.warnings)
            if not window.ok:
                result.status = window.status
                result.detail = window.reason or window.status
            else:
                try:
                    champion = self.current_champion()
                except ModelUnavailable as exc:
                    champion = None
                    result.status = "MODEL_UNAVAILABLE"
                    result.detail = str(exc)
                if champion is not None:
                    try:
                        forecast = generate_forecast(
                            self.store, self.cfg, self.model_cfg, champion, window,
                            now=now, source=self.source.name,
                        )
                    except InferenceFailure as exc:
                        forecast = None
                        result.status = "NO_OUTPUT"
                        result.detail = str(exc)
                        self.store.log_event("inference_failed", None, {"error": str(exc)})
                    if forecast is not None:
                        result.warnings.extend(forecast.warnings or [])
                        result.status = forecast.status
                        if forecast.status == "OK" and forecast.record is not None:
                            if self.store.insert_forecast(forecast.record):
                                result.forecast_id = forecast.forecast_id
                                publish_forecast(self.store, self.cfg, forecast.envelope, performance)
                                append_history(self.cfg, forecast.envelope)
                                self.store.log_event(
                                    "forecast_created", forecast.forecast_id,
                                    {"candle_ts": candle_ts, "model": forecast.record["model_version"]},
                                )
                            else:
                                result.detail = "forecast insert collided (idempotent)"
                        else:
                            result.detail = "; ".join(forecast.warnings or [result.status])
                            publish_status(self.cfg, result.status, self.store, detail=result.detail,
                                           champion=champion_row, now=now)

        # 7. PERSIST CYCLE + PUBLISH -------------------------------------------
        self.store.mark_cycle(
            cycle_key, candle_ts, result.status, result.detail,
            duration_ms=(time.perf_counter() - started) * 1000,
        )
        self._publish(result, now, performance=performance, drift=drift, degradation=degradation)

        # 8. LEARNING TRIGGERS (background, never blocks inference) ------------
        if self.allow_training:
            self.manager.tick()
            if self.manager.maybe_start(drift=drift, degradation=degradation) is not None:
                publish_training(
                    self.cfg, self.store, self.store.get_kv("learning.last_decision"),
                    self.store.jobs(10),
                )

        result.duration_ms = (time.perf_counter() - started) * 1000
        return result

    def _mark_failed_cycle(self, now: datetime, result: CycleResult, started: float) -> None:
        """Record a non-forecast cycle so health/staleness stays truthful."""
        key = f"{self.cfg.symbol}:{self.cfg.timeframe}:attempt:{utc_iso(now)}"
        self.store.mark_cycle(
            key, utc_iso(now), result.status, result.detail,
            duration_ms=(time.perf_counter() - started) * 1000,
        )

    # ------------------------------------------------------------------ publish
    def _publish(
        self,
        result: CycleResult,
        now: datetime,
        performance: dict | None = None,
        drift: dict | None = None,
        degradation: dict | None = None,
    ) -> None:
        champion = self.store.champion()
        publish_status(
            self.cfg, result.status, self.store, detail=result.detail,
            last_cycle=self.store.last_cycle(), champion=champion,
            extra={"cycle": result.to_dict()}, now=now,
        )
        if performance is not None:
            publish_performance(self.cfg, performance, degradation or {})
        if drift is not None:
            publish_drift(self.cfg, drift)
        refresh_latest(self.store, self.cfg, result.status, performance, now=now)
        publish_models(self.cfg, self.store)
        publish_health(self.cfg, self.health())

    def health(self) -> dict:
        last = self.store.last_cycle()
        champion = self.store.champion()
        disk = shutil.disk_usage(str(PROJECT_ROOT))
        age = None
        healthy = True
        reasons: list[str] = []
        if last is None:
            healthy = False
            reasons.append("no completed cycle yet")
        else:
            finished = parse_ts(last["finished_at"]) if last.get("finished_at") else None
            if finished is not None:
                age = (self.clock() - finished.to_pydatetime()).total_seconds()
                if age > self.cfg.monitoring.stale_after_seconds:
                    healthy = False
                    reasons.append(f"last cycle {age:.0f}s ago (stale after {self.cfg.monitoring.stale_after_seconds:.0f}s)")
            if last["status"] not in {"OK", "NO_OUTPUT"}:
                healthy = False
                reasons.append(f"last cycle status {last['status']}")
        if champion is None:
            healthy = False
            reasons.append("no champion registered")
        try:
            self.store.execute("SELECT 1")
        except Exception as exc:  # pragma: no cover - disk failure path
            healthy = False
            reasons.append(f"database error: {exc}")
        return {
            "healthy": healthy,
            "reasons": reasons,
            "database": str(self.store.path),
            "last_cycle": last,
            "last_cycle_age_seconds": None if age is None else round(age, 1),
            "champion": None if champion is None else champion["version"],
            "source": self.source.name,
            "disk_free_gb": round(disk.free / 1e9, 2),
            "uptime_seconds": round((self.clock() - self.started_at).total_seconds(), 1),
            "training_active": self.store.active_job("candidate") is not None,
            "matured_horizons": matured_horizon_count(self.store),
        }

    # --------------------------------------------------------------------- loop
    def next_boundary(self, now: datetime | None = None) -> datetime:
        now = now or self.clock()
        minutes = TIMEFRAME_MINUTES[self.cfg.timeframe]
        ts = pd.Timestamp(now).tz_convert("UTC") if pd.Timestamp(now).tzinfo else pd.Timestamp(now, tz="UTC")
        floored = ts.floor(f"{minutes}min")
        if ts - floored >= pd.Timedelta(seconds=self.cfg.source.confirm_delay_seconds):
            boundary = floored + pd.Timedelta(minutes=minutes)
        else:
            boundary = floored
        return (boundary + pd.Timedelta(seconds=self.cfg.source.confirm_delay_seconds)).to_pydatetime()

    def run_forever(
        self,
        max_cycles: int | None = None,
        stop_event: Event | None = None,
        sleep=time.sleep,
    ) -> list[CycleResult]:
        stop_event = stop_event or Event()
        results: list[CycleResult] = []
        limit = self.cfg.server.max_cycles_per_run if max_cycles is None else max_cycles
        backoff = 0.0
        while not stop_event.is_set():
            if limit and len(results) >= limit:
                break
            try:
                result = self.cycle()
                results.append(result)
                backoff = 0.0
            except Exception as exc:  # keep the service alive, never fabricate a forecast
                LOGGER.exception("cycle failed")
                self.store.log_event("cycle_exception", None, {"error": str(exc)})
                publish_status(self.cfg, "RECOVERING", self.store, detail=str(exc),
                               last_cycle=self.store.last_cycle(), champion=self.store.champion())
                backoff = min(max(backoff * 2, 5.0), 120.0)
            if limit and len(results) >= limit:
                break
            try:
                wait = (self.next_boundary() - self.clock()).total_seconds()
            except Exception:
                wait = 0.0
            if backoff:
                sleep(min(backoff, 120.0))
            elif wait > 0:
                sleep(min(wait + 0.5, 900.0))
            else:
                sleep(max(self.cfg.server.poll_seconds, 1.0))
        return results
