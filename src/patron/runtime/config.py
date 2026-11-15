"""Typed runtime configuration for the autonomous PATRON service.

The model config (``config/model_config.yaml``) stays frozen: it describes the
network and the inference contract. Everything an *operations* layer needs
(poll cadence, confirmation delay, retraining thresholds, promotion gates) lives
here so that changing policy never changes model semantics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from patron.paths import CONFIG_DIR

RUNTIME_CONFIG_PATH = CONFIG_DIR / "runtime.yaml"


class SourceCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    venue: str = "binance"
    market_type: str = "spot"
    confirm_delay_seconds: float = Field(ge=0, default=20)
    lookback_hours: int = Field(gt=0, default=48)
    request_pause_seconds: float = Field(ge=0, default=0.15)


class HistoryCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sequence_length: int = Field(gt=0, default=96)
    warmup_bars: int = Field(ge=0, default=384)


class StorageCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    database: str = "data/runtime/patron.db"
    output_dir: str = "reports/live"
    forecast_ledger: str = "data/runtime/forecasts"


class DegradationCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    metric: str = "barrier_accuracy"
    floor: float = 0.30
    min_samples: int = Field(gt=0, default=200)
    window_forecasts: int = Field(gt=0, default=300)


class MonitoringCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    window_forecasts: int = Field(gt=0, default=500)
    min_samples: int = Field(gt=0, default=50)
    coverage_bounds: list[float] = Field(default_factory=lambda: [0.60, 0.99])
    stale_after_seconds: float = Field(gt=0, default=1200)
    degradation: DegradationCfg = Field(default_factory=DegradationCfg)

    @model_validator(mode="after")
    def _check_bounds(self) -> "MonitoringCfg":
        lo, hi = self.coverage_bounds
        if not 0 < lo < hi <= 1:
            raise ValueError("coverage_bounds must be two ascending probabilities")
        return self


class DriftCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    min_samples: int = Field(gt=0, default=100)
    feature_z_threshold: float = Field(gt=0, default=3.0)
    psi_threshold: float = Field(gt=0, default=0.25)
    min_drifting_features: int = Field(gt=0, default=3)
    outcome_psi_threshold: float = Field(gt=0, default=0.25)


class LearningCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    min_new_matured_examples: int = Field(gt=0, default=288)
    schedule_hours: float = Field(gt=0, default=168)
    cooldown_hours: float = Field(ge=0, default=24)
    replay_dataset: str = "data/features/BTCUSDT_dataset.parquet"
    recent_fraction: float = Field(gt=0, le=1, default=0.35)
    train_epochs: int = Field(gt=0, default=6)
    train_stride: int = Field(ge=1, default=2)
    threads: int = Field(gt=0, default=6)
    max_runtime_minutes: float = Field(gt=0, default=120)
    command: list[str] = Field(default_factory=list)


class GovernanceCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    min_validation_samples: int = Field(gt=0, default=2000)
    max_logloss_delta: float = Field(ge=0, default=0.02)
    max_accuracy_drop: float = Field(ge=0, default=0.01)
    coverage_bounds: list[float] = Field(default_factory=lambda: [0.60, 0.99])
    min_seeds: int = Field(gt=0, default=1)
    mandatory_gates: list[str] = Field(
        default_factory=lambda: [
            "schema_valid",
            "probabilities_valid",
            "min_samples",
            "logloss_not_worse",
            "accuracy_not_worse",
            "coverage_ok",
        ]
    )


class ServerCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    poll_seconds: float = Field(gt=0, default=20)
    heartbeat_seconds: float = Field(gt=0, default=60)
    max_cycles_per_run: int = Field(ge=0, default=0)
    shutdown_timeout_seconds: float = Field(gt=0, default=5)


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: str = "1.0.0"
    symbol: str = "BTC/USDT"
    timeframe: str = "15m"
    source: SourceCfg = Field(default_factory=SourceCfg)
    history: HistoryCfg = Field(default_factory=HistoryCfg)
    storage: StorageCfg = Field(default_factory=StorageCfg)
    monitoring: MonitoringCfg = Field(default_factory=MonitoringCfg)
    drift: DriftCfg = Field(default_factory=DriftCfg)
    learning: LearningCfg = Field(default_factory=LearningCfg)
    governance: GovernanceCfg = Field(default_factory=GovernanceCfg)
    server: ServerCfg = Field(default_factory=ServerCfg)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "RuntimeConfig":
        cfg_path = Path(path) if path is not None else RUNTIME_CONFIG_PATH
        if not cfg_path.exists():
            raise FileNotFoundError(f"runtime config not found: {cfg_path}")
        with cfg_path.open("r", encoding="utf-8") as fh:
            return cls.model_validate(yaml.safe_load(fh))

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")


def load_runtime_config(path: str | Path | None = None) -> RuntimeConfig:
    return RuntimeConfig.load(path)


#: Explicit statuses the runtime may publish. A forecast is only ever ``OK``
#: when every upstream check passed.
STATUSES = (
    "OK",
    "PROVISIONAL",
    "NO_OUTPUT",
    "INVALID_INPUT",
    "INSUFFICIENT_HISTORY",
    "STALE_DATA",
    "DATA_GAP",
    "MODEL_UNAVAILABLE",
    "TRAINING_IN_PROGRESS",
    "RECOVERING",
)

Status = Literal[
    "OK",
    "PROVISIONAL",
    "NO_OUTPUT",
    "INVALID_INPUT",
    "INSUFFICIENT_HISTORY",
    "STALE_DATA",
    "DATA_GAP",
    "MODEL_UNAVAILABLE",
    "TRAINING_IN_PROGRESS",
    "RECOVERING",
]
