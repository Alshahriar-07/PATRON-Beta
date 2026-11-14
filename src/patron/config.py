"""Typed configuration loading and validation.

The YAML in ``config/model_config.yaml`` is the single source of truth. Every
value used by the code path is validated here so a malformed config fails fast
instead of silently changing model semantics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from patron.paths import DEFAULT_CONFIG_PATH, ensure_dirs


class ProjectCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    full_name: str
    version: str
    task: str


class MarketCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    exchange: str
    market_type: str
    symbol: str
    timeframe: str
    context_timeframes: list[str]
    history_years_target: float
    only_closed_candles: bool = True


class InputCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sequence_length: int = Field(gt=0)
    feature_count: int = Field(gt=0)
    dtype: str
    order: str
    feature_order: list[str]

    @model_validator(mode="after")
    def _check_features(self) -> "InputCfg":
        if len(self.feature_order) != self.feature_count:
            raise ValueError(
                f"feature_order has {len(self.feature_order)} entries but "
                f"feature_count is {self.feature_count}"
            )
        if len(set(self.feature_order)) != len(self.feature_order):
            raise ValueError("feature_order contains duplicates")
        if self.order != "oldest_to_newest":
            raise ValueError("PATRON v1 requires oldest_to_newest input order")
        return self


class BarrierCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    atr_period: int = Field(gt=0)
    tp_atr_multiplier: float = Field(gt=0)
    sl_atr_multiplier: float = Field(gt=0)
    ambiguous_same_candle: str


class RangeCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fields: list[str]
    quantiles: list[float]
    per_future_candle: bool
    target_space: str

    @model_validator(mode="after")
    def _check_range(self) -> "RangeCfg":
        if sorted(self.fields) != sorted(["high", "low", "close"]):
            raise ValueError("range fields must be exactly high/low/close")
        if len(self.quantiles) != 3 or self.quantiles != sorted(self.quantiles):
            raise ValueError("range quantiles must be 3 ascending values")
        return self


class MomentumCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    classes: list[str]
    normalized_return_boundaries: list[float]

    @model_validator(mode="after")
    def _check_momentum(self) -> "MomentumCfg":
        if len(self.classes) != 5:
            raise ValueError("momentum head requires exactly 5 classes")
        if len(self.normalized_return_boundaries) != 4:
            raise ValueError("expected 4 momentum boundaries for 5 classes")
        if self.normalized_return_boundaries != sorted(
            self.normalized_return_boundaries
        ):
            raise ValueError("momentum boundaries must be ascending")
        return self


class ForecastCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    horizon_candles: int = Field(gt=0)
    horizon_minutes: int
    barrier: BarrierCfg
    range: RangeCfg
    momentum: MomentumCfg
    pattern_tags: list[str]

    @model_validator(mode="after")
    def _check_horizon(self) -> "ForecastCfg":
        if self.horizon_candles * 15 != self.horizon_minutes:
            raise ValueError("horizon_minutes must equal horizon_candles * 15")
        if len(self.pattern_tags) != 12:
            raise ValueError("PATRON v1 expects 12 pattern tag families")
        return self


class ModelCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    architecture: str
    input_projection_dim: int
    cnn_channels: list[int]
    cnn_kernels: list[int]
    bilstm_hidden_per_direction: int
    bilstm_layers: int
    attention_heads: int
    attention_dim: int
    shared_dim: int
    dropout: float
    target_trainable_parameters: list[int]


class LossWeightsCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    barrier: float
    quantile: float
    momentum: float
    pattern: float


class TrainingCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    framework: str
    optimizer: str
    learning_rate: float
    weight_decay: float
    batch_size: int
    max_epochs: int
    early_stopping_patience: int
    gradient_clip_norm: float
    random_seed: int
    loss_weights: LossWeightsCfg


class ValidationCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    method: str
    purge_bars: int
    embargo_bars: int
    final_test_untouched: bool
    calibrator: str
    baseline_models: list[str]


class RuntimeCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    inference_batch_size: int
    reject_nan_inf: bool
    reject_unclosed_candle: bool
    reject_stale_data: bool
    live_order_execution: bool


class DataCfg(BaseModel):
    """Download/storage settings (additive section, see DECISIONS in memory.md)."""

    model_config = ConfigDict(extra="forbid")
    source: str
    base_url: str
    rest_base_url: str
    start: str
    end: str | None = None
    timeframes: list[str]
    request_pause_seconds: float = 0.15
    raw_overwrite: bool = False


class PatronConfig(BaseModel):
    """Fully validated PATRON configuration."""

    model_config = ConfigDict(extra="forbid")
    project: ProjectCfg
    market: MarketCfg
    input: InputCfg
    forecast: ForecastCfg
    model: ModelCfg
    training: TrainingCfg
    validation: ValidationCfg
    runtime: RuntimeCfg
    data: DataCfg

    # ---------------------------------------------------------------- loading
    @classmethod
    def load(cls, path: str | Path | None = None) -> "PatronConfig":
        cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
        if not cfg_path.exists():
            raise FileNotFoundError(f"config file not found: {cfg_path}")
        with cfg_path.open("r", encoding="utf-8") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh)
        return cls.model_validate(raw)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def save_snapshot(self, directory: str | Path) -> Path:
        """Persist the exact config used by a run (experiment reproducibility)."""
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "config_snapshot.yaml"
        with out.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)
        return out

    # --------------------------------------------------------------- helpers
    @property
    def sequence_length(self) -> int:
        return self.input.sequence_length

    @property
    def feature_count(self) -> int:
        return self.input.feature_count

    @property
    def feature_order(self) -> list[str]:
        return list(self.input.feature_order)

    @property
    def horizon(self) -> int:
        return self.forecast.horizon_candles

    @property
    def pattern_tags(self) -> list[str]:
        return list(self.forecast.pattern_tags)

    @property
    def quantiles(self) -> list[float]:
        return list(self.forecast.range.quantiles)

    @property
    def momentum_classes(self) -> list[str]:
        return list(self.forecast.momentum.classes)

    @property
    def symbol(self) -> str:
        return self.market.symbol

    @property
    def timeframe(self) -> str:
        return self.market.timeframe

    def describe(self) -> str:
        return (
            f"{self.project.name} {self.project.version} | {self.symbol} "
            f"{self.market.timeframe} | {self.sequence_length}x{self.feature_count} "
            f"-> {self.horizon} candles"
        )


def load_config(path: str | Path | None = None) -> PatronConfig:
    """Load and validate the PATRON config, creating output dirs as a side effect."""
    ensure_dirs()
    return PatronConfig.load(path)
