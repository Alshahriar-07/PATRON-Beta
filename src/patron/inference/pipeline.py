"""PATRON inference: strict input validation -> features -> model -> JSON output.

The contract is `schemas/input.schema.json` in, `schemas/output.schema.json` out.
Features need history beyond the 96 input candles (96-bar volume baseline and
4h ATR warm-up), so a warm-up frame must be supplied; without it the predictor
returns `NO_OUTPUT` instead of silently using a cold-started feature row.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from patron.config import PatronConfig, load_config
from patron.features.builder import build_features
from patron.features.schema import FEATURE_ORDER, FEATURE_VERSION
from patron.features.scaler import RobustFeatureScaler
from patron.labels.barrier import BARRIER_CLASSES
from patron.labels.momentum import MOMENTUM_CLASSES
from patron.models.patron_net import PatronNet, build_model
from patron.paths import CLEAN_DIR, SCHEMA_DIR, symbol_slug
from patron.training.trainer import load_checkpoint

LOGGER = logging.getLogger(__name__)

TIMEFRAME_MINUTES = 15
WARMUP_BARS = 384  # 4 days of 15m bars: enough for the 4h ATR(14) warm-up
CANDLE_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")


class InputValidationError(ValueError):
    """Raised when the inference payload violates the input contract."""


@dataclass
class Predictor:
    cfg: PatronConfig
    model: PatronNet
    scaler: RobustFeatureScaler
    checkpoint: Path
    temperature: float | None = None
    warmup_frame: pd.DataFrame | None = None
    metadata: dict = field(default_factory=dict)


def _load_schema(name: str = "input.schema.json") -> dict:
    candidates = [SCHEMA_DIR / name, Path("Plan") / "schemas" / name]
    for path in candidates:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"schema {name} not found in {candidates}")


def validate_payload(payload: dict, schema: dict | None = None) -> dict:
    """Validate against the JSON schema and additional semantic rules."""
    import jsonschema

    schema = schema or _load_schema()
    try:
        jsonschema.validate(payload, schema)
    except jsonschema.ValidationError as exc:
        raise InputValidationError(f"input schema violation at {list(exc.absolute_path)}: {exc.message}") from exc

    candles = payload["candles"]
    timestamps = pd.to_datetime([c["timestamp"] for c in candles], utc=True, errors="coerce")
    if timestamps.isna().any():
        raise InputValidationError("candle timestamps must be ISO-8601 date-times")
    if not timestamps.is_monotonic_increasing:
        raise InputValidationError("candles must be ordered oldest -> newest")
    if timestamps.duplicated().any():
        raise InputValidationError("duplicate candle timestamps")
    deltas = timestamps.to_series().diff().dropna().unique()
    if len(deltas) != 1 or pd.Timedelta(deltas[0]) != pd.Timedelta(minutes=TIMEFRAME_MINUTES):
        raise InputValidationError("candles must be exactly 15 minutes apart with no gaps")

    frame = pd.DataFrame(candles)
    frame["timestamp_open"] = timestamps
    frame["timestamp_close"] = timestamps + pd.Timedelta(minutes=TIMEFRAME_MINUTES)
    for col in ("open", "high", "low", "close", "volume"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
        if frame[col].isna().any():
            raise InputValidationError(f"non-numeric {col} value")
    if (frame[["open", "high", "low", "close"]] <= 0).any().any():
        raise InputValidationError("prices must be positive")
    if (frame["volume"] < 0).any():
        raise InputValidationError("volume must be non-negative")
    if (frame["high"] < frame[["open", "close", "low"]].max(axis=1)).any():
        raise InputValidationError("high below open/close/low")
    if (frame["low"] > frame[["open", "close", "high"]].min(axis=1)).any():
        raise InputValidationError("low above open/close/high")
    return frame.sort_values("timestamp_open").reset_index(drop=True)


def load_warmup_from_clean(start: pd.Timestamp, path: Path | None = None) -> pd.DataFrame | None:
    """Slice contiguous 15m history immediately before ``start`` from local data."""
    path = path or CLEAN_DIR / symbol_slug("BTC/USDT") / "15m.parquet"
    if not path.exists():
        return None
    frame = pd.read_parquet(path)
    frame["timestamp_open"] = pd.to_datetime(frame["timestamp_open"], utc=True)
    frame = frame[frame["timestamp_open"] < start].sort_values("timestamp_open").tail(WARMUP_BARS)
    if frame.empty or frame["timestamp_open"].iloc[-1] + pd.Timedelta(minutes=TIMEFRAME_MINUTES) != start:
        return None
    diffs = frame["timestamp_open"].diff().dropna()
    if (diffs != pd.Timedelta(minutes=TIMEFRAME_MINUTES)).any():
        return None
    if "is_gap" in frame.columns and int(frame["is_gap"].iloc[1:].sum()) > 0:
        return None
    return frame.reset_index(drop=True)


def load_predictor(
    checkpoint_path: str | Path,
    config_path: str | Path | None = None,
    warmup_path: str | Path | None = None,
) -> Predictor:
    payload = load_checkpoint(checkpoint_path)
    cfg = PatronConfig.model_validate(payload["config"]) if config_path is None else load_config(config_path)
    model = build_model(cfg)
    model.load_state_dict(payload["model_state"])
    model.eval()
    scaler = RobustFeatureScaler.load_from_dict(payload["scaler"])
    warmup = None
    if warmup_path is not None:
        warmup = pd.read_parquet(warmup_path)
        warmup["timestamp_open"] = pd.to_datetime(warmup["timestamp_open"], utc=True)
    return Predictor(
        cfg=cfg,
        model=model,
        scaler=scaler,
        checkpoint=Path(checkpoint_path),
        temperature=payload.get("temperature"),
        warmup_frame=warmup,
        metadata={
            "epoch": payload.get("epoch"),
            "best_epoch": payload.get("best_epoch"),
            "val_loss": payload.get("val_loss"),
            "feature_order": payload.get("feature_order"),
        },
    )


def _neutral_payload(symbol: str, as_of: str, version: str, warnings: list[str]) -> dict:
    return {
        "status": "NO_OUTPUT",
        "symbol": symbol,
        "as_of": as_of,
        "horizon_minutes": 90,
        "model_version": version,
        "feature_version": FEATURE_VERSION,
        "pattern_scores": {},
        "barrier_probabilities": {"tp_first": 0.0, "sl_first": 0.0, "timeout": 0.0},
        "range_quantiles": [
            {
                "step": step,
                "minutes_ahead": step * TIMEFRAME_MINUTES,
                "high": {"q10": 0.0, "q50": 0.0, "q90": 0.0},
                "low": {"q10": 0.0, "q50": 0.0, "q90": 0.0},
                "close": {"q10": 0.0, "q50": 0.0, "q90": 0.0},
            }
            for step in range(1, 7)
        ],
        "momentum": {"class_probabilities": {}, "predicted_class": "sideways"},
        "data_quality": {
            "valid": False,
            "closed_candles": 0,
            "sequence_length": 96,
            "warnings": warnings,
        },
    }


def predict(payload: dict, predictor: Predictor, now: datetime | None = None) -> dict:
    cfg = predictor.cfg
    version = f"{cfg.project.name.lower()}-{cfg.project.version}"
    frame = validate_payload(payload)
    warnings: list[str] = []
    now = now or datetime.now(timezone.utc)

    last_close = frame["timestamp_close"].iloc[-1].to_pydatetime()
    if cfg.runtime.reject_unclosed_candle and last_close > now:
        raise InputValidationError(
            f"last candle closes at {last_close.isoformat()} which is in the future"
        )
    as_of = last_close.isoformat()

    if len(frame) != cfg.sequence_length:
        raise InputValidationError(f"expected {cfg.sequence_length} candles, got {len(frame)}")

    warmup = predictor.warmup_frame
    start = frame["timestamp_open"].iloc[0]
    if warmup is None:
        warmup = load_warmup_from_clean(start)
    if warmup is not None:
        warmup = warmup[warmup["timestamp_open"] < start].sort_values("timestamp_open").tail(WARMUP_BARS)
        contiguous = (
            len(warmup) >= WARMUP_BARS // 2
            and warmup["timestamp_open"].iloc[-1] + pd.Timedelta(minutes=TIMEFRAME_MINUTES) == start
            and (warmup["timestamp_open"].diff().dropna() == pd.Timedelta(minutes=TIMEFRAME_MINUTES)).all()
        )
        if not contiguous:
            warnings.append("warm-up history is not contiguous with the input window; ignored")
            warmup = None
    else:
        warnings.append("no warm-up history available")

    if warmup is None:
        return _neutral_payload(
            cfg.symbol,
            as_of,
            version,
            warnings
            + [
                "insufficient history for a 96-bar volume baseline and 4h ATR warm-up; "
                "features of the final candle would be invalid (NO_OUTPUT)"
            ],
        )

    combined = pd.concat(
        [warmup[["timestamp_open", "timestamp_close", "open", "high", "low", "close", "volume"]], frame[["timestamp_open", "timestamp_close", "open", "high", "low", "close", "volume"]]],
        ignore_index=True,
    )
    feature_frame = build_features(combined, atr_period=cfg.forecast.barrier.atr_period)
    last = feature_frame.iloc[-1]
    if not bool(last["feature_valid"]):
        return _neutral_payload(
            cfg.symbol, as_of, version, warnings + ["final feature row invalid (NaN/inf)"]
        )

    window_raw = feature_frame[list(FEATURE_ORDER)].to_numpy(dtype=np.float64)[-cfg.sequence_length :]
    if not np.isfinite(window_raw).all():
        return _neutral_payload(cfg.symbol, as_of, version, warnings + ["window contains invalid features"])
    x = torch.from_numpy(predictor.scaler.transform(window_raw)[None, :, :])

    with torch.no_grad():
        out = predictor.model(x)

    barrier_probs = torch.softmax(out["barrier"][0], dim=-1).numpy()
    momentum_probs = torch.softmax(out["momentum"][0], dim=-1).numpy()
    pattern_probs = torch.sigmoid(out["pattern"][0]).numpy()
    quantiles = out["quantile"][0].numpy()  # (H, F, Q)

    result = {
        "status": "OK",
        "symbol": cfg.symbol,
        "as_of": as_of,
        "horizon_minutes": cfg.forecast.horizon_minutes,
        "model_version": version,
        "feature_version": FEATURE_VERSION,
        "pattern_scores": {
            tag: round(float(pattern_probs[i]), 6) for i, tag in enumerate(cfg.pattern_tags)
        },
        "barrier_probabilities": {
            name: round(float(barrier_probs[i]), 6) for i, name in enumerate(BARRIER_CLASSES)
        },
        "range_quantiles": [
            {
                "step": step,
                "minutes_ahead": step * TIMEFRAME_MINUTES,
                "high": _quantile_obj(quantiles[step - 1, 0]),
                "low": _quantile_obj(quantiles[step - 1, 1]),
                "close": _quantile_obj(quantiles[step - 1, 2]),
            }
            for step in range(1, cfg.horizon + 1)
        ],
        "momentum": {
            "class_probabilities": {
                name: round(float(momentum_probs[i]), 6)
                for i, name in enumerate(MOMENTUM_CLASSES)
            },
            "predicted_class": MOMENTUM_CLASSES[int(momentum_probs.argmax())],
        },
        "data_quality": {
            "valid": True,
            "closed_candles": int(len(frame)),
            "sequence_length": cfg.sequence_length,
            "warnings": warnings,
        },
    }
    return result


def _quantile_obj(values: np.ndarray) -> dict:
    return {
        "q10": round(float(values[0]), 6),
        "q50": round(float(values[1]), 6),
        "q90": round(float(values[2]), 6),
    }


def validate_output(payload: dict) -> None:
    import jsonschema

    jsonschema.validate(payload, _load_schema("output.schema.json"))
