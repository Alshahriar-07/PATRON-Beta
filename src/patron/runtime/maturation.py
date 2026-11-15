"""Independent, idempotent maturation of every forecast horizon.

A forecast made at 10:00 has six target candles (10:15 ... 11:30). Each horizon is
evaluated as soon as *its own* candle is stored and confirmed — the first horizon is
never held back waiting for the sixth. Metrics use the same target space as training:
relative returns versus the close at the forecast origin.

Only the six-step horizon can carry a momentum class and a triple-barrier outcome,
because both labels need all six future candles. Until then those fields are marked
``immature`` rather than guessed.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from patron.config import PatronConfig
from patron.labels.barrier import AMBIGUOUS, BARRIER_CLASSES, triple_barrier_labels
from patron.labels.momentum import MOMENTUM_CLASSES, momentum_labels
from patron.runtime.config import RuntimeConfig
from patron.runtime.store import RuntimeStore, utc_iso

LOGGER = logging.getLogger(__name__)

QUANTILE_LEVELS = (0.10, 0.50, 0.90)
RANGE_FIELDS = ("high", "low", "close")


def pinball_loss(actual: float, forecast: float, level: float) -> float:
    diff = actual - forecast
    return max(level * diff, (level - 1.0) * diff)


def _quantiles(step_payload: dict, field: str) -> list[float]:
    band = step_payload[field]
    return [float(band["q10"]), float(band["q50"]), float(band["q90"])]


def evaluate_horizon(
    envelope: dict,
    step: int,
    candle: dict,
    reference_close: float,
    reference_atr: float,
    model_cfg: PatronConfig,
    barrier: dict | None = None,
) -> tuple[dict, dict]:
    """Return ``(actual, metrics)`` for one matured horizon."""
    actual_prices = {
        "high": float(candle["high"]),
        "low": float(candle["low"]),
        "close": float(candle["close"]),
    }
    actual_rel = {field: actual_prices[field] / reference_close - 1.0 for field in RANGE_FIELDS}
    step_payload = envelope["model_output"]["range_quantiles"][step - 1]

    abs_error: dict[str, float] = {}
    pct_error: dict[str, float] = {}
    pinball: dict[str, float] = {}
    coverage: dict[str, bool] = {}
    for field in RANGE_FIELDS:
        q10, q50, q90 = _quantiles(step_payload, field)
        predicted_price = reference_close * (1.0 + q50)
        abs_error[field] = abs(predicted_price - actual_prices[field])
        pct_error[field] = abs_error[field] / actual_prices[field] if actual_prices[field] else float("nan")
        losses = [
            pinball_loss(actual_rel[field], q10, QUANTILE_LEVELS[0]),
            pinball_loss(actual_rel[field], q50, QUANTILE_LEVELS[1]),
            pinball_loss(actual_rel[field], q90, QUANTILE_LEVELS[2]),
        ]
        pinball[field] = float(np.mean(losses))
        coverage[field] = bool(q10 <= actual_rel[field] <= q90)

    q50_close = _quantiles(step_payload, "close")[1]
    predicted_direction = "up" if q50_close > 0 else "down" if q50_close < 0 else "flat"
    actual_direction = "up" if actual_rel["close"] > 0 else "down" if actual_rel["close"] < 0 else "flat"

    horizon = model_cfg.horizon
    metrics: dict = {
        "abs_error": abs_error,
        "pct_error": pct_error,
        "pinball": pinball,
        "pinball_mean": float(np.mean(list(pinball.values()))),
        "coverage": coverage,
        "direction": {
            "predicted": predicted_direction,
            "actual": actual_direction,
            "correct": predicted_direction == actual_direction,
        },
        "predicted_close_return": q50_close,
        "actual_close_return": actual_rel["close"],
        "maturity": {
            "range": "matured",
            "momentum": "matured" if step == horizon else "immature",
            "barrier": "matured" if step == horizon and barrier is not None else "immature",
        },
    }
    if step == horizon:
        metrics["momentum"] = _momentum_eval(
            envelope, reference_close, reference_atr, actual_prices, model_cfg
        )
        metrics["barrier"] = _barrier_eval(
            envelope,
            None if barrier is None else barrier.get("label"),
            None if barrier is None else barrier.get("ambiguous"),
            None if barrier is None else barrier.get("tp_step"),
            None if barrier is None else barrier.get("sl_step"),
        )

    actual = {
        "ts_open": utc_iso(candle["ts_open"]),
        "high": actual_prices["high"],
        "low": actual_prices["low"],
        "close": actual_prices["close"],
        "relative": actual_rel,
    }
    if barrier is not None:
        actual["barrier_label"] = barrier.get("label")
    return actual, metrics


def _momentum_eval(
    envelope: dict, reference_close: float, reference_atr: float, actual_prices: dict, model_cfg: PatronConfig
) -> dict:
    close = np.array([reference_close, actual_prices["close"]], dtype=np.float64)
    atr = np.array([reference_atr, reference_atr], dtype=np.float64)
    labels, norm = momentum_labels(
        close, atr, horizon=1, boundaries=model_cfg.forecast.momentum.normalized_return_boundaries
    )
    actual_class = MOMENTUM_CLASSES[int(labels[0])] if labels[0] >= 0 else None
    predicted_class = envelope["model_output"]["momentum"]["predicted_class"]
    return {
        "predicted": predicted_class,
        "actual": actual_class,
        "correct": bool(actual_class is not None and predicted_class == actual_class),
        "normalized_return": float(norm[0]) if np.isfinite(norm[0]) else None,
    }


def _barrier_eval(
    envelope: dict, label: int | None, ambiguous: bool | None, tp_step: int | None, sl_step: int | None
) -> dict:
    probs = envelope["model_output"]["barrier_probabilities"]
    predicted = max(probs, key=probs.get)
    unusable = label is None or bool(ambiguous) or int(label) == AMBIGUOUS
    actual = None if unusable else BARRIER_CLASSES[int(label)]
    return {
        "predicted": predicted,
        "actual": actual,
        "correct": bool(actual is not None and predicted == actual),
        "ambiguous": bool(unusable),
        "tp_step": tp_step,
        "sl_step": sl_step,
        "probs": probs,
    }


def evaluate_barrier(
    origin_close: float,
    origin_atr: float,
    future: list[dict],
    model_cfg: PatronConfig,
) -> dict:
    """Full six-bar triple-barrier label for the forecast origin.

    ``future`` must hold the six verified closed candles after the origin, oldest
    first. Fewer than six means the label is immature, not "timeout".
    """
    horizon = model_cfg.horizon
    if len(future) < horizon:
        return {"label": None, "ambiguous": True, "tp_step": None, "sl_step": None, "reason": "immature"}
    closes = np.array([origin_close] + [float(row["close"]) for row in future], dtype=np.float64)
    highs = np.array([origin_close] + [float(row["high"]) for row in future], dtype=np.float64)
    lows = np.array([origin_close] + [float(row["low"]) for row in future], dtype=np.float64)
    atr = np.full(len(closes), float(origin_atr), dtype=np.float64)
    result = triple_barrier_labels(
        close=closes,
        high=highs,
        low=lows,
        atr=atr,
        horizon=horizon,
        tp_mult=model_cfg.forecast.barrier.tp_atr_multiplier,
        sl_mult=model_cfg.forecast.barrier.sl_atr_multiplier,
    )
    label = int(result.label[0])
    return {
        "label": None if label == AMBIGUOUS else label,
        "ambiguous": label == AMBIGUOUS or not bool(result.mask[0]),
        "tp_step": int(result.tp_step[0]),
        "sl_step": int(result.sl_step[0]),
    }


def _future_candles(
    store: RuntimeStore, symbol: str, timeframe: str, origin_open: str, horizon: int
) -> list[dict]:
    frame = store.candles(symbol, timeframe, start=origin_open)
    if frame.empty:
        return []
    rows = frame[frame["timestamp_open"] > pd.Timestamp(origin_open)].head(horizon)
    return rows.to_dict("records")


def mature_forecasts(
    store: RuntimeStore,
    cfg: RuntimeConfig,
    model_cfg: PatronConfig,
    now: datetime | None = None,
) -> dict:
    """Evaluate every maturity-ready horizon; never rewrite an existing evaluation."""
    now = now or datetime.now(timezone.utc)
    evaluated: list[dict] = []
    skipped = 0
    awaiting = 0

    for record in store.forecasts_without_evaluations():
        envelope = json.loads(record["payload"])
        done = {
            int(row["step"])
            for row in store.evaluations(record["forecast_id"])
        }
        reference_close = float(record["reference_close"])
        reference_atr = float(record["reference_atr"])

        for target in envelope["targets"]:
            step = int(target["step"])
            if step in done:
                continue
            candle = store.candle(cfg.symbol, cfg.timeframe, target["target_ts"])
            if candle is None:
                awaiting += 1
                continue
            candle = dict(candle)
            if int(candle.get("revision", 0)) > 0:
                store.log_event(
                    "evaluated_on_revised_candle",
                    record["forecast_id"],
                    {"step": step, "revision": int(candle["revision"])},
                )

            barrier = None
            if step == model_cfg.horizon:
                future = _future_candles(
                    store, cfg.symbol, cfg.timeframe, record["candle_ts"], model_cfg.horizon
                )
                barrier = evaluate_barrier(reference_close, reference_atr, future, model_cfg)

            actual, metrics = evaluate_horizon(
                envelope, step, candle, reference_close, reference_atr, model_cfg, barrier
            )
            inserted = store.record_evaluation(
                record["forecast_id"],
                step,
                target["target_ts"],
                "EVALUATED",
                actual,
                metrics,
                "ambiguous" if metrics.get("barrier", {}).get("ambiguous") else "matured",
            )
            if inserted:
                evaluated.append(
                    {"forecast_id": record["forecast_id"], "step": step, "target_ts": target["target_ts"]}
                )
                store.log_event(
                    "horizon_evaluated", record["forecast_id"],
                    {"step": step, "target_ts": target["target_ts"]},
                )
            else:
                skipped += 1

    return {
        "evaluated": evaluated,
        "count": len(evaluated),
        "duplicates_skipped": skipped,
        "awaiting_candles": awaiting,
    }
