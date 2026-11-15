"""Inference contract: input validation and schema-valid output."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from patron.features.schema import FEATURE_ORDER
from patron.features.scaler import RobustFeatureScaler
from patron.inference import pipeline as inference
from patron.inference.pipeline import (
    InputValidationError,
    Predictor,
    predict,
    validate_output,
    validate_payload,
)
from patron.models.patron_net import build_model
from tests.conftest import make_candles


@pytest.fixture(scope="module")
def predictor(cfg):
    candles = make_candles(520, seed=21)
    warmup = candles.iloc[:410].reset_index(drop=True)
    window = candles.iloc[410:506].reset_index(drop=True)
    model = build_model(cfg).eval()
    from patron.features.builder import build_features

    features = build_features(candles)
    scaler = RobustFeatureScaler.fit(
        features.loc[features["feature_valid"], list(FEATURE_ORDER)].to_numpy(), list(FEATURE_ORDER)
    )
    p = Predictor(cfg=cfg, model=model, scaler=scaler, checkpoint=Path("unused.pt"), warmup_frame=warmup)
    return p, window


def _payload(window):
    return {
        "symbol": "BTC/USDT",
        "timeframe": "15m",
        "candles": [
            {
                "timestamp": row.timestamp_open.isoformat(),
                "open": float(row.open),
                "high": float(row.high),
                "low": float(row.low),
                "close": float(row.close),
                "volume": float(row.volume),
            }
            for row in window.itertuples()
        ],
    }


def test_rejects_sample_placeholder_with_duplicate_timestamps():
    import json

    # ``examples/invalid_input.json`` is the deliberately malformed placeholder that
    # ships with the repository (duplicate/gapped timestamps). ``Plan/`` is the local
    # spec bundle and is only a fallback for development checkouts.
    candidate = Path("examples/invalid_input.json")
    if not candidate.exists():
        candidate = Path("Plan/examples/sample_input.json")
    if not candidate.exists():
        pytest.skip("invalid input fixture not available")
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    with pytest.raises(InputValidationError, match="ordered|duplicate|15 minutes"):
        validate_payload(payload)


def test_rejects_gapped_timestamps(predictor):
    p, window = predictor
    frame = window.copy()
    frame.loc[40, "timestamp_open"] = frame.loc[40, "timestamp_open"] + pd.Timedelta(minutes=7)
    with pytest.raises(InputValidationError, match="15 minutes"):
        validate_payload(_payload(frame))


def test_rejects_wrong_candle_count(predictor):
    p, window = predictor
    with pytest.raises(InputValidationError):
        validate_payload(_payload(window.iloc[:95]))


def test_rejects_inconsistent_ohlc(predictor):
    p, window = predictor
    bad = window.copy()
    bad.loc[3, "high"] = bad.loc[3, "low"] * 0.5
    with pytest.raises(InputValidationError, match="high"):
        validate_payload(_payload(bad))


def test_ok_output_matches_schema(predictor):
    p, window = predictor
    now = window["timestamp_close"].iloc[-1].to_pydatetime() + timedelta(minutes=1)
    result = predict(_payload(window), p, now=now)
    assert result["status"] == "OK"
    assert result["horizon_minutes"] == 90
    assert len(result["range_quantiles"]) == 6
    assert result["range_quantiles"][0]["minutes_ahead"] == 15
    assert sum(result["barrier_probabilities"].values()) == pytest.approx(1.0, abs=1e-4)
    assert sum(result["momentum"]["class_probabilities"].values()) == pytest.approx(1.0, abs=1e-4)
    q = result["range_quantiles"][0]["close"]
    assert q["q10"] <= q["q50"] <= q["q90"]
    validate_output(result)


def test_no_output_without_warmup(predictor, monkeypatch):
    p, window = predictor
    cold = Predictor(cfg=p.cfg, model=p.model, scaler=p.scaler, checkpoint=p.checkpoint, warmup_frame=None)
    monkeypatch.setattr(inference, "load_warmup_from_clean", lambda *a, **k: None)
    now = window["timestamp_close"].iloc[-1].to_pydatetime() + timedelta(minutes=1)
    result = predict(_payload(window), cold, now=now)
    assert result["status"] == "NO_OUTPUT"
    assert result["data_quality"]["valid"] is False
    assert result["data_quality"]["warnings"]
    validate_output(result)


def test_unclosed_candle_rejected(predictor):
    p, window = predictor
    now = window["timestamp_open"].iloc[-1].to_pydatetime()
    with pytest.raises(InputValidationError, match="future"):
        predict(_payload(window), p, now=now)
