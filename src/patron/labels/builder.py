"""Combine features, pattern tags and all forward-looking labels into one table."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from patron.config import PatronConfig
from patron.features.builder import build_features
from patron.features.patterns import PATTERN_TAGS, annotate_patterns
from patron.features.schema import FEATURE_ORDER, FEATURE_VERSION
from patron.labels.barrier import triple_barrier_labels
from patron.labels.momentum import momentum_labels
from patron.labels.range_labels import forward_range_targets, range_target_frame
from patron.utils.io import read_parquet, write_parquet

LOGGER = logging.getLogger(__name__)

LABEL_VERSION = "labels-1.0.0"

PATTERN_COLUMNS = tuple(f"pat_{tag}" for tag in PATTERN_TAGS)


def build_dataset_frame(candles: pd.DataFrame, cfg: PatronConfig) -> pd.DataFrame:
    """Build the supervised table: 12 features + patterns + labels.

    Input windows end at bar ``t``; every target starts at ``t+1``. No target
    column is ever read by the feature builder.
    """
    candles = candles.sort_values("timestamp_open").reset_index(drop=True)
    features = build_features(
        candles,
        atr_period=cfg.forecast.barrier.atr_period,
        volume_baseline=cfg.sequence_length,
        feature_order=cfg.feature_order,
    )

    patterns = annotate_patterns(candles)
    patterns.columns = list(PATTERN_COLUMNS)

    barrier = triple_barrier_labels(
        close=candles["close"].to_numpy(),
        high=candles["high"].to_numpy(),
        low=candles["low"].to_numpy(),
        atr=features["atr14"].to_numpy(),
        horizon=cfg.horizon,
        tp_mult=cfg.forecast.barrier.tp_atr_multiplier,
        sl_mult=cfg.forecast.barrier.sl_atr_multiplier,
        conservative_ambiguous=False,
    )

    range_targets = forward_range_targets(
        candles["close"].to_numpy(),
        candles["high"].to_numpy(),
        candles["low"].to_numpy(),
        horizon=cfg.horizon,
    )
    range_frame = range_target_frame(range_targets, horizon=cfg.horizon)

    momentum_cls, momentum_norm = momentum_labels(
        candles["close"].to_numpy(),
        features["atr14"].to_numpy(),
        horizon=cfg.horizon,
        boundaries=cfg.forecast.momentum.normalized_return_boundaries,
    )

    frame = pd.concat(
        [
            features.reset_index(drop=True),
            patterns.reset_index(drop=True),
            barrier.to_frame().reset_index(drop=True),
            range_frame.reset_index(drop=True),
            pd.DataFrame(
                {
                    "momentum_class": momentum_cls,
                    "momentum_norm_return": momentum_norm,
                }
            ),
        ],
        axis=1,
    )

    # A row is trainable only when features, barrier label, momentum class and
    # every range target are available (i.e. the full 6-candle horizon exists).
    target_cols = [f"tgt_{f}_{s}" for f in ("high", "low", "close") for s in range(1, cfg.horizon + 1)]
    frame["sample_valid"] = (
        frame["feature_valid"].to_numpy()
        & frame["barrier_mask"].to_numpy().astype(bool)
        & (frame["momentum_class"].to_numpy() >= 0)
        & np.isfinite(frame[target_cols].to_numpy()).all(axis=1)
    )
    frame.attrs["feature_version"] = FEATURE_VERSION
    frame.attrs["label_version"] = LABEL_VERSION
    return frame


def dataset_columns(cfg: PatronConfig) -> dict[str, list[str]]:
    horizon = cfg.horizon
    return {
        "features": list(FEATURE_ORDER),
        "patterns": list(PATTERN_COLUMNS),
        "barrier": [
            "barrier_label",
            "barrier_mask",
            "barrier_tp_step",
            "barrier_sl_step",
            "barrier_ambiguous",
            "barrier_upper",
            "barrier_lower",
        ],
        "range": [f"tgt_{f}_{s}" for f in ("high", "low", "close") for s in range(1, horizon + 1)],
        "momentum": ["momentum_class", "momentum_norm_return"],
        "aux": ["timestamp_open", "timestamp_close", "close", "atr14", "is_gap", "feature_valid", "sample_valid"],
    }


def build_and_save_dataset(
    cfg: PatronConfig,
    clean_path: str | Path,
    out_path: str | Path,
) -> tuple[pd.DataFrame, dict]:
    candles = read_parquet(clean_path)
    frame = build_dataset_frame(candles, cfg)
    write_parquet(frame, out_path)

    target_cols = [c for c in frame.columns if c.startswith("tgt_")]
    stats = {
        "rows": int(len(frame)),
        "valid_rows": int(frame["sample_valid"].sum()),
        "invalid_rows": int((~frame["sample_valid"]).sum()),
        "feature_version": FEATURE_VERSION,
        "label_version": LABEL_VERSION,
        "barrier_counts": {
            str(k): int(v)
            for k, v in frame.loc[frame["barrier_mask"] == 1, "barrier_label"]
            .value_counts()
            .sort_index()
            .items()
        },
        "barrier_ambiguous": int(frame["barrier_ambiguous"].sum()),
        "momentum_counts": {
            str(k): int(v) for k, v in frame["momentum_class"].value_counts().sort_index().items()
        },
        "pattern_positive_rates": {
            col: float(frame[col].mean()) for col in PATTERN_COLUMNS
        },
        "target_nan_cells": int(frame[target_cols].isna().sum().sum()),
        "start": frame["timestamp_open"].iloc[0].isoformat() if len(frame) else None,
        "end": frame["timestamp_open"].iloc[-1].isoformat() if len(frame) else None,
    }
    LOGGER.info(
        "dataset built: %d rows (%d trainable), barrier=%s",
        stats["rows"],
        stats["valid_rows"],
        stats["barrier_counts"],
    )
    return frame, stats
