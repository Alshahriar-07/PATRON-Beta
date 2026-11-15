"""Load the built dataset and assemble train/val/test sequence datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from patron.config import PatronConfig
from patron.datasets.windows import PatronDatasets, make_datasets, valid_window_positions
from patron.features.schema import FEATURE_ORDER
from patron.features.scaler import RobustFeatureScaler
from patron.paths import FEATURE_DIR, symbol_slug
from patron.utils.io import read_json, read_parquet


@dataclass
class PreparedData:
    frame: pd.DataFrame
    scaler: RobustFeatureScaler
    scaled_features: np.ndarray
    bundles: PatronDatasets


def dataset_path_for(cfg: PatronConfig) -> Path:
    return FEATURE_DIR / f"{symbol_slug(cfg.symbol)}_dataset.parquet"


def scaler_path() -> Path:
    return FEATURE_DIR / "scaler.json"


def load_frame(cfg: PatronConfig, path: str | Path | None = None) -> pd.DataFrame:
    p = Path(path) if path is not None else dataset_path_for(cfg)
    if not p.exists():
        raise FileNotFoundError(
            f"dataset not found at {p}; run `python scripts/build_dataset.py` first"
        )
    frame = read_parquet(p)
    missing = [c for c in FEATURE_ORDER if c not in frame.columns]
    if missing:
        raise ValueError(f"dataset missing features: {missing}")
    return frame


def prepare_data(
    cfg: PatronConfig,
    dataset_path: str | Path | None = None,
    scaler_file: str | Path | None = None,
    stride: int = 1,
    train_stride: int | None = None,
) -> PreparedData:
    frame = load_frame(cfg, dataset_path)
    scaler = RobustFeatureScaler.load(scaler_file or scaler_path())
    scaled = scaler.transform_frame(frame)
    bundles = make_datasets(scaled, frame, cfg, stride=stride, train_stride=train_stride)
    return PreparedData(frame=frame, scaler=scaler, scaled_features=scaled, bundles=bundles)


def dataset_manifest() -> dict:
    path = FEATURE_DIR.parent / "manifests" / "feature_manifest.json"
    return read_json(path) if path.exists() else {}


def prepare_windows_only(cfg: PatronConfig, frame: pd.DataFrame) -> np.ndarray:
    return valid_window_positions(
        frame, seq_len=cfg.sequence_length, horizon=cfg.horizon
    ).positions
