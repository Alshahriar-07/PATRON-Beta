"""Train-only robust feature scaling.

Fit on the training partition only; persist median/IQR + versions so inference
cannot silently drift from training. Scaling is a clipped robust z-score:
``clip((x - median) / IQR, -clip, clip)``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from patron.features.schema import FEATURE_ORDER, FEATURE_VERSION
from patron.utils.io import read_json, write_json


class RobustFeatureScaler:
    def __init__(
        self,
        median: np.ndarray,
        scale: np.ndarray,
        feature_order: tuple[str, ...] | list[str] = FEATURE_ORDER,
        clip: float = 5.0,
        feature_version: str = FEATURE_VERSION,
        fit_metadata: dict | None = None,
    ) -> None:
        self.median = np.asarray(median, dtype=np.float64)
        self.scale = np.asarray(scale, dtype=np.float64)
        self.feature_order = list(feature_order)
        self.clip_value = float(clip)
        self.feature_version = feature_version
        self.fit_metadata = fit_metadata or {}
        if len(self.median) != len(self.feature_order) or len(self.scale) != len(
            self.feature_order
        ):
            raise ValueError("scaler statistics do not match feature order length")
        if np.any(self.scale <= 0) or not np.isfinite(self.scale).all():
            raise ValueError("scaler scale must be positive and finite")

    # ------------------------------------------------------------------ fit
    @classmethod
    def fit(
        cls,
        X: np.ndarray,
        feature_order: tuple[str, ...] | list[str] = FEATURE_ORDER,
        clip: float = 5.0,
        metadata: dict | None = None,
    ) -> "RobustFeatureScaler":
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != len(feature_order):
            raise ValueError(f"expected [N, {len(feature_order)}], got {arr.shape}")
        if arr.shape[0] < 100:
            raise ValueError("refusing to fit scaler on fewer than 100 rows")
        if not np.isfinite(arr).all():
            raise ValueError("scaler input contains NaN/inf; filter invalid rows first")

        median = np.median(arr, axis=0)
        q25, q75 = np.percentile(arr, [25, 75], axis=0)
        iqr = q75 - q25
        std = arr.std(axis=0)
        scale = np.where(iqr > 1e-12, iqr, std)
        scale = np.where(scale > 1e-12, scale, 1.0)
        return cls(median, scale, feature_order, clip, FEATURE_VERSION, metadata or {})

    # -------------------------------------------------------------- transform
    def transform(self, X: np.ndarray) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] != len(self.feature_order):
            raise ValueError(f"expected {len(self.feature_order)} features, got {arr.shape[1]}")
        out = (arr - self.median) / self.scale
        np.clip(out, -self.clip_value, self.clip_value, out=out)
        return out.astype(np.float32)

    def transform_frame(self, frame: pd.DataFrame) -> np.ndarray:
        return self.transform(frame[self.feature_order].to_numpy(dtype=np.float64))

    # ------------------------------------------------------------------- io
    def to_dict(self) -> dict:
        return {
            "scaler": "robust_clipped_zscore",
            "feature_version": self.feature_version,
            "feature_order": self.feature_order,
            "median": self.median.tolist(),
            "scale": self.scale.tolist(),
            "clip": self.clip_value,
            "fit_metadata": self.fit_metadata,
        }

    def save(self, path: str | Path) -> Path:
        return write_json(self.to_dict(), path)

    @classmethod
    def load(cls, path: str | Path) -> "RobustFeatureScaler":
        return cls.load_from_dict(read_json(path))

    @classmethod
    def load_from_dict(cls, payload: dict) -> "RobustFeatureScaler":
        expected = list(payload.get("feature_order", []))
        if expected != list(FEATURE_ORDER):
            raise ValueError(
                f"scaler feature order {expected} does not match code schema {list(FEATURE_ORDER)}"
            )
        if payload.get("feature_version") != FEATURE_VERSION:
            raise ValueError(
                f"scaler feature version {payload.get('feature_version')} != {FEATURE_VERSION}"
            )
        return cls(
            payload["median"],
            payload["scale"],
            expected,
            payload.get("clip", 5.0),
            payload["feature_version"],
            payload.get("fit_metadata", {}),
        )
