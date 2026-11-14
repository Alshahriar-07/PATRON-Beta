"""96-candle sequence windowing and the PyTorch dataset.

Windows are built lazily: only the `[N, 12]` feature matrix lives in memory while
each sample slices its own 96-candle window. Windows that contain an invalid
feature row or a market-data gap (including the 6-candle target horizon) are
excluded entirely, so no sample silently spans missing candles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from patron.datasets.splits import ChronologicalSplit, chronological_split, walk_forward_folds
from patron.features.schema import FEATURE_ORDER, SEQUENCE_LENGTH
from patron.labels.barrier import BARRIER_CLASSES
from patron.labels.momentum import MOMENTUM_CLASSES
from patron.labels.range_labels import frame_to_targets

TARGET_TENSOR_NAME = "targets"


@dataclass(frozen=True)
class WindowBuildStats:
    positions: np.ndarray
    n_rejected_feature_warmup: int
    n_rejected_gap: int
    n_rejected_labels: int


def valid_window_positions(
    frame: pd.DataFrame,
    seq_len: int = SEQUENCE_LENGTH,
    horizon: int = 6,
    stride: int = 1,
) -> WindowBuildStats:
    """Return end-bar indices whose 96-candle window and targets are clean."""
    n = len(frame)
    if n < seq_len + horizon:
        raise ValueError(f"dataset too short: {n} rows for {seq_len}+{horizon}")

    feature_ok = frame["feature_valid"].to_numpy(dtype=bool)
    sample_ok = frame["sample_valid"].to_numpy(dtype=bool)
    gap = frame["is_gap"].to_numpy(dtype=np.int32)

    feat_valid_series = pd.Series(feature_ok.astype(np.int32))
    window_feature_ok = (
        feat_valid_series.rolling(seq_len, min_periods=seq_len).sum().to_numpy() == seq_len
    )

    # A window is usable only if no gap occurs in [i-seq_len+1, i+horizon].
    gap_series = pd.Series(gap)
    span = seq_len + horizon
    gap_in_span = gap_series.rolling(span, min_periods=1).max().shift(-horizon).to_numpy()
    no_gap = np.nan_to_num(gap_in_span, nan=1.0) == 0

    positions = np.flatnonzero(window_feature_ok & sample_ok & no_gap)
    positions = positions[positions >= seq_len - 1]
    positions = positions[::stride]
    n_core = max(0, len(positions))
    if n_core == 0:
        raise ValueError("no valid windows produced")
    return WindowBuildStats(
        positions=positions.astype(np.int64),
        n_rejected_feature_warmup=int((~window_feature_ok).sum()),
        n_rejected_gap=int((~no_gap).sum()),
        n_rejected_labels=int((~sample_ok).sum()),
    )


class SequenceDataset(Dataset):
    """Lazy 96x12 window dataset with all four task targets."""

    def __init__(
        self,
        scaled_features: np.ndarray,
        frame: pd.DataFrame,
        positions: np.ndarray,
        seq_len: int = SEQUENCE_LENGTH,
        horizon: int = 6,
    ) -> None:
        if scaled_features.shape[1] != len(FEATURE_ORDER):
            raise ValueError("scaled feature matrix width does not match schema")
        self.features = np.ascontiguousarray(scaled_features, dtype=np.float32)
        self.positions = np.asarray(positions, dtype=np.int64)
        self.seq_len = int(seq_len)
        self.horizon = int(horizon)

        self.barrier = frame["barrier_label"].to_numpy(dtype=np.int64)
        self.barrier_mask = frame["barrier_mask"].to_numpy(dtype=np.float32)
        self.barrier_ambiguous = frame["barrier_ambiguous"].to_numpy(dtype=np.int64)
        self.momentum = frame["momentum_class"].to_numpy(dtype=np.int64)
        self.patterns = frame[[c for c in frame.columns if c.startswith("pat_")]].to_numpy(dtype=np.float32)
        self.range_targets = frame_to_targets(frame, horizon=horizon).astype(np.float32)
        self.timestamps = frame["timestamp_open"].to_numpy()
        self.reference_close = frame["close"].to_numpy(dtype=np.float64)

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        end = int(self.positions[idx])
        start = end - self.seq_len + 1
        x = self.features[start : end + 1]  # oldest -> newest

        pattern_mask = np.isfinite(self.patterns[end])
        pattern_labels = np.nan_to_num(self.patterns[end], nan=0.0)
        barrier = int(self.barrier[end])
        barrier_label = barrier if barrier >= 0 else 0

        return {
            "x": torch.from_numpy(x.copy()),
            "barrier": torch.tensor(barrier_label, dtype=torch.long),
            "barrier_mask": torch.tensor(self.barrier_mask[end], dtype=torch.float32),
            "momentum": torch.tensor(int(self.momentum[end]), dtype=torch.long),
            "pattern": torch.from_numpy(pattern_labels.astype(np.float32)),
            "pattern_mask": torch.from_numpy(pattern_mask.astype(np.float32)),
            "range": torch.from_numpy(self.range_targets[end].copy()),
            "index": torch.tensor(end, dtype=torch.long),
        }


@dataclass
class PatronDatasets:
    train: SequenceDataset
    val: SequenceDataset
    test: SequenceDataset
    split: ChronologicalSplit
    positions: np.ndarray
    stats: WindowBuildStats


def make_datasets(
    scaled_features: np.ndarray,
    frame: pd.DataFrame,
    cfg,
    stride: int = 1,
) -> PatronDatasets:
    """Build train/val/test sequence datasets with purge + embargo."""
    stats = valid_window_positions(
        frame, seq_len=cfg.sequence_length, horizon=cfg.horizon, stride=stride
    )
    positions = stats.positions
    split_local = chronological_split(
        len(positions),
        purge=cfg.validation.purge_bars,
        embargo=cfg.validation.embargo_bars,
    )
    train_pos = positions[split_local.train]
    val_pos = positions[split_local.val]
    test_pos = positions[split_local.test]

    def _build(pos: np.ndarray) -> SequenceDataset:
        return SequenceDataset(scaled_features, frame, pos, cfg.sequence_length, cfg.horizon)

    return PatronDatasets(
        train=_build(train_pos),
        val=_build(val_pos),
        test=_build(test_pos),
        split=split_local,
        positions=positions,
        stats=stats,
    )


def make_walk_forward_datasets(
    scaled_features: np.ndarray,
    frame: pd.DataFrame,
    cfg,
    n_folds: int = 4,
    stride: int = 1,
) -> list[tuple[SequenceDataset, SequenceDataset]]:
    stats = valid_window_positions(
        frame, seq_len=cfg.sequence_length, horizon=cfg.horizon, stride=stride
    )
    positions = stats.positions
    folds = walk_forward_folds(
        len(positions),
        n_folds=n_folds,
        purge=cfg.validation.purge_bars,
        embargo=cfg.validation.embargo_bars,
    )
    return [
        (
            SequenceDataset(scaled_features, frame, positions[tr], cfg.sequence_length, cfg.horizon),
            SequenceDataset(scaled_features, frame, positions[va], cfg.sequence_length, cfg.horizon),
        )
        for tr, va in folds
    ]


def num_classes() -> tuple[int, int]:
    return len(BARRIER_CLASSES), len(MOMENTUM_CLASSES)


def iter_positions(dataset: SequenceDataset) -> Iterable[int]:
    return (int(p) for p in dataset.positions)
