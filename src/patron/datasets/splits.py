"""Chronological splits with purge and embargo (no random shuffling)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ChronologicalSplit:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    train_end: int
    val_start: int
    val_end: int
    test_start: int
    purge: int
    embargo: int

    def summary(self) -> dict:
        return {
            "train": [int(self.train[0]), int(self.train[-1])] if len(self.train) else None,
            "val": [int(self.val[0]), int(self.val[-1])] if len(self.val) else None,
            "test": [int(self.test[0]), int(self.test[-1])] if len(self.test) else None,
            "purge_bars": self.purge,
            "embargo_bars": self.embargo,
        }


def chronological_split(
    n_samples: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    purge: int = 6,
    embargo: int = 6,
) -> ChronologicalSplit:
    """Split sample indices in time order.

    ``purge`` bars are dropped at the end of each block so targets of a block
    cannot overlap the following block; ``embargo`` bars are skipped at the start
    of the following block as an extra buffer.
    """
    if n_samples <= 0:
        raise ValueError("no samples to split")
    if not 0 < train_ratio < 1 or not 0 < val_ratio < 1 or train_ratio + val_ratio >= 1:
        raise ValueError("invalid split ratios")

    train_end = int(np.floor(n_samples * train_ratio))
    val_start = train_end + embargo
    val_end = int(np.floor(n_samples * (train_ratio + val_ratio)))
    test_start = val_end + embargo

    train = np.arange(0, max(0, train_end - purge), dtype=np.int64)
    val = np.arange(val_start, max(val_start, val_end - purge), dtype=np.int64)
    test = np.arange(test_start, n_samples, dtype=np.int64)
    if len(test) == 0:
        raise ValueError("final test split is empty; not enough samples")
    return ChronologicalSplit(
        train=train,
        val=val,
        test=test,
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        test_start=test_start,
        purge=purge,
        embargo=embargo,
    )


def walk_forward_folds(
    n_samples: int,
    n_folds: int = 4,
    min_train_ratio: float = 0.5,
    purge: int = 6,
    embargo: int = 6,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window walk-forward folds over the development period.

    Fold ``k`` trains on everything before the fold block and validates on the
    block itself, with purge before and embargo after the boundary.
    """
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    start = int(np.floor(n_samples * min_train_ratio))
    remaining = n_samples - start
    if remaining < n_folds:
        raise ValueError("not enough samples for walk-forward folds")
    block = remaining // n_folds
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for k in range(n_folds):
        val_start = start + k * block
        val_end = start + (k + 1) * block if k < n_folds - 1 else n_samples
        train_end = max(0, val_start - embargo)
        val_from = min(n_samples, val_start + (purge if k > 0 else 0))
        if train_end <= 0 or val_from >= val_end:
            continue
        folds.append(
            (np.arange(0, train_end, dtype=np.int64), np.arange(val_from, val_end, dtype=np.int64))
        )
    if not folds:
        raise ValueError("walk-forward produced no usable folds")
    return folds


def split_dates(
    frame: pd.DataFrame,
    split: ChronologicalSplit,
    positions: np.ndarray | None = None,
) -> dict:
    """Human-readable date ranges for experiment records.

    ``split`` holds indices into the array of *valid window positions*; when
    ``positions`` (bar indices) is supplied the indices are mapped through it.
    Without it the indices are treated as bar indices, which is only correct for
    contiguous position arrays — window builds drop rows, so callers should pass
    ``positions``.
    """

    def _range(indices: np.ndarray) -> dict | None:
        if len(indices) == 0:
            return None
        bars = positions[indices] if positions is not None else indices
        ts = pd.to_datetime(frame["timestamp_open"].iloc[bars], utc=True)
        return {"start": ts.iloc[0].isoformat(), "end": ts.iloc[-1].isoformat(), "count": int(len(indices))}

    return {"train": _range(split.train), "val": _range(split.val), "test": _range(split.test)}
