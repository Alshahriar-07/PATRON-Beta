"""Window/split integrity: gaps, purging, shapes, no label leakage across splits."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from patron.datasets.splits import chronological_split
from patron.datasets.windows import make_datasets, valid_window_positions
from patron.features.builder import build_features
from patron.features.schema import FEATURE_ORDER
from patron.features.scaler import RobustFeatureScaler
from patron.labels.builder import build_dataset_frame
from tests.conftest import make_candles


@pytest.fixture(scope="module")
def dataset_frame(cfg):
    candles = make_candles(1400, seed=11)
    frame = build_dataset_frame(candles, cfg)
    return frame


def test_windows_never_span_a_gap(dataset_frame, cfg):
    frame = dataset_frame.copy()
    gap_at = 700
    frame.loc[gap_at, "is_gap"] = 1
    stats = valid_window_positions(frame, seq_len=cfg.sequence_length, horizon=cfg.horizon)
    bad = stats.positions[
        (stats.positions >= gap_at - cfg.horizon)
        & (stats.positions <= gap_at + cfg.sequence_length - 1)
    ]
    assert len(bad) == 0
    assert stats.n_rejected_gap >= 1


def test_windows_have_only_valid_features(dataset_frame, cfg):
    stats = valid_window_positions(dataset_frame, seq_len=cfg.sequence_length, horizon=cfg.horizon)
    valid = dataset_frame["feature_valid"].to_numpy()
    for pos in stats.positions[:200]:
        assert valid[pos - cfg.sequence_length + 1 : pos + 1].all()
        assert dataset_frame["sample_valid"].iloc[pos]


def test_split_is_chronological_and_purged():
    split = chronological_split(1000, purge=6, embargo=6)
    assert split.train[-1] < split.val[0] < split.val[-1] < split.test[0]
    assert split.val[0] - split.train[-1] > 6
    assert split.test[0] - split.val[-1] > 6


def test_dataset_item_shapes(dataset_frame, cfg):
    scaler = RobustFeatureScaler.fit(
        dataset_frame.loc[dataset_frame["feature_valid"], list(FEATURE_ORDER)].to_numpy(), list(FEATURE_ORDER)
    )
    scaled = scaler.transform_frame(dataset_frame)
    bundles = make_datasets(scaled, dataset_frame, cfg)
    assert len(bundles.train) > 0 and len(bundles.val) > 0 and len(bundles.test) > 0
    item = bundles.train[0]
    assert item["x"].shape == (cfg.sequence_length, len(FEATURE_ORDER))
    assert item["x"].dtype.is_floating_point
    assert item["range"].shape == (cfg.horizon, 3)
    assert item["pattern"].shape == (len(cfg.pattern_tags),)
    assert item["barrier"].ndim == 0 and item["momentum"].ndim == 0
    assert float(item["x"].abs().max()) <= 5.0 + 1e-6


def test_split_targets_do_not_overlap(dataset_frame, cfg):
    scaler = RobustFeatureScaler.fit(
        dataset_frame.loc[dataset_frame["feature_valid"], list(FEATURE_ORDER)].to_numpy(), list(FEATURE_ORDER)
    )
    scaled = scaler.transform_frame(dataset_frame)
    bundles = make_datasets(scaled, dataset_frame, cfg)
    train_end = int(bundles.train.positions[-1])
    val_start = int(bundles.val.positions[0])
    assert train_end + cfg.horizon < val_start
    val_end = int(bundles.val.positions[-1])
    test_start = int(bundles.test.positions[0])
    assert val_end + cfg.horizon < test_start


def test_dataset_is_not_shuffled(dataset_frame, cfg):
    scaler = RobustFeatureScaler.fit(
        dataset_frame.loc[dataset_frame["feature_valid"], list(FEATURE_ORDER)].to_numpy(), list(FEATURE_ORDER)
    )
    scaled = scaler.transform_frame(dataset_frame)
    bundles = make_datasets(scaled, dataset_frame, cfg)
    assert np.all(np.diff(bundles.train.positions) > 0)
    assert np.all(np.diff(bundles.test.positions) > 0)
    assert bundles.train.positions.max() < bundles.val.positions.min() < bundles.test.positions.min()
