"""Consolidated leakage-protocol checks for the expanded dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd

from patron.datasets.splits import chronological_split
from patron.features.scaler import RobustFeatureScaler
from patron.features.schema import FEATURE_ORDER


def test_no_train_test_time_overlap_with_purge(dataset_frame, cfg):
    purge, embargo = cfg.validation.purge_bars, cfg.validation.embargo_bars
    split = chronological_split(len(dataset_frame), purge=purge, embargo=embargo)
    assert split.train.max() < split.val.min()
    assert split.val.max() < split.test.min()
    gap_train_val = int(split.val.min() - split.train.max())
    assert gap_train_val >= purge + embargo - 1
    ts = pd.to_datetime(dataset_frame["timestamp_open"], utc=True)
    assert ts.iloc[split.train.max()] < ts.iloc[split.val.min()] < ts.iloc[split.test.min()]


def test_scaler_is_fit_on_train_only(dataset_frame, cfg):
    split = chronological_split(
        len(dataset_frame), purge=cfg.validation.purge_bars, embargo=cfg.validation.embargo_bars
    )
    train_rows = dataset_frame.loc[
        dataset_frame["feature_valid"] & dataset_frame.index.isin(split.train), list(FEATURE_ORDER)
    ].to_numpy()
    scaler = RobustFeatureScaler.fit(train_rows, list(FEATURE_ORDER))
    full_median = np.nanmedian(dataset_frame[list(FEATURE_ORDER)].to_numpy(), axis=0)
    # Fitting on the full frame instead of train would shift the medians.
    assert not np.allclose(np.asarray(scaler.median), full_median, rtol=0, atol=1e-6)


def test_window_targets_never_cross_into_the_next_split(dataset_frame, cfg):
    from patron.datasets.windows import valid_window_positions

    stats = valid_window_positions(dataset_frame)
    split = chronological_split(
        len(stats.positions), purge=cfg.validation.purge_bars, embargo=cfg.validation.embargo_bars
    )
    train_end_bar = stats.positions[split.train.max()]
    val_start_bar = stats.positions[split.val.min()]
    # A 96-bar window plus the 6-bar target horizon must fit before the boundary.
    assert train_end_bar + cfg.horizon <= val_start_bar


def test_canonical_candles_have_no_future_timestamps(candles):
    now = pd.Timestamp.now("UTC")
    assert (candles["timestamp_close"] <= now).all()
    assert (candles["timestamp_close"] > candles["timestamp_open"]).all()


def test_symbol_mapping_cannot_silently_collapse_assets():
    from patron.data.markets import VENUES

    for slug, venue in VENUES.items():
        native = list(venue.symbols.values())
        assert len(native) == len(set(native)), f"{slug} maps two assets to one symbol"
        assert venue.venue_symbol("BTC") != venue.venue_symbol("ETH")
