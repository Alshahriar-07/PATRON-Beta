"""One-epoch training smoke test on synthetic candles (CPU, seconds)."""

from __future__ import annotations

import json

import numpy as np

from patron.datasets.loader import PreparedData
from patron.datasets.windows import make_datasets
from patron.features.scaler import RobustFeatureScaler
from patron.features.schema import FEATURE_ORDER
from patron.labels.builder import build_dataset_frame
from patron.models.patron_net import build_model
from patron.training import trainer as trainer_module
from patron.training.trainer import load_checkpoint, train
from tests.conftest import make_candles


def test_one_epoch_training_writes_a_loadable_checkpoint(tmp_path, cfg, monkeypatch):
    monkeypatch.setattr(trainer_module, "EXPERIMENT_DIR", tmp_path)

    frame = build_dataset_frame(make_candles(700, seed=3), cfg)
    valid = frame["feature_valid"].to_numpy(dtype=bool)
    scaler = RobustFeatureScaler.fit(
        np.nan_to_num(frame.loc[valid, list(FEATURE_ORDER)].to_numpy()), list(FEATURE_ORDER)
    )
    scaled = scaler.transform_frame(frame)
    bundles = make_datasets(scaled, frame, cfg, train_stride=4)
    data = PreparedData(frame=frame, scaler=scaler, scaled_features=scaled, bundles=bundles)

    assert len(bundles.train) > 0 and len(bundles.val) > 0
    history = train(
        build_model(cfg),
        data,
        cfg,
        run_name="smoke_test",
        max_epochs=1,
        batch_size=64,
        seed=0,
    )

    assert history.best_epoch == 1
    assert history.records[0].train_terms["loss"] > 0
    run_dir = history.run_dir
    assert (run_dir / "training_history.json").exists()
    assert (run_dir / "train_log.jsonl").exists()
    payload = load_checkpoint(run_dir / "best.pt")
    assert payload["sequence_length"] == 96
    assert payload["horizon"] == 6
    assert list(payload["feature_order"]) == list(FEATURE_ORDER)

    record = json.loads((run_dir / "training_history.json").read_text(encoding="utf-8"))
    assert record["epochs_run"] == 1
    assert record["best_val_loss"] > 0
