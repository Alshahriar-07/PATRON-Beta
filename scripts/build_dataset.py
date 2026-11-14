"""Build the supervised PATRON dataset (features + labels), scaler and manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from patron.config import load_config
from patron.datasets.splits import chronological_split, split_dates
from patron.datasets.windows import valid_window_positions
from patron.features.builder import save_feature_schema
from patron.features.schema import FEATURE_ORDER, FEATURE_VERSION
from patron.features.scaler import RobustFeatureScaler
from patron.labels.builder import build_dataset_frame, dataset_columns
from patron.logging_utils import setup_logging
from patron.paths import DATA_DIR, FEATURE_DIR, REPORT_DIR, ensure_dirs, symbol_slug
from patron.utils.io import file_sha256, write_json
from patron.utils.seed import set_seed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build PATRON features + labels")
    parser.add_argument("--config", default="config/model_config.yaml")
    parser.add_argument("--clean", default=None, help="clean 15m parquet path")
    parser.add_argument("--out", default=None, help="output dataset parquet")
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args(argv)

    ensure_dirs()
    setup_logging()
    cfg = load_config(args.config)
    set_seed(cfg.training.random_seed)

    slug = symbol_slug(cfg.symbol)
    clean_path = Path(args.clean) if args.clean else DATA_DIR / "clean" / slug / "15m.parquet"
    out_path = Path(args.out) if args.out else FEATURE_DIR / f"{slug}_dataset.parquet"
    if not clean_path.exists():
        print(f"clean data missing: {clean_path}", file=sys.stderr)
        return 2

    frame = build_dataset_frame(_read(clean_path), cfg)

    stats = valid_window_positions(frame, seq_len=cfg.sequence_length, horizon=cfg.horizon, stride=args.stride)
    positions = stats.positions
    split = chronological_split(
        len(positions), purge=cfg.validation.purge_bars, embargo=cfg.validation.embargo_bars
    )
    train_positions = positions[split.train]

    train_raw = frame.iloc[train_positions][list(FEATURE_ORDER)].to_numpy(dtype=np.float64)
    scaler = RobustFeatureScaler.fit(
        train_raw,
        feature_order=FEATURE_ORDER,
        metadata={
            "fit_split": "train",
            "train_windows": int(len(train_positions)),
            "symbol": cfg.symbol,
            "timeframe": cfg.timeframe,
        },
    )
    scaler_path = FEATURE_DIR / "scaler.json"
    scaler.save(scaler_path)
    save_feature_schema(FEATURE_DIR / "feature_schema.json")

    frame.to_parquet(out_path, index=False)

    target_cols = [c for c in frame.columns if c.startswith("tgt_")]
    barrier_counts = {
        str(k): int(v)
        for k, v in frame.loc[frame["barrier_mask"] == 1, "barrier_label"].value_counts().sort_index().items()
    }
    momentum_counts = {
        str(k): int(v) for k, v in frame["momentum_class"].value_counts().sort_index().items()
    }
    pattern_cols = [c for c in frame.columns if c.startswith("pat_")]
    manifest = {
        "symbol": cfg.symbol,
        "timeframe": cfg.timeframe,
        "sequence_length": cfg.sequence_length,
        "horizon": cfg.horizon,
        "feature_version": FEATURE_VERSION,
        "feature_order": list(FEATURE_ORDER),
        "dataset_path": str(out_path),
        "dataset_sha256": file_sha256(out_path),
        "rows": int(len(frame)),
        "columns": dataset_columns(cfg),
        "feature_warmup_valid_rows": int(frame["feature_valid"].sum()),
        "trainable_rows": int(frame["sample_valid"].sum()),
        "target_nan_cells": int(frame[target_cols].isna().sum().sum()),
        "barrier_counts": barrier_counts,
        "barrier_ambiguous": int(frame["barrier_ambiguous"].sum()),
        "momentum_counts": momentum_counts,
        "pattern_positive_rates": {c: float(frame[c].mean()) for c in pattern_cols},
        "windows": {
            "valid_windows": int(len(positions)),
            "stride": args.stride,
            "rejected_feature_or_warmup": int(stats.n_rejected_feature_warmup),
            "rejected_gap_span": int(stats.n_rejected_gap),
            "rejected_labels": int(stats.n_rejected_labels),
        },
        "split": {**split.summary(), **{"dates": split_dates(frame, split)}},
        "scaler": {
            "path": str(scaler_path),
            "sha256": file_sha256(scaler_path),
            "feature_order": scaler.feature_order,
        },
        "range": {
            "start": frame["timestamp_open"].iloc[0].isoformat(),
            "end": frame["timestamp_open"].iloc[-1].isoformat(),
        },
    }
    write_json(manifest, DATA_DIR / "manifests" / "feature_manifest.json")
    write_json(
        {
            "barrier_counts": barrier_counts,
            "momentum_counts": momentum_counts,
            "pattern_positive_rates": manifest["pattern_positive_rates"],
        },
        REPORT_DIR / "label_distribution.json",
    )

    print(json.dumps({k: manifest[k] for k in ("rows", "trainable_rows", "windows", "barrier_counts", "momentum_counts")}, indent=2))
    print(f"dataset: {out_path}")
    print(f"scaler : {scaler_path}")
    return 0


def _read(path: Path):
    import pandas as pd

    return pd.read_parquet(path)


if __name__ == "__main__":
    raise SystemExit(main())
