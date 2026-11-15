"""Controlled continuous learning: eligibility, replay mixture, retraining triggers.

Rules enforced here:

* only **verified, closed** candles already in the store are used — the store only
  accepts validated closed bars;
* only **fully matured** labels are eligible (barrier resolved, momentum assigned,
  all six range targets present);
* predictions are never used as inputs to training;
* newest data cannot dominate: the recent share is capped by
  ``learning.recent_fraction`` and the effective mixture is logged with the run;
* triggers need a minimum amount of new evidence, respect a cooldown, and never
  fire while another candidate job is active.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from patron.config import PatronConfig
from patron.labels.builder import build_dataset_frame
from patron.paths import DATA_DIR, PROJECT_ROOT
from patron.runtime.config import RuntimeConfig
from patron.runtime.store import RuntimeStore, utc_iso

LOGGER = logging.getLogger(__name__)

LAST_TRAIN_AT = "learning.last_train_at"
LAST_TRAIN_MATURED = "learning.last_train_matured"
LAST_DATASET = "learning.last_dataset"
LAST_JOB_AT = "learning.last_job_at"


@dataclass
class TriggerDecision:
    should_train: bool
    reasons: list[str] = field(default_factory=list)
    evidence: dict = field(default_factory=dict)
    blocked_by: str | None = None

    def to_dict(self) -> dict:
        return {
            "should_train": self.should_train,
            "reasons": self.reasons,
            "evidence": self.evidence,
            "blocked_by": self.blocked_by,
        }


def matured_horizon_count(store: RuntimeStore) -> int:
    row = store.execute("SELECT COUNT(*) AS n FROM evaluations WHERE step=6").fetchone()
    return int(row["n"])


def eligibility_counts(store: RuntimeStore) -> dict:
    row = store.execute(
        "SELECT COUNT(*) AS n FROM evaluations WHERE step=6 AND status='EVALUATED'"
    ).fetchone()
    ambiguous = store.execute(
        "SELECT COUNT(*) AS n FROM evaluations WHERE step=6 AND label_state='ambiguous'"
    ).fetchone()
    return {
        "matured_horizons": int(row["n"]),
        "ambiguous_barriers": int(ambiguous["n"]),
        "usable_examples": int(row["n"]) - int(ambiguous["n"]),
    }


def build_recent_frame(
    store: RuntimeStore, cfg: RuntimeConfig, model_cfg: PatronConfig
) -> pd.DataFrame:
    candles = store.candles(cfg.symbol, cfg.timeframe)
    horizon = model_cfg.horizon
    minimum = model_cfg.sequence_length + cfg.history.warmup_bars + horizon
    if len(candles) < minimum:
        return pd.DataFrame()
    dataset = build_dataset_frame(candles, model_cfg)
    target_cols = [col for col in dataset.columns if col.startswith("tgt_")]
    eligible = (
        dataset["feature_valid"].astype(bool)
        & (dataset["barrier_mask"] == 1)
        & (dataset["barrier_label"] >= 0)
        & (dataset["momentum_class"] >= 0)
        & dataset[target_cols].notna().all(axis=1)
    )
    if "sample_valid" in dataset.columns:
        eligible &= dataset["sample_valid"].astype(bool)
    matured = dataset[eligible].copy()
    if matured.empty:
        return matured
    # The final `horizon` bars cannot have matured labels yet.
    matured = matured.iloc[:-horizon] if len(matured) > horizon else matured.iloc[0:0]
    return matured.reset_index(drop=True)


def _sample_replay(replay: pd.DataFrame, recent_rows: int, fraction: float, seed: int) -> pd.DataFrame:
    """Cap the newest-data share so replay stays dominant (forgetting guard)."""
    if recent_rows == 0 or fraction >= 1.0:
        return replay
    max_recent_share = min(max(fraction, 0.05), 0.95)
    replay_wanted = int(round(recent_rows * (1 - max_recent_share) / max_recent_share))
    if replay_wanted >= len(replay):
        return replay
    rng = np.random.default_rng(seed)
    index = np.sort(rng.choice(len(replay), size=max(replay_wanted, 1), replace=False))
    return replay.iloc[index]


def assemble_training_dataset(
    store: RuntimeStore,
    cfg: RuntimeConfig,
    model_cfg: PatronConfig,
    now: datetime | None = None,
    seed: int | None = None,
) -> dict:
    """Blend the historical replay set with recently matured live examples."""
    now = now or datetime.now(timezone.utc)
    seed = model_cfg.training.random_seed if seed is None else seed
    replay_path = Path(cfg.learning.replay_dataset)
    if not replay_path.is_absolute():
        replay_path = PROJECT_ROOT / replay_path

    recent = build_recent_frame(store, cfg, model_cfg)
    if not replay_path.exists() and recent.empty:
        return {
            "status": "NO_DATA",
            "reason": f"replay dataset missing ({replay_path}) and no matured live examples",
            "replay_rows": 0,
            "recent_rows": 0,
        }

    replay = pd.read_parquet(replay_path) if replay_path.exists() else pd.DataFrame()
    columns = list(recent.columns) if replay.empty else list(replay.columns)
    if not recent.empty:
        keep = [col for col in columns if col in recent.columns]
        recent = recent[keep]
    if not replay.empty:
        replay = replay[[col for col in columns if col in replay.columns]]

    replay_sample = _sample_replay(replay, len(recent), cfg.learning.recent_fraction, seed)
    frames = [frame for frame in (replay_sample, recent) if not frame.empty]
    if not frames:
        return {"status": "NO_DATA", "reason": "no eligible rows", "replay_rows": 0, "recent_rows": 0}
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["timestamp_open"], keep="last")
    combined = combined.sort_values("timestamp_open").reset_index(drop=True)

    out_dir = DATA_DIR / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp(now).strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"runtime_replay_{stamp}.parquet"
    combined.to_parquet(out_path, index=False)

    digest = hashlib.sha256()
    digest.update(combined["timestamp_open"].astype("int64").to_numpy().tobytes())
    digest.update(combined[[*model_cfg.feature_order]].to_numpy(dtype=np.float64).tobytes())

    recent_share = len(recent) / len(combined) if len(combined) else 0.0
    info = {
        "status": "OK",
        "path": str(out_path),
        "rows": int(len(combined)),
        "replay_rows": int(len(replay_sample)),
        "recent_rows": int(len(recent)),
        "recent_share": round(float(recent_share), 4),
        "requested_recent_fraction": cfg.learning.recent_fraction,
        "hash": digest.hexdigest(),
        "start": utc_iso(combined["timestamp_open"].iloc[0]),
        "end": utc_iso(combined["timestamp_open"].iloc[-1]),
        "seed": int(seed),
    }
    store.set_kv(LAST_DATASET, info)
    store.log_event("training_dataset_built", info["path"], {"rows": info["rows"],
                                                            "recent_rows": info["recent_rows"],
                                                            "hash": info["hash"]})
    return info


def evaluate_triggers(
    store: RuntimeStore,
    cfg: RuntimeConfig,
    model_cfg: PatronConfig,
    drift: dict | None = None,
    degradation: dict | None = None,
    now: datetime | None = None,
    active_job: bool = False,
) -> TriggerDecision:
    now = now or datetime.now(timezone.utc)
    evidence = {
        "evaluated_at": utc_iso(now),
        "eligible_examples": eligibility_counts(store),
        "drift_triggered": bool(drift and drift.get("triggered")),
        "degradation": degradation or {},
        "active_job": active_job,
    }
    if not cfg.learning.enabled:
        return TriggerDecision(False, evidence=evidence, blocked_by="learning disabled")

    matured = matured_horizon_count(store)
    last_matured = int(store.get_kv(LAST_TRAIN_MATURED, 0) or 0)
    new_examples = max(0, matured - last_matured)
    evidence["new_matured_horizons"] = new_examples
    evidence["matured_horizons"] = matured

    reasons: list[str] = []
    if new_examples >= cfg.learning.min_new_matured_examples:
        reasons.append(
            f"{new_examples} newly matured horizons (>= {cfg.learning.min_new_matured_examples})"
        )
    last_train_at = store.get_kv(LAST_TRAIN_AT)
    if last_train_at:
        elapsed = pd.Timestamp(now) - pd.Timestamp(last_train_at)
        evidence["hours_since_training"] = round(elapsed.total_seconds() / 3600, 2)
        if elapsed >= pd.Timedelta(hours=cfg.learning.schedule_hours):
            reasons.append(f"scheduled retraining window reached ({elapsed.total_seconds() / 3600:.1f}h)")
    else:
        evidence["hours_since_training"] = None

    if degradation and degradation.get("degraded"):
        reasons.append(f"performance degradation: {degradation.get('reason')}")
    if drift and drift.get("triggered"):
        reasons.append(f"drift confirmed: {'; '.join(drift.get('reasons', []))}")

    evidence["reasons"] = reasons
    if active_job:
        return TriggerDecision(False, reasons, evidence, blocked_by="candidate job already active")
    if not reasons:
        return TriggerDecision(False, reasons, evidence, blocked_by="no trigger condition met")

    last_attempt = store.get_kv(LAST_JOB_AT) or last_train_at
    if last_attempt:
        elapsed = pd.Timestamp(now) - pd.Timestamp(last_attempt)
        if elapsed < pd.Timedelta(hours=cfg.learning.cooldown_hours):
            return TriggerDecision(
                False, reasons, evidence,
                blocked_by=f"cooldown active ({elapsed.total_seconds() / 3600:.1f}h of {cfg.learning.cooldown_hours}h)",
            )

    if matured < cfg.learning.min_new_matured_examples:
        return TriggerDecision(
            False, reasons, evidence,
            blocked_by=f"only {matured} matured horizons, need {cfg.learning.min_new_matured_examples}",
        )
    return TriggerDecision(True, reasons, evidence)


def default_train_command(
    cfg: RuntimeConfig, run_name: str, dataset_path: str
) -> list[str]:
    import sys

    return [
        sys.executable,
        "scripts/train.py",
        "--dataset",
        dataset_path,
        "--run-name",
        run_name,
        "--epochs",
        str(cfg.learning.train_epochs),
        "--stride",
        str(cfg.learning.train_stride),
        "--threads",
        str(cfg.learning.threads),
    ]


def default_evaluate_command(run_dir: str, dataset_path: str) -> list[str]:
    import sys

    return [
        sys.executable,
        "scripts/evaluate.py",
        "--checkpoint",
        f"{run_dir}/best.pt",
        "--dataset",
        dataset_path,
        "--out",
        f"{run_dir}/evaluation_report.json",
        "--no-baselines",
    ]
