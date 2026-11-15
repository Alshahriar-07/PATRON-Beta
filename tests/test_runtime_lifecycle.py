"""End-to-end runtime lifecycle tests.

These tests drive the real service through consecutive candle closes using a
deterministic synthetic source and the shipped checkpoint. They cover the full
chain: ingest -> validate -> mature -> metrics -> window -> forecast -> persist ->
publish, plus governed retraining, promotion, rejection and rollback.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import jsonschema
import pandas as pd
import pytest

from patron.runtime.collector import StaticSource
from patron.runtime.config import load_runtime_config
from patron.runtime.engine import check_output_consistency
from patron.runtime.governance import evaluate_gates, load_run_metrics
from patron.runtime.learning import (
    LAST_JOB_AT,
    LAST_TRAIN_AT,
    LAST_TRAIN_MATURED,
    assemble_training_dataset,
    build_recent_frame,
    evaluate_triggers,
)
from patron.runtime.jobs import TrainingJobManager
from patron.runtime.service import DEFAULT_CHECKPOINT, PatronRuntime, best_available_checkpoint
from patron.runtime.store import RuntimeStore, utc_iso

pytestmark = pytest.mark.skipif(
    best_available_checkpoint() is None, reason="no checkpoint available for live inference tests"
)

COLUMNS = ["timestamp_open", "timestamp_close", "open", "high", "low", "close", "volume"]


class DeadRunner:
    """A runner whose child has already exited (no real process is started)."""

    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    def spawn(self, command, cwd, log_path) -> int:
        self.commands.append(list(command))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("simulated\n", encoding="utf-8")
        return 424242

    def alive(self, pid) -> bool:
        return False


@pytest.fixture()
def env(tmp_path, candles):
    cfg = load_runtime_config()
    cfg.storage.database = str(tmp_path / "patron.db")
    cfg.storage.output_dir = str(tmp_path / "live")
    cfg.storage.forecast_ledger = str(tmp_path / "forecasts")
    cfg.source.confirm_delay_seconds = 5
    cfg.source.lookback_hours = 240
    cfg.learning.enabled = False
    cfg.learning.replay_dataset = str(tmp_path / "missing_replay.parquet")
    cfg.monitoring.stale_after_seconds = 10_000

    frame = candles[list(COLUMNS)].copy()
    store = RuntimeStore(cfg.storage.database)
    runtime = PatronRuntime(
        runtime_cfg=cfg,
        store=store,
        source=StaticSource(frame=frame, name="test-source"),
        allow_training=False,
        clock=lambda: pd.Timestamp("2026-01-10T00:00:00Z").to_pydatetime(),
    )
    return {"cfg": cfg, "store": store, "runtime": runtime, "frame": frame, "tmp": tmp_path}


def _now(frame: pd.DataFrame, index: int) -> pd.Timestamp:
    return pd.Timestamp(frame["timestamp_close"].iloc[index]) + pd.Timedelta(seconds=10)


def _run_cycles(runtime, frame, start: int, count: int):
    results = []
    for index in range(start, start + count):
        results.append(runtime.cycle(_now(frame, index).to_pydatetime()))
    return results


def _live(env, name: str) -> dict:
    path = Path(env["cfg"].storage.output_dir) / name
    return json.loads(path.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ end-to-end
def test_each_closed_candle_produces_one_forecast(env):
    frame = env["frame"]
    results = _run_cycles(env["runtime"], frame, len(frame) - 5, 4)
    assert [r.status for r in results] == ["OK"] * 4
    assert len({r.forecast_id for r in results}) == 4

    store = env["store"]
    forecasts = store.recent_forecasts(10)
    assert len(forecasts) == 4
    for record in forecasts:
        envelope = json.loads(record["payload"])
        assert len(envelope["targets"]) == 6
        assert [t["minutes_ahead"] for t in envelope["targets"]] == [15, 30, 45, 60, 75, 90]
        assert envelope["input_window"]["candles"] == 96
        assert envelope["input_window"]["hash"] == record["input_hash"]
        # The 96 rows end at the candle that triggered the forecast.
        assert envelope["input_window"]["end"] == record["as_of"]

    latest = _live(env, "latest_forecast.json")
    assert latest["status"] == "OK"
    assert latest["stale"] is False
    assert latest["forecast_id"] == results[-1].forecast_id
    history = (Path(env["cfg"].storage.output_dir) / "forecast_history.jsonl").read_text().splitlines()
    assert len(history) == 4


def test_forecast_matches_runtime_schema(env, cfg):
    _run_cycles(env["runtime"], env["frame"], len(env["frame"]) - 2, 1)
    latest = _live(env, "latest_forecast.json")
    schema = json.loads(Path("schemas/forecast.schema.json").read_text(encoding="utf-8"))
    jsonschema.validate(latest, schema)
    # The nested model output still satisfies the frozen v1 output schema.
    from patron.inference.pipeline import validate_output

    validate_output(latest["model_output"])
    assert check_output_consistency(latest["model_output"]) == []


def test_reprocessing_a_candle_is_idempotent(env):
    frame = env["frame"]
    runtime = env["runtime"]
    now = _now(frame, len(frame) - 3).to_pydatetime()
    first = runtime.cycle(now)
    second = runtime.cycle(now)
    assert first.forecast_id == second.forecast_id
    assert second.detail == "candle already has an immutable forecast"
    assert len(env["store"].recent_forecasts(10)) == 1
    # No duplicate evaluation rows either.
    rows = env["store"].evaluations(first.forecast_id)
    assert len(rows) == len({row["step"] for row in rows})


def test_restart_continues_without_duplicates(env, tmp_path):
    frame = env["frame"]
    _run_cycles(env["runtime"], frame, len(frame) - 4, 2)
    before = len(env["store"].recent_forecasts(50))

    reopened = RuntimeStore(env["cfg"].storage.database)
    runtime = PatronRuntime(
        runtime_cfg=env["cfg"], store=reopened,
        source=StaticSource(frame=frame, name="test-source"), allow_training=False,
    )
    assert runtime.store.champion()["version"] == env["store"].champion()["version"]
    results = _run_cycles(runtime, frame, len(frame) - 2, 1)
    assert results[0].status == "OK"
    assert len(reopened.recent_forecasts(50)) == before + 1


def test_matures_each_horizon_as_its_candle_closes(env):
    frame = env["frame"]
    runtime = env["runtime"]
    start = len(frame) - 7
    results = _run_cycles(runtime, frame, start, 7)
    store = env["store"]

    first_id = results[0].forecast_id
    first_rows = {row["step"]: row for row in store.evaluations(first_id)}
    assert sorted(first_rows) == [1, 2, 3, 4, 5, 6]

    # After exactly one further candle, only horizon 1 could have matured.
    assert results[1].matured == 1
    # The sixth horizon only appears once all six future candles exist.
    sixth = json.loads(first_rows[6]["metrics"])
    assert sixth["maturity"] == {"range": "matured", "momentum": "matured", "barrier": "matured"}
    assert "barrier" in sixth and "momentum" in sixth
    # Earlier horizons explicitly say the six-bar labels are immature.
    assert json.loads(first_rows[1]["metrics"])["maturity"]["barrier"] == "immature"

    # Newest forecast has no matured horizon yet.
    assert store.evaluations(results[-1].forecast_id) == []


def test_stale_source_never_publishes_a_fresh_forecast(env):
    frame = env["frame"]
    runtime = env["runtime"]
    _run_cycles(runtime, frame, len(frame) - 3, 1)
    runtime.source.fail_times = 1
    result = runtime.cycle(_now(frame, len(frame) - 2).to_pydatetime())
    assert result.status == "STALE_DATA"
    status = _live(env, "status.json")
    assert status["status"] == "STALE_DATA"

    latest = _live(env, "latest_forecast.json")
    assert latest["stale"] is True
    assert latest["current_status"] == "STALE_DATA"

    health = runtime.health()
    assert health["healthy"] is False
    assert any("STALE_DATA" in reason for reason in health["reasons"])


def test_gap_blocks_the_forecast_and_reports_data_gap(env):
    frame = env["frame"]
    runtime = env["runtime"]
    _run_cycles(runtime, frame, len(frame) - 4, 1)
    gap_index = len(frame) - 2
    runtime.source.frame = frame.drop(index=gap_index).reset_index(drop=True)
    result = runtime.cycle(_now(frame, len(frame) - 1).to_pydatetime())
    assert result.status == "DATA_GAP"
    assert len(env["store"].recent_forecasts(10)) == 1
    assert _live(env, "status.json")["status"] == "DATA_GAP"


def test_published_files_and_health_are_consistent(env):
    _run_cycles(env["runtime"], env["frame"], len(env["frame"]) - 3, 2)
    for name in ("status.json", "performance.json", "models.json", "drift.json", "health.json"):
        assert (Path(env["cfg"].storage.output_dir) / name).exists()
    health = _live(env, "health.json")
    assert health["healthy"] is True
    assert health["champion"] == env["store"].champion()["version"]
    assert health["database"] == env["cfg"].storage.database


# --------------------------------------------------------------------- learning
def test_training_frame_uses_only_matured_candles_and_ignores_predictions(env):
    frame = env["frame"]
    runtime = env["runtime"]
    _run_cycles(runtime, frame, len(frame) - 8, 6)

    recent = build_recent_frame(env["store"], env["cfg"], runtime.model_cfg)
    assert not recent.empty
    assert recent["barrier_mask"].eq(1).all()
    assert (recent["momentum_class"] >= 0).all()
    last_candle = env["store"].latest_candles("BTC/USDT", "15m", 1)["timestamp_open"].iloc[0]
    assert recent["timestamp_open"].max() <= last_candle - pd.Timedelta(minutes=15 * runtime.model_cfg.horizon)

    # Predictions are not ground truth: adding/removing forecasts cannot change the frame.
    before = recent[["timestamp_open", "close"]].copy()
    env["store"].execute("DELETE FROM forecasts")
    env["store"]._conn.commit()
    after = build_recent_frame(env["store"], env["cfg"], runtime.model_cfg)
    pd.testing.assert_frame_equal(before, after[["timestamp_open", "close"]])


def test_trigger_needs_evidence_and_respects_cooldown(env):
    cfg = env["cfg"]
    runtime = env["runtime"]
    _run_cycles(runtime, env["frame"], len(env["frame"]) - 12, 8)
    cfg.learning.enabled = True
    cfg.learning.min_new_matured_examples = 2

    # Two newly matured horizons are enough evidence under this test policy.
    decision = evaluate_triggers(env["store"], cfg, runtime.model_cfg)
    assert decision.should_train is True
    assert any("newly matured" in reason for reason in decision.reasons)
    assert decision.evidence["eligible_examples"]["matured_horizons"] >= 2

    # Mark those examples as already consumed: no new evidence, training is blocked.
    env["store"].set_kv(LAST_TRAIN_MATURED, decision.evidence["matured_horizons"])
    decision = evaluate_triggers(env["store"], cfg, runtime.model_cfg)
    assert decision.should_train is False
    assert decision.blocked_by == "no trigger condition met"

    # Pretend the last successful training was long ago: the schedule trigger fires.
    env["store"].set_kv(LAST_TRAIN_AT, utc_iso(pd.Timestamp("2025-01-01T00:00:00Z")))
    env["store"].set_kv(LAST_JOB_AT, utc_iso(pd.Timestamp("2025-01-01T00:00:00Z")))
    decision = evaluate_triggers(env["store"], cfg, runtime.model_cfg)
    assert decision.should_train is True
    assert any("scheduled" in reason for reason in decision.reasons)

    # A recent attempt starts the cooldown again.
    env["store"].set_kv(LAST_JOB_AT, utc_iso(pd.Timestamp.now(tz="UTC")))
    decision = evaluate_triggers(env["store"], cfg, runtime.model_cfg)
    assert decision.should_train is False
    assert "cooldown" in (decision.blocked_by or "")


def test_dataset_blend_respects_recent_share_and_maturity(env, tmp_path):
    cfg = env["cfg"]
    runtime = env["runtime"]
    cfg.learning.min_new_matured_examples = 1
    cfg.learning.recent_fraction = 0.25
    _run_cycles(runtime, env["frame"], len(env["frame"]) - 30, 20)

    recent = build_recent_frame(env["store"], cfg, runtime.model_cfg)
    if recent.empty:
        pytest.skip("not enough store history for a matured recent block")
    replay = pd.concat([recent] * 8, ignore_index=True)
    replay["timestamp_open"] = pd.date_range("2020-01-01", periods=len(replay), freq="15min", tz="UTC")
    replay["timestamp_close"] = replay["timestamp_open"] + pd.Timedelta(minutes=15)
    replay_path = tmp_path / "replay.parquet"
    replay.to_parquet(replay_path, index=False)
    cfg.learning.replay_dataset = str(replay_path)

    info = assemble_training_dataset(env["store"], cfg, runtime.model_cfg)
    assert info["status"] == "OK"
    assert info["recent_rows"] == len(recent)
    assert info["recent_share"] <= 0.25 + 0.02
    assert Path(info["path"]).exists()
    blended = pd.read_parquet(info["path"])
    assert blended["timestamp_open"].is_monotonic_increasing
    assert not blended.duplicated(subset=["timestamp_open"]).any()


# ------------------------------------------------------------------- governance
def _make_candidate_run(tmp_path: Path, name: str, val_loss: float, coverage: float, accuracy: float) -> Path:
    run = tmp_path / "experiments" / name
    run.mkdir(parents=True, exist_ok=True)
    shutil.copy(DEFAULT_CHECKPOINT, run / "best.pt")
    (run / "training_history.json").write_text(
        json.dumps(
            {
                "epochs_run": 2,
                "best_epoch": 1,
                "best_val_loss": val_loss,
                "history": [
                    {
                        "epoch": 1,
                        "val_loss": val_loss,
                        "val_barrier_accuracy": accuracy,
                        "val_momentum_accuracy": 0.24,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (run / "evaluation_report.json").write_text(
        json.dumps(
            {
                "val": {
                    "quantile": {
                        "n": 5000,
                        "per_field": {
                            field: {"coverage_q10_q90": coverage} for field in ("high", "low", "close")
                        },
                    },
                    "barrier": {"accuracy": accuracy},
                    "momentum": {"accuracy": 0.24},
                }
            }
        ),
        encoding="utf-8",
    )
    return run


def _queue_job(store, run: Path) -> str:
    job_id = store.create_job(
        "candidate",
        config={"run_name": run.name, "dataset": "unused.parquet", "seeds": 1},
        run_dir=str(run),
    )
    store.update_job(job_id, status="TRAINING", pid=424242)
    return job_id


def test_candidate_that_passes_every_gate_is_promoted_and_can_be_rolled_back(env, tmp_path):
    store = env["store"]
    runtime = env["runtime"]
    _run_cycles(runtime, env["frame"], len(env["frame"]) - 3, 1)
    champion_before = store.champion()["version"]
    champion_metrics = json.loads(store.champion()["metrics"])

    run = _make_candidate_run(
        tmp_path, "candidate_test_pass",
        val_loss=champion_metrics["val_loss"] - 0.05,
        coverage=0.88,
        accuracy=champion_metrics["val_barrier_accuracy"] + 0.02,
    )
    cfg = env["cfg"]
    cfg.learning.command = ["fake-trainer"]
    job_id = _queue_job(store, run)

    manager = TrainingJobManager(store, cfg, runtime.model_cfg, runner=DeadRunner())
    manager.tick()

    job = store.job(job_id)
    assert job["status"] == "PROMOTED", job["result"]
    assert store.champion()["version"] == run.name
    assert store.model(champion_before)["role"] == "retired"
    decision = json.loads((run / "promotion_decision.json").read_text(encoding="utf-8"))
    assert decision["passed"] is True

    rolled = store.rollback()
    assert rolled["version"] == champion_before
    assert store.champion()["version"] == champion_before


def test_candidate_with_worse_metrics_is_rejected_and_champion_survives(env, tmp_path):
    store = env["store"]
    runtime = env["runtime"]
    _run_cycles(runtime, env["frame"], len(env["frame"]) - 3, 1)
    champion_before = store.champion()["version"]
    champion_metrics = json.loads(store.champion()["metrics"])

    run = _make_candidate_run(
        tmp_path, "candidate_test_fail",
        val_loss=champion_metrics["val_loss"] + 0.4,
        coverage=0.40,
        accuracy=champion_metrics["val_barrier_accuracy"] - 0.2,
    )
    cfg = env["cfg"]
    cfg.learning.command = ["fake-trainer"]
    job_id = _queue_job(store, run)

    manager = TrainingJobManager(store, cfg, runtime.model_cfg, runner=DeadRunner())
    manager.tick()

    job = store.job(job_id)
    assert job["status"] == "REJECTED"
    assert store.champion()["version"] == champion_before
    result = json.loads(job["result"])
    failed = {gate["name"] for gate in result["gates"] if not gate["passed"]}
    assert {"logloss_not_worse", "accuracy_not_worse", "coverage_ok"} <= failed
    assert store.events(kind="candidate_rejected")


def test_training_crash_is_isolated_from_inference(env, tmp_path):
    store = env["store"]
    runtime = env["runtime"]
    _run_cycles(runtime, env["frame"], len(env["frame"]) - 3, 1)
    champion_before = store.champion()["version"]

    broken = tmp_path / "experiments" / "candidate_test_broken"
    broken.mkdir(parents=True)
    job_id = store.create_job("candidate", config={"run_name": broken.name}, run_dir=str(broken))
    store.update_job(job_id, status="TRAINING", pid=424242)

    cfg = env["cfg"]
    cfg.learning.command = ["fake-trainer"]
    TrainingJobManager(store, cfg, runtime.model_cfg, runner=DeadRunner()).tick()

    assert store.job(job_id)["status"] == "FAILED"
    assert store.champion()["version"] == champion_before
    # Live inference still works after the failed candidate.
    result = runtime.cycle(_now(env["frame"], len(env["frame"]) - 2).to_pydatetime())
    assert result.status == "OK"


def test_gates_require_an_evaluation_report(tmp_path):
    cfg = load_runtime_config()
    candidate = {"val_loss": 1.0, "val_barrier_accuracy": 0.4, "evaluation": None}
    champion = {"val_loss": 1.4, "val_barrier_accuracy": 0.37}
    result = evaluate_gates(candidate, champion, cfg, {"schema_valid": True, "output_valid": True})
    names = {gate["name"]: gate["passed"] for gate in result["gates"]}
    assert names["min_samples"] is False
    assert names["coverage_ok"] is False
    assert result["passed"] is False


def test_load_run_metrics_reads_best_epoch(tmp_path):
    run = _make_candidate_run(tmp_path, "candidate_metrics", 1.11, 0.9, 0.39)
    metrics = load_run_metrics(run)
    assert metrics["available"] is True
    assert metrics["val_loss"] == 1.11
    assert metrics["evaluation"]["coverage"]["close"] == 0.9
