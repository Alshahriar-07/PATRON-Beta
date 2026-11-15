"""Background candidate training and evaluation.

Live inference is never blocked: training runs in a separate process, one candidate
job at a time, and its failure cannot touch the champion. The manager owns a small
explicit state machine persisted in the ``jobs`` table so a restart resumes where it
stopped instead of silently losing or duplicating work.

    QUEUED -> TRAINING -> EVALUATING -> GATING -> PROMOTED | REJECTED | FAILED
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from patron.config import PatronConfig
from patron.paths import EXPERIMENT_DIR, LOG_DIR, PROJECT_ROOT
from patron.runtime.config import RuntimeConfig
from patron.runtime.store import RuntimeStore, utc_iso

LOGGER = logging.getLogger(__name__)


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    if os.name == "nt":
        try:
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {int(pid)}", "/NH"],
                capture_output=True, text=True, timeout=15,
            )
        except Exception:
            return False
        return str(int(pid)) in (out.stdout or "")
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


class SubprocessRunner:
    """Spawn a child process detached from the prediction loop."""

    def spawn(self, command: list[str], cwd: Path, log_path: Path) -> int:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env.setdefault("PYTHONUNBUFFERED", "1")
        with log_path.open("ab") as log:
            process = subprocess.Popen(  # noqa: S603 - explicit command from config
                command,
                cwd=str(cwd),
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                env=env,
            )
        return process.pid

    def alive(self, pid: int | None) -> bool:
        return pid_alive(pid)


class TrainingJobManager:
    def __init__(
        self,
        store: RuntimeStore,
        cfg: RuntimeConfig,
        model_cfg: PatronConfig,
        runner: SubprocessRunner | None = None,
        repo_root: Path | None = None,
    ) -> None:
        self.store = store
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.runner = runner or SubprocessRunner()
        self.root = repo_root or PROJECT_ROOT
        self.transitions: list[dict] = []

    # ------------------------------------------------------------------ public
    def maybe_start(self, drift: dict | None = None, degradation: dict | None = None) -> dict | None:
        from patron.runtime.learning import (
            LAST_JOB_AT,
            LAST_TRAIN_MATURED,
            assemble_training_dataset,
            default_train_command,
            evaluate_triggers,
            matured_horizon_count,
        )

        active = self.store.active_job("candidate")
        if active is not None:
            return None

        decision = evaluate_triggers(
            self.store, self.cfg, self.model_cfg, drift=drift, degradation=degradation,
            active_job=False,
        )
        if not decision.should_train:
            self.store.set_kv("learning.last_decision", decision.to_dict())
            return None

        dataset = assemble_training_dataset(self.store, self.cfg, self.model_cfg)
        if dataset.get("status") != "OK":
            self.store.set_kv("learning.last_decision", decision.to_dict() | {"blocked_by": dataset.get("reason")})
            self.store.log_event("training_blocked", None, {"reason": dataset.get("reason")})
            return None

        now = datetime.now(timezone.utc)
        run_name = f"candidate_{now.strftime('%Y%m%d-%H%M%S')}"
        run_dir = self.root / "experiments" / run_name
        job_id = self.store.create_job(
            "candidate",
            config={
                "run_name": run_name,
                "dataset": dataset["path"],
                "dataset_hash": dataset["hash"],
                "rows": dataset["rows"],
                "recent_rows": dataset["recent_rows"],
                "recent_share": dataset["recent_share"],
                "seed": dataset["seed"],
                "epochs": self.cfg.learning.train_epochs,
                "stride": self.cfg.learning.train_stride,
                "triggers": decision.reasons,
            },
            run_dir=str(run_dir),
        )
        self.store.set_kv(LAST_JOB_AT, utc_iso(now))
        self.store.set_kv(LAST_TRAIN_MATURED, matured_horizon_count(self.store))
        self.store.set_kv("learning.last_decision", decision.to_dict() | {"job_id": job_id, "started": True})
        command = self.cfg.learning.command or default_train_command(
            self.cfg, run_name, dataset["path"]
        )
        log_path = LOG_DIR / "runtime" / f"{run_name}.log"
        try:
            pid = self.runner.spawn(command, self.root, log_path)
        except Exception as exc:
            self.store.update_job(job_id, status="FAILED", result={"error": f"spawn failed: {exc}"})
            self.store.log_event("training_spawn_failed", job_id, {"error": str(exc)})
            return self.store.job(job_id)

        self.store.update_job(
            job_id, status="TRAINING", pid=pid, started_at=utc_iso(now), log_path=str(log_path),
            run_dir=str(run_dir),
        )
        self.store.log_event(
            "training_started", job_id, {"command": command, "dataset": dataset["path"], "pid": pid}
        )
        LOGGER.info("candidate training started: %s (pid %s)", job_id, pid)
        return self.store.job(job_id)

    def tick(self) -> list[dict]:
        """Advance every non-terminal job whose process has exited."""
        transitions: list[dict] = []
        for job in self.store.jobs(limit=20):
            status = job["status"]
            if status in {"PROMOTED", "REJECTED", "FAILED", "CANCELLED"}:
                continue
            if status == "QUEUED":
                job = self._spawn_training(job)
                status = job["status"]
            if status in {"TRAINING", "EVALUATING"} and self.runner.alive(job.get("pid")):
                continue
            if status == "TRAINING":
                job = self._advance_after_training(job)
                transitions.append({"job_id": job["job_id"], "status": job["status"]})
            elif status == "EVALUATING":
                job = self._advance_after_evaluation(job)
                transitions.append({"job_id": job["job_id"], "status": job["status"]})
            elif status == "GATING":
                job = self._finalize(job)
                transitions.append({"job_id": job["job_id"], "status": job["status"]})
        self.transitions.extend(transitions)
        return transitions

    # ------------------------------------------------------------------ stages
    def _spawn_training(self, job: dict) -> dict:
        from patron.runtime.learning import default_train_command

        config = json.loads(job["config"]) if job.get("config") else {}
        run_name = config.get("run_name") or Path(job["run_dir"]).name
        command = self.cfg.learning.command or default_train_command(
            self.cfg, run_name, config.get("dataset", "")
        )
        log_path = LOG_DIR / "runtime" / f"{run_name}.log"
        pid = self.runner.spawn(command, self.root, log_path)
        self.store.update_job(
            job["job_id"], status="TRAINING", pid=pid,
            started_at=utc_iso(datetime.now(timezone.utc)), log_path=str(log_path),
        )
        return self.store.job(job["job_id"]) or job

    def _advance_after_training(self, job: dict) -> dict:
        from patron.runtime.learning import default_evaluate_command

        run_dir = Path(job["run_dir"]) if job.get("run_dir") else None
        checkpoint = run_dir / "best.pt" if run_dir else None
        if checkpoint is None or not checkpoint.exists():
            return self._fail(job, "training finished without best.pt")
        config = json.loads(job["config"]) if job.get("config") else {}

        if self.cfg.learning.command:
            # A custom command owns its own evaluation; gate straight away.
            self.store.update_job(job["job_id"], status="GATING", finished_at=None)
            return self._finalize(self.store.job(job["job_id"]) or job)

        command = default_evaluate_command(str(run_dir), config.get("dataset", ""))
        log_path = LOG_DIR / "runtime" / f"{run_dir.name}_eval.log"
        pid = self.runner.spawn(command, self.root, log_path)
        self.store.update_job(
            job["job_id"], status="EVALUATING", pid=pid,
            started_at=utc_iso(datetime.now(timezone.utc)), log_path=str(log_path),
        )
        self.store.log_event("candidate_evaluation_started", job["job_id"], {"command": command})
        return self.store.job(job["job_id"]) or job

    def _advance_after_evaluation(self, job: dict) -> dict:
        run_dir = Path(job["run_dir"])
        if not (run_dir / "evaluation_report.json").exists():
            return self._fail(job, "evaluation finished without evaluation_report.json")
        self.store.update_job(job["job_id"], status="GATING")
        return self._finalize(self.store.job(job["job_id"]))

    def _finalize(self, job: dict) -> dict:
        from patron.runtime.governance import (
            champion_metrics,
            evaluate_gates,
            load_run_metrics,
            promote_candidate,
            register_candidate,
            reject_candidate,
            validate_candidate_checkpoint,
        )
        from patron.runtime.learning import LAST_TRAIN_AT, LAST_TRAIN_MATURED, matured_horizon_count
        from patron.runtime.window import build_window

        run_dir = Path(job["run_dir"])
        config = json.loads(job["config"]) if job.get("config") else {}
        try:
            metrics = load_run_metrics(run_dir)
            window = build_window(
                self.store, self.cfg.symbol, self.cfg.timeframe,
                self.cfg.history.sequence_length, self.cfg.history.warmup_bars,
            )
            validity = validate_candidate_checkpoint(
                run_dir / "best.pt", window, self.cfg, self.model_cfg
            )
            version = register_candidate(
                self.store, run_dir, metrics,
                reference=self.store.champion().get("reference") if self.store.champion() else None,
            )
            gate_result = evaluate_gates(
                metrics, champion_metrics(self.store), self.cfg, validity,
                seeds=int(config.get("seeds", 1)),
            )
            decision = {
                "candidate": version,
                "champion": (self.store.champion() or {}).get("version"),
                "metrics": metrics,
                "validity": validity,
                **gate_result,
            }
            (run_dir / "promotion_decision.json").write_text(
                json.dumps(decision, indent=2, default=str), encoding="utf-8"
            )
            if gate_result["passed"]:
                promote_candidate(self.store, version)
                self.store.set_kv(LAST_TRAIN_AT, utc_iso(datetime.now(timezone.utc)))
                self.store.set_kv(LAST_TRAIN_MATURED, matured_horizon_count(self.store))
                self.store.update_job(job["job_id"], status="PROMOTED", finished_at=utc_iso(datetime.now(timezone.utc)),
                                      result=decision)
            else:
                reject_candidate(self.store, version, gate_result)
                self.store.update_job(job["job_id"], status="REJECTED", finished_at=utc_iso(datetime.now(timezone.utc)),
                                      result=decision)
                self.store.log_event("champion_retained", version, {"reasons": gate_result["reasons"]})
        except Exception as exc:  # gating failure must never disturb inference
            return self._fail(job, f"gating failed: {exc}")
        return self.store.job(job["job_id"]) or job

    def _fail(self, job: dict, reason: str) -> dict:
        self.store.update_job(
            job["job_id"], status="FAILED", finished_at=utc_iso(datetime.now(timezone.utc)),
            result={"error": reason},
        )
        self.store.log_event("candidate_failed", job["job_id"], {"reason": reason})
        LOGGER.warning("candidate job %s failed: %s", job["job_id"], reason)
        return self.store.job(job["job_id"]) or job
