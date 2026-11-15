"""Durable state for the autonomous runtime (SQLite, WAL, append-only by design).

Three invariants this store enforces:

1. **One forecast per closed candle.** ``UNIQUE(symbol, timeframe, candle_ts)`` plus
   an insert-or-ignore means reprocessing a candle can never duplicate a forecast.
2. **One evaluation per (forecast, horizon).** The primary key makes maturation
   idempotent across retries and restarts.
3. **Predictions are immutable.** Forecast payloads are inserted once; matured actuals
   and metrics go into the ``evaluations`` table and are linked, never written back.

All timestamps are stored as UTC ISO-8601 strings so they sort correctly and remain
readable in an audit without a parser.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from patron.paths import DATA_DIR

CANDLE_COLUMNS = ("ts_open", "ts_close", "open", "high", "low", "close", "volume")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS candles (
    symbol     TEXT NOT NULL,
    timeframe  TEXT NOT NULL,
    ts_open    TEXT NOT NULL,
    ts_close   TEXT NOT NULL,
    open       REAL NOT NULL,
    high       REAL NOT NULL,
    low        REAL NOT NULL,
    close      REAL NOT NULL,
    volume     REAL NOT NULL,
    source     TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    revision   INTEGER NOT NULL DEFAULT 0,
    checksum   TEXT,
    PRIMARY KEY (symbol, timeframe, ts_open)
);

CREATE TABLE IF NOT EXISTS cycles (
    cycle_key   TEXT PRIMARY KEY,
    candle_ts   TEXT NOT NULL,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    status      TEXT NOT NULL,
    detail      TEXT,
    duration_ms REAL
);

CREATE TABLE IF NOT EXISTS forecasts (
    forecast_id         TEXT PRIMARY KEY,
    created_at          TEXT NOT NULL,
    as_of               TEXT NOT NULL,
    symbol              TEXT NOT NULL,
    timeframe           TEXT NOT NULL,
    candle_ts           TEXT NOT NULL,
    input_start         TEXT NOT NULL,
    input_end           TEXT NOT NULL,
    input_hash          TEXT NOT NULL,
    model_name          TEXT NOT NULL,
    model_version       TEXT NOT NULL,
    checkpoint          TEXT,
    checkpoint_sha      TEXT,
    feature_version     TEXT NOT NULL,
    calibration_version TEXT,
    reference_close     REAL,
    reference_atr       REAL,
    source              TEXT,
    payload             TEXT NOT NULL,
    model_output        TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS forecasts_one_per_candle
    ON forecasts(symbol, timeframe, candle_ts);

CREATE TABLE IF NOT EXISTS evaluations (
    forecast_id  TEXT NOT NULL,
    step         INTEGER NOT NULL,
    target_ts    TEXT NOT NULL,
    evaluated_at TEXT NOT NULL,
    status       TEXT NOT NULL,
    actual       TEXT,
    metrics      TEXT,
    label_state  TEXT,
    PRIMARY KEY (forecast_id, step)
);

CREATE TABLE IF NOT EXISTS registry (
    version          TEXT PRIMARY KEY,
    name             TEXT NOT NULL,
    path             TEXT NOT NULL,
    sha256           TEXT NOT NULL,
    role             TEXT NOT NULL,
    created_at       TEXT NOT NULL,
    promoted_at      TEXT,
    metrics          TEXT,
    notes            TEXT,
    reference        TEXT,
    previous_version TEXT
);

CREATE TABLE IF NOT EXISTS jobs (
    job_id      TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,
    status      TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    started_at  TEXT,
    finished_at TEXT,
    config      TEXT,
    result      TEXT,
    log_path    TEXT,
    pid         INTEGER,
    run_dir     TEXT
);

CREATE TABLE IF NOT EXISTS events (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    at      TEXT NOT NULL,
    kind    TEXT NOT NULL,
    subject TEXT,
    payload TEXT
);

CREATE INDEX IF NOT EXISTS events_kind_subject ON events(kind, subject);

CREATE TABLE IF NOT EXISTS kv (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def utc_iso(value: datetime | pd.Timestamp | str) -> str:
    """Normalise any timestamp to a UTC ISO-8601 ``...Z`` string."""
    if isinstance(value, str):
        ts = pd.Timestamp(value)
    elif isinstance(value, pd.Timestamp):
        ts = value
    else:
        ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_ts(value: str) -> pd.Timestamp:
    return pd.Timestamp(value).tz_convert("UTC")


def _json(payload: Any) -> str:
    def default(obj: Any) -> Any:
        if isinstance(obj, (pd.Timestamp, datetime)):
            return utc_iso(obj)
        if hasattr(obj, "item"):
            return obj.item()
        raise TypeError(f"not serialisable: {type(obj)!r}")

    return json.dumps(payload, default=default, sort_keys=True)


class RuntimeStore:
    """Thin, explicit SQLite wrapper. No ORM, no hidden state."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else DATA_DIR / "runtime" / "patron.db"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), timeout=30)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA_SQL)
        self._conn.commit()

    # ------------------------------------------------------------------ plumbing
    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "RuntimeStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def execute(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        return self._conn.execute(sql, tuple(params))

    # ------------------------------------------------------------------ candles
    def upsert_candles(
        self,
        frame: pd.DataFrame,
        symbol: str,
        timeframe: str,
        source: str,
        fetched_at: datetime | None = None,
    ) -> dict[str, int]:
        """Insert closed candles; record revisions instead of silently overwriting."""
        fetched_at = fetched_at or datetime.now(timezone.utc)
        inserted = revised = unchanged = 0
        with self._conn:
            for row in frame.itertuples(index=False):
                ts_open = utc_iso(getattr(row, "timestamp_open"))
                ts_close = utc_iso(getattr(row, "timestamp_close"))
                values = (
                    float(row.open),
                    float(row.high),
                    float(row.low),
                    float(row.close),
                    float(row.volume),
                )
                existing = self._conn.execute(
                    "SELECT open, high, low, close, volume, revision FROM candles "
                    "WHERE symbol=? AND timeframe=? AND ts_open=?",
                    (symbol, timeframe, ts_open),
                ).fetchone()
                if existing is None:
                    self._conn.execute(
                        "INSERT INTO candles (symbol, timeframe, ts_open, ts_close, open, high, "
                        "low, close, volume, source, fetched_at, revision, checksum) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?,?,0,?)",
                        (symbol, timeframe, ts_open, ts_close, *values, source, utc_iso(fetched_at), source),
                    )
                    inserted += 1
                    continue
                old = tuple(existing[key] for key in ("open", "high", "low", "close", "volume"))
                if all(abs(a - b) <= 1e-12 for a, b in zip(old, values)):
                    unchanged += 1
                    continue
                revision = int(existing["revision"]) + 1
                self._conn.execute(
                    "UPDATE candles SET ts_close=?, open=?, high=?, low=?, close=?, volume=?, "
                    "source=?, fetched_at=?, revision=? WHERE symbol=? AND timeframe=? AND ts_open=?",
                    (ts_close, *values, source, utc_iso(fetched_at), revision, symbol, timeframe, ts_open),
                )
                self.log_event(
                    "candle_revised",
                    ts_open,
                    {"symbol": symbol, "timeframe": timeframe, "revision": revision,
                     "before": dict(zip(CANDLE_COLUMNS[2:], old)), "after": dict(zip(CANDLE_COLUMNS[2:], values))},
                    commit=False,
                )
                revised += 1
            self._conn.commit()
        return {"inserted": inserted, "revised": revised, "unchanged": unchanged}

    def candles(
        self, symbol: str, timeframe: str, start: str | None = None, end: str | None = None
    ) -> pd.DataFrame:
        sql = (
            "SELECT ts_open, ts_close, open, high, low, close, volume, source, fetched_at, revision "
            "FROM candles WHERE symbol=? AND timeframe=?"
        )
        params: list[Any] = [symbol, timeframe]
        if start is not None:
            sql += " AND ts_open>=?"
            params.append(utc_iso(start))
        if end is not None:
            sql += " AND ts_open<=?"
            params.append(utc_iso(end))
        sql += " ORDER BY ts_open"
        frame = pd.read_sql_query(sql, self._conn, params=params)
        if frame.empty:
            return frame
        frame["timestamp_open"] = pd.to_datetime(frame["ts_open"], utc=True)
        frame["timestamp_close"] = pd.to_datetime(frame["ts_close"], utc=True)
        return frame

    def latest_candles(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        frame = pd.read_sql_query(
            "SELECT ts_open, ts_close, open, high, low, close, volume, source, fetched_at, revision "
            "FROM candles WHERE symbol=? AND timeframe=? ORDER BY ts_open DESC LIMIT ?",
            self._conn,
            params=(symbol, timeframe, int(count)),
        )
        if frame.empty:
            return frame
        frame = frame.iloc[::-1].reset_index(drop=True)
        frame["timestamp_open"] = pd.to_datetime(frame["ts_open"], utc=True)
        frame["timestamp_close"] = pd.to_datetime(frame["ts_close"], utc=True)
        return frame

    def candle(self, symbol: str, timeframe: str, ts_open: str) -> pd.Series | None:
        row = self._conn.execute(
            "SELECT ts_open, ts_close, open, high, low, close, volume, source, fetched_at, revision "
            "FROM candles WHERE symbol=? AND timeframe=? AND ts_open=?",
            (symbol, timeframe, utc_iso(ts_open)),
        ).fetchone()
        return None if row is None else pd.Series(dict(row))

    def candle_count(self, symbol: str, timeframe: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM candles WHERE symbol=? AND timeframe=?", (symbol, timeframe)
        ).fetchone()
        return int(row["n"])

    # ------------------------------------------------------------------- cycles
    def mark_cycle(
        self, cycle_key: str, candle_ts: str, status: str, detail: str = "", duration_ms: float | None = None
    ) -> None:
        now = utc_iso(datetime.now(timezone.utc))
        with self._conn:
            self._conn.execute(
                "INSERT INTO cycles (cycle_key, candle_ts, started_at, finished_at, status, detail, duration_ms) "
                "VALUES (?,?,?,?,?,?,?) ON CONFLICT(cycle_key) DO UPDATE SET finished_at=excluded.finished_at, "
                "status=excluded.status, detail=excluded.detail, duration_ms=excluded.duration_ms",
                (cycle_key, candle_ts, now, now, status, detail, duration_ms),
            )

    def last_cycle(self) -> dict | None:
        # Chronological by completion time: failed attempts carry a synthetic candle_ts.
        row = self._conn.execute(
            "SELECT * FROM cycles ORDER BY finished_at DESC, rowid DESC LIMIT 1"
        ).fetchone()
        return None if row is None else dict(row)

    def cycle_done(self, candle_ts: str, statuses: tuple[str, ...] = ("OK", "NO_OUTPUT", "STALE_DATA")) -> bool:
        row = self._conn.execute(
            "SELECT status FROM cycles WHERE candle_ts=? ORDER BY finished_at DESC LIMIT 1",
            (utc_iso(candle_ts),),
        ).fetchone()
        return bool(row and row["status"] in statuses)

    # ---------------------------------------------------------------- forecasts
    def insert_forecast(self, record: dict) -> bool:
        """Insert once per candle. Returns False when the candle already had a forecast."""
        cols = (
            "forecast_id", "created_at", "as_of", "symbol", "timeframe", "candle_ts",
            "input_start", "input_end", "input_hash", "model_name", "model_version",
            "checkpoint", "checkpoint_sha", "feature_version", "calibration_version",
            "reference_close", "reference_atr", "source", "payload", "model_output",
        )
        values = tuple(record.get(col) for col in cols)
        with self._conn:
            cur = self._conn.execute(
                f"INSERT OR IGNORE INTO forecasts ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                values,
            )
        return cur.rowcount > 0

    def forecast(self, forecast_id: str) -> dict | None:
        row = self._conn.execute("SELECT * FROM forecasts WHERE forecast_id=?", (forecast_id,)).fetchone()
        return None if row is None else dict(row)

    def forecast_for_candle(self, symbol: str, timeframe: str, candle_ts: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM forecasts WHERE symbol=? AND timeframe=? AND candle_ts=?",
            (symbol, timeframe, utc_iso(candle_ts)),
        ).fetchone()
        return None if row is None else dict(row)

    def recent_forecasts(self, limit: int = 50) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM forecasts ORDER BY candle_ts DESC LIMIT ?", (int(limit),)
        ).fetchall()
        return [dict(row) for row in rows]

    def forecasts_without_evaluations(self) -> list[dict]:
        """Forecasts that still have at least one horizon without an evaluation row.

        Done in Python instead of JSON1 SQL so the store works on any SQLite build.
        """
        done = {
            (row["forecast_id"], int(row["step"]))
            for row in self._conn.execute("SELECT forecast_id, step FROM evaluations")
        }
        out: list[dict] = []
        for row in self._conn.execute("SELECT * FROM forecasts ORDER BY candle_ts ASC"):
            record = dict(row)
            payload = json.loads(record["payload"])
            steps = {int(target["step"]) for target in payload.get("targets", [])}
            if any((record["forecast_id"], step) not in done for step in steps):
                out.append(record)
        return out

    # -------------------------------------------------------------- evaluations
    def record_evaluation(
        self,
        forecast_id: str,
        step: int,
        target_ts: str,
        status: str,
        actual: dict | None,
        metrics: dict | None,
        label_state: str | None = None,
    ) -> bool:
        with self._conn:
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO evaluations "
                "(forecast_id, step, target_ts, evaluated_at, status, actual, metrics, label_state) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (
                    forecast_id,
                    int(step),
                    utc_iso(target_ts),
                    utc_iso(datetime.now(timezone.utc)),
                    status,
                    _json(actual) if actual is not None else None,
                    _json(metrics) if metrics is not None else None,
                    label_state,
                ),
            )
        return cur.rowcount > 0

    def evaluations(self, forecast_id: str | None = None, limit: int = 1000) -> list[dict]:
        if forecast_id is None:
            rows = self._conn.execute(
                "SELECT * FROM evaluations ORDER BY target_ts DESC LIMIT ?", (int(limit),)
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM evaluations WHERE forecast_id=? ORDER BY step", (forecast_id,)
            ).fetchall()
        return [dict(row) for row in rows]

    def matured_evaluation_count(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(DISTINCT forecast_id) AS n FROM evaluations WHERE status='EVALUATED'"
        ).fetchone()
        return int(row["n"])

    # ----------------------------------------------------------------- registry
    def register_model(
        self,
        version: str,
        name: str,
        path: str,
        sha256: str,
        role: str,
        metrics: dict | None,
        notes: str | None = None,
        reference: dict | None = None,
        previous_version: str | None = None,
    ) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO registry (version, name, path, sha256, role, created_at, promoted_at, "
                "metrics, notes, reference, previous_version) VALUES (?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(version) DO UPDATE SET role=excluded.role, path=excluded.path, "
                "sha256=excluded.sha256, metrics=excluded.metrics, notes=excluded.notes, "
                "reference=COALESCE(excluded.reference, registry.reference), previous_version=excluded.previous_version",
                (
                    version,
                    name,
                    path,
                    sha256,
                    role,
                    utc_iso(datetime.now(timezone.utc)),
                    utc_iso(datetime.now(timezone.utc)) if role == "champion" else None,
                    _json(metrics) if metrics is not None else None,
                    notes,
                    _json(reference) if reference is not None else None,
                    previous_version,
                ),
            )

    def champion(self) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM registry WHERE role='champion' ORDER BY promoted_at DESC LIMIT 1"
        ).fetchone()
        return None if row is None else dict(row)

    def model(self, version: str) -> dict | None:
        row = self._conn.execute("SELECT * FROM registry WHERE version=?", (version,)).fetchone()
        return None if row is None else dict(row)

    def models(self) -> list[dict]:
        rows = self._conn.execute("SELECT * FROM registry ORDER BY created_at").fetchall()
        return [dict(row) for row in rows]

    def set_champion(self, version: str) -> dict:
        """Atomically swap the champion pointer, demoting the previous champion."""
        now = utc_iso(datetime.now(timezone.utc))
        with self._conn:
            current = self._conn.execute(
                "SELECT version FROM registry WHERE role='champion' ORDER BY promoted_at DESC LIMIT 1"
            ).fetchone()
            previous = current["version"] if current else None
            self._conn.execute("UPDATE registry SET role='retired' WHERE role='champion'")
            self._conn.execute(
                "UPDATE registry SET role='champion', promoted_at=?, previous_version=? WHERE version=?",
                (now, previous, version),
            )
        self.log_event("champion_promoted", version, {"previous": previous})
        return {"version": version, "previous": previous}

    def rollback(self, version: str | None = None) -> dict:
        champion = self.champion()
        if champion is None:
            raise RuntimeError("no champion to roll back from")
        target = version or champion.get("previous_version")
        if not target:
            raise RuntimeError("no previous champion recorded; cannot roll back")
        if self.model(target) is None:
            raise RuntimeError(f"previous champion {target} is not in the registry")
        with self._conn:
            self._conn.execute("UPDATE registry SET role='retired' WHERE role='champion'")
            self._conn.execute(
                "UPDATE registry SET role='champion', promoted_at=? WHERE version=?",
                (utc_iso(datetime.now(timezone.utc)), target),
            )
        self.log_event("champion_rolled_back", target, {"from": champion["version"]})
        return {"version": target, "from": champion["version"]}

    # -------------------------------------------------------------------- jobs
    def create_job(self, kind: str, config: dict | None = None, run_dir: str | None = None) -> str:
        job_id = f"{kind}-{uuid.uuid4().hex[:12]}"
        with self._conn:
            self._conn.execute(
                "INSERT INTO jobs (job_id, kind, status, created_at, config, run_dir) VALUES (?,?,?,?,?,?)",
                (job_id, kind, "QUEUED", utc_iso(datetime.now(timezone.utc)),
                 _json(config) if config is not None else None, run_dir),
            )
        return job_id

    def update_job(self, job_id: str, **fields: Any) -> None:
        allowed = {"status", "started_at", "finished_at", "result", "log_path", "pid", "run_dir", "config"}
        unknown = set(fields) - allowed
        if unknown:
            raise KeyError(f"unknown job fields: {sorted(unknown)}")
        assignments = ", ".join(f"{key}=?" for key in fields)
        values = []
        for key, value in fields.items():
            if key in {"result", "config"} and value is not None and not isinstance(value, str):
                value = _json(value)
            values.append(value)
        with self._conn:
            self._conn.execute(f"UPDATE jobs SET {assignments} WHERE job_id=?", (*values, job_id))

    def job(self, job_id: str) -> dict | None:
        row = self._conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return None if row is None else dict(row)

    def jobs(self, limit: int = 20) -> list[dict]:
        rows = self._conn.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (int(limit),)).fetchall()
        return [dict(row) for row in rows]

    def active_job(self, kind: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM jobs WHERE kind=? AND status IN ('QUEUED','RUNNING','EVALUATING') "
            "ORDER BY created_at DESC LIMIT 1",
            (kind,),
        ).fetchone()
        return None if row is None else dict(row)

    # ------------------------------------------------------------------ events
    def log_event(self, kind: str, subject: str | None, payload: dict | None = None, commit: bool = True) -> None:
        self._conn.execute(
            "INSERT INTO events (at, kind, subject, payload) VALUES (?,?,?,?)",
            (utc_iso(datetime.now(timezone.utc)), kind, subject, _json(payload) if payload is not None else None),
        )
        if commit:
            self._conn.commit()

    def events(self, limit: int = 100, kind: str | None = None) -> list[dict]:
        if kind is None:
            rows = self._conn.execute("SELECT * FROM events ORDER BY id DESC LIMIT ?", (int(limit),)).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM events WHERE kind=? ORDER BY id DESC LIMIT ?", (kind, int(limit))
            ).fetchall()
        return [dict(row) for row in rows]

    # ---------------------------------------------------------------------- kv
    def get_kv(self, key: str, default: Any = None) -> Any:
        row = self._conn.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        if row is None:
            return default
        return json.loads(row["value"])

    def set_kv(self, key: str, value: Any) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO kv (key, value, updated_at) VALUES (?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (key, json.dumps(value, default=str), utc_iso(datetime.now(timezone.utc))),
            )
