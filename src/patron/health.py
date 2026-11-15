"""Environment, contract and readiness checks for PATRON.

``python -m patron health`` prints a human-readable report; ``--json`` returns the
same result as machine-readable JSON. The check never writes to project data:
the runtime database is opened read-only and nothing is created or deleted.
"""

from __future__ import annotations

import importlib
import json
import os
import platform
import sqlite3
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"

# import name -> distribution name (they differ for sklearn and PyYAML)
REQUIRED_MODULES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "pyarrow": "pyarrow",
    "sklearn": "scikit-learn",
    "torch": "torch",
    "yaml": "PyYAML",
    "pydantic": "pydantic",
    "jsonschema": "jsonschema",
    "requests": "requests",
}
EXPECTED_PARAMETERS = 272_618
MIN_PYTHON = (3, 12)


@dataclass
class Check:
    name: str
    status: str
    detail: str
    hint: str = ""


def _check_python() -> Check:
    version = sys.version_info
    ok = (version.major, version.minor) >= MIN_PYTHON
    return Check(
        "python_version",
        PASS if ok else FAIL,
        f"{platform.python_version()} ({platform.python_implementation()}) on {platform.system()}",
        "" if ok else f"PATRON requires Python >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]}",
    )


def _check_dependencies() -> Check:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as package_version

    missing: list[str] = []
    versions: dict[str, str] = {}
    for module_name, distribution in REQUIRED_MODULES.items():
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - depends on environment
            missing.append(f"{distribution} ({type(exc).__name__})")
            continue
        try:
            versions[distribution] = package_version(distribution)
        except PackageNotFoundError:  # pragma: no cover - vendored installs
            versions[distribution] = "unknown"
    if missing:
        return Check(
            "dependencies",
            FAIL,
            "missing: " + ", ".join(missing),
            "install with: pip install -r requirements.txt",
        )
    return Check(
        "dependencies",
        PASS,
        ", ".join(f"{k}=={v}" for k, v in versions.items()),
    )


def _check_torch_runtime() -> Check:
    try:
        import torch

        tensor = torch.ones(2, 2) @ torch.ones(2, 2)
        if float(tensor.sum()) != 8.0:
            return Check("torch_runtime", FAIL, "matrix multiply returned an unexpected value")
    except Exception as exc:  # pragma: no cover - environment specific
        return Check(
            "torch_runtime",
            FAIL,
            f"{type(exc).__name__}: {exc}",
            "reinstall the CPU wheel from requirements.txt",
        )
    return Check(
        "torch_runtime",
        PASS,
        f"torch {torch.__version__}, threads={torch.get_num_threads()}, device=cpu",
    )


def _check_config(config_path: Path) -> tuple[Check, object | None]:
    from patron.config import load_config

    if not config_path.exists():
        return Check("config", FAIL, f"not found: {config_path}", "run from the repository root"), None
    try:
        cfg = load_config(str(config_path))
    except Exception as exc:
        return Check("config", FAIL, f"invalid config: {exc}"), None
    return (
        Check(
            "config",
            PASS,
            f"{cfg.symbol} {cfg.timeframe}, seq {cfg.sequence_length}, "
            f"horizon {cfg.horizon}, features {cfg.feature_count}",
        ),
        cfg,
    )


def _check_model(cfg) -> Check:
    from patron.models.patron_net import build_model, count_parameters

    try:
        params = count_parameters(build_model(cfg))
    except Exception as exc:
        return Check("model_contract", FAIL, f"cannot build model: {exc}")
    if params != EXPECTED_PARAMETERS:
        return Check(
            "model_contract",
            FAIL,
            f"{params:,} trainable parameters (expected {EXPECTED_PARAMETERS:,})",
            "architecture or config changed; this release expects the frozen v2 model",
        )
    return Check(
        "model_contract",
        PASS,
        f"{params:,} trainable parameters, input (96, 12), 4 heads",
    )


def _check_scaler() -> Check:
    from patron.paths import FEATURE_DIR
    from patron.features.schema import FEATURE_ORDER

    path = FEATURE_DIR / "scaler.json"
    if not path.exists():
        return Check("feature_schema", WARN, "scaler.json not found", "run scripts/build_dataset.py")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        order = payload.get("feature_order") or payload.get("features")
    except Exception as exc:
        return Check("feature_schema", FAIL, f"unreadable scaler: {exc}")
    if list(order or []) != list(FEATURE_ORDER):
        return Check(
            "feature_schema",
            FAIL,
            "scaler feature order does not match the 12-feature contract",
            "rebuild the dataset with the current feature builder",
        )
    return Check("feature_schema", PASS, "scaler present, 12-feature order matches the contract")


def _check_dataset(dataset_path: Path) -> Check:
    if not dataset_path.exists():
        return Check(
            "dataset",
            WARN,
            f"not found: {dataset_path.name}",
            "download data and run scripts/build_dataset.py (the release does not ship the dataset)",
        )
    try:
        import pyarrow.parquet as pq

        schema = pq.read_schema(dataset_path)
        names = set(schema.names)
        required = {"timestamp_open", "close", "barrier_label", "momentum_class", "sample_valid"}
        missing = sorted(required - names)
        if missing:
            return Check("dataset", FAIL, f"missing columns: {missing}")
        meta = pq.read_metadata(dataset_path)
    except Exception as exc:
        return Check("dataset", FAIL, f"unreadable parquet: {exc}")
    return Check(
        "dataset",
        PASS,
        f"{meta.num_rows:,} rows, {len(schema.names)} columns, {dataset_path.name}",
    )


def _check_checkpoint(checkpoint: Path) -> Check:
    if not checkpoint.exists():
        return Check(
            "checkpoint",
            WARN,
            f"not found: {checkpoint}",
            "train a model (python -m patron train) or pass --checkpoint",
        )
    try:
        from patron.training.trainer import build_model_from_checkpoint, load_checkpoint

        payload = load_checkpoint(checkpoint)
        model = build_model_from_checkpoint(payload)
    except Exception as exc:
        return Check("checkpoint", FAIL, f"cannot load safely: {exc}")
    params = sum(p.numel() for p in model.parameters())
    if params != EXPECTED_PARAMETERS:
        return Check("checkpoint", FAIL, f"checkpoint has {params:,} parameters")
    return Check(
        "checkpoint",
        PASS,
        f"{checkpoint.name}: {params:,} parameters, epoch {payload.get('epoch')}, "
        f"val_loss {payload.get('val_loss')}",
    )


def _check_schemas(schema_root: Path) -> Check:
    names = ("input.schema.json", "output.schema.json", "forecast.schema.json")
    missing = [name for name in names if not (schema_root / name).exists()]
    if missing:
        return Check(
            "schemas",
            FAIL,
            f"missing files: {missing} in {schema_root}",
            "restore the files from the repository or reinstall the package",
        )
    for name in names:
        try:
            json.loads((schema_root / name).read_text(encoding="utf-8"))
        except Exception as exc:
            return Check("schemas", FAIL, f"{name}: {exc}")
    return Check("schemas", PASS, "input / output / forecast JSON schemas present and parseable")


def _check_runtime(database: Path) -> Check:
    if not database.exists():
        return Check(
            "runtime_state",
            WARN,
            "runtime database not initialised",
            "start it with: python -m patron runtime once",
        )
    try:
        conn = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
        try:
            tables = {
                row[0]
                for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
        finally:
            conn.close()
    except Exception as exc:
        return Check("runtime_state", FAIL, f"cannot open read-only: {exc}")
    expected = {"candles", "forecasts", "evaluations", "registry"}
    if not expected.issubset(tables):
        return Check("runtime_state", FAIL, f"unexpected schema, tables={sorted(tables)}")
    return Check("runtime_state", PASS, f"{database.name}: {len(tables)} tables, read-only check")


def _check_write_access(root: Path) -> Check:
    targets = [root / "logs", root / "experiments"]
    unwritable = [str(p) for p in targets if p.exists() and not os.access(p, os.W_OK)]
    if unwritable:
        return Check("write_access", FAIL, f"not writable: {unwritable}")
    return Check("write_access", PASS, "logs/ and experiments/ are writable")


def _check_inference(checkpoint: Path) -> Check:
    if not checkpoint.exists():
        return Check("inference", WARN, "skipped: no checkpoint")
    try:
        from patron.inference.pipeline import load_predictor

        predictor = load_predictor(checkpoint)
        if predictor.scaler is None:
            return Check("inference", FAIL, "predictor has no scaler")
    except Exception as exc:
        return Check("inference", FAIL, f"predictor init failed: {exc}")
    return Check("inference", PASS, "predictor initialises from the checkpoint")


def _redact_path(value: str) -> str:
    """Hide the local home directory so reports can be shared safely."""
    home = str(Path.home())
    text = str(value)
    if home and text.startswith(home):
        text = "~" + text[len(home):]
    return text.replace("\\", "/")


def run_health(
    root: Path,
    config_path: Path,
    dataset_path: Path,
    checkpoint_path: Path,
    database_path: Path,
) -> dict:
    checks: list[Check] = [_check_python(), _check_dependencies(), _check_torch_runtime()]
    config_check, cfg = _check_config(config_path)
    checks.append(config_check)
    # Installed wheels resolve contracts from the package; source checkouts use the
    # repository copy (a test asserts the two stay identical).
    schema_root = (root / "schemas") if (root / "schemas").is_dir() else Path(__file__).resolve().parent / "schemas"
    checks.append(_check_schemas(schema_root))
    if cfg is not None:
        checks.append(_check_model(cfg))
        checks.append(_check_scaler())
    checks.append(_check_dataset(dataset_path))
    checks.append(_check_checkpoint(checkpoint_path))
    checks.append(_check_inference(checkpoint_path))
    checks.append(_check_runtime(database_path))
    checks.append(_check_write_access(root))

    statuses = [c.status for c in checks]
    overall = FAIL if FAIL in statuses else WARN if WARN in statuses else PASS
    return {
        "status": overall,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "executable": _redact_path(sys.executable),
            "root": _redact_path(root),
        },
        "counts": {
            PASS: statuses.count(PASS),
            WARN: statuses.count(WARN),
            FAIL: statuses.count(FAIL),
        },
        "checks": [asdict(c) for c in checks],
    }


def format_report(result: dict) -> str:
    lines = [
        f"PATRON health: {result['status']}  "
        f"({result['counts'][PASS]} pass, {result['counts'][WARN]} warn, "
        f"{result['counts'][FAIL]} fail)",
        f"python {result['environment']['python']} | {result['environment']['platform']}",
        "",
    ]
    for check in result["checks"]:
        lines.append(f"[{check['status']:4}] {check['name']}: {check['detail']}")
        if check["hint"]:
            lines.append(f"       -> {check['hint']}")
    return "\n".join(lines)
