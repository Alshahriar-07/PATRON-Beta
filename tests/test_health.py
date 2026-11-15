"""Health-check behaviour: statuses, JSON shape, and no side effects."""

from __future__ import annotations

import json
from pathlib import Path

from patron.health import FAIL, PASS, WARN, format_report, run_health
from patron.paths import PROJECT_ROOT


def _paths():
    return dict(
        root=PROJECT_ROOT,
        config_path=PROJECT_ROOT / "config" / "model_config.yaml",
        dataset_path=PROJECT_ROOT / "data" / "features" / "BTCUSDT_dataset.parquet",
        checkpoint_path=PROJECT_ROOT / "experiments" / "patron_v2_20261114-203859" / "best.pt",
        database_path=PROJECT_ROOT / "data" / "runtime" / "patron.db",
    )


def test_health_reports_statuses_and_json_shape():
    result = run_health(**_paths())
    assert result["status"] in {PASS, WARN}
    names = {check["name"] for check in result["checks"]}
    required = {
        "python_version",
        "dependencies",
        "torch_runtime",
        "config",
        "schemas",
        "model_contract",
        "dataset",
        "checkpoint",
        "inference",
        "runtime_state",
        "write_access",
    }
    assert required.issubset(names)
    assert result["counts"][PASS] + result["counts"][WARN] == len(result["checks"])
    json.dumps(result)  # must be JSON serialisable
    assert "PATRON health:" in format_report(result)
    assert "model_contract" in format_report(result)


def test_missing_dataset_is_a_warning_not_a_failure(tmp_path):
    paths = _paths()
    paths["dataset_path"] = tmp_path / "absent.parquet"
    result = run_health(**paths)
    dataset = next(c for c in result["checks"] if c["name"] == "dataset")
    assert dataset["status"] == WARN
    assert "not found" in dataset["detail"]


def test_invalid_config_fails(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("project: [not, a, mapping]\n", encoding="utf-8")
    paths = _paths()
    paths["config_path"] = bad
    result = run_health(**paths)
    assert result["status"] == FAIL
    assert any(c["name"] == "config" and c["status"] == FAIL for c in result["checks"])


def test_health_does_not_modify_the_runtime_database():
    database = _paths()["database_path"]
    before = (database.stat().st_mtime_ns, database.read_bytes()[:64]) if database.exists() else None
    run_health(**_paths())
    after = (database.stat().st_mtime_ns, database.read_bytes()[:64]) if database.exists() else None
    assert before == after


def test_package_schema_directory_is_valid():
    """An installed wheel resolves contracts from patron/schemas, not the repo root."""
    import patron
    from patron.health import _check_schemas

    package_schemas = Path(patron.__file__).resolve().parent / "schemas"
    check = _check_schemas(package_schemas)
    assert check.status == PASS, check.detail


def test_health_exposes_no_absolute_home_path():
    result = run_health(**_paths())
    home = str(Path.home())
    payload = json.dumps(result)
    assert home not in payload, "health output leaks the local home directory"
