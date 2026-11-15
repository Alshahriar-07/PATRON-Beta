"""CLI contract: help, version, exit codes and machine-readable output."""

from __future__ import annotations

import json

import pytest

from patron import __version__
from patron.cli import build_parser, main
from patron.paths import FEATURE_DIR


def test_parser_exposes_all_commands():
    parser = build_parser()
    actions = [a for a in parser._actions if getattr(a, "choices", None)]
    commands = set()
    for action in actions:
        commands.update(action.choices.keys())
    assert {"health", "train", "evaluate", "predict", "runtime", "experiment"}.issubset(commands)


@pytest.mark.parametrize(
    "argv",
    [
        ["health", "--help"],
        ["train", "--help"],
        ["evaluate", "--help"],
        ["predict", "--help"],
        ["runtime", "--help"],
        ["experiment", "--help"],
    ],
)
def test_subcommand_help_exits_zero(argv, capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(argv)
    assert excinfo.value.code == 0
    assert "usage" in capsys.readouterr().out.lower()


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_health_json_is_machine_readable(capsys):
    assert main(["health", "--json"]) in {0, 1}
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] in {"PASS", "WARN", "FAIL"}
    assert "checks" in payload and "counts" in payload


def test_predict_missing_input_returns_2(capsys):
    code = main(["predict", "--input", "does/not/exist.json"])
    assert code == 2
    assert "not found" in capsys.readouterr().err


def test_predict_missing_checkpoint_returns_2(tmp_path, capsys):
    npayload = tmp_path / "in.json"
    npayload.write_text("{}", encoding="utf-8")
    code = main(["predict", "--checkpoint", str(tmp_path / "nope.pt"), "--input", str(npayload)])
    assert code == 2


def test_experiment_list_runs(capsys):
    assert main(["experiment", "list"]) == 0
    out = capsys.readouterr().out
    assert "run" in out


def test_experiment_compare_requires_two_runs(capsys):
    code = main(["experiment", "compare", "--runs", "only-one"])
    assert code == 2
    assert "at least two" in capsys.readouterr().err


def test_train_dry_run(capsys):
    if not (FEATURE_DIR / "BTCUSDT_dataset.parquet").exists():
        pytest.skip("dataset not built")
    code = main(["train", "--dry-run", "--train-stride", "8"])
    assert code == 0
    out = capsys.readouterr().out
    assert "parameters: 272,618" in out
    assert "dry run" in out


def test_runtime_status_reads_published_state(capsys):
    code = main(["runtime", "status"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert "runtime" in payload
