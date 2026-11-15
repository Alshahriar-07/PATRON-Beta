"""Security-relevant behaviour: safe checkpoint loading, validation, scanning."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import scan_release  # noqa: E402
from patron.inference.pipeline import InputValidationError, validate_payload  # noqa: E402
from patron.training.trainer import load_checkpoint  # noqa: E402


class _Unpicklable:
    """Stand-in for a malicious payload: requires full unpickling."""

    def __init__(self) -> None:
        self.value = 1


def test_checkpoint_with_custom_object_is_rejected_in_safe_mode(tmp_path):
    path = tmp_path / "evil.pt"
    torch.save({"payload": _Unpicklable()}, path)
    with pytest.raises(ValueError, match="safe mode"):
        load_checkpoint(path)


def test_non_checkpoint_payload_is_rejected(tmp_path):
    path = tmp_path / "list.pt"
    torch.save([1, 2, 3], path)
    with pytest.raises(ValueError, match="not a PATRON checkpoint"):
        load_checkpoint(path)


def test_missing_keys_are_reported(tmp_path):
    path = tmp_path / "partial.pt"
    torch.save({"model_state": {}}, path)
    with pytest.raises(ValueError, match="missing keys"):
        load_checkpoint(path)


def test_missing_file_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "absent.pt")


def test_inference_rejects_malformed_payload():
    with pytest.raises(InputValidationError):
        validate_payload({"symbol": "BTC/USDT", "timeframe": "15m", "candles": []})


def test_inference_rejects_non_finite_prices():
    candles = [
        {
            "timestamp": f"2026-01-01T{hour:02d}:{minute:02d}:00+00:00",
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": float("nan"),
            "volume": 1.0,
        }
        for hour, minute in ((0, 0), (0, 15))
    ]
    with pytest.raises((InputValidationError, ValueError)):
        validate_payload({"symbol": "BTC/USDT", "timeframe": "15m", "candles": candles})


def test_scanner_detects_planted_secrets_and_paths(tmp_path):
    # Built at runtime so this test file does not itself contain a secret-looking
    # literal (the scanner would flag the test as a leak).
    secret_line = "API" + '_KEY = "' + "a" * 32 + '"\n'
    path_line = "HOME = '" + "C:" + "\\" + "Users" + "\\" + "someone" + "'\n"
    (tmp_path / "leak.py").write_text(secret_line + path_line, encoding="utf-8")
    findings = scan_release.scan(tmp_path)
    kinds = {f.kind for f in findings}
    assert "assigned_secret" in kinds
    assert "windows_home" in kinds


def test_scanner_detects_artifacts_and_oversized_files(tmp_path):
    (tmp_path / "model.pt").write_bytes(b"\x00" * 16)
    big = tmp_path / "big.txt"
    big.write_bytes(b"a" * (scan_release.MAX_BYTES + 1))
    kinds = {f.kind for f in scan_release.scan(tmp_path)}
    assert "forbidden_artifact" in kinds
    assert "large_file" in kinds


def test_scanner_clean_tree(tmp_path):
    (tmp_path / "README.md").write_text("# hello\n", encoding="utf-8")
    assert scan_release.scan(tmp_path) == []


def test_scanner_allows_example_env(tmp_path):
    (tmp_path / ".env.example").write_text(
        'BINANCE_API_KEY="replace-me-unused-placeholder"\n', encoding="utf-8"
    )
    assert scan_release.scan(tmp_path) == []
