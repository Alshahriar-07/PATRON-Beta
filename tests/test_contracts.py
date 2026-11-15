"""Frozen-contract regression tests for the release."""

from __future__ import annotations

import json
import re
from pathlib import Path

from patron import __version__
from patron.config import load_config
from patron.features.schema import FEATURE_ORDER
from patron.models.patron_net import build_model, count_parameters
from patron.paths import PROJECT_ROOT, SCHEMA_DIR

PACKAGE_SCHEMAS = Path(__file__).resolve().parents[1] / "src" / "patron" / "schemas"


def test_parameter_count_is_frozen():
    cfg = load_config()
    assert count_parameters(build_model(cfg)) == 272_618


def test_feature_contract():
    cfg = load_config()
    assert cfg.sequence_length == 96
    assert cfg.feature_count == 12
    assert list(cfg.feature_order) == list(FEATURE_ORDER)
    assert len(FEATURE_ORDER) == 12


def test_label_contract():
    cfg = load_config()
    assert cfg.horizon == 6
    assert tuple(cfg.quantiles) == (0.1, 0.5, 0.9)
    assert len(cfg.pattern_tags) == 12
    assert len(cfg.momentum_classes) == 5
    assert cfg.forecast.barrier.tp_atr_multiplier == 1.5
    assert cfg.forecast.barrier.sl_atr_multiplier == 1.0
    assert cfg.forecast.momentum.normalized_return_boundaries == [-1.0, -0.35, 0.35, 1.0]


def test_package_schemas_match_repository_schemas():
    for name in ("input.schema.json", "output.schema.json", "forecast.schema.json"):
        root_copy = (SCHEMA_DIR / name).read_bytes()
        package_copy = (PACKAGE_SCHEMAS / name).read_bytes()
        assert root_copy == package_copy, f"{name} drifted between schemas/ and patron/schemas/"


def test_output_schema_required_fields():
    schema = json.loads((SCHEMA_DIR / "output.schema.json").read_text(encoding="utf-8"))
    required = {
        "status",
        "symbol",
        "as_of",
        "horizon_minutes",
        "model_version",
        "pattern_scores",
        "barrier_probabilities",
        "range_quantiles",
        "momentum",
        "data_quality",
    }
    assert required.issubset(set(schema.get("required", [])))
    assert set(schema["properties"]["status"].get("enum", [])) >= {"OK", "NO_OUTPUT"}


def test_version_matches_pyproject():
    text = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    declared = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M).group(1)
    assert declared == __version__
    assert __version__ == "1.0.0b1"


def test_license_is_apache_and_notice_exists():
    license_text = (PROJECT_ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "Apache License" in license_text and "Version 2.0" in license_text
    notice = (PROJECT_ROOT / "NOTICE").read_text(encoding="utf-8")
    assert "Apache License" in notice


def test_release_documents_exist_at_root():
    required = [
        "README.md",
        "CHANGELOG.md",
        "RELEASE_NOTES.md",
        "SECURITY.md",
        "CONTRIBUTING.md",
        "CODE_OF_CONDUCT.md",
        "MODEL_CARD.md",
        "DATASET_CARD.md",
        "REPRODUCIBILITY.md",
        "INSTALLATION.md",
        "QUICKSTART.md",
        "CONFIGURATION.md",
        "LIMITATIONS.md",
        "THIRD_PARTY_NOTICES.md",
        "LICENSE",
        "NOTICE",
        "requirements.txt",
        ".gitignore",
        ".gitattributes",
    ]
    missing = [name for name in required if not (PROJECT_ROOT / name).exists()]
    assert not missing, missing


def test_checkpoint_reload_keeps_parameter_count(tmp_path):
    """A checkpoint written by the trainer must reload under the safe loader."""
    from patron.training.trainer import build_model_from_checkpoint

    checkpoint = PROJECT_ROOT / "experiments" / "patron_v2_20261114-203859" / "best.pt"
    if not checkpoint.exists():
        import pytest

        pytest.skip("reference checkpoint not present")
    from patron.training.trainer import load_checkpoint

    payload = load_checkpoint(checkpoint)
    model = build_model_from_checkpoint(payload)
    assert sum(p.numel() for p in model.parameters()) == 272_618
    assert list(payload["feature_order"]) == list(FEATURE_ORDER)
