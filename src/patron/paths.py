"""Canonical paths for the PATRON project.

All paths are resolved relative to the repository root so that scripts can be
executed from any working directory.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_DIR = PROJECT_ROOT / "config"
SCHEMA_DIR = PROJECT_ROOT / "schemas"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"
FEATURE_DIR = DATA_DIR / "features"
MANIFEST_DIR = DATA_DIR / "manifests"
EXPERIMENT_DIR = PROJECT_ROOT / "experiments"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
REPORT_DIR = PROJECT_ROOT / "reports"
PLAN_DIR = PROJECT_ROOT / "plan"
LOG_DIR = PROJECT_ROOT / "logs"

DEFAULT_CONFIG_PATH = CONFIG_DIR / "model_config.yaml"


def ensure_dirs() -> None:
    """Create every directory the pipeline writes to."""
    for path in (
        CONFIG_DIR,
        SCHEMA_DIR,
        RAW_DIR,
        CLEAN_DIR,
        FEATURE_DIR,
        MANIFEST_DIR,
        EXPERIMENT_DIR,
        CHECKPOINT_DIR,
        REPORT_DIR,
        LOG_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def symbol_slug(symbol: str) -> str:
    """`BTC/USDT` -> `BTCUSDT` for filesystem and API use."""
    return symbol.replace("/", "").replace("-", "").upper()
