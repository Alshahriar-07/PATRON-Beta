"""Autonomous runtime: autonomous ingestion, live forecasting, maturation,
monitoring, controlled retraining and champion/challenger governance.

Nothing in this package changes the frozen model contract; it orchestrates the
existing dataset, feature, inference and evaluation code.
"""

from patron.runtime.config import STATUSES, RuntimeConfig, load_runtime_config

__all__ = ["STATUSES", "RuntimeConfig", "load_runtime_config"]
