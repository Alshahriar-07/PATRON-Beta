"""Candlestick pattern registry, chart structures and the pattern-outcome dataset."""

from patron.patterns.detectors import CandleContext, PatternThresholds, build_context
from patron.patterns.registry import (
    CANDLESTICK_PATTERNS,
    PATTERN_BY_KEY,
    PatternDefinition,
    detect_all,
    detect_pattern,
    registry_summary,
)
from patron.patterns.structures import STRUCTURE_DEFINITIONS, all_definitions

__all__ = [
    "CandleContext",
    "PatternThresholds",
    "build_context",
    "CANDLESTICK_PATTERNS",
    "STRUCTURE_DEFINITIONS",
    "PATTERN_BY_KEY",
    "PatternDefinition",
    "all_definitions",
    "detect_all",
    "detect_pattern",
    "registry_summary",
]
