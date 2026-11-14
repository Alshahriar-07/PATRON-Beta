"""Feature schema constants shared by training and inference."""

from __future__ import annotations

FEATURE_VERSION = "features-1.0.0"
FEATURE_SCHEMA_VERSION = "feature-schema-1.0.0"

# Fixed order mandated by the ZIP specification. Never reorder: the scaler,
# saved checkpoints and inference inputs all depend on this exact order.
FEATURE_ORDER: tuple[str, ...] = (
    "open_rel",
    "high_rel",
    "low_rel",
    "close_rel",
    "volume_log_rel",
    "body_to_range",
    "atr14_pct",
    "rsi14_scaled",
    "ema9_gap",
    "ema21_gap",
    "volume_spike",
    "htf_trend",
)

FEATURE_COUNT = len(FEATURE_ORDER)
SEQUENCE_LENGTH = 96

#: Rolling lookback needed before the first 12-feature row is valid. The 96-bar
#: volume baseline dominates; ATR/RSI/EMA warm-ups are shorter.
FEATURE_WARMUP_BARS = 96
