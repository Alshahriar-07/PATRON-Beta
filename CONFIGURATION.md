# Configuration

Two YAML files matter, and neither is read from environment variables.

| File | Controls | Change frequency |
|---|---|---|
| `config/model_config.yaml` | data contract, model architecture, training hyperparameters, validation protocol | rarely; a change requires a new model version |
| `config/runtime.yaml` | autonomous-runtime policy: cadence, thresholds, drift, retraining, promotion gates | operational |

## `config/model_config.yaml`

### `project`

Name and `version` (currently `1.0.0`). This is the **model** version, distinct
from the Python package version (`1.0.0b1`). Checkpoints record it.

### `market`

`symbol: BTC/USDT`, `timeframe: 15m`, `context_timeframes: [1h, 4h]`,
`only_closed_candles: true`. Other venues/assets are context-only; the supervised
target is BTC/USDT.

### `input`

`sequence_length: 96`, `feature_count: 12`, `order: oldest_to_newest`, and the
explicit `feature_order` list. The scaler validates this order on load; reordering
features without rebuilding the dataset will fail loudly (by design).

### `forecast`

| Key | Value | Meaning |
|---|---|---|
| `horizon_candles` | 6 | forecast 90 minutes ahead |
| `barrier.tp_atr_multiplier` | 1.5 | upper barrier = close + 1.5 × ATR14 |
| `barrier.sl_atr_multiplier` | 1.0 | lower barrier = close − 1.0 × ATR14 |
| `barrier.ambiguous_same_candle` | `exclude_from_primary_loss` | bars touching both sides are masked |
| `range.quantiles` | [0.1, 0.5, 0.9] | per-field, per-step quantiles |
| `range.target_space` | `relative_to_last_close` | targets are relative returns |
| `momentum.classes` | 5 classes | strong bearish … strong bullish |
| `momentum.normalized_return_boundaries` | [-1.0, -0.35, 0.35, 1.0] | ATR-normalised 6-bar return bins |
| `pattern_tags` | 12 tags | rule-based binary pattern targets |

These are contract values. Changing them invalidates checkpoints and the frozen
inference schema.

### `model`

Architecture hyperparameters: projection 32, CNN `[32, 64, 64]` with kernels
`[3, 5, 7]`, BiLSTM 64 per direction, 4 attention heads at d=128, shared dim 64,
dropout 0.15, parameter budget `[200000, 1000000]` (actual 272,618). There is no
switch to change the architecture; that is a new model, not a configuration.

### `training`

`optimizer: AdamW`, `learning_rate: 0.0003`, `weight_decay: 0.0001`,
`batch_size: 128`, `max_epochs: 100`, `early_stopping_patience: 12`,
`gradient_clip_norm: 1.0`, `random_seed: 42`, and per-head `loss_weights`
(barrier 1.0, quantile 0.5, momentum 0.2, pattern 0.15).

These are the **champion's** recipe. Experiments for the training upgrade use
separate configs; only a promoted candidate may change this file (see
`plan/ADVANCED_TRAINING_UPGRADE.md` §0 and §7).

CLI overrides exist for the common knobs: `--epochs`, `--batch-size`,
`--train-stride`, `--threads`, `--seed`, `--run-name`.

### `validation`

`method: walk_forward`, `purge_bars: 6`, `embargo_bars: 6`,
`final_test_untouched: true`, `calibrator: temperature_scaling`,
`baseline_models: [majority, momentum_rule, mean_reversion, lightgbm]`.

### `runtime` (inside the model config)

Inference-only flags: `inference_batch_size: 1`, `reject_nan_inf`,
`reject_unclosed_candle`, `reject_stale_data`, `live_order_execution: false`.

### `data` (additive block)

Download parameters: archive base URL, REST base URL, start date, timeframes,
request pause, overwrite flag. `end: null` means "latest available".

## `config/runtime.yaml`

Operational policy for `python -m patron runtime`. Actual sections:

| Section | Controls |
|---|---|
| `source` | venue, market type, confirmation delay after candle close, lookback hours, request pause |
| `history` | model sequence length (96) and indicator warm-up bars (384) |
| `storage` | SQLite database, published JSON directory, immutable forecast ledger |
| `monitoring` | rolling window size, minimum samples, coverage bounds, stale threshold, degradation floor |
| `drift` | feature z-score and PSI thresholds, minimum drifting features, outcome PSI |
| `learning` | evidence minimums (matured examples), schedule, cooldown, replay dataset, recent-data cap, epochs/stride/threads, time limit, optional custom command |
| `governance` | promotion gates: validation samples, log-loss delta, accuracy drop, coverage bounds, seeds, mandatory gate list |
| `server` | poll interval, heartbeat, cycle cap, shutdown timeout |

Thresholds here change operations, never model semantics. The file is validated on
load; unknown keys raise an error instead of being silently ignored. `symbol` and
`timeframe` must agree with the model config, and `history.sequence_length` must
be 96.

## Precedence and paths

1. Explicit CLI flags win.
2. YAML values.
3. Built-in defaults.

Commands resolve `config/…` and `schemas/…` relative to the repository root, so run
them from the project directory (or pass absolute paths). Checkpoints embed their
own config and scaler; `--config` on `evaluate` exists to override the architecture
when comparing an older run.

## What is *not* configurable (and why)

- Feature definitions and order (frozen contract; version bump required).
- Label rules (triple barrier, momentum bins).
- Output schema (`schemas/output.schema.json`).
- Device selection: CPU only in this release.
- Exchange credentials: none are used; the project does not place orders.
