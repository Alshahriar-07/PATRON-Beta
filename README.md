# PATRON

Multi-task deep learning model for short-horizon BTC/USDT forecasting on 15-minute candles.
Given the last **96 closed 15m candles**, PATRON predicts the next **6 candles (90 minutes)**:

| Head | Output |
|---|---|
| Pattern | 12 rule-based candlestick/structure pattern probabilities |
| Barrier | P(TP first) / P(SL first) / P(timeout) for the next 90 min |
| Range | q10 / q50 / q90 of high, low and close, relative to the current close, for each of the 6 future candles |
| Momentum | 5-class distribution: strong bearish → strong bullish |

The frozen input/output contracts are in [`schemas/`](schemas/); the training audit and the
advanced-training roadmap are in [`reports/training_upgrade/training_audit.md`](reports/training_upgrade/training_audit.md)
and [`plan/ADVANCED_TRAINING_UPGRADE.md`](plan/ADVANCED_TRAINING_UPGRADE.md), and the build log
is [`plan/memory.md`](plan/memory.md). Implemented with PyTorch on CPU; no GPU required.

## Status

| Phase | Scope | State |
|---|---|---|
| 0 | Setup, config, logging | complete |
| 1 | Data download + validation | complete |
| 2 | Features + scaler + patterns | complete |
| 3 | Labels (barrier / range / momentum) | complete |
| 4 | Dataset, splits, baselines | complete |
| 5 | Model + losses | complete |
| 6 | Training loop | complete |
| 7 | Evaluation + calibration | complete |
| 8 | Inference CLI | complete |
| 9 | Test suite (162 tests) | complete |
| U1–U10 | Scale-up: 6 venues, 10 assets, 103-pattern registry, walk-forward, multi-seed | complete |
| A1–A8 | Autonomous runtime: candle-by-candle service, forecast ledger, maturation, drift, retraining, champion/challenger | complete (live cycles run; candidate retraining gated, see Limits) |
| T1 | Training pipeline audit (read-only diagnostics) | complete — `reports/training_upgrade/training_audit.md` |
| T2–T12 | Advanced training upgrade (optimization, loss balancing, hard examples, regimes, patterns, error intelligence, continual learning) | designed, not run — `plan/ADVANCED_TRAINING_UPGRADE.md` |
| R1 | Packaging, CLI, health check, docs, CI, release build | complete — `v1.0.0-beta.1` (see [`RELEASE_NOTES.md`](RELEASE_NOTES.md)) |

## Layout

```
config/model_config.yaml     all hyperparameters, feature order, thresholds (spec + additive data block)
src/patron/
  data/        downloader (Binance archives + REST), validation, HTF resampling,
               markets/venues/collect/quality/align/timeframes (6 venues, 1m–1d)
  features/    12 causal features, indicators, pattern rules, robust scaler,
               advanced.py (34 grouped causal features, ablation-ready)
  patterns/    detectors, 103-pattern registry, 16 chart structures, dataset, analysis
  labels/      triple barrier, forward range targets, momentum bins
  datasets/    chronological splits with purge/embargo, gap-aware windowing
  models/      PatronNet, multi-task losses
  training/    training loop, checkpoints, JSONL logs
  evaluation/  metrics, temperature calibration, baselines, report, regimes
  inference/   input validation, warm-up handling, schema-checked output
src/patron/cli.py    `python -m patron` command line;  health.py = PASS/WARN/FAIL checks
src/patron/schemas/  package copy of the JSON contracts (kept identical by test)
scripts/       download_data, build_dataset, train, evaluate, predict,
               collect_market_data, build_data_report, build_pattern_dataset,
               run_feature_ablation, run_walk_forward, train_multiseed,
               build_final_comparison, audit_diagnostics, install.sh/.ps1,
               run_training.*, run_inference.*, health_check, scan_release,
               build_release
examples/      runnable inference / evaluation / training examples + sample payloads
.github/       CI, security and release workflows, issue and PR templates
tests/         pytest suite (162 tests)
data/          downloaded + cleaned candles, feature dataset, manifests
runtime/       autonomous service (see below): store, collector, engine, maturation,
               monitoring, drift, learning, jobs, governance, publisher, service
config/runtime.yaml   operational policy (cadence, thresholds, gates) — separate from the model
schemas/forecast.schema.json   versioned live-forecast envelope (v2)
experiments/   per-run config snapshot, checkpoints, JSONL metrics, candidate runs
reports/       data-quality, label distribution, evaluation, predictions,
               training_upgrade/ (T1 audit + machine-readable evidence)
reports/live/  published runtime state: latest forecast, status, performance, models, drift,
               training, health, forecast_history.jsonl
data/runtime/  SQLite state (candles, forecasts, evaluations, registry, jobs, events) +
               immutable per-forecast JSON ledger
plan/memory.md build log (decisions, commands, results, next steps)
```

## Setup

```bash
python -m venv .venv
./.venv/Scripts/pip install -r requirements.txt
./.venv/Scripts/python.exe -m pip install -e . --no-deps
```

Python 3.12. `torch==2.4.1` (CPU) is pinned: newer CPU wheels fail on this
Windows host with `WinError 126 ... shm.dll`. Installers: `scripts/install.sh`
(Linux/macOS) and `scripts/install.ps1` (Windows). Full guide: [`INSTALLATION.md`](INSTALLATION.md).

## Command line

Everything is available as subcommands of one entry point:

```bash
python -m patron health                    # PASS / WARN / FAIL report
python -m patron health --json             # machine-readable, no secrets
python -m patron train --config config/model_config.yaml
python -m patron evaluate --checkpoint PATH
python -m patron predict --checkpoint PATH --input JSON [--warmup PARQUET]
python -m patron runtime status|once|serve|mature|rollback
python -m patron experiment list|compare --runs A B
```

Exit codes: `0` success, `1` check/runtime failure, `2` missing input,
`3` invalid input, `4` model produced no usable output. Every subcommand has
`--help`. `health` checks Python, dependencies, the PyTorch runtime, config,
schemas, the 272,618-parameter contract, scaler feature order, dataset,
checkpoint (safe-load), predictor init, the runtime database (opened read-only)
and write access; it never modifies project data.

A five-minute walkthrough is in [`QUICKSTART.md`](QUICKSTART.md); every setting is
documented in [`CONFIGURATION.md`](CONFIGURATION.md).

## Pipeline

```bash
# 1. download 15m/1h/4h BTC/USDT candles from 2021-11-01, validate, cross-check HTF
./.venv/Scripts/python.exe scripts/download_data.py --start 2021-11-01

# 2. features + labels + scaler -> data/features/BTCUSDT_dataset.parquet
./.venv/Scripts/python.exe scripts/build_dataset.py

# 3. train (CPU: ~4 min/epoch, AdamW 3e-4, ReduceLROnPlateau, early stop patience 12)
./.venv/Scripts/python.exe scripts/train.py --run-name patron_v2

# 4. evaluate the best checkpoint on the held-out test split
./.venv/Scripts/python.exe scripts/evaluate.py \
    --checkpoint experiments/patron_v2_*/best.pt

# 5. inference from 96 closed candles
./.venv/Scripts/python.exe scripts/predict.py \
    --input data/clean/BTCUSDT/latest_input.json \
    --checkpoint experiments/patron_v2_*/best.pt
```

Every script is idempotent and logs to `logs/` plus `experiments/<run>/metrics.jsonl`.

### Inference input / output

Input must match [`schemas/input.schema.json`](schemas/input.schema.json):
exactly 96 candles, 15 minutes apart, strictly ascending, all closed, consistent OHLC.
The model also needs 384 prior 15m candles for ATR/EMA/4h warm-up; these are read from
`data/clean/BTCUSDT/15m.parquet` (or `--warmup <parquet>`). Without enough warm-up
history the API returns `status: "NO_OUTPUT"` with a warning instead of silently
cold-starting features.

Output conforms to [`schemas/output.schema.json`](schemas/output.schema.json)
and is validated before it is written:

```json
{
  "status": "OK",
  "symbol": "BTC/USDT",
  "as_of": "2026-11-14T12:45:00+00:00",
  "horizon_minutes": 90,
  "model_version": "...",
  "pattern_scores": { "bullish_engulfing": 0.12, "...": 0.0 },
  "barrier_probabilities": { "tp_first": 0.31, "sl_first": 0.55, "timeout": 0.14 },
  "range_quantiles": [
    { "step": 1, "minutes_ahead": 15,
      "high":  { "q10": -0.001, "q50": 0.004, "q90": 0.009 },
      "low":   { "q10": -0.008, "q50": -0.004, "q90": 0.001 },
      "close": { "q10": -0.005, "q50": 0.0004, "q90": 0.006 } }
  ],
  "momentum": { "class_probabilities": { "...": 0.2 }, "predicted_class": "sideways" },
  "data_quality": { "valid": true, "closed_candles": 96, "sequence_length": 96, "warnings": [] }
}
```

## Data and label rules

- **Source:** Binance spot monthly archives (`data.binance.vision`) plus REST `/api/v3/klines`
  for the open month. No API key; one gap of 5 bars (2023-03) exists and is flagged,
  never filled.
- **Features (12):** log return, candle body/range, wick ratios, ATR%, RSI, EMA9/EMA21 gaps,
  volume z-score, 1h and 4h trend. All ratios — no absolute prices — and strictly causal.
- **Triple barrier:** TP = 1.5 × ATR14, SL = 1.0 × ATR14, horizon 6 bars. First touch wins;
  a candle that touches both is marked ambiguous and masked out of the primary loss.
- **Range targets:** each future candle's high/low/close relative to the window's last close.
- **Momentum:** 6-bar return divided by ATR% and binned at ±0.35 and ±1.0.
- **Splits:** chronological 70/15/15 with a 6-bar purge and 6-bar embargo; the test split
  is evaluated once.

## Scale-up: multi-venue, multi-asset, pattern research

Dataset version **2.0.0** is additive; the shipped model still trains on dataset 1.0.0.

```bash
# 6 public venues, BTC 15m, 2023-11 → now (per-venue parquet + checksums + manifest)
./.venv/Scripts/python.exe scripts/collect_market_data.py --assets BTC --start 2023-11-01
./.venv/Scripts/python.exe scripts/build_data_report.py

# 10 assets on Binance + Bybit (context only, never labelled with BTC's future)
./.venv/Scripts/python.exe scripts/collect_market_data.py --all-assets --start 2023-11-01

# 103 candlestick patterns + 16 chart structures → occurrences and outcomes
./.venv/Scripts/python.exe scripts/build_pattern_dataset.py

# ablation, walk-forward, multi-seed, final comparison
./.venv/Scripts/python.exe scripts/run_feature_ablation.py
./.venv/Scripts/python.exe scripts/run_walk_forward.py
./.venv/Scripts/python.exe scripts/train_multiseed.py --seeds 42 7 2026 --epochs 6 --stride 2
./.venv/Scripts/python.exe scripts/build_final_comparison.py
```

Coverage limits are recorded, not patched: Kraken's OHLC endpoint caps at 720 bars
(~7.5 days), Bitget's v2 spot candles only reach ~10 days back, and Coinbase/Kraken
quote BTC in USD (compared by returns only). Binance, Bybit and OKX serve the full
2023-11 → present window. Report: `reports/data_expansion_report.md`.

The pattern registry (`src/patron/patterns/registry.py`) stores a formula, executed
conditions, context requirement, base confidence and literature reference for every
pattern; the occurrence dataset keeps outcomes at 1/2/3/6/12/24 candles separately from
the input features. Report: `reports/pattern_research_report.md`.

**Negative result, stated plainly:** the 34 advanced features did not beat the original
12 features in the tabular proxy ablation (`reports/feature_ablation_report.md`), so they
are not used by the shipped model. Full comparison: `reports/final_model_comparison.md`.

## Autonomous runtime

After setup the system runs itself: every newly closed 15-minute candle triggers one
idempotent cycle, a forecast is written to an append-only ledger, and every horizon is
evaluated the moment its own candle matures.

```bash
# one cycle now (fetch → validate → mature → metrics → window → forecast → publish)
./.venv/Scripts/python.exe scripts/patron_service.py once

# continuous service (sleeps until the next candle close + confirmation delay)
./.venv/Scripts/python.exe scripts/patron_service.py serve

# inspect published state / health / pending maturation / model rollback
./.venv/Scripts/python.exe scripts/patron_service.py status
./.venv/Scripts/python.exe scripts/patron_service.py health
./.venv/Scripts/python.exe scripts/patron_service.py mature
./.venv/Scripts/python.exe scripts/patron_service.py rollback
```

Lifecycle, in order, each stage logged: fetch and confirm → validate (dedupe, drop
forming bars, OHLC sanity) → mature pending horizons → recompute rolling metrics →
rebuild the 96-candle window from durable state → run the active champion → persist →
publish. A cycle that ran for a candle is keyed in the `cycles` table, and the forecast
table has `UNIQUE(symbol, timeframe, candle_ts)`, so re-running the same candle returns
the existing forecast instead of creating a second one. Restarts rebuild the window and
champion from the database: there is no volatile state to lose.

**Statuses** are explicit and never cosmetic: `OK`, `NO_OUTPUT`, `INVALID_INPUT`,
`INSUFFICIENT_HISTORY`, `STALE_DATA`, `DATA_GAP`, `MODEL_UNAVAILABLE`, `RECOVERING`,
plus `PROVISIONAL` reserved for the separate forming-candle view. A failed fetch, a gap
inside the 96-candle window, or a champion whose file hash no longer matches all produce
a non-`OK` cycle and no forecast. `latest_forecast.json` is a pointer, not the ledger:
it carries `stale`, `current_status`, `freshness.age_seconds`, and per-horizon statuses.

**Forecast ledger.** Each forecast is stored twice (SQLite row + immutable
`data/runtime/forecasts/<id>.json`) with its input hash, checkpoint hash, feature and
calibration versions, source provenance, six exact target timestamps, a feature snapshot
and the untouched model output. Maturation results are linked rows keyed
`(forecast_id, step)`; the original prediction is never rewritten.

**Maturation** is per horizon: a 10:00 forecast is scored at 10:15 for step 1, at 10:30
for step 2, and so on. Metrics use the training target space (relative returns versus the
close at the origin): absolute/percentage error, pinball loss for q10/q50/q90, quantile
coverage, directional correctness, and — only at step 6 — the momentum class and the
triple-barrier outcome. Records say `maturity: immature` until those labels can exist.

**Monitoring and drift.** Performance is computed only from matured evaluations, per
step and per head, with sample counts. Drift compares live feature snapshots and matured
momentum outcomes against the distribution stored with the champion (z-score + PSI,
minimum samples). Every trigger decision is logged with its evidence in
`reports/live/training.json` and the `events` table.

**Controlled continuous learning.** Candidates are trained only from verified, closed
candles with fully matured labels (the store never holds a forming bar), blended into the
historical replay set with the newest-data share capped by `learning.recent_fraction`.
Training runs in a separate process (`scripts/train.py`), one job at a time, and live
inference keeps using the champion. A candidate is promoted only if it passes the gate
list configured before evaluation: checkpoint loads, output is schema-valid with
probabilities summing to 1 and ordered quantiles, enough validation windows, validation
log loss and barrier accuracy not worse than the champion, and q10-q90 coverage within
bounds. The final test split is never used for selection. Rejected candidates keep their
`promotion_decision.json` and reasons; a promoted candidate can be rolled back to the
previous champion.

Defaults live in `config/runtime.yaml` (confirmation delay, poll cadence, stale
threshold, drift/drift thresholds, trigger minimums, cooldown, gate tolerances). The
service ran live on 2026-11-15: a fetch failure produced `STALE_DATA` with no forecast,
the retry fix recovered on the next attempt, and a real cycle published an `OK` forecast
for the 10:45 candle that was then scored at 11:00 — all documented in
`reports/runtime_operations_report.md`.

## Results

Final run `patron_v2_20261114-203859` (best epoch 3, val loss 1.4048; 5 years of
15m BTC/USDT, test split 2026-02-12 → 2026-11-14, 26,269 windows, evaluated once):

| Head | Metric (test) | PATRON | Baseline |
|---|---|---|---|
| Barrier | accuracy | 0.381 | 0.469 (majority `sl_first`) |
| Barrier | balanced accuracy | 0.409 | 0.333 (majority) |
| Barrier | macro-F1 | 0.383 | 0.213 (majority), 0.239 (logistic) |
| Barrier | ROC-AUC (OvR) | 0.594 | — |
| Barrier | ECE (calibrated) | 0.028 | — |
| Momentum | accuracy | 0.238 | 0.208 (majority), 0.258 (logistic) |
| Momentum | macro-F1 | 0.224 | 0.069 (majority) |
| Range | pinball (mean) | 0.00097 | constant-zero 0.00128 |
| Range | median MAE (close) | 0.00262 | persistence 0.00254 |
| Pattern | mAP | 0.086 | base-rate reference ≈ 0.065 |

Honest reading (full evidence in [`reports/training_upgrade/training_audit.md`](reports/training_upgrade/training_audit.md)):

- The barrier head carries real but weak ranking signal — ROC-AUC 0.594 and balanced
  accuracy 0.409 — while plain accuracy stays below the majority baseline. The cause is
  measured: inverse-frequency class weights shift the decision boundary so the model
  predicts `tp_first` on ~38 % of windows against a 26 % base rate. On validation, a
  prior-corrected argmax on the *same logits* reaches 0.486 accuracy vs 0.473 majority.
  This is an operating-point choice, not a missing-signal problem, and the release does
  not retrain to hide it.
- The momentum head is near-uninformative: log loss 1.594 against 1.609 for a uniform
  prior, accuracy 0.238, macro-F1 0.224.
- Quantiles never cross (0 violations) and pinball beats a constant-zero predictor, but the
  10–90 band covers ~90 % of outcomes instead of 80 % and the predicted close median has a
  regression slope of −0.013 against the realised close — it estimates dispersion, not
  direction. Median MAE is slightly worse than “no change”.
- These are research baselines, not a trading edge, and no post-audit improvement has been
  demonstrated yet (T2–T12 are designed, not run).

## Tests

```bash
./.venv/Scripts/python.exe -m pytest -q
```

162 tests cover data validation (gaps, duplicates, invalid OHLC, unclosed candles),
feature causality for the 12 v1 features and the advanced groups, label horizon and
leakage, gap-aware windows and purged splits, model shapes / 272,618-param budget /
non-crossing quantiles, masked losses, the inference contract including `NO_OUTPUT`,
venue symbol mapping and 15m-grid alignment, multi-timeframe resampling (partial buckets
dropped, no look-ahead), the 100+ pattern registry (causality by truncation), and the
pattern-outcome dataset (outcomes stored separately from inputs), train-only window
subsampling that leaves validation/test intact, and the autonomous
runtime: candle confirmation, one forecast per close, per-horizon maturation timing,
idempotent retries and restarts, gap/stale refusal, schema-valid envelopes, drift and
retraining triggers, and governed promotion / rejection / rollback. The release adds
coverage for the health check (statuses, JSON shape, read-only database access),
the CLI (help, version, exit codes, dry-run training), safe checkpoint loading
(pickle refusal, missing keys), the release scanner (planted secrets, private
paths, oversized artifacts), frozen contracts (272,618 params, schema copies,
version alignment), and a one-epoch training smoke test on synthetic candles.

## Limitations

- Research model: no live trading, no order execution, no exchange API keys.
- Trained on 15m BTC/USDT only; other symbols/timeframes need retraining.
- CPU-trained on 5 years of data; accuracy is modest by design (the target is a
  calibrated distribution, not a point forecast).
- The expanded multi-venue data is not used by the shipped model: no cross-venue or
  advanced-feature variant demonstrated better validation results, so none was adopted.
- Walk-forward is a rolling evaluation of the fixed checkpoint, not per-fold retraining;
  the multi-seed study uses a short budget (4 epochs, stride 4, seeds 42/7/2026) and
  reports validation dispersion only: val loss 1.4122 ± 0.0007, barrier acc 0.3180 ± 0.0039.
- Probabilities are temperature-calibrated on the validation split and reported with
  ECE/Brier in `reports/evaluation_report.md`; they are not a guarantee of future returns.
- CI workflows are committed but have not executed on GitHub; the same commands pass
  locally. No linter or formatter is enforced (syntax check + tests only), and dependency
  CVE auditing is a documented manual step rather than a CI gate.
- macOS is untested; Linux is expected to work but Windows is the reference environment.
- The autonomous retraining path was verified with test doubles (promotion, rejection,
  crash isolation, rollback) but **no live candidate retrain finished on this host**: the
  runtime requires 288 matured horizons (~3 days of continuous operation) before the
  evidence trigger fires, and a real 1-epoch run on this CPU is minutes-per-epoch. The
  machinery, its gates and its refusal to promote are tested; a live promotion is not
  claimed.
- Only two live forecast cycles had been executed at the time of writing, so the live
  performance files contain one matured horizon — far below the configured minimum for
  degradation decisions. Those numbers are startup evidence, not an evaluation.
- The runtime is a single-machine worker loop (SQLite + subprocesses), not a distributed
  service: no queue broker, no horizontal scaling, no dashboard. The published JSON files
  are the API surface.
- `serve` sleeps until the next candle close; if the host is suspended across several
  closes it processes the newest closed candle only (older ones are backfilled as candles
  for maturation, not as new forecasts, because a forecast would be stale on arrival).

## Documentation

| Document | Contents |
|---|---|
| [`INSTALLATION.md`](INSTALLATION.md) | Windows / Linux / macOS setup and troubleshooting |
| [`QUICKSTART.md`](QUICKSTART.md) | five-minute path from clone to first verify |
| [`INSTRUCTION.md`](INSTRUCTION.md) | universal integration guide: API, CLI, subprocess, runtime, examples |
| [`CONFIGURATION.md`](CONFIGURATION.md) | every key in both YAML files |
| [`MODEL_CARD.md`](MODEL_CARD.md) | architecture, intended use, measured metrics |
| [`DATASET_CARD.md`](DATASET_CARD.md) | data source, construction, splits, licensing |
| [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) | environment, seeds, determinism caveats |
| [`LIMITATIONS.md`](LIMITATIONS.md) | the complete honest list |
| [`SECURITY.md`](SECURITY.md) | reporting, scope, hardening |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | workflow and ground rules |
| [`CHANGELOG.md`](CHANGELOG.md) · [`RELEASE_NOTES.md`](RELEASE_NOTES.md) | what changed and what is in this release |
| [`plan/memory.md`](plan/memory.md) | build log with decisions and failures |

## Security

No credentials are needed or stored; the project reads public market data and
writes local files only. Checkpoints are loaded with `weights_only=True`, inference
inputs are JSON-Schema validated, and runtime SQL is parameterised. Report issues
privately per [`SECURITY.md`](SECURITY.md) — never in a public issue. The audit is
in `reports/security/security_audit.md`.

## License

Apache License 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). Third-party
components keep their own licenses ([`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)).
Market data is not distributed.

## Disclaimer

PATRON is experimental research software. Predictions are probabilistic estimates
from incomplete data; historical metrics do not guarantee future results; the
software is not financial advice, does not execute orders, and offers no profit
guarantee. You are responsible for your own decisions and for complying with the
terms of any data provider you use.

## Notes

- The original Bengali specification is a local planning bundle and is not part of this
  repository or its releases; `schemas/` is the authoritative contract.
- The model config's `project.version` (`1.0.0`) is the model version; the Python package
  release is `1.0.0b1` (`v1.0.0-beta.1`).
- `plan/memory.md` records every step, command, test result, error and technical decision.
