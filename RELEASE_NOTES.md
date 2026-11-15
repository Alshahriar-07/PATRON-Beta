# PATRON v1.0.0-beta.1 — Release Notes

**Status: READY FOR PUBLIC TESTING.** Not production-validated. Not financial
advice and not a trading bot — it never places orders.

## Purpose

This is the first packaged, installable, testable release of PATRON, a
multi-task PyTorch model that forecasts the next six 15-minute BTC/USDT candles
from the previous 96: candlestick-pattern probabilities, triple-barrier
probabilities, per-step high/low/close quantiles, and a five-class momentum
distribution. The release makes the research reproducible end to end: data
pipeline, training, evaluation, inference, and the autonomous forecasting
runtime.

## What is in it

| Component | State |
|---|---|
| Frozen v2 architecture, 272,618 parameters, 96 × 12 input | implemented + tested |
| Dataset build, labels, purged chronological splits | implemented + tested |
| Training loop, checkpoints, JSONL histories | implemented + tested |
| Evaluation (metrics, calibration, baselines, regimes) | implemented + tested |
| Inference with JSON-Schema input/output contracts | implemented + tested |
| Autonomous runtime (ledger, maturation, drift, governance) | implemented + tested; live cycles run |
| `python -m patron` CLI and health check | implemented + tested |
| 162 automated tests | passing locally (see `reports/release/release_validation.md`) |
| CI / security / release workflows | committed; not yet run on GitHub |
| Universal integration guide (`INSTRUCTION.md`) | written against this release; every example executed |
| Training improvement plan (T2–T12) | designed, not run |

## Install

```bash
git clone <repository-url> && cd PATRON
python -m venv .venv
# Windows
./.venv/Scripts/pip install -r requirements.txt
./.venv/Scripts/pip install -e . --no-deps
./.venv/Scripts/python.exe -m patron health
# Linux / macOS
./.venv/bin/pip install -r requirements.txt
./.venv/bin/pip install -e . --no-deps
./.venv/bin/python -m patron health
```

Or run `scripts/install.sh` / `scripts/install.ps1`. Python 3.12; CPU is enough.

## Quick test

```bash
python -m patron health                 # expect PASS, or WARN without data/checkpoint
python -m pytest                       # 162 tests, synthetic fixtures, no network
python -m patron experiment list       # runs found under experiments/
```

A full walkthrough is in `QUICKSTART.md`. To integrate PATRON into your own application —
Python API, CLI, subprocess, runtime, output interpretation, error handling — read
`INSTRUCTION.md`.

## Compatibility

- Python 3.12 (3.11 not supported: the code targets 3.12 language features).
- `torch==2.4.1` CPU wheels; newer CPU builds fail on the reference Windows host.
- Input contract: 96 closed 15-minute candles, 12 features, oldest → newest.
- Output contract: `schemas/output.schema.json` (unchanged since v2).
- Old checkpoints load through the hardened loader; unknown keys are ignored,
  missing required keys raise an error.

## Checkpoint and dataset availability

Not shipped. This keeps the release small, avoids redistributing exchange data,
and prevents accidental use of stale weights. Both are reproducible from public
Binance data; see `DATASET_CARD.md` and `REPRODUCIBILITY.md`. The reference
training run used dataset sha256 `f0d4862984c110d2c98983b072b8c9ccb7c0703b0ba1728e3fa3732d012b4421`.

## Known issues

- Momentum head is at chance (accuracy 0.2382 test, log loss 1.5939 vs uniform
  1.6094). Quantile medians barely beat "no change" and the 10–90 band is too
  wide (90.3 % coverage for a nominal 80 %). Barrier accuracy at the shipped
  operating point (0.3813) is below the majority baseline (0.4692) while
  macro-F1 (0.3826) is far above it — a decision-rule issue analysed in
  `reports/training_upgrade/training_audit.md`.
- Runtime candidate promotion has not been exercised live (needs ~3 days of
  accumulated matured outcomes).
- CI workflows are untested on GitHub; the local equivalents pass.
- Windows + OneDrive paths can slow parquet writes; use a local path for datasets.
- macOS is untested (no hardware at hand); dependencies are pure-CPU wheels.

## Upgrade notes

Nothing to migrate: this is the first packaged release. If you used the
pre-release scripts, the only behaviour change is that `--stride` no longer
subsamples validation/test (use `--train-stride` for that) and checkpoints are
loaded with `weights_only=True` — pass `allow_unsafe=True` only for files you
produced yourself and trust.

## Verify the download

```bash
sha256sum -c checksums/SHA256SUMS.txt      # Linux/macOS
Get-FileHash <file> -Algorithm SHA256      # Windows PowerShell
```
