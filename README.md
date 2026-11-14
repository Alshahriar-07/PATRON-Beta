# PATRON

Multi-task deep learning model for short-horizon BTC/USDT forecasting on 15-minute candles.
Given the last **96 closed 15m candles**, PATRON predicts the next **6 candles (90 minutes)**:

| Head | Output |
|---|---|
| Pattern | 12 rule-based candlestick/structure pattern probabilities |
| Barrier | P(TP first) / P(SL first) / P(timeout) for the next 90 min |
| Range | q10 / q50 / q90 of high, low and close, relative to the current close, for each of the 6 future candles |
| Momentum | 5-class distribution: strong bearish → strong bullish |

The full specification lives in [`Plan/`](Plan/) (architecture, contract, data/feature/label
rules, roadmap). Implemented with PyTorch on CPU; no GPU required.

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
| 9 | Test suite (44 tests) | complete |

## Layout

```
config/model_config.yaml     all hyperparameters, feature order, thresholds (spec + additive data block)
src/patron/
  data/        downloader (Binance archives + REST), validation, HTF resampling
  features/    12 causal features, indicators, pattern rules, robust scaler
  labels/      triple barrier, forward range targets, momentum bins
  datasets/    chronological splits with purge/embargo, gap-aware windowing
  models/      PatronNet, multi-task losses
  training/    training loop, checkpoints, JSONL logs
  evaluation/  metrics, temperature calibration, baselines, report
  inference/   input validation, warm-up handling, schema-checked output
scripts/       download_data.py, build_dataset.py, train.py, evaluate.py, predict.py
tests/         pytest suite
data/          downloaded + cleaned candles, feature dataset, manifests
experiments/   per-run config snapshot, checkpoints, JSONL metrics
reports/       data-quality, label distribution, evaluation, predictions
plan/memory.md build log (decisions, commands, results, next steps)
```

## Setup

```bash
python -m venv .venv
./.venv/Scripts/pip install -r requirements.txt
./.venv/Scripts/python.exe -m pip install -e . --no-deps
```

Python 3.12. `torch==2.4.1` (CPU) is pinned: newer CPU wheels fail on this
Windows host with `WinError 126 ... shm.dll`.

## Pipeline

```bash
# 1. download 15m/1h/4h BTC/USDT candles from 2021-11-01, validate, cross-check HTF
./.venv/Scripts/python.exe scripts/download_data.py --start 2021-11-01

# 2. features + labels + scaler -> data/features/BTCUSDT_dataset.parquet
./.venv/Scripts/python.exe scripts/build_dataset.py

# 3. train (CPU: ~4 min/epoch, early stopping patience 12)
./.venv/Scripts/python.exe scripts/train.py --run-name patron_v1

# 4. evaluate the best checkpoint on the held-out test split
./.venv/Scripts/python.exe scripts/evaluate.py \
    --checkpoint experiments/patron_v1_*/best.pt

# 5. inference from 96 closed candles
./.venv/Scripts/python.exe scripts/predict.py \
    --input data/clean/BTCUSDT/latest_input.json \
    --checkpoint experiments/patron_v1_*/best.pt
```

Every script is idempotent and logs to `logs/` plus `experiments/<run>/metrics.jsonl`.

### Inference input / output

Input must match [`Plan/schemas/input.schema.json`](Plan/schemas/input.schema.json):
exactly 96 candles, 15 minutes apart, strictly ascending, all closed, consistent OHLC.
The model also needs 384 prior 15m candles for ATR/EMA/4h warm-up; these are read from
`data/clean/BTCUSDT/15m.parquet` (or `--warmup <parquet>`). Without enough warm-up
history the API returns `status: "NO_OUTPUT"` with a warning instead of silently
cold-starting features.

Output conforms to [`Plan/schemas/output.schema.json`](Plan/schemas/output.schema.json)
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

## Tests

```bash
./.venv/Scripts/python.exe -m pytest -q
```

44 tests cover data validation (gaps, duplicates, invalid OHLC, unclosed candles),
feature causality (future candles cannot change past features), label horizon and
leakage, gap-aware windows and purged splits, model shapes / 272,618-param budget /
non-crossing quantiles, masked losses, and the inference contract including `NO_OUTPUT`.

## Limitations

- Research model: no live trading, no order execution, no exchange API keys.
- Trained on 15m BTC/USDT only; other symbols/timeframes need retraining.
- CPU-trained on 5 years of data; accuracy is modest by design (the target is a
  calibrated distribution, not a point forecast).
- Probabilities are temperature-calibrated on the validation split and reported with
  ECE/Brier in `reports/evaluation_report.md`; they are not a guarantee of future returns.

## Notes

- `Plan/README_BN.md` contains the original Bengali specification.
- `plan/memory.md` records every step, command, test result, error and technical decision.
