# Model Card — PATRON v2 (BTC/USDT 15m)

## Overview

| Field | Value |
|---|---|
| Model | PATRON v2 (`cnn_bilstm_self_attention`) |
| Version | model `1.0.0`, release `1.0.0-beta.1` |
| Task | short-horizon probabilistic forecasting of the next six 15-minute candles |
| Market | BTC/USDT spot, Binance |
| Parameters | 272,618 trainable (frozen for this release) |
| Input | `(96, 12)` float32, oldest → newest, 12 causal features |
| Outputs | 4 heads (below) |
| Framework | PyTorch 2.4.1 CPU |
| License | Apache-2.0 (weights produced by this code inherit it) |
| Contact | project issue tracker; security per `SECURITY.md` |

## Architecture

Input projection (12→32) → three residual Conv1D blocks (32/k3, 64/k5, 64/k7,
BatchNorm, GELU, dropout 0.15) → BiLSTM (64 per direction) → 4-head
self-attention (d=128) with residual → FFN (128→128→64) with residual →
attention pooling → shared 64-d vector → four linear heads. The quantile head
parameterises `q50` directly and `q10 = q50 − softplus(a)`,
`q90 = q50 + softplus(b)`, so quantiles cannot cross (0 violations on 26,269 test
windows).

## Outputs

| Head | Shape | Meaning |
|---|---|---|
| Pattern | 12 logits | rule-based candlestick/structure tags at the window's last bar |
| Barrier | 3 logits | P(TP first) / P(SL first) / P(timeout) over the next 6 candles; TP = +1.5·ATR14, SL = −1.0·ATR14 |
| Range | 6 × 3 × 3 | q10/q50/q90 of high, low, close for each future candle, relative to the current close |
| Momentum | 5 logits | 6-candle ATR-normalised return bins: strong bearish … strong bullish |

## Intended use

- Research on short-horizon probabilistic forecasting and calibration.
- A reference implementation for multi-task sequence models on financial data.
- Local experimentation, teaching, and benchmarking against baselines.

## Out-of-scope use

- Live trading or automated order execution (not implemented; no exchange
  credentials are used).
- Investment advice, risk management, or portfolio construction.
- Any market other than BTC/USDT spot 15m without retraining and re-validation.
- Any horizon other than 6 bars / 90 minutes.
- Safety-critical decisions of any kind.

## Training data

BTC/USDT spot 15m from Binance public archives, 2021-11-01 → 2026-11-14
(176,591 rows; 122,608 training windows). Features: 12 causal indicators.
Labels: triple barrier, forward high/low/close, momentum bins. Chronological
70/15/15 split with 6-bar purge and 6-bar embargo; scaler fit on the training split
only. Details and hashes: `DATASET_CARD.md`.

## Evaluation

Held-out test split, n = 26,269 windows (2026-02-12 → 2026-11-14), checkpoint
`patron_v2_20261114-203859`. Reference values, not a claim of current or future
performance:

| Head | Metric | Test |
|---|---|---|
| Barrier | accuracy | 0.3813 |
| Barrier | balanced accuracy | 0.4093 |
| Barrier | macro-F1 | 0.3826 |
| Barrier | log loss | 1.0592 |
| Barrier | ROC-AUC (OvR) | 0.5937 |
| Barrier | ECE | 0.0287 |
| Momentum | accuracy | 0.2382 |
| Momentum | macro-F1 | 0.2244 |
| Momentum | log loss | 1.5939 |
| Quantile | mean pinball | 0.000974 |
| Quantile | 10–90 coverage | 0.903–0.907 (nominal 0.80) |
| Pattern | mAP | 0.0864 |

Baselines on the same split: majority-class barrier accuracy 0.4692; one-bar
logistic regression 0.4708 accuracy / 0.2389 macro-F1; persistence close MAE
0.002541 vs model median MAE 0.002622.

## Honest reading of those numbers

- The barrier head ranks classes better than chance (AUC 0.594) and beats the
  majority baseline on macro-F1 and balanced accuracy, but at the shipped
  class-weighted operating point it predicts fewer than half the windows
  correctly. A validation-fitted prior correction lifts accuracy above the
  majority baseline without retraining — the decision rule, not the
  representation, is the limiting factor.
- The momentum head is effectively at chance: 0.2382 accuracy vs 0.20 for five
  balanced classes, log loss within 0.016 of uniform.
- The quantile head predicts an over-wide, nearly unconditional band: the
  regression slope of the true close on the predicted median is −0.013, and 90 %
  of outcomes fall inside the 10–90 interval instead of 80 %.
- Walk-forward slices of the fixed checkpoint range 0.3648–0.4029 barrier
  accuracy (four chronological blocks); regime slices range 0.3576 (bull) to
  0.4107 (bear).
- Seed-to-seed standard deviation of validation barrier accuracy is ≈ 0.0039
  (three seeds, short budget), so small differences are not signal.

The full audit with gradient-norm evidence is in
`reports/training_upgrade/training_audit.md`.

## Ethical, legal and environmental notes

- **Not financial advice.** Probabilistic estimates only; markets can and do
  invalidate them. Historical performance does not guarantee future results.
- **Data licensing.** Binance market data is not redistributed; users must comply
  with Binance's terms. No personal data is used.
- **Compute.** Training runs on CPU at ~4 minutes per epoch at stride 1
  (~2 minutes at `--train-stride 2`); the reference model was trained in under
  an hour on a 6-core laptop-class CPU. No cloud training was used.
- **Surveillance.** The model forecasts price only; it has no access to user data.

## Maintenance

Model versions are immutable artifacts: a new training configuration produces a
new `experiments/<name>/` directory and, if promoted, a new model version.
Promotion is gated (`plan/ADVANCED_TRAINING_UPGRADE.md` §7) and the previous
champion stays available for rollback.
