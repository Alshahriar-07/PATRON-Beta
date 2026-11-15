# Dataset Card — PATRON BTC/USDT 15m (dataset 1.0.0)

## Summary

| Field | Value |
|---|---|
| Name | `BTCUSDT_dataset.parquet` (dataset 1.0.0, features 1.0.0, labels 1.0.0) |
| Content | BTC/USDT spot 15-minute candles with 12 causal features, 12 pattern tags and four label families |
| Source | Binance public market data (`data.binance.vision` monthly archives + REST for the current month) |
| Period | 2021-11-01 00:00 UTC → 2026-11-14 12:45 UTC |
| Rows | 176,591 (one row per closed 15m candle) |
| Size | ≈ 57 MB parquet (58 columns) |
| SHA-256 | `f0d4862984c110d2c98983b072b8c9ccb7c0703b0ba1728e3fa3732d012b4421` |
| License | **Not redistributed.** Binance data remains subject to Binance's terms; the code that builds it is Apache-2.0 |
| Distribution | Local artifact only; excluded from git and from releases |
| Maintainer | Repository maintainer (see `NOTICE`) |

## Why it is not shipped

Redistributing exchange market data may violate the provider's terms, the file is
large, and a stale snapshot invites results that cannot be reproduced. The build
is deterministic (same months → same parquet, modulo REST-tail growth), so the
dataset can be regenerated:

```bash
python scripts/download_data.py          # monthly archives + REST tail
python scripts/build_dataset.py          # features, labels, manifest, scaler
```

`data/manifests/feature_manifest.json` records the dataset hash, row count,
feature order and column groups for the build you actually produced.

## Construction

1. **Download** monthly `klines` archives (15m, plus 1h/4h for context) from
   `data.binance.vision`; the current month comes from the REST endpoint
   `/api/v3/klines`. Only closed candles are kept.
2. **Validate**: OHLC sanity, duplicate removal, gap detection, monotonic time.
   A canonical CLEAN parquet is written with an `is_gap` flag.
3. **Features** (12, all causal, computed per bar): `open_rel`, `high_rel`,
   `low_rel`, `close_rel`, `volume_log_rel`, `body_to_range`, `atr14_pct`,
   `rsi14_scaled`, `ema9_gap`, `ema21_gap`, `volume_spike`, `htf_trend`.
   Warm-up rows are marked `feature_valid = False`.
4. **Pattern tags** (12 rule-based binary tags) from the same bar's geometry.
5. **Labels** for the next 6 bars: triple barrier (TP 1.5·ATR14, SL 1.0·ATR14,
   first touch wins, same-candle double touch marked ambiguous and masked),
   relative high/low/close targets, and the 5-class momentum bin from the
   ATR-normalised 6-bar return with boundaries [-1.0, -0.35, 0.35, 1.0].

## Splits

| Split | Dates | Windows |
|---|---|---|
| Train (70 %) | 2021-11 → 2025-05 | 122,608 |
| Validation (15 %) | 2025-05-13 → 2026-02-12 | 26,262 |
| Test (15 %) | 2026-02-12 → 2026-11-14 | 26,269 |

Chronological, with 6-bar purge before each boundary and 6-bar embargo after it.
The scaler is fitted on training windows only. Windows are dropped if any of the
96 input bars is invalid or if a gap falls inside the window plus its 6-bar
target horizon, so no sample silently spans missing candles.

## Known properties and issues

- **Barrier balance**: sl_first 45.9 %, timeout 27.9 %, tp_first 25.9 %,
  ambiguous 0.58 % (1,028 rows, excluded from the primary loss).
- **Momentum balance**: 28,984–39,802 per class; 19 rows have no valid class and
  are excluded.
- **Feature nulls**: ≤ 0.13 % in any feature, concentrated in `htf_trend` warm-up.
- **Overlapping windows**: 96-bar windows at stride 1 overlap by 95 bars; treat
  ~1,277 as the independent-sample count for the training split.
- **Regime labels** are computed separately (expanding, shifted thresholds) and
  are causal; they are not stored in the parquet.
- **Gaps**: missing Binance bars are flagged, not imputed. A window containing a
  flagged bar is excluded.
- **No personal data**, no user identifiers, no order-book or account data.

## Reproducibility

The hash above corresponds to the build used for the reference model. A rebuild
will extend the REST-backed tail, producing a different hash and slightly
different split dates; record the new manifest and re-register any promotion
baseline. See `REPRODUCIBILITY.md`.

## Intended use

Training and evaluating research models for this repository. Not for
redistribution, resale, or constructing financial products. Verify that your use
of the underlying data complies with the provider's terms and your local
regulations.
