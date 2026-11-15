# Limitations

Everything here is measured or directly observable in the repository. Nothing in
this file is a hedge added for safety; each item has a number or a reproduction
path behind it.

## Model quality

1. **The barrier head is below the majority baseline on accuracy at the shipped
   operating point.** Test accuracy 0.3813 vs 0.4692 for always predicting
   `sl_first`. Its ranking is better than chance (ROC-AUC 0.5937) and its
   macro-F1 (0.3826) is far above the baseline (0.2129). The inverse-frequency
   class weights move the decision boundary; a validation-fitted prior correction
   reaches 0.4860 accuracy on validation with the same logits.
2. **The momentum head is at chance.** Test accuracy 0.2382 (5 classes), log loss
   1.5939 against 1.6094 for a uniform prediction, macro-F1 0.2244. A one-bar
   logistic regression scores 0.2578 accuracy. Reported, not hidden.
3. **Quantile forecasts are over-dispersed and nearly unconditional.** Median
   regression slope of the true close on the predicted median is −0.013 (1.0 would
   be perfect location); the 10–90 interval covers 90.3–90.7 % of outcomes instead
   of 80 %; test median MAE 0.002622 is worse than the persistence baseline
   0.002541. Pinball loss improves only because the band is wide.
4. **Pattern probabilities sit near base rates.** mAP 0.0864 for 12 tags; only the
   two engulfing tags clearly separate (AP 0.128 vs base rate 0.096).
5. **Improvements were not demonstrated.** The T1 audit identified causes; T2–T12
   have not been run. No claim of better performance is made anywhere in this
   release.
6. **Seed noise is comparable to small differences.** Three short-budget seeds give
   a validation barrier-accuracy standard deviation of ≈ 0.0039. Differences
   below ~0.008 are not meaningful.

## Data and evaluation

7. **Single asset, single venue, single timeframe.** BTC/USDT spot 15m only. Other
   assets and venues exist as context research; they are not supervised targets.
8. **The test split has been read.** It was used for reporting the shipped model,
   so it can no longer serve as an untouched hold-out for *that* model. Future
   selection must use validation plus walk-forward; the final confirmation should
   use a newly matured period.
9. **Walk-forward results are slices of a fixed checkpoint**, not fold-wise
   retraining. The retraining path exists (`walk_forward_folds`) but costs ~1 h per
   fold on this CPU and was not executed for the report.
10. **Overlapping windows.** 96-bar windows at stride 1 overlap by 95 bars;
    122,608 training windows correspond to ~1,277 independent chunks. Effective
    sample size is much smaller than the row count suggests.
11. **Regime labels are causal but coarse** (expanding z-score thresholds, 96-bar
    lookback). They are for slicing, not for trading decisions.
12. **Ambiguous triple-barrier bars (0.58 %)** are excluded from the primary loss;
    their true intrabar order is unknowable from 15m OHLC.

## Runtime

13. **No live promotion has occurred.** The champion/challenger machinery is tested
    with stand-in runners; a real candidate retrain needs ≈ 3 days of accumulated
    matured outcomes and has not been run end to end.
14. **The runtime is a single-process worker loop**, not a distributed system. It
    has no HTTP API, no authentication layer, and no horizontal scaling. Local
    JSON/SQLite files are its interface.
15. **Regime-sliced live metrics are not wired**; the offline labeller exists.
16. **No order execution.** The runtime publishes forecasts; it cannot trade and
    has no exchange credentials.
17. **A background `serve` process dies with its terminal session** on Windows
    unless wrapped in a service/scheduler. Nothing restarts it automatically.

## Engineering

18. **CPU only.** Training is ~4 min/epoch at stride 1 (~2 min at
    `--train-stride 2`) on 6 cores. No GPU path is validated, and cloud training
    was not used.
19. **CI is committed but has not run on GitHub.** The same commands pass locally.
20. **No linter/formatter is enforced.** `compileall` and the test suite are the
    static checks; style is by convention.
21. **Dependency CVE auditing is manual.** `pip check` and the repository secret
    scanner run; a full `pip-audit` is a documented manual step, not a CI gate.
22. **macOS is untested.** Linux is expected to work but was not the reference
    environment.
23. **Bitwise reproducibility is not claimed** — CPU kernels, thread counts and
    BLAS implementations differ; see `REPRODUCIBILITY.md`.
24. **The dataset is not distributed.** Rebuilding it later produces a different
    hash because the REST tail grows; recorded numbers then refer to a historical
    dataset.

## Legal / financial

25. **Not financial advice, not a trading system, no profit guarantee.** Predictions
    are probabilistic estimates on incomplete data. Historical metrics do not
    predict future performance.
26. **Market data licensing is the user's responsibility.** Binance terms apply;
    redistribution of derived datasets may be restricted.
27. **Model weights inherit Apache-2.0 from this code**, but no warranty of
    fitness or non-infringement is provided.
