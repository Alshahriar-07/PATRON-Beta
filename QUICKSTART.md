# Quick start

Five minutes from clone to first verified command. CPU only.

## 1. Install

```bash
python -m venv .venv
# Windows
./.venv/Scripts/pip install -r requirements.txt && ./.venv/Scripts/pip install -e . --no-deps
# Linux / macOS
./.venv/bin/pip install -r requirements.txt && ./.venv/bin/pip install -e . --no-deps
```

or `scripts/install.sh` / `scripts/install.ps1`.

## 2. Check the environment

```bash
python -m patron health
```

```
PATRON health: PASS  (12 pass, 0 warn, 0 fail)
[PASS] model_contract: 272,618 trainable parameters, input (96, 12), 4 heads
[PASS] checkpoint: best.pt: 272,618 parameters, epoch 3, val_loss 1.4048
...
```

Without data or a checkpoint you will see `WARN` entries instead — that is
expected for a fresh clone and not an error.

## 3. Run the tests

```bash
python -m pytest
```

Synthetic fixtures only: no network, no market data, no checkpoint required.

## 4. Train something small

```bash
python -m patron train --run-name quickstart --epochs 2 --train-stride 8 --batch-size 64
```

*Needs the dataset.* If `data/features/BTCUSDT_dataset.parquet` is missing:

```bash
python scripts/download_data.py
python scripts/build_dataset.py
```

A 2-epoch run is a smoke test, not a usable model. A full run is
`--epochs 15` (about an hour on 6 CPU cores at stride 1, ~30 minutes at
`--train-stride 2`).

## 5. Inspect runs

```bash
python -m patron experiment list
python -m patron experiment compare --runs <run_a> <run_b>
```

Each run directory contains `config_snapshot.yaml`, `train_log.jsonl`,
`training_history.json`, `best.pt`, `last.pt` and (for runs created through the
CLI) `experiment.json` with the dataset hash and hardware record.

## 6. Inference

Inference needs a checkpoint and a 96-candle JSON window **plus a warm-up parquet**
(384 prior bars) because two features look further back than the window:

```bash
python -m patron predict \
  --checkpoint experiments/<run>/best.pt \
  --input examples/sample_input.json \
  --warmup data/clean/BTCUSDT/15m.parquet \
  --output reports/predictions/latest.json
```

`examples/sample_input.json` is a real, schema-valid 96-bar window. Without
`--warmup` the predictor returns `NO_OUTPUT` with an explicit reason rather than
inventing feature values — that is the designed behaviour.

## 7. Evaluate a checkpoint

```bash
python -m patron evaluate --checkpoint experiments/<run>/best.pt
```

Writes `reports/evaluation_report.json` and `.md` with barrier, momentum,
quantile and pattern metrics on the held-out split, plus baselines.

## 8. Start the autonomous runtime (optional)

```bash
python -m patron runtime once --no-train     # one confirmed-candle cycle
python -m patron runtime status              # published state
python -m patron runtime serve --no-train    # loop (Ctrl+C to stop)
```

The runtime writes to `data/runtime/` and `reports/live/`. It never places orders.

## Next steps

- `CONFIGURATION.md` for every setting.
- `MODEL_CARD.md` and `LIMITATIONS.md` for what the numbers do and do not mean.
- `plan/ADVANCED_TRAINING_UPGRADE.md` for the training-improvement programme.
