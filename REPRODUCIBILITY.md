# Reproducibility

PATRON is deterministic where it can be and explicit where it cannot. This page
records what is pinned, what is recorded per run, and what will still vary.

## Environment

| Item | Reference value |
|---|---|
| OS tested | Windows 11 (10.0.26300), CPU-only |
| Python | 3.12.10 |
| PyTorch | 2.4.1+cpu |
| NumPy / pandas / pyarrow | 2.5.4 / 3.1.1 / 26.0.1 |
| scikit-learn / LightGBM | 1.9.1 / 4.7.0 |
| Hardware | 6-core laptop-class CPU, no GPU; `torch.set_num_threads(6)` in training scripts |
| Threads | 6 (training), 1 (inference) |

`requirements.txt` holds the installable ranges plus the torch pin;
`requirements-lock.txt` is generated from the working environment of this release
(exact versions actually used to run the test suite).

## Determinism

Set by code (`patron.utils.seed.set_seed`): Python `random`, NumPy, PyTorch CPU,
`PYTHONHASHSEED`, and the DataLoader generator seed. Training uses a seeded
`torch.Generator` for shuffling.

What can still differ between machines, and why:

- **CPU kernels.** PyTorch does not promise bitwise-identical results across CPU
  architectures, thread counts, or BLAS implementations. `torch.use_deterministic_algorithms(True)`
  is *not* enabled because several recurrent/attention kernels have no
  deterministic CPU implementation; enabling it would fail rather than help.
- **Thread count** changes floating-point reduction order. Keep `--threads`
  fixed when comparing runs.
- **Data tail.** `download_data.py` appends the current month from the REST API, so
  a rebuild on a later date produces more rows and a different dataset hash.
- **Wall-clock.** Timestamps, run directory names and durations are recorded, not
  reproduced.

Consequence: **exact bitwise reproduction is not claimed.** What is claimed is
that the same code, same dataset hash, same seed and same thread count reproduce
validation metrics within normal numerical noise (~1e-3 on loss, ~0.004 on
accuracy across seeds).

## What every run records

`experiments/<run>/` contains:

| File | Content |
|---|---|
| `config_snapshot.yaml` | the full model config used |
| `train_log.jsonl` | start event (windows, params, class weights, scheduler, seed), one line per epoch, finish event |
| `training_history.json` | per-epoch train/val losses and per-term losses, LR, seconds, best epoch |
| `best.pt`, `last.pt` | checkpoints with config, scaler, feature order, split summary, epoch metrics |
| `experiment.json` | *(new in this release, written by `python -m patron train`)* dataset sha256, feature version, seed, stride, batch, optimizer, scheduler, wall-clock, hardware |

The dataset hash ties a run to bytes, not to a date; the training audit found the
older logs hard-coded `dataset_version: "1.0.0"` and this release stops that.

## Reproduce the reference pipeline

```bash
# 1. Environment
python -m venv .venv && ./.venv/Scripts/pip install -r requirements.txt
./.venv/Scripts/pip install -e . --no-deps
./.venv/Scripts/python.exe -m patron health

# 2. Data (network; ~5 years of 15m klines)
python scripts/download_data.py
python scripts/build_dataset.py
# record data/manifests/feature_manifest.json -> dataset_sha256

# 3. Train (CPU; hours for a full run, minutes for a short one)
python -m patron train --run-name repro_s42 --epochs 6 --train-stride 2 --seed 42

# 4. Evaluate on the held-out split (writes reports/evaluation_report.{json,md})
python -m patron evaluate --checkpoint experiments/repro_s42_*/best.pt

# 5. Walk-forward slices and multi-seed dispersion
python scripts/run_walk_forward.py
python scripts/train_multiseed.py --seeds 42 7 2026 --epochs 4 --stride 4
```

## Reference numbers

The shipped analysis refers to `experiments/patron_v2_20261114-203859/best.pt`
(dataset hash `f0d486…b4421`, 15 epochs, best epoch 3, val loss 1.4048). Its test
metrics are in `MODEL_CARD.md`; the audit that explains them is in
`reports/training_upgrade/training_audit.md`. Those numbers are **historical
measurements**, not a guarantee for a rebuild.

## Experiment bookkeeping (T9 target)

The advanced-training plan adds a registry and a resumable runner. Until then,
`python -m patron experiment list` enumerates runs by reading
`training_history.json`, and reports the checkpoint SHA-256 so two runs with the
same name can still be distinguished.

## Reporting a reproduction failure

Open an issue with: dataset hash, config snapshot, seed, thread count, exact
command, and the observed vs expected metric. Without the hash and seed the
report cannot be checked.
