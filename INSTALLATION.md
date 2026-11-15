# Installation

PATRON is a pure-Python CPU project. No GPU, no database server, no credentials.

## Requirements

| Item | Requirement |
|---|---|
| Python | **3.12** (the code uses 3.12-only syntax; 3.11 will not import) |
| Disk | ~2.5 GB with PyTorch and the 57 MB dataset; ~400 MB without a dataset |
| RAM | 8 GB recommended for training (peak ≈ 3 GB) |
| OS | Windows 11 (tested), Linux x86_64 (expected to work), macOS (untested) |
| Network | only to download market data and install packages |

## Windows 10/11

```powershell
git clone <repository-url> PATRON
cd PATRON
powershell -ExecutionPolicy Bypass -File scripts/install.ps1
```

The script checks the Python version, creates `.venv`, installs
`requirements.txt`, installs the package in editable mode, verifies PyTorch can
run a matrix multiply, and finishes with `python -m patron health`.

Manual equivalent:

```powershell
python -m venv .venv
./.venv/Scripts/python.exe -m pip install --upgrade pip
./.venv/Scripts/pip install -r requirements.txt
./.venv/Scripts/pip install -e . --no-deps
./.venv/Scripts/python.exe -m patron health
```

## Linux / macOS

```bash
git clone <repository-url> PATRON && cd PATRON
bash scripts/install.sh
```

Manual equivalent:

```bash
python3.12 -m venv .venv
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -r requirements.txt
./.venv/bin/pip install -e . --no-deps
./.venv/bin/python -m patron health
```

## PyTorch note

`requirements.txt` pins `torch==2.4.1` because newer CPU wheels fail on the
reference Windows host with `WinError 126 … shm.dll` / `torch_cpu.dll`. If you are
on Linux and prefer a newer wheel, install it explicitly and re-run
`python -m patron health`; do not mix CUDA wheels into a CPU environment.

CUDA is **not** required or configured. If you have a working CUDA stack, training
still runs on CPU by default because device selection is not exposed; this release
does not claim GPU support.

## Verify

```bash
python -m patron health          # PASS with data/checkpoint, WARN without them
python -m pytest                # full suite, synthetic fixtures, no network
```

`health` returns exit code 0 for PASS/WARN and 1 for FAIL. Use `--json` for
machine-readable output and `--strict` to fail on warnings.

## Data and checkpoint are optional

A fresh clone has neither the dataset nor a checkpoint. Both are intentionally not
distributed:

```bash
python scripts/download_data.py      # public Binance archives (network, ~5 years)
python scripts/build_dataset.py      # features + labels + manifest + scaler
python -m patron train --run-name my_first --epochs 3 --train-stride 4
```

`health` reports `WARN` for each missing artifact and still exits successfully.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ImportError: cannot import name …` | Python < 3.12, or the package is not installed: `pip install -e . --no-deps` |
| `WinError 126 shm.dll` | a newer/incompatible torch wheel; install `torch==2.4.1` from requirements |
| `ModuleNotFoundError: patron` | editable install missing, or running from another directory |
| `schema … not found` | run commands from the repository root (schemas live in `schemas/`) |
| `checkpoint could not be loaded in safe mode` | file is not a PATRON checkpoint or was saved with full pickle; verify its origin before using `allow_unsafe=True` |
| very slow epochs | OneDrive/network paths and antivirus scanning; move the repo to a local disk and keep `--threads 6` |
| `pytest` needs `Plan/` | only in old checkouts; this release stores fixtures in `examples/` |
