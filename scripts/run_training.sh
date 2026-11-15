#!/usr/bin/env bash
# Train a PATRON candidate. Arguments are passed to `python -m patron train`.
# Examples:
#   scripts/run_training.sh --dry-run
#   scripts/run_training.sh --run-name e0_control_s42 --epochs 6 --train-stride 2 --seed 42
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV_PY=".venv/bin/python"
if [ ! -x "$VENV_PY" ]; then
  VENV_PY=".venv/Scripts/python.exe"
fi
if [ ! -x "$VENV_PY" ]; then
  echo "No virtual environment found. Run scripts/install.sh first." >&2
  exit 2
fi

if [ "$#" -eq 0 ]; then
  echo "usage: $0 [patron train options]   (e.g. --epochs 6 --train-stride 2)" >&2
  exit 2
fi

exec "$VENV_PY" -m patron train "$@"
