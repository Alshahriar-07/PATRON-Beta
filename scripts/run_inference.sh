#!/usr/bin/env bash
# Run one schema-validated inference.
#
# Required: CHECKPOINT (path to best.pt) and INPUT (96-candle JSON).
# Optional: WARMUP (parquet with >=384 prior 15m bars), OUTPUT (JSON path).
#
#   CHECKPOINT=experiments/patron_v2_20261114-203859/best.pt \
#   INPUT=examples/sample_input.json \
#   WARMUP=data/clean/BTCUSDT/15m.parquet \
#   scripts/run_inference.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${CHECKPOINT:?set CHECKPOINT to a best.pt path}"
: "${INPUT:?set INPUT to a 96-candle JSON file}"

VENV_PY=".venv/bin/python"
if [ ! -x "$VENV_PY" ]; then
  VENV_PY=".venv/Scripts/python.exe"
fi
if [ ! -x "$VENV_PY" ]; then
  echo "No virtual environment found. Run scripts/install.sh first." >&2
  exit 2
fi

ARGS=(--checkpoint "$CHECKPOINT" --input "$INPUT")
if [ -n "${WARMUP:-}" ]; then ARGS+=(--warmup "$WARMUP"); fi
if [ -n "${OUTPUT:-}" ]; then ARGS+=(--output "$OUTPUT"); fi

exec "$VENV_PY" -m patron predict "${ARGS[@]}"
