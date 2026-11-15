#!/usr/bin/env bash
# PATRON installer for Linux / macOS.
# Creates .venv, installs dependencies, verifies PyTorch and runs the health check.
# No sudo, no system packages, no CUDA.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON:-}"
if [ -z "$PYTHON_BIN" ]; then
  for candidate in python3.12 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then PYTHON_BIN="$candidate"; break; fi
  done
fi
if [ -z "$PYTHON_BIN" ]; then
  echo "ERROR: no Python interpreter found. Install Python 3.12 and re-run." >&2
  exit 1
fi

VERSION="$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [ "$VERSION" != "3.12" ]; then
  echo "ERROR: PATRON requires Python 3.12, found $VERSION ($PYTHON_BIN)." >&2
  exit 1
fi

if [ ! -d .venv ]; then
  echo "==> creating virtual environment"
  "$PYTHON_BIN" -m venv .venv
fi

VENV_PY=".venv/bin/python"
echo "==> upgrading pip"
"$VENV_PY" -m pip install --quiet --upgrade pip

echo "==> installing requirements (CPU PyTorch, a few minutes on first run)"
"$VENV_PY" -m pip install --quiet -r requirements.txt

echo "==> installing patron (editable, no dependency resolution)"
"$VENV_PY" -m pip install --quiet -e . --no-deps

echo "==> verifying PyTorch"
"$VENV_PY" - <<'PY'
import torch
x = torch.ones(2, 2) @ torch.ones(2, 2)
assert float(x.sum()) == 8.0, "torch matrix multiply failed"
print(f"    torch {torch.__version__} ok (threads={torch.get_num_threads()})")
PY

echo "==> health check"
set +e
"$VENV_PY" -m patron health
STATUS=$?
set -e
if [ "$STATUS" -ne 0 ]; then
  echo "Health check reported FAIL. Fix the entries above, then re-run:" >&2
  echo "  ./.venv/bin/python -m patron health" >&2
  exit "$STATUS"
fi

echo
echo "Installed. Next:"
echo "  ./.venv/bin/python -m pytest"
echo "  ./.venv/bin/python -m patron health --json"
