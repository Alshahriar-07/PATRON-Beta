"""Thin wrapper around ``python -m patron health``.

Useful for schedulers and installers that call a script path. Exit codes match
the CLI: 0 for PASS/WARN, 1 for FAIL (or any WARN with ``--strict``).

    python scripts/health_check.py
    python scripts/health_check.py --json
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:  # allow running from a source checkout without install
    sys.path.insert(0, str(SRC))


def main(argv: list[str] | None = None) -> int:
    try:
        from patron.cli import main as cli_main
    except Exception as exc:  # pragma: no cover - broken environment
        print(f"cannot import patron: {exc}", file=sys.stderr)
        print("install with: pip install -e . --no-deps", file=sys.stderr)
        return 2
    args = list(argv if argv is not None else sys.argv[1:])
    return cli_main(["health", *args])


if __name__ == "__main__":
    raise SystemExit(main())
