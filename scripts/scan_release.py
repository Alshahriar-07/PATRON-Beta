"""Scan a tree for secrets, private paths and artifacts that must not ship.

Used by the release builder, by CI and by contributors before opening a PR.
It is a heuristic scanner, not a guarantee: a clean run means "nothing obvious",
not "provably secret-free".

    python scripts/scan_release.py                 # scan the repository
    python scripts/scan_release.py --root dist/release/PATRON-v1.0.0-beta.1
    python scripts/scan_release.py --json

Exit codes: 0 clean, 1 findings, 2 invalid root.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

TEXT_SUFFIXES = {
    ".py", ".md", ".txt", ".yaml", ".yml", ".json", ".toml", ".cfg", ".ini",
    ".sh", ".ps1", ".bat", ".gitignore", ".gitattributes", ".env", ".example", "",
}
SKIP_DIR_NAMES = {
    ".git", ".venv", "venv", "env", "__pycache__", ".pytest_cache", ".ruff_cache",
    "node_modules", "dist", "build", "Plan", "data", "experiments", "logs",
}
ALLOWLIST_FILES = {".env.example", "scan_release.py", "scan_release.pyc"}

SECRET_PATTERNS = [
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("private_key_block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    (
        "assigned_secret",
        re.compile(
            r"(?i)\b(api[_-]?key|api[_-]?secret|secret[_-]?key|access[_-]?token|"
            r"client[_-]?secret|password)\b\s*[:=]\s*['\"][A-Za-z0-9_\-/+=]{16,}['\"]"
        ),
    ),
    ("bearer_token", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9\-._~+/]{20,}")),
]
PRIVATE_PATH_PATTERNS = [
    ("windows_home", re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+[^\\/\s\"']+")),
    ("posix_home", re.compile(r"(?<![\w.])/(?:home|Users)/[A-Za-z0-9._-]+")),
]
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".parquet", ".db", ".sqlite", ".sqlite3", ".pkl", ".pickle"}
FORBIDDEN_NAMES = {".env", "id_rsa", "id_ed25519", ".netrc"}
MAX_BYTES = 5 * 1024 * 1024


@dataclass
class Finding:
    path: str
    kind: str
    detail: str


def iter_files(root: Path):
    for path in sorted(root.rglob("*")):
        if any(part in SKIP_DIR_NAMES for part in path.parts):
            continue
        if path.is_file():
            yield path


def scan(root: Path, include_reports: bool = False) -> list[Finding]:
    findings: list[Finding] = []
    for path in iter_files(root):
        rel = path.relative_to(root).as_posix()
        if not include_reports and rel.startswith("reports/"):
            continue
        if path.name in FORBIDDEN_NAMES:
            findings.append(Finding(rel, "forbidden_file", "secret-bearing filename"))
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            findings.append(Finding(rel, "forbidden_artifact", f"binary artifact {path.suffix}"))
        size = path.stat().st_size
        if size > MAX_BYTES:
            findings.append(Finding(rel, "large_file", f"{size / 1e6:.1f} MB > 5 MB limit"))
        if path.suffix.lower() not in TEXT_SUFFIXES and path.suffix != "":
            continue
        if size > 2 * 1024 * 1024:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:  # pragma: no cover - unreadable file
            findings.append(Finding(rel, "unreadable", str(exc)))
            continue
        if path.name in ALLOWLIST_FILES:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            for name, pattern in SECRET_PATTERNS:
                if pattern.search(line):
                    findings.append(Finding(rel, name, f"line {lineno}"))
            for name, pattern in PRIVATE_PATH_PATTERNS:
                if pattern.search(line):
                    findings.append(Finding(rel, name, f"line {lineno}"))
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scan for secrets and private artifacts")
    parser.add_argument("--root", default=".", help="tree to scan (default: repository root)")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--include-reports", action="store_true", help="also scan reports/")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 2

    findings = scan(root, include_reports=args.include_reports)
    if args.json:
        print(json.dumps(
            {"root": str(root), "findings": [asdict(f) for f in findings], "clean": not findings},
            indent=2,
        ))
    elif not findings:
        print(f"scan clean: {root}")
    else:
        print(f"{len(findings)} finding(s) under {root}:")
        for f in findings:
            print(f"  [{f.kind}] {f.path} ({f.detail})")
        print("\nReview each item. Do not commit secrets; rotate anything already exposed.")
    return 0 if not findings else 1


if __name__ == "__main__":
    raise SystemExit(main())
