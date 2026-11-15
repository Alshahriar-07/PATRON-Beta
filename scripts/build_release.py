"""Build a reproducible PATRON release directory, source archives and packages.

    python scripts/build_release.py --version 1.0.0-beta.1

Steps: validate version and docs, stage a clean tree (no data, checkpoints,
databases, venvs, caches or secrets), scan it, write the source .zip and .tar.gz,
build the wheel and sdist, assemble ``dist/release/PATRON-v<version>/`` with
checksums and a build manifest, then verify the archives and checksums.

Exit code 0 only if every gate passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import scan_release  # noqa: E402  (local script import)

INCLUDE_FILES = [
    "pyproject.toml",
    "requirements.txt",
    "requirements-lock.txt",
    "README.md",
    "CHANGELOG.md",
    "RELEASE_NOTES.md",
    "LICENSE",
    "NOTICE",
    "SECURITY.md",
    "CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md",
    "MODEL_CARD.md",
    "DATASET_CARD.md",
    "REPRODUCIBILITY.md",
    "INSTALLATION.md",
    "QUICKSTART.md",
    "CONFIGURATION.md",
    "LIMITATIONS.md",
    "THIRD_PARTY_NOTICES.md",
    "INSTRUCTION.md",
    ".gitignore",
    ".gitattributes",
    ".env.example",
]
INCLUDE_TREES = ["src", "tests", "config", "schemas", "scripts", "examples", "plan"]
REQUIRED_DOCS = [
    "LICENSE",
    "NOTICE",
    "README.md",
    "CHANGELOG.md",
    "SECURITY.md",
    "CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md",
    "MODEL_CARD.md",
    "DATASET_CARD.md",
    "LIMITATIONS.md",
    "THIRD_PARTY_NOTICES.md",
    "REPRODUCIBILITY.md",
    "INSTALLATION.md",
    "QUICKSTART.md",
    "CONFIGURATION.md",
    "INSTRUCTION.md",
]
EXCLUDE_DIR_NAMES = {
    "__pycache__", ".pytest_cache", ".ruff_cache", ".git", ".venv", "venv",
    "node_modules", "dist", "build", "data", "experiments", "logs", "reports",
    "Plan", ".mypy_cache", ".idea", ".vscode",
}
FIXED_MTIME = 1_763_164_800  # 2025-11-15T00:00:00Z-ish, keeps archives stable


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized(version: str) -> str:
    return version.replace("-beta.", "b").replace("-alpha.", "a").replace("-rc.", "rc")


def project_version() -> str:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.strip().startswith("version"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("cannot read version from pyproject.toml")


def stage_tree(target: Path) -> list[Path]:
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    copied: list[Path] = []
    for name in INCLUDE_FILES:
        source = ROOT / name
        if not source.exists():
            continue
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(destination)
    for tree in INCLUDE_TREES:
        base = ROOT / tree
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            if any(part in EXCLUDE_DIR_NAMES for part in path.parts):
                continue
            if path.suffix in {".pyc", ".pyo"}:
                continue
            destination = target / path.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            copied.append(destination)
    for path in copied:
        os.utime(path, (FIXED_MTIME, FIXED_MTIME))
    return copied


def write_zip(source: Path, archive: Path) -> None:
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(source.rglob("*")):
            if not path.is_file():
                continue
            info = zipfile.ZipInfo(str(source.name / path.relative_to(source)))
            info.date_time = (2025, 11, 15, 0, 0, 0)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            handle.writestr(info, path.read_bytes())


def write_tar(source: Path, archive: Path) -> None:
    with tarfile.open(archive, "w:gz") as handle:
        for path in sorted(source.rglob("*")):
            if not path.is_file():
                continue
            info = handle.gettarinfo(str(path), arcname=str(source.name / path.relative_to(source)))
            info.mtime = FIXED_MTIME
            info.uid = info.gid = 0
            info.uname = info.gname = "patron"
            with path.open("rb") as payload:
                handle.addfile(info, payload)


def package_build(outdir: Path) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "build", "--no-isolation", "--outdir", str(outdir)]
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"package build failed:\n{result.stdout}\n{result.stderr}")
    artifacts = sorted(p.name for p in outdir.iterdir() if p.suffix in {".whl", ".gz"})
    return {"command": " ".join(cmd[1:]), "artifacts": artifacts}


RELEASE_README = """# PATRON {version}

Pre-release source distribution. Not production-validated; not financial advice.

## Contents

| Path | What it is |
|---|---|
| `PATRON-v{version}-source.zip`, `.tar.gz` | complete source tree (code, tests, configs, schemas, examples, docs) |
| `patron-{package}-py3-none-any.whl`, `.tar.gz` | Python package artifacts |
| `README.md`, `CHANGELOG.md`, `RELEASE_NOTES.md` | project overview and this release |
| `LICENSE`, `NOTICE`, `THIRD_PARTY_NOTICES.md` | Apache-2.0 and third-party attributions |
| `MODEL_CARD.md`, `DATASET_CARD.md`, `LIMITATIONS.md` | honest scope and measured limits |
| `INSTALLATION.md`, `QUICKSTART.md`, `CONFIGURATION.md` | install and use |
| `REPRODUCIBILITY.md` | environment, seeds, what is and is not reproducible |
| `configs/` | the two real configuration files (`model_config.yaml`, `runtime.yaml`) |
| `examples/` | runnable inference / evaluation / training examples |
| `scripts/` | installers, run wrappers, health check, release builder and scanner |
| `reports/` | training audit, baseline metrics, release validation report |
| `checksums/SHA256SUMS.txt` | hashes for every file listed above |
| `BUILD-MANIFEST.json` | provenance of this build |

The dataset and checkpoints are **not** included: they are reproducible from
public data (`DATASET_CARD.md`) and are never redistributed by this project.

## Verify

```bash
sha256sum -c checksums/SHA256SUMS.txt
```

## Install from an archive

```bash
tar -xzf PATRON-v{version}-source.tar.gz
cd PATRON-v{version}
python -m venv .venv && ./.venv/bin/pip install -r requirements.txt
./.venv/bin/pip install -e . --no-deps
./.venv/bin/python -m patron health
```

Canonical copies of all documents live in the repository root; this directory is a
build artifact generated by `scripts/build_release.py`.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a PATRON release")
    parser.add_argument("--version", default="1.0.0-beta.1", help="release version, e.g. 1.0.0-beta.1")
    parser.add_argument("--dist", default=str(ROOT / "dist"))
    parser.add_argument("--skip-packages", action="store_true", help="skip wheel/sdist build")
    args = parser.parse_args(argv)

    started = time.time()
    py_version = project_version()
    if normalized(args.version) != py_version:
        print(
            f"version mismatch: --version {args.version} normalises to {normalized(args.version)}, "
            f"pyproject.toml says {py_version}",
            file=sys.stderr,
        )
        return 1

    missing = [name for name in REQUIRED_DOCS if not (ROOT / name).exists()]
    if missing:
        print(f"missing required documentation: {missing}", file=sys.stderr)
        return 1
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    if "Apache License" not in license_text or "Version 2.0" not in license_text:
        print("LICENSE does not look like Apache-2.0", file=sys.stderr)
        return 1
    if "Apache License" not in (ROOT / "NOTICE").read_text(encoding="utf-8"):
        print("NOTICE does not mention the Apache License", file=sys.stderr)
        return 1

    dist = Path(args.dist)
    release_name = f"PATRON-v{args.version}"
    release_dir = dist / "release" / release_name
    staging = dist / "_staging" / f"PATRON-v{args.version}"

    print(f"==> staging {release_name}")
    copied = stage_tree(staging)
    print(f"    {len(copied)} files staged")

    print("==> scanning staged tree")
    findings = scan_release.scan(staging, include_reports=True)
    if findings:
        for finding in findings:
            print(f"    [{finding.kind}] {finding.path} ({finding.detail})", file=sys.stderr)
        print("aborting: staged tree has findings", file=sys.stderr)
        return 1
    print("    clean")

    if release_dir.exists():
        shutil.rmtree(release_dir)
    release_dir.mkdir(parents=True)

    print("==> writing source archives")
    zip_path = dist / f"{release_name}-source.zip"
    tar_path = dist / f"{release_name}-source.tar.gz"
    write_zip(staging, zip_path)
    write_tar(staging, tar_path)

    packages: dict = {"artifacts": []}
    if not args.skip_packages:
        print("==> building wheel and sdist")
        artifacts_dir = dist / "artifacts"
        packages = package_build(artifacts_dir)

    print("==> assembling release directory")
    for name in (
        "README.md", "CHANGELOG.md", "RELEASE_NOTES.md", "LICENSE", "NOTICE",
        "SECURITY.md", "MODEL_CARD.md", "DATASET_CARD.md", "LIMITATIONS.md",
        "THIRD_PARTY_NOTICES.md", "REPRODUCIBILITY.md", "INSTALLATION.md",
        "QUICKSTART.md", "CONFIGURATION.md", "INSTRUCTION.md", "CONTRIBUTING.md",
        "CODE_OF_CONDUCT.md",
        "requirements.txt", "requirements-lock.txt", "pyproject.toml", ".gitignore",
        ".gitattributes",
    ):
        source = ROOT / name
        if source.exists():
            shutil.copy2(source, release_dir / name)
    (release_dir / "README.md").write_text(
        RELEASE_README.format(version=args.version, package=py_version), encoding="utf-8"
    )
    configs = release_dir / "configs"
    configs.mkdir(exist_ok=True)
    shutil.copy2(ROOT / "config" / "model_config.yaml", configs / "model_config.yaml")
    shutil.copy2(ROOT / "config" / "runtime.yaml", configs / "runtime.yaml")
    (configs / "README.md").write_text(
        "PATRON reads `model_config.yaml` for the model/training contract and "
        "`runtime.yaml` for autonomous-runtime policy. Inference has no separate "
        "config: it uses the checkpoint's embedded config plus CLI flags "
        "(`--checkpoint`, `--input`, `--warmup`, `--output`).\n",
        encoding="utf-8",
    )
    for tree in ("examples", "scripts"):
        shutil.copytree(ROOT / tree, release_dir / tree, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    reports_dir = release_dir / "reports"
    reports_dir.mkdir()
    audit = ROOT / "reports" / "training_upgrade" / "training_audit.md"
    if audit.exists():
        shutil.copy2(audit, reports_dir / "training_audit.md")
    baseline = ROOT / "reports" / "release" / "baseline_metrics.json"
    if baseline.exists():
        shutil.copy2(baseline, reports_dir / "baseline_metrics.json")
    validation = ROOT / "reports" / "release" / "release_validation.md"
    if validation.exists():
        shutil.copy2(validation, reports_dir / "release_validation.md")
    else:
        (reports_dir / "release_validation.md").write_text(
            "Release validation has not been recorded yet. Run the checks in "
            "`reports/release/FINAL_RELEASE_REPORT.md` and rebuild.\n",
            encoding="utf-8",
        )
    shutil.copy2(zip_path, release_dir / zip_path.name)
    shutil.copy2(tar_path, release_dir / tar_path.name)
    for artifact in packages.get("artifacts", []):
        shutil.copy2(dist / "artifacts" / artifact, release_dir / artifact)

    print("==> verifying archives")
    with zipfile.ZipFile(zip_path) as handle:
        zip_members = [m for m in handle.namelist() if not m.endswith("/")]
    with tarfile.open(tar_path) as handle:
        tar_members = [m.name for m in handle.getmembers() if m.isfile()]
    if len(zip_members) != len(copied) or len(tar_members) != len(copied):
        print(
            f"archive mismatch: staged {len(copied)}, zip {len(zip_members)}, tar {len(tar_members)}",
            file=sys.stderr,
        )
        return 1
    bad = [
        name for name in zip_members
        if any(part in {".git", "data", "experiments", "logs", "__pycache__"} for part in Path(name).parts)
    ]
    if bad:
        print(f"forbidden archive entries: {bad[:5]}", file=sys.stderr)
        return 1

    print("==> writing checksums and manifest")
    checksum_lines = []
    for path in sorted(release_dir.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            checksum_lines.append(f"{sha256(path)}  {path.relative_to(release_dir).as_posix()}")
    checksums_dir = release_dir / "checksums"
    checksums_dir.mkdir(exist_ok=True)
    # newline="\n" matters: sha256sum -c rejects CRLF on Linux/macOS.
    (checksums_dir / "SHA256SUMS.txt").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8", newline="\n"
    )

    git = {"commit": None, "branch": None, "dirty": None}
    try:
        git["commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip() or None
        git["branch"] = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip() or None
        git["dirty"] = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True
            ).stdout.strip()
        )
    except OSError:
        pass
    manifest = {
        "release": release_name,
        "release_version": args.version,
        "package_version": py_version,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "built_on": {
            "python": sys.version.split()[0],
            "platform": sys.platform,
        },
        "git": git,
        "file_count": len(copied),
        "archives": {
            zip_path.name: {"sha256": sha256(zip_path), "bytes": zip_path.stat().st_size},
            tar_path.name: {"sha256": sha256(tar_path), "bytes": tar_path.stat().st_size},
        },
        "packages": {
            name: {"sha256": sha256(dist / "artifacts" / name),
                   "bytes": (dist / "artifacts" / name).stat().st_size}
            for name in packages.get("artifacts", [])
        },
        "excluded": sorted(EXCLUDE_DIR_NAMES),
        "verification": {"staging_scan": "clean", "archive_members_match": True},
    }
    (release_dir / "BUILD-MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("==> verifying checksums")
    for line in checksum_lines:
        digest, rel = line.split("  ", 1)
        if sha256(release_dir / rel) != digest:
            print(f"checksum mismatch: {rel}", file=sys.stderr)
            return 1

    report_path = ROOT / "reports" / "release" / "release_build.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "release": release_name,
        "release_dir": str(release_dir),
        "files": len(checksum_lines),
        "archives": manifest["archives"],
        "packages": manifest["packages"],
        "duration_seconds": round(time.time() - started, 1),
        "status": "built and verified",
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        f"\nrelease built: {release_dir}\n"
        f"  {len(checksum_lines)} files checksummed\n"
        f"  {zip_path.name} ({zip_path.stat().st_size / 1024:.0f} KiB)\n"
        f"  {tar_path.name} ({tar_path.stat().st_size / 1024:.0f} KiB)\n"
        f"  report: {report_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
