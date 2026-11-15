# Changelog

All notable changes to PATRON are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Status labels used below: **implemented** (in the code), **tested** (covered by
automated tests or a recorded run), **experimental** (works but unproven),
**planned** (not implemented).

## [1.0.0-beta.1] — 2026-11-15

First public testing release. This is a pre-release: it is feature-complete for
local research use, not validated for production or live trading.

### Added

- **Unified CLI**: `python -m patron health | train | evaluate | predict |
  runtime | experiment` with `--help`, validation, logging and explicit exit
  codes. *Implemented, tested.*
- **Health check**: `python -m patron health [--json] [--strict]` verifies Python,
  dependencies, PyTorch runtime, config, JSON schemas, model contract
  (272,618 parameters), scaler feature order, dataset presence, checkpoint
  loadability, runtime database (read-only), write access and predictor
  initialisation. Returns PASS / WARN / FAIL. *Implemented, tested.*
- **Release tooling**: `scripts/build_release.py` builds a versioned release
  directory, source `.zip`/`.tar.gz`, wheel and sdist, `SHA256SUMS.txt` and
  `BUILD-MANIFEST.json`, with exclusions for data, checkpoints, databases,
  virtualenvs and caches. *Implemented, tested.*
- **Release scanner**: `scripts/scan_release.py` checks for secrets, private
  paths, oversized files and forbidden artifacts before packaging.
  *Implemented, tested.*
- **Installation scripts**: `scripts/install.sh`, `scripts/install.ps1` (venv,
  dependency install, PyTorch check, health check) and `run_training` /
  `run_inference` wrappers for both shells. *Implemented, smoke-tested.*
- **Health-check entry point for scripts**: `scripts/health_check.py` (thin
  wrapper around `python -m patron health`). *Implemented, tested.*
- **Examples**: `examples/inference_example.py` (schema-validated inference from a
  JSON window), `examples/evaluate_example.py` (metrics for a checkpoint) and
  `examples/train_example.py` (small-budget training into a candidate run
  directory). *Implemented, exercised by tests.*
- **Release documentation**: `MODEL_CARD.md`, `DATASET_CARD.md`,
  `REPRODUCIBILITY.md`, `INSTALLATION.md`, `QUICKSTART.md`, `CONFIGURATION.md`,
  `LIMITATIONS.md`, `THIRD_PARTY_NOTICES.md`, `SECURITY.md`, `CONTRIBUTING.md`,
  `CODE_OF_CONDUCT.md`, `RELEASE_NOTES.md`. *Written against the actual code.*
- **CI and security workflows**: `.github/workflows/ci.yml` (tests, CLI smoke,
  packaging check, secret scan on Linux + Windows, Python 3.12),
  `security.yml` (secret scan, `pip check`, dependency inventory) and
  `release.yml` (tag-gated build; publishing requires a manual approval step).
  *Implemented; not yet exercised on GitHub.*
- **GitHub templates**: bug report, feature request, pull request template.
- **License**: Apache License 2.0 (`LICENSE`, `NOTICE`, `pyproject.toml` metadata).

### Changed

- `pyproject.toml`: version `1.0.0b1` (PEP 440 for `v1.0.0-beta.1`), Apache-2.0
  license, authors, classifiers, project URLs, and package-data for JSON schemas.
  The model version in `config/model_config.yaml` remains `1.0.0`.
- `patron.__version__` now reports `1.0.0b1`.
- `scripts/patron_service.py` delegates to the same runtime implementation as
  `python -m patron runtime …`; behaviour and flags are unchanged.
- Training window subsampling: `--train-stride` (added with the training audit)
  replaces the earlier behaviour where `--stride` halved validation and test too.
- README rewritten for the public release: install, quick start, CLI, contracts,
  runtime, testing, security, license, limitations.

### Fixed

- **Unsafe checkpoint loading**: `torch.load` now uses `weights_only=True` for
  CLI and library paths; unsafe unpickling requires an explicit, never-default
  `allow_unsafe=True`. Loading a non-checkpoint or a payload missing required keys
  raises a clear `ValueError`. *Tested.*
- **Tests depended on `Plan/`** (a local, non-distributed spec bundle) for the
  sample inference payload; the fixture now lives in `examples/` with a fallback.
- **Broken documentation links** to `Plan/` that would 404 on GitHub now point at
  `schemas/` and the in-repo reports.

### Security

- Repository scan for API keys, tokens, passwords, private paths and machine
  identifiers: no secrets found in tracked files or the single existing commit.
- `.env.example` added (dummy values only); `.env`, `*.pt`, `*.db`, `dist/` and
  virtualenvs are ignored by git.
- Runtime SQLite access is parameterised and the health check opens the database
  read-only.
- Inputs to the public inference path are JSON-Schema validated before use.

### Known limitations

- No published checkpoint and no bundled market dataset: the release is source +
  reproducible pipeline. Training a comparable model takes hours on CPU.
- The model's momentum head is at chance level and its quantile head has weak
  conditional location skill; the shipped barrier operating point sits below the
  majority-class accuracy. Details and measured numbers are in
  `reports/training_upgrade/training_audit.md` and `LIMITATIONS.md`.
- The autonomous runtime has been exercised live, but a live candidate promotion
  has not occurred; promotion/rejection/rollback paths are covered by tests with
  stand-in runners.
- CI workflows are committed but have not run on GitHub yet.
- No live trading, no order execution, no exchange credentials.

### Planned

- T2–T12 of `plan/ADVANCED_TRAINING_UPGRADE.md` (optimization, loss balancing,
  hard examples, regimes, patterns, error intelligence, continual training,
  experiment runner, evaluation and governance upgrades).
