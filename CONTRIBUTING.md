# Contributing to PATRON

Thanks for taking the time to contribute. PATRON is an experimental research
project, so correctness and honest reporting matter more than feature count.

## Ground rules

- **Do not change the frozen contracts casually.** The 12 input features, 96-candle
  window, six-bar horizon, triple-barrier definition, quantile levels, momentum
  bin boundaries, four output heads and 272,618-parameter architecture are
  versioned. A change needs a written proposal, an ablation, and a new model
  version — not a drive-by edit.
- **No look-ahead.** Features, labels, regime labels and sample weights must be
  computable at the decision time. Add a test if you touch any of them.
- **No fabricated results.** Report what a run produced, with its split, seed
  count and dataset hash. Negative results belong in the changelog too.
- **No secrets, no private data.** Market datasets, checkpoints, runtime
  databases and logs are not committed (see `.gitignore`).
- Be civil: see `CODE_OF_CONDUCT.md`.

## Development setup

```bash
python -m venv .venv
# Windows
./.venv/Scripts/pip install -r requirements.txt
./.venv/Scripts/pip install -e . --no-deps
./.venv/Scripts/python.exe -m patron health
# Linux / macOS
./.venv/bin/pip install -r requirements.txt
./.venv/bin/pip install -e . --no-deps
./.venv/bin/python -m patron health
```

Python 3.12 is required. CPU-only PyTorch is sufficient; `requirements.txt` pins
`torch==2.4.1` because newer CPU wheels fail on the reference Windows host.

## Before opening a pull request

```bash
python -m pytest                 # full suite, synthetic fixtures only
python -m patron health          # environment and contract checks
python scripts/scan_release.py   # secrets / private paths / artifact scan
python -m compileall -q src      # syntax check
```

Guidelines:

1. Add or update a test for behaviour changes. Tests must not require network
   access or the real market dataset.
2. Keep the public CLI stable; add flags rather than renaming them.
3. Log with the `patron.<module>` logger, never `print` inside library code.
4. No new runtime dependency without a justification in the PR description.
5. If you change training or evaluation, include the command, seed(s), budget and
   validation numbers in the PR. "Training loss went down" is not evidence.

## Areas where help is welcome

- Reviewing the training audit (`reports/training_upgrade/training_audit.md`) and
  the T2–T12 plan (`plan/ADVANCED_TRAINING_UPGRADE.md`).
- Reproducing a run on Linux/macOS and reporting timing.
- Additional leakage and causality tests.
- Documentation fixes where the docs and the code disagree — the code wins; tell
  us which sentence is wrong.

## Commit and PR style

- Present tense, imperative subject ("Add safe checkpoint loading").
- Explain *why* in the body; link the issue or the failing behaviour.
- Keep unrelated refactors out of functional PRs.
- One approval from a maintainer is required. Release tags are cut by a
  maintainer only (see `REPRODUCIBILITY.md`).

## Reporting security issues

Do **not** open a public issue; follow `SECURITY.md`.
