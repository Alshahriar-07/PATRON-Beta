## What changed

<!-- One paragraph. Link the issue if there is one. -->

## Why

<!-- The behaviour or defect this addresses. -->

## How it was verified

<!-- Paste the commands you ran and their result. "Tests pass" is not enough: say which. -->

- [ ] `python -m pytest` — result:
- [ ] `python -m patron health` — result:
- [ ] `python scripts/scan_release.py` — result:

## Contracts

- [ ] Model architecture / parameter count unchanged (272,618)
- [ ] 96 × 12 input contract and output schema unchanged
- [ ] No look-ahead introduced (features, labels, sample weights, regimes)
- [ ] No dataset, checkpoint, database, log or secret added to the repository
- [ ] Champion checkpoint untouched; new training in `experiments/`

If any box cannot be ticked, explain why in this description.

## Notes for reviewers

<!-- Known limitations, follow-up work, or decisions you want a second opinion on. -->
