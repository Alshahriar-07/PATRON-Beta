# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| `1.0.0-beta.1` (current testing release) | ✅ security fixes |
| `1.0.0` development snapshots before the beta | ❌ |
| Anything older / unversioned forks | ❌ |

PATRON is an experimental research system. No version is approved for live trading
or unattended operation with real funds.

## Reporting a vulnerability

Please **do not open a public issue** for security problems.

1. Preferred: GitHub → *Security* → *Advisories* → *Report a vulnerability*
   (private advisory on this repository).
2. Alternative: email **alshahriarsayon425@gmail.com** with the subject
   `PATRON security`.

Include: affected version/commit, operating system, a minimal reproduction, the
impact you believe it has, and whether you want to be credited.

**What to expect**

| Stage | Target |
|---|---|
| Acknowledgement | within 72 hours |
| Triage and severity assessment | within 7 days |
| Fix or mitigation plan | within 30 days for high/critical, best effort otherwise |
| Public disclosure | coordinated, normally 90 days after report or after a fix ships |

There is no bug bounty. Reporters are credited in the advisory and CHANGELOG
unless they ask otherwise.

## Scope

In scope:

- Code execution, path traversal or file overwrite through the CLI, inference
  input handling, runtime service, dataset/report writers, or the release builder.
- Unsafe deserialisation of model checkpoints or cached data.
- SQL injection or state corruption in the runtime SQLite store.
- Credential or personal-data exposure in code, logs, reports or release artifacts.
- Dependency vulnerabilities reachable through PATRON's documented usage.

Out of scope:

- Financial losses, forecast accuracy, or market outcomes. PATRON produces
  probabilistic estimates, not advice; see `LIMITATIONS.md`.
- Vulnerabilities requiring an attacker who already controls the machine, the
  Python environment, or the user's checkpoint files.
- Denial of service through deliberately huge local input files.
- Third-party services (GitHub, Binance, PyPI) outside this repository.

## Security posture of this release

Implemented in code:

- **Safe checkpoint loading.** `patron.training.trainer.load_checkpoint` uses
  `torch.load(..., weights_only=True)`. Full unpickling requires an explicit
  `allow_unsafe=True`, which no CLI path sets. The loader also verifies the
  payload is a dict with `model_state`, `config`, `feature_order`, `scaler`.
- **Strict input contract.** Inference payloads are validated against
  `schemas/input.schema.json` plus semantic checks (96 bars, 15-minute spacing,
  monotonic timestamps, no duplicates, no future candles, finite OHLCV).
- **No credentials required.** The project reads only public market data. There is
  no API-key handling, no order execution and no outbound write access.
- **Parameterised SQL.** All runtime queries bind values; dynamic identifiers come
  from fixed internal tuples, never from user input.
- **Subprocess use is explicit.** The runtime spawns `sys.executable` with an
  argument list (no shell), from a command built in code, and writes the child
  log to a fixed path.
- **No secrets in the repository or its git history** (checked with the release
  scanner in `scripts/scan_release.py`; see `reports/security/security_audit.md`).

Known limitations, stated rather than hidden:

- The runtime has **no network authentication layer** because it exposes no
  network service; it writes JSON files and a SQLite database locally. Do not put
  it behind a public HTTP server without adding authentication.
- Checkpoints are trusted artifacts in the sense that a *valid* checkpoint is
  loaded into a model and executed; safe loading removes code execution, not the
  possibility of a model that behaves badly.
- Dependency advisories are not automatically enforced in CI. `pip check` and the
  secret scanner run; a full CVE audit is a manual step (`pip-audit`).
- No fuzzing or formal verification has been performed.

## If you suspect a leaked secret

PATRON needs none. If you nonetheless committed a key to your fork: revoke it at
the provider first, then remove it from history (`git filter-repo`), then rotate
anything derived from it. Deleting the file in a new commit does not remove it
from history.
