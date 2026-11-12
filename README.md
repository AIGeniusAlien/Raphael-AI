# Raphael — comprehensive clinical copilot (synthetic development)

**Current release status: NOT AUTHORIZED for real patient care, real PHI, or public distribution.** Raphael's long-term scope covers medical reasoning, multispecialty imaging, clinical documentation, charting and hospital operations. Present implementations are bounded synthetic software features, not a clinically validated replacement for clinicians or hospital systems.

## Run the current app locally

From this repository on Python 3.12:

```bash
python -m dev_start --setup --provision
```

This provisions synthetic-only clinician/patient accounts, asks for account setup credentials through the existing bootstrap workflow, migrates a development SQLite database, and binds the FastAPI app to **127.0.0.1:8000** (not the public internet). For persistent local state, use `--database <private-synthetic-path>` at initial setup and reuse the same exact path at startup. By default live AI provider access is disabled; do not assume configured/validated diagnostic inference. Do **not** enter actual patient data.

Run all locally supported verification checks:

```bash
python -m dev_verify --all --require-browser --output <private-synthetic-results-directory>
```

`--require-browser` requires working Chromium/Playwright. GitHub native iOS simulator checks and dependency advisory audit remain separate CI evidence. See [controlled synthetic preview handoff](docs/CONTROLLED_SYNTHETIC_PREVIEW_HANDOFF.md) for the source-scoped release preflight and human signoff gates.

**Launch tracks are not interchangeable:** The local clinician/patient development app demonstrates broader synthetic workflows. `consumer_candidate` is a more restricted *patient records/account preview* only. Neither is a clinical deployment or approved unrestricted public release.

---

> Ultimate increment,2026-11-01: [Current implementation and release gates](docs/RAPHAEL_FINAL_LAUNCH_READINESS.md). Reports below retain their historical source scope.

Release readiness:see [the2026-10-28 source-scoped audit](docs/RC_RELEASE_AUDIT.md). Historical test/live reports do not imply TestFlight or clinical-production approval.

One-command synthetic setup: `python -m dev_start --setup --provision` (Python3.12). Automated verification: `python -m dev_verify --all --require-browser`. See [automated development](docs/AUTOMATED_DEVELOPMENT.md) and [current engineering report](docs/PRODUCTION_ENGINEERING_REPORT.md) for actual platform results and release blockers.

Raphael AI development prototype: FastAPI, SQLite, browser chat and evidence retrieval scaffolds.

See [development setup and tested limitations](docs/DEVELOPMENT.md), the
[master requirements](docs/RAPHAEL_MASTER_SPEC.md),
[requirements matrix](docs/REQUIREMENTS_MATRIX.md), and [roadmap](docs/ROADMAP.md).

The opt-in synthetic patient/physician portal and its account-provisioning steps
are described in [release readiness](docs/RELEASE_READINESS.md).
The [feature inventory](docs/FEATURE_INVENTORY.md) distinguishes present capabilities,
historical routes, mocks and missing clinical platform features.

This build supports local development with synthetic data. Portal mode adds scoped
development authentication and blocks legacy APIs. Legacy mode remains unauthenticated;
neither mode is approved for clinical use or public deployment.

The [core intelligence workspace](docs/CORE_INTELLIGENCE.md) connects scoped streaming conversations, browser voice controls and versioned consent-controlled encounter notes to the clinician portal. Live service and speech-validation limitations are explicit.

[Connected care](docs/CARE_JOURNEY.md) adds expiring patient-authorized intake snapshots and receiving-clinician review. [Infrastructure status](docs/INFRASTRUCTURE_STATUS.md) records the actual live-service blockers; [mobile decision](docs/MOBILE_ADR.md) distinguishes the working mobile web view from future native distribution.
