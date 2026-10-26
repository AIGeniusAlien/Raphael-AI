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
