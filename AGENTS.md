# Raphael engineering instructions

- Read docs/RAPHAEL_MASTER_SPEC.md and relevant rows of docs/REQUIREMENTS_MATRIX.md before changes. Keep the matrix and docs/ROADMAP.md consistent with actual implementation and test evidence.
- Reconcile new/historical requirements additively without removing old IDs. Maintain docs/FEATURE_INVENTORY.md and docs/RELEASE_READINESS.md; distinguish local/mocked tests, live service results and independent clinical/regulatory validation.
- Preserve existing working functionality and APIs. Implement complete, testable features; label incomplete scaffolding and mock integrations plainly.
- Add meaningful automated tests and regression coverage. Run `python -m unittest discover -s tests -v` and `node --test tests/frontend.test.cjs` from the repository root for recovery changes. Record what ran, passed, failed, or remained untested.
- Use temporary databases and synthetic data in development and tests. Never access, erase or modify production patient data for testing.
- Never fabricate citations, source verification, clinical validation or deployment-readiness claims. Require clinician oversight and confirmation of diagnostic outputs; never enable autonomous clinical decisions.
- Never put production secrets in source code, test fixtures or logs. Use scoped development credentials via secure settings.
- Do not deploy to production, merge changes, activate billing or publish clinical claims without explicit approval.
- Cloud tasks are already isolated. Use the existing checkout and do not create a Git worktree unless explicitly requested.
- Portal work is opt-in with RAPHAEL_DEV_PORTAL=1 and RAPHAEL_DEV_SYNTHETIC_ONLY=1. Use a new dedicated synthetic SQLite DB and locally provisioned accounts, never default/shared credentials. Keep legacy APIs blocked in portal mode and never equate internal message storage with external delivery.
