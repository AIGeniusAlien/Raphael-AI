# Development build and operations

This guide is for an explicitly isolated **synthetic-data development instance**. It is not an instruction or approval to deploy a clinical service, expose production data, merge a protected branch, activate billing or send external clinical messages. Full release gates are in RELEASE_READINESS.md and REQUIREMENTS_MATRIX.md. The complete platform remains incomplete.

## Retained cloud environment

Work in `/workspace/Raphael-AI` on the current development branch. Do not create a Git worktree. The prepared Python 3.12 virtual environment is `/workspace/.raphael/venv`; the reviewed setup script in environment settings refreshes runtime and optional browser-test dependencies. Installed processes do not survive publication; retained packages/files do. Saving setup instructions or environment requirements does not execute them, inject credentials or deploy Raphael.

```bash
cd /workspace/Raphael-AI
. /workspace/.raphael/venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/workspace/.cache/huggingface
export RAPHAEL_DEV_PORTAL=1
export RAPHAEL_DEV_SYNTHETIC_ONLY=1
mkdir -p /workspace/.raphael/state
export DATABASE_URL=sqlite:////workspace/.raphael/state/portal-synthetic.db
```

Choose a new dedicated synthetic SQLite file. Never point this workflow at an existing patient database. The portal refuses populated legacy tables before initialization, but neither this guard nor the operator flag can identify real information pasted into a synthetic chart. SQLite is the supported portal persistence backend in this increment; do not substitute PostgreSQL and claim migration support.

Create accounts once using private password prompts; do not paste passwords into source, terminal command arguments, chat or logs. The bootstrap command rejects duplicate usernames instead of overwriting credentials or records.

```bash
python -m auth.dev_bootstrap --database-url "$DATABASE_URL" --tenant synthetic --username dev-clinician --role clinician
python -m auth.dev_bootstrap --database-url "$DATABASE_URL" --tenant synthetic --username dev-patient --role patient
# Available after the org-admin implementation/migration has passed the current checks:
python -m auth.dev_bootstrap --database-url "$DATABASE_URL" --tenant synthetic --username dev-admin --role org_admin
bash run_local.sh
```

The local entrypoint keeps the service on loopback, preserves an explicit DATABASE_URL and uses PORT (default 8000). With no explicit database URL, portal mode chooses a separate `portal-synthetic.db`; legacy mode chooses `raphael.db`. The portal root selects the role-aware interface and denies legacy unauthenticated `/api` endpoints. Leave legacy mode restricted to trusted local compatibility testing.

## Build and verification commands

```bash
python -m pip check
python -m unittest discover -s tests -v
node --test tests/*.test.cjs
# Optional: install requirements-dev.txt and use a local Chromium executable.
python -m unittest discover -s e2e -v
```

On another development machine, create a Python 3.12 virtual environment, install CPU PyTorch from its verified official package index and `requirements.txt`, then optionally `requirements-dev.txt`. The prepared cloud machine already has those packages; do not install Electron for the browser/API workflow. Use the final TEST_RESULTS.md ledger to distinguish executed checks, skipped optional checks and unavailable services.

## Dedicated external service configuration

The environment draft declares `LLM_API_KEY` scoped to `api.openai.com`, a suggested `LLM_BASE_URL=https://api.openai.com/v1`, suggested `LLM_MODEL=gpt-4o-mini`, and an unset `QDRANT_URL` requirement. These are requirement metadata, not supplied secrets or successful integration evidence. Inspect existing binding names/presence before requesting additional values. Supply only dedicated development values securely in environment settings. A proxy secret may be a placeholder locally and must use its supported configured HTTPS destination.

For live assessments, additionally configure the development Qdrant URL and optional QDRANT_API_KEY, cache EMBED_MODEL with matching EMBED_DIM, ingest reviewed legally reusable source excerpts, and record both source/provenance and real retrieval/LLM results. Do not broaden unknown network policies. Citation-ID membership and schema validation do not establish medical factuality or licensing. Missing services must report unavailable status instead of returning runtime mock answers.

FHIR preview/import uses an explicitly configured authorized sandbox destination and synthetic resources only. No arbitrary patient-supplied URL or production EHR credentials should be accepted. An HTTP test fixture or mock transport validates adapter behavior, not a live external EHR connection. External email/SMS/EHR delivery and live billing remain disabled unless separately configured, tested and explicitly authorized.

## Migrations and data preservation

Keep a verified backup of the dedicated synthetic SQLite file before schema upgrades. Development schema initialization is additive/repeatable; migrations that rebuild a constrained account table must preserve user IDs, password hashes, sessions and chart relationships and validate foreign keys. A migration test is not a PostgreSQL migration system or disaster-recovery certification. Never delete/recreate a populated database to make setup pass.

To recover a failed development upgrade, stop only the process started for this instance, preserve the failed database for diagnosis, restore the last verified synthetic backup to a separate path, select the known-good source revision and point DATABASE_URL at that restored copy. Re-run integrity, startup, account authorization and persistent-record checks before use. Do not silently reset production or overwrite the original data. Record actual backup/restore/rollback tests separately in TEST_RESULTS.md.

## Staging and Render boundary

A Render-compatible configuration/build artifact is a development/staging preparation artifact, not a performed deployment. Review the final render.yaml and environment validation commands before using it. Keep automatic deployment disabled for the current review branch. Public/shared hosting is not approved until the residual identity, TLS, consent, operational controls and external-service gates have been reviewed. Local cookie sessions use HttpOnly/SameSite controls; HTTPS must set Secure for any explicitly authorized hosted instance.

Do not add real patient accounts, production secrets, live charges or real clinical transports to a demonstration. There is no claim of independent security review, HIPAA/BAA readiness, FDA authorization, applicable IRB clearance or clinical validation. Publication of the cloud snapshot is separate from application deployment.


## Current staging entrypoint
Use activated Python3.12 with requirements.txt installed. Set RAPHAEL_DEV_PORTAL=1,RAPHAEL_DEV_SYNTHETIC_ONLY=1 and DATABASE_URL to a new dedicated synthetic SQLite file. Provision separate org_admin,clinician,patient accounts using `python -m auth.dev_bootstrap --database-url "$DATABASE_URL" --tenant synthetic --username <unique-name> --role <role>`; passwords are prompted privately. Run `bash run_staging.sh` on loopback. `python -m dev_runtime migrate --database-url "$DATABASE_URL"` is repeatable; `python -m dev_runtime backup --database-url "$DATABASE_URL" --output <new-file>` creates an integrity-checked copy and refuses overwriting. Restore only into a new file with matching code and stopped processes; do not overwrite an original database. Render blueprint autoDeploy is false; configure persistent storage explicitly or expect data loss on platform redeploy. No deployment was performed.


## Integrated intelligence workspace
The existing start/secure bootstrap commands remain valid. Automatic dev_runtime migrations preserve data and add intelligence tables. Open the clinician portal link to `/static/intelligence.html`, optionally with `?patient=<authorized-id>`. Provide HTTPS LLM_BASE_URL plus LLM_API_KEY securely, approve locally owned/legally usable excerpts in the evidence workspace, then ask a matching question. Streaming uses the approved local SQLite index directly; structured assessments need RAPHAEL_EVIDENCE_BACKEND=sqlite or a configured Qdrant service. No live credentials were present in this build. Follow CORE_INTELLIGENCE.md to create an encounter,confirm consent,append a labeled synthetic transcript,edit the draft and explicitly approve. Browser speech requires actual permission/support and is not validated by fixture tests. No source rewrite is required. No persistent/browser preview is published in this onboarding environment; local Chromium tests exercise the real UI.


## Patient-to-clinician care and mobile preview
Use the same secure synthetic bootstrap and `bash run_local.sh` workflow. Add a distinct receiving-clinician account through the private-password CLI. On the authenticated portal choose **Connected intake and emergency information**. Follow CARE_JOURNEY.md in separate patient/receiver browser sessions. The mobile web client uses the same backend/cookie;resize to390px or use a real HTTPS mobile test host when separately authorized. A web manifest/home-screen metadata does not establish native installation or App Store approval. No sensitive data is cached offline. No hosted preview is published by this onboarding tool;real local Chromium tests exercise the application.
