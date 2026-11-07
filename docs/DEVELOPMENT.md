# Development recovery and verification

The recovery results below are the historical baseline. The current opt-in
synthetic patient/physician portal, full integrated test results, updated service
gates and secure account-provisioning commands are in [RELEASE_READINESS.md](RELEASE_READINESS.md).
Legacy mode remains unauthed; portal mode blocks those APIs, adds scoped cookie
sessions and uses dedicated `dev_*` tables. [FEATURE_INVENTORY.md](FEATURE_INVENTORY.md)
and the requirements matrix are the current capability checklist. No live inference
or clinical validation is established by mock tests.

The API now imports and completes normal startup with SQLite. All tables and all three FTS5 triggers are retained. Schema initialization uses `sqlite3.complete_statement` to preserve complete trigger bodies and executes each statement through the existing SQLAlchemy connection, without DBAPI `executescript`'s implicit commit. Repeated startup does not drop tables or erase records.

The UI boot no longer calls an absent `/api/env` endpoint. Changing patients selects a conversation belonging to the newly selected patient or clears the old selection. `run_local.sh` starts the actual `main:app` entrypoint, uses the activated Python environment, and keeps its default database outside the checkout. Qdrant ingestion uses typed points with deterministic UUID storage IDs while preserving full SHA-256 citation IDs in payloads. Reingesting identical chunks upserts instead of duplicating them. Existing Qdrant collections are not deleted or migrated.

## Reproduce locally

Use Python 3.12 (the runtime version family in runtime.txt) and activate a virtual environment. No package lock exists; requirements.txt is the Render runtime manifest. pyproject.toml alone omits packages imported by main.py.

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install torch==2.6.0+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip check
python -m unittest discover -s tests -v
node --test tests/frontend.test.cjs
bash run_local.sh
```

The cloud environment already has `/workspace/.raphael/venv`; activate it with `. /workspace/.raphael/venv/bin/activate`. Its reusable install script is saved in environment settings and `/workspace/.raphael/install.sh`. Node 24 is available for the dependency-free controller tests. No Electron installation is required for the browser/API workflow.

`run_local.sh` binds to loopback on PORT (default 8000). Its default SQLite file is `$XDG_STATE_HOME/raphael/raphael.db`, falling back to `$HOME/.local/state/raphael/raphael.db`. To keep all development state in the cloud workspace, set `XDG_STATE_HOME=/workspace/.raphael/state`. An explicit `DATABASE_URL` is preserved. Use a new synthetic-data path, never a production database. The SQLite schema is not a PostgreSQL migration system.

## Actual validation

- `python -m unittest discover -s tests -v`: **16 tests passed**, including the original failure reproduction, intact triggers, rollback behavior, repeated startup, temporary-file persistence, FTS insert/update/delete, feedback, route validation, actual OpenAI SDK calls over mock HTTP, structured synthetic answers, insufficient-evidence abstention, and local Qdrant ingestion/retrieval for all three ingestion paths with mocked embeddings and PubMed responses.
- The suite starts `bash run_local.sh` in a real subprocess twice, requests health, frontend assets, patient and conversation APIs over HTTP, verifies persistence after restart, and stops only those test process groups. This uses a temporary database and no production credentials.
- `node --test tests/frontend.test.cjs`: **3 tests passed** for real controller code with a minimal DOM double: boot/control binding and selection of the correct patient's conversations. This is not a real browser rendering or visual test.
- Dependency compatibility checks passed for the installed 72 packages. Installed third-party packages and existing application date handling emit deprecation/syntax warnings; they do not block startup.
- Authentication helper token creation/validation and missing/invalid-token rejection pass. **API routes currently have no authentication dependencies**; the tests explicitly record this limitation. There is no integrated login or enforced tenant isolation.
- No live LLM inference, external Qdrant server, model download, verified medical source ingestion, clinical evaluation, Electron window or real PACS/FHIR integration was validated.

## Feature inventory

| Area and files | Current capability | Limits |
| --- | --- | --- |
| main.py; database.py; feedback/store.py | SQLite patients, conversations, messages, text document FTS, feedback, health, active API | Basic records only; no longitudinal model, migration history or relationship/tenant enforcement |
| static/Raphael.html; app.js; styles.css; logo; standard.html; handsfree.html; manifest.json | Served browser chat shell, patient/conversation selection, upload/search; redirect pages | No login; remaining untrusted HTML interpolation; no browser-level E2E or voice implementation |
| main.py; corpus/indexer.py; pubmed/ingest.py; evidence/ingest.py | Evidence ingestion, Qdrant indexing/search, draft/critic JSON pipeline, citation-presence gate, low-evidence abstention | Mock/local validation only; source licensing, citation validity and clinical safety not established |
| corpus/fetch.py; registry.py; pubmed/client.py; classify.py; query_packs.py | URL/PDF extraction, source metadata registry, PubMed clients and heuristic classification | Registry is not an ingested verified corpus; URL destination controls and update safeguards need work |
| rag/*.py | Alternate typed RAG schemas, retrieval, prompts, citation and safety helpers | Legacy evidence-tier vocabulary differs from active main.py; unintegrated paths require contract tests |
| radiology/cant_miss.py; templates.py | Static study checklists, report formatting | Not image interpretation or clinically validated radiology reasoning |
| auth/jwt_auth.py; tenancy/tenant.py | JWT and tenant-header helpers | Unwired; default JWT secret is unsafe; no access control or tenant separation in active routes |
| integrations/fhir/smart.py; static/fhir-demo.html | SMART auth/token/GET helper scaffold and demo page | Demo endpoints absent from active API; real OAuth security/PKCE/state workflow untested |
| integrations/pacs/*.py | Local DICOM metadata and DICOMweb request helpers | No validation, PHI protection, pixel viewing or authenticated API workflow |
| integrations/billing/stripe_api.py; compliance/audit.py | Stripe checkout helper and stdout audit helper | Unwired; billing inactive; no immutable audit storage or compliance evidence |
| main.js; preload.js; renderer.js; index.html; package.json | Electron shell, isolated preload bridge and structured-output UI | Not launched; relative API URLs from file-based window need a backend origin design |
| popup.html; voice_test.py; data_ingest.py; guideline_samples.jsonl; example_patient.json | Extension/voice/ingest scaffolds and sample data | `/v1/intents` is absent; samples are not verified medical evidence; placeholder tests are not validation |
| eval/run_eval.py; eval_cases.json; SaMD-Validation.md | Legacy case evaluation skeleton and safety-document skeleton | Harness targets absent `/cases` and `/query` routes; no clinical benchmarks or validation |
| requirements.txt; pyproject.toml; runtime.txt; render.yaml; run_local.sh; deploy_render.sh | Dependency declarations, runtime pin, Render scaffold and corrected local startup | Manifest mismatch, no lock/CI; deploy_render.sh is only a message and its backend path is obsolete; no deployment performed |
| API-Reference.md; Pilot-Setup-Guide.md; README.md | Original prototype documentation | Legacy reference includes absent `/v1` and FHIR routes; use `/openapi.json` for current route contracts |

## Remaining steps for a development deployment

1. Review this branch and run the checks above. Local, synthetic-data API development is supported now.
2. Complete ROADMAP.md DEV-01/02 before sharing a network-accessible instance: authentication, authorization, tenant boundaries, safe HTML rendering and browser tests. Keep this build restricted to trusted local development until then.
3. Supply dedicated development `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`, `QDRANT_URL` and optional `QDRANT_API_KEY`. Choose the provider first so secrets and network destinations can be scoped correctly. Cache `EMBED_MODEL` (default `sentence-transformers/all-MiniLM-L6-v2`); ensure its dimension matches `EMBED_DIM` (default 384). Enter secrets securely, never in chat or source.
4. Execute DEV-03 with legally reusable verified sources and record real service results; complete DEV-04/05 release gates. Mock responses do not satisfy the live AI acceptance criterion.
5. Review and save the updated cloud install/start configuration, then publish the environment snapshot if desired. This does not deploy Raphael. Any application deployment remains a separate explicit approval and must use synthetic data and development credentials.

No production data was accessed or modified. No clinical validation, full test-release completion, merge or application deployment is claimed.
