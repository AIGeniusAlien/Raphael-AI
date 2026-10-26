# Test evidence ledger

Only executed results belong in this file. Developer protocol tests, real local browser tests, external-service validation and independent clinical studies are different evidence classes. All development tests use synthetic data and temporary databases; no production credentials or patient data are needed.

## Previous verified checkpoint

Before the latest platform expansion, the branch recorded:

| Check | Actual historical outcome | Scope and limits |
| --- | --- | --- |
| `python -m unittest discover -s tests -v` | 36 passed | Recovery, portal authorization/session, synthetic persistence, FTS triggers, assessment contract and real main-app HTTP checks |
| `node --test tests/*.test.cjs` | 19 passed | Legacy and portal controller logic, safe literal rendering and stale session/chart/thread response guards |
| `python -m unittest discover -s e2e -v` | 1 passed | Real Chromium with separate patient/clinician contexts, internal messaging, note approval visibility, mobile layout and logout |
| Dependency compatibility | 74 packages compatible | Runtime plus optional Playwright tooling; no package verification bypass |

These counts are a baseline, not a claim that the latest added modules have passed a final integrated run.

## Current expansion validation

The current implementation adds account administration/throttling/migration, structured charts, secure document sharing, messaging review/status workflows, synthetic DICOM preview, sandbox FHIR adapters and runtime/operations work. Results below must be updated from the actual final runner output. The exact integrated outcome is pending while implementation agents finish their modules.

| Target | Required meaningful evidence | Current result |
| --- | --- | --- |
| `tests/test_dev_admin.py` | Scoped org-admin management, no admin chart privilege, disable/session revocation, self/final-admin protection, migration preservation, login throttling and expiry | 11 passed in the agent’s isolated run (2.936s); existing 14 portal tests also passed (5.450s). Final integrated run remains separate. |
| `tests/test_dev_chart.py` | Typed chart/encounter entries, lab trends/ranges, draft publication, document authorization and bounded PDF/text handling, privacy/consent and persistence | 11 passed in the agent’s isolated run (5.504s); JavaScript syntax check passed. Real-browser/main-app validation remains separate. |
| `tests/test_dev_messaging.py` | Authorized inbox/receipts/attachments, manual drafts and explicit approval/send, at-most-once concurrency and no external delivery | Not yet recorded in this ledger |
| `tests/test_dev_imaging.py` | Actual valid synthetic DICOM metadata/pixels, bounded decoding, authorized image download/report publication, invalid/compressed/oversized inputs | 10 passed in the agent’s isolated run (2.408s), including atomic duplicate handling. No diagnostic model tested. |
| `tests/test_fhir_sandbox.py` | Configured destination validation, synthetic FHIR resources/provenance, failure handling and actual local HTTP fixture | 6 passed in the agent’s isolated run (0.719s), including actual local HTTP fixture and traversal/resource-reference denial. No external sandbox tested. |
| Runtime/operations and full legacy suite | Real normal startup, extension installation, readiness reporting, repeat migration/persistence and compatibility regressions | Not yet recorded in this ledger |
| Integrated browser scenarios | Actual new UI flows, role permissions, errors and no fabricated service success | Not yet recorded in this ledger |

## Service and clinical exclusions

- OpenAI SDK mock HTTP is not a real development inference request. No dedicated LLM credential has been supplied through this task.
- Local Qdrant memory with synthetic embeddings/PubMed fixtures does not validate an external Qdrant server, real model cache, licensed corpus or source-supporting citations.
- A synthetic DICOM PNG preview is pixel viewing, not diagnostic image interpretation or validation of an imaging model.
- A local FHIR HTTP fixture or mock transport is not a live external SMART/EHR service, authorization flow or production writeback.
- Deterministic transcript notes are not AI summaries. Internal stored/read status is not provider-confirmed email/SMS/EHR delivery.
- Clinical diagnostic accuracy, medication safety, transcription quality, subgroup/bias analysis, regulatory compliance and independent clinical/security validation have not been established by developer tests.

Failed, skipped, unrun and blocked outcomes must remain distinct. Do not replace the final run with a zero-test success or assume a PID/open port proves readiness.


## Current execution evidence and limits

95 Python backend/integration tests passed in57.106s.19 Node controller tests passed. The expanded browser run initially failed on an obsolete error-message assertion and Playwright string evaluation prohibited by CSP; assertions were corrected without weakening CSP and rerun results are recorded below. CI workflow `.github/workflows/development.yml` was added but has not run on GitHub. `run_staging.sh` and Render blueprint are prepared, not deployed; automatic Render deployment is disabled. `/readyz` checks development database/schema readiness, not clinical release readiness.

Actual dependency audit found reported advisories in ecdsa,filelock,pillow,pydicom,pypdf,python-dotenv,python-jose,python-multipart,sentence-transformers,setuptools,starlette and transformers. Dependency remediation and rerunning file/model security regressions are release blockers; no clean vulnerability scan or public-deployment readiness is claimed. Full local audit artifact: /workspace/.raphael/vulnerability-audit.json. Existing component tests pass but do not negate these advisories.

New modules:dev_chart.py,dev_messaging.py,dev_imaging.py,dev_knowledge.py,dev_runtime.py and integrations/fhir/sandbox.py. Local evidence is genuinely indexed SQLiteFTS5 over manually supplied reviewed excerpts; it is not independently verified medical evidence or embeddings. Live OpenAI/Qdrant/FHIR, voice,postgres migrations, full roles,caregiver,external clinical messaging and clinical validation remain incomplete.

Final corrected Chromium run:2 tests passed in29.088s. Expanded platform scenario covers internal draft/edit/approval/send, administration and actual DICOM preview/report review; the preserved original portal scenario also passed. No external clinical service was validated.
