## Current production-engineering increment

Read PRODUCTION_ENGINEERING_REPORT.md and AUTOMATED_DEVELOPMENT.md first. New one-command verification/startup,request budgets/CSRF/header safeguards,safety protocol evaluation,native readiness utility,CI and compliance evidence are integrated. Local189Python/26frontend/6Chromium/5mobile source tests pass. Native compilation,Windows/macOS runtime and clinical validation are not established. Source assets preserved;no merge/deploy. Earlier results below are historical checkpoints.

## Current verified recovery result — 2026-10-26

See [recovery implementation report](RECOVERY_IMPLEMENTATION_REPORT.md) for exact owner commands and next actions,and IMPLEMENTATION_REPORT.json/RECOVERY_LIVE_RESULTS.json for current results. Python181/181,frontend26/26,Chromium6/6 and mobileLinux5/5passed. Actual physician AND patient Chromium workflows completed with realOpenAI/Qdrant,384-dimensional text-embedding-3-small,6CDCdocuments/31vectors,repeat0writes,resolving citations,fresh original-source checks and automated support screens passed. Earlier failed screens are retained in RECOVERY_LIVE_ATTEMPTS.json. This does not establish clinical accuracy. Chart grants/revocation,patient AI and account controls are integrated. Account deletion is review-only;iOS source exists but Xcode/device/signing unverified;Android native absent. No staging deployment,AppStore readiness,merge or production-data claim. All1,112prior requirement IDs/descriptions preserved;machine ledger now1,117rows.

Earlier checkpoint text below is historical and may describe resolved infrastructure or feature gaps;use the current report first.

Current live medical intelligence result:real physician Chromium conversation passed with actualOpenAI,Qdrantandhosted384-dimensional embeddings;3CDCdocuments/20vectors,repeat ingestion0writes,sixverified excerpts andresolving citations. LocalMiniLMload failed;hostedfallback is explicit andmodel-space guards preserve data. Source membership is not clinical factuality/population validation;the response remains unapproved. See LIVE_MEDICAL_INTELLIGENCE.md and resultJSONs;earlierblocked runtime observations below are historical.

Current live verification:real OpenAI authentication,model access and streaming passed;real Qdrant collection access passed overHTTPS443. Pinned SDK port6333fallback caused the initial403 and is corrected in all existing client factories. Target raphael_evidence_v1 is absent. Actual embedding verification and medical ingestion remain blocked byProxyError fetching Hugging Face/CDC;network changes were saved for publication/reconnect. No remote collection mutation,corpus or grounded medical-conversation claim. See INFRASTRUCTURE_STATUS.md;older401/403credential observations below are historical.

Current connected-care sprint:all five provider variables are injected;OpenAI401 invalid-key indication,Qdrant403Forbidden. No live ingestion/retrieval/citation success. Evidence snapshots now persist and completed intake AI responses can be explicitly staged,edited,saved and approved as care summaries with source provenance. Consent is rechecked before initial provider transmission. See CONNECTED_RELEASE_CANDIDATE.md and INFRASTRUCTURE_STATUS.md;earlier runtime checkpoints below are historical. No full-product/native/mobile-device/staging readiness claim.

Published runtime checkpoint (2026-10-26): four required variables now present; QDRANT_API_KEY absent. A real bounded OpenAI request returned401; streaming/model access unverified. No Qdrant request/mutation occurred. See [INFRASTRUCTURE_STATUS.md](INFRASTRUCTURE_STATUS.md) for current evidence and exact secure binding actions; earlier missing-binding reports below are historical. No full product,live retrieval or deployment-readiness claim.

# Development roadmap

The immediate recovery is a local synthetic-data development build. It is not a clinically validated product or an approved clinical deployment. Feature IDs and acceptance gates are in REQUIREMENTS_MATRIX.md. Mocked integrations cannot satisfy live-service gates.

| Milestone | Deliverable and exit criteria | Dependencies / scoped implementation tasks |
| --- | --- | --- |
| M1: Working development application | Recovery tests pass; browser works; dev identity and authorization work; synthetic patients and conversations persist; dedicated development LLM and evidence retrieval work end to end. | REC-001–006 complete. Finish DEV-01–05 below before claiming the full test release. |
| M2: Evidence-grounded medical intelligence | Licensed/public source acquisition, provenance, verified citations, hybrid retrieval and reranking, source updates and conservative structured answers tested on reviewed fixtures. | MK requirements, CI-001, CI-006. Build on M1 live-service verification; do not treat presence-only citation checks as verification. |
| M3: Advanced diagnostic reasoning | Differential reasoning, supporting/contradictory evidence, uncertainty, red flags, deterministic safety checks and clinician oversight evaluated. | CI and CS requirements. Requires M2 sources and clinical evaluation design. |
| M4: Multimodal imaging, voice, and clinical integrations | DICOM/PACS, validated image models, voice transcription, reviewed documentation and SMART-on-FHIR workflows with provenance. | RI, PI, VD requirements. Sandbox integrations can proceed alongside M2/M3 after M1 identity boundaries. |
| M5: Clinical evaluation and security hardening | Expert-reviewed benchmarks, red-team tests, tenant isolation, RBAC, encryption, backups, monitoring, reproducible evaluation and clinical safety review. | ER and EP requirements. Evaluation design and identity hardening start in M1; final gate needs M2–M4 integration results. |
| M6: Regulated clinical deployment readiness | Regulatory and risk-management documentation, external validation, recovery exercises, operational controls and explicit deployment approval. | Completed M5 gates, intended-use definition, relevant jurisdictional review. No automatic deployment or autonomous clinical decisions. |

Next implementation tasks, in priority order:

1. **DEV-01 — Development identity and access enforcement (EP-001/002, PI-008).** Design a dev-only identity source with no default shared secret; wire auth dependencies and tenant boundaries into APIs and UI; test missing/invalid/expired credentials, cross-tenant requests and role denial. Existing JWT helpers alone do not satisfy this task.
2. **DEV-02 — Browser safety and realistic browser tests (REL-002).** Replace remaining untrusted `innerHTML` interpolation for patient names, conversation titles and document search results. Add browser automation covering navigation, creation, upload, chat, reload and API failures. Current Node tests use a minimal DOM double.
3. **DEV-03 — Dedicated service verification (REL-006/007/008).** Configure an approved development LLM provider, Qdrant instance and embedding model cache; ingest verified, legally reusable source excerpts; verify draft/critic outputs, citation membership and uncertainty with live development services. Never use production credentials or represent local synthetic Qdrant and mock HTTP as live functionality.
4. **DEV-04 — Input, persistence and output contracts (PI, CI, MK).** Add typed API inputs, relationship validation, migrations, consistent evidence tiers between legacy `rag/` and active `main.py`, verified source identities, source-count semantics, schema-constrained model outputs, robust failure handling and cross-patient tests. General chat currently bypasses evidence grounding. Preserve existing APIs with compatibility tests.
5. **DEV-05 — Reproducible release checks (EP-010, ER-010/011).** Align pyproject/requirements runtime dependencies, establish a reviewed dependency lock and CI, replace the obsolete `/cases` evaluation harness with current API evaluation, add dedicated-service smoke gates, document and review a development release candidate. Electron and FHIR demo pages currently have integration gaps.

Later implementation tasks should be split by each matrix ID, with explicit acceptance tests, source licensing and clinician review where relevant. Maintain statuses as evidence changes. Publishing a cloud environment is separate from deploying the application.


# Expanded reconciliation and release sequencing

All six milestones and DEV-01–05 above remain required. The original next-task descriptions are retained as planning history; the table below records the expanded release work and its dependencies. Basic synthetic portal implementation does not fulfill the complete PP/PH/MSG requirement groups.

| Increment | Requirement IDs | Exit evidence | Dependencies and remaining work |
| --- | --- | --- | --- |
| R1 — Retain recovery | REC-001–006, REL-001/005/009 | Normal startup, health, FTS triggers and synthetic persistence still pass | Do not discard existing recovery fixes; all follow-on changes rerun the relevant suites |
| R2 — Isolated synthetic accounts and portals | TEST-002/003/008/013/014; PP-001/003; PH-001/002; SEC-001/002/004 | Opt-in development bootstrap, session login/logout, authorized assigned chart access, role/cross-tenant denial tests, safe UI rendering | No real registration, identity proofing, production credentials or real-patient data; full PP/PH security remains M5 |
| R3 — Internal synthetic messaging | TEST-009; MSG-001–004/006/008/011/012/014/016/017 | Two synthetic users share authorized durable threads, unauthorized recipients are rejected, AI drafts require explicit clinician approval | Internal storage only; delivery services, routing, attachments, notifications, urgent policies and EHR integration stay open |
| R4 — Draft clinical documentation | TEST-010; PH-012/015; VD-003/006 | Draft/save/review behavior and patient visibility restrictions tested with synthetic content | No autonomous orders, external writeback or implied clinical validation |
| R5 — Evidence contract hardening | TEST-007; KNOW-021/026; CI-006; MK-009 | Structured output and citation membership reject malformed or unknown references; low-evidence abstention tests pass | Valid schema and membership do not prove clinical factuality, source licensing, comprehensive coverage or citation support |
| R6 — Live development inference and verified retrieval | TEST-005/006; REL-006/007; KNOW-017–026 | Dedicated development OpenAI/compatible provider and Qdrant/model execute real requests; reviewed reusable sources and citations independently verified | Requires scoped bindings/model downloads/provider choice; no production credentials; remain blocked until actual evidence |
| R7 — Reviewable release package | TEST-011/012; GOV-001–012; SEC-019 | Complete evidence report, documentation consistency checks and authorized draft PR | GitHub API access is restored; draft PR creation must be confirmed separately; no auto merge/deploy |

## Platform implementation after the first test release

- **M2:** complete KNOW/MK source licensing/provenance/versioning/freshness/deduplication/coverage and retrieval evaluation; restore direct PDF ingestion safely (HIST-003/004/022).
- **M3:** complete ADV/CI reasoning, patient-specific applicability, contradiction and guideline comparison, medication/interaction/contraindication safeguards, clinician-reviewed order drafts and coding (PH-013/014, HIST-011). Each output capability needs specialist-reviewed fixtures.
- **M4:** expand PP/PH beyond basic records: longitudinal timelines, labs, meds, allergies, reports, summaries, appointments, sharing, caregiver and care-team workflows. Complete MSG attachments/routing/notifications/urgent policies/EHR transport and clear provider-confirmed delivery semantics. Implement MM/RI/VD/PACS/FHIR/desktop/voice features with sandbox services and synthetic multimodal fixtures; preserve HIST-005–020.
- **M5:** complete identity verification, account recovery, consent, RBAC/ABAC, independent tenant/security review, audit integrity, operations, backup/restore, evaluation/clinical note quality and bias/subgroup studies; replace historical evaluation harness (HIST-023); permit billing integration only with explicit separate approval.
- **M6:** intended-use and jurisdictional regulatory assessment, HIPAA/BAA and applicable IRB review, independent clinical validation, operational release controls and explicit deployment approval. Plan Kubernetes only after a reviewed deployment architecture (HIST-021), not as an untested manifest claim.

Requirements may proceed in parallel once their dependencies and ownership boundaries are explicit. Security/evaluation design begins early; clinical deployment completion cannot be inferred from an M1 demonstration. Every unmet or blocked requirement stays visible in REQUIREMENTS_MATRIX.md.


# Current dependency-aware implementation checkpoint

The complete-platform directive is reconciled additively in the matrix. All earlier milestones, task IDs and unfinished requirements remain required; prior R/DEV entries describe their original checkpoints. The active increment implements the next dependency-safe portions rather than restarting the portal.

| Increment | Implemented files / acceptance IDs | Exit evidence and next dependent work |
| --- | --- | --- |
| P1 — Platform identity/admin | dev_portal.py; auth/dev_bootstrap.py; tests/test_dev_admin.py; DBX/ADX/UIX | Account-table migration, tenant-admin lifecycle and persistent throttling. Validate integrated admin UI before FLOW-008 is complete; verified medical identity, additional staff/radiologist/platform roles and recovery remain open. |
| P2 — Authorized longitudinal chart | dev_chart.py; static/chart.*; PTX/PHX/DBX | Typed records, encounter edits, range trends, approved plans/reports, appointments and protected documents. Finish exact role/privacy/browser checks; no clinical interpretation or full consent/proxy claim. |
| P3 — Review-before-send inbox | dev_messaging.py; static/inbox.*; MSX/FLOW-002 | Internal receipts, attachments, manual response drafting/editing/approval/explicit send. Concurrency must yield at most one message; urgent/external transport remains disabled. |
| P4 — Synthetic images and FHIR adapter | dev_imaging.py; integrations/fhir/sandbox.py; IMX/FHX/FLOW-006/007 | Valid pixel preview and approved reports can satisfy bounded viewing; adapter fixture tests do not satisfy a real external sandbox/SMART integration gate. |
| P5 — Repeatable runtime/release package | dev_runtime.py; CI/runtime/build artifacts; OPX/ACCEPT | Verify migration/backup/status/startup and current-run unit/controller/browser suites. No automatic staging/production deployment or branch merge. |
| P6 — Real evidence/inference demonstration | AIX/KNX/FLOW-003/004/005 | Dedicated LLM/Qdrant/model configuration, reviewed legally reusable sources, resolving/supporting citations and AI-assisted editable drafts. This remains blocked until actual service/evidence prerequisites and meaningful results exist. |
| P7 — Remaining complete platform | All unmet UIX–OPX and existing CI–GOV rows | Continue role/consent/proxy/security, clinical safety/coding/order contracts, sources/freshness/coverage, audio providers, SMART/OAuth/PACS, PostgreSQL/operations, test billing and independent evaluation. Keep each missing/blocked criterion visible; M1 is not clinical readiness. |

Use CONTINUATION.md for exact resume commands and the next executable integrated acceptance scenarios. Test evidence and source/credential scope, not a generic “implemented” label, determine release readiness.


## Completed core intelligence increment
CORE-001/002/004/005 have bounded tested implementations:provider stream transport with scoped persistence,consent-controlled versioned template notes and explicit patient-note publication. CORE-003 has real browser API wiring with fixture tests,not live speech validation. CORE-006 reuses structured assessment with provider configuration still required. CORE-007 chat attachments remains unimplemented. Do not repeat this implementation when continuing; validate actual configured providers and extend attachment/audio/AI note workflows without overwriting clinician edits.


## Combined mission priorities
Preserve the complete global continuum-of-care mission in the appended directives. This run follows P0/P1 beforeP2:actual runtime variable checks conclusively blocked live providers,so the independent patient-consented intake/review/summary workflow was implemented and tested. P3 adds responsive emergency information and a shared mobile-web manifest,not a native-app claim. Next three tasks:(1)apply runtime provider bindings and verify real model/collection/schema/corpus/citations;(2)extend snapshot-scoped consent into the broader longitudinal chart model with caregiver/identity/jurisdiction safeguards;(3)validate physical mobile/voice behavior and implement native secure-device features only when justified. International sharing,multilingual medical translation,participating facility acknowledgments,prevention/integrative/WimHof evidence appraisal and contraindication safeguards remain full tracked requirements,not this increment's achievements. Breath-holding in water,while driving or other dangerous settings must never be recommended;unvalidated alternatives must not replace necessary proven care.


## Recovered development checkpoint — 2026-10-26

Recovered chart-scoped access,patient AI,account controls,population filtering,claim-support screening and actual native iOS source. See RECOVER-001–005 in REQUIREMENTS_MATRIX.md. Initial integrated Python177/177,frontend26/26,Chromium6/6,extended patient/account browser1/1,mobile structural audit and5/5tests passed. Two additional clinician regressions passed in targeted15/15suite. Live ingestion expanded6CDCdocuments/31vectors,repeat0writes;first live physician claim screen failed closed. Complete live acceptance and final combined regression are still pending. iOS compilation/signing unverified;Android absent;no staging deployment or App Store readiness claim. Account deletion is review-only.
