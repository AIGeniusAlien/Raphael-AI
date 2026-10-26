Current live verification:real OpenAI authentication,model access and streaming passed;real Qdrant collection access passed overHTTPS443. Pinned SDK port6333fallback caused the initial403 and is corrected in all existing client factories. Target raphael_evidence_v1 is absent. Actual embedding verification and medical ingestion remain blocked byProxyError fetching Hugging Face/CDC;network changes were saved for publication/reconnect. No remote collection mutation,corpus or grounded medical-conversation claim. See INFRASTRUCTURE_STATUS.md;older401/403credential observations below are historical.

Current connected-care sprint:all five provider variables are injected;OpenAI401 invalid-key indication,Qdrant403Forbidden. No live ingestion/retrieval/citation success. Evidence snapshots now persist and completed intake AI responses can be explicitly staged,edited,saved and approved as care summaries with source provenance. Consent is rechecked before initial provider transmission. See CONNECTED_RELEASE_CANDIDATE.md and INFRASTRUCTURE_STATUS.md;earlier runtime checkpoints below are historical. No full-product/native/mobile-device/staging readiness claim.

Published runtime checkpoint (2026-10-26): four required variables now present; QDRANT_API_KEY absent. A real bounded OpenAI request returned401; streaming/model access unverified. No Qdrant request/mutation occurred. See [INFRASTRUCTURE_STATUS.md](INFRASTRUCTURE_STATUS.md) for current evidence and exact secure binding actions; earlier missing-binding reports below are historical. No full product,live retrieval or deployment-readiness claim.

# Exact continuation after this increment

Continue in `/workspace/Raphael-AI` on `fix/sqlite-startup-recovery` and the existing draft PR #1. Preserve the startup recovery and all implemented portal/platform changes. Do not create a new Git worktree, repeat completed implementation from scratch, reset the checkout, overwrite user data or merge/deploy automatically.

## First commands

```bash
cd /workspace/Raphael-AI
git status --short
. /workspace/.raphael/venv/bin/activate
python -m unittest discover -s tests -v
node --test tests/*.test.cjs
python -m unittest discover -s e2e -v
```

Read AGENTS.md, RAPHAEL_MASTER_SPEC.md, REQUIREMENTS_MATRIX.md, RELEASE_READINESS.md and TEST_RESULTS.md first. The matrix preserves prior IDs and adds the latest 15-system directive plus FLOW-001–011 (scenarios A–K) and ACCEPT-001–014 release gates. Fix a reproducible failing implementation/test before expanding features; never turn an unavailable integration into a mock success. Inspect the latest test results rather than relying on historical baseline totals.

## Resume instruction

> Continue implementing the existing Raphael platform from the current branch. Do not redo the preserved SQLite/FTS recovery, cookie-session portal, authorization, internal messaging, approved-note visibility or any newly passing extension. Read the release ledger and exact remaining matrix rows. First complete any failed integrated extension or browser workflow, then supply dedicated development service configuration through secure settings and validate the real LLM/Qdrant/model/corpus workflow. Implement the next unmet end-to-end scenario with backend, UI, migration, authorization, meaningful automated tests and actual execution. Preserve all requirement IDs, keep incomplete/blocked/clinical-validation states explicit, update the draft PR and report exact remaining executable steps. Use only temporary synthetic datasets; no merge, production deployment, external clinical delivery or live billing.

## Dependency-aware next execution

1. Read the latest `TEST_RESULTS.md` result ledger and module contracts. The expanded synthetic modules are `dev_chart.py`, `dev_messaging.py`, `dev_imaging.py`, `dev_runtime.py` and `integrations/fhir/sandbox.py`; they integrate through the returned portal context. Do not make a second incompatible auth/data layer. Re-run the relevant tests for any change.
2. Check configuration metadata and variable **names/presence only**, never values. Existing saved requirements for LLM_API_KEY/LLM_BASE_URL/LLM_MODEL/QDRANT_URL do not mean values are available. Securely supply a dedicated development provider/key and Qdrant endpoint, configure only their actual destinations, cache the matching embedding model and ingest a reviewed legally reusable source set.
3. Execute FLOW-003 (complex synthetic structured assessment) and FLOW-004 (retrieval with resolving, supporting citations) against the real configured development services. Record response structure, source provenance, citation membership/support, failure handling and clinician review; no fabricated clinical evidence or claims. If blocked, name the exact missing prerequisite and continue independent scenarios.
4. Execute the implemented chart, review-before-send, administration, DICOM and FHIR scenarios against the integrated UI/API. Complete any remaining portions of FLOW-001/002/005/006/007/008/009/010/011 that the final ledger still marks partial. An adapter fixture does not satisfy the live sandbox portion of FLOW-007.
5. After the synthetic demonstration gates, continue the complete-platform backlog: patient consent/proxy/identity/recovery, remaining professional/admin roles, clinical order/coding/safety contracts, source licensing/versioning/freshness/coverage/retrieval evaluation, real audio providers, SMART/OAuth resource mapping, PostgreSQL migrations, independent clinical/security evaluation and controlled release operations. Each new capability needs its own reviewed acceptance tests.

The complete platform is not finished while any matrix acceptance criterion remains missing, partial, blocked or awaiting independent validation. A locally runnable build is not a clinically validated or regulator-authorized device.


## Latest security checkpoint
Dependency advisory remediation is implemented,not a pending rerun of the prior vulnerable versions. Use requirements.lock plus torch2.6CPU and SECURITY_REMEDIATION.md. Next executable feature: add an authorized clinician conversation router with persistent generation records,actual OpenAI SSE streaming,cancellation and provider-error handling; test via mock HTTP transport and actual browser,never simulated live streaming. Then connect consent-gated microphone/voice and versioned editable encounter notes. Dedicated live credentials remain missing. Preserve all existing clinical/chart/DICOM/messaging regressions and requirement IDs.


## Core intelligence now implemented — resume here
Do not rebuild streaming,voice-controller or versioned transcript-template code. Read CORE_INTELLIGENCE.md and inspect dev_intelligence.py/static/intelligence.*/static/voice.js. Run full Python,Node andChromium suites. Next:configure dedicated LLM credentials securely and approve a legally usable source set to validate actual streaming and structured medical outputs. Independently exercise real microphone/browser speech with consent. Then implement authorized chat attachments,provider realtime audio/diarization and AI note synthesis with strict provenance/edit-preservation; these are not current capabilities. Maintain clinician signoff,atomic patient-note publication and the existing DICOM/messaging/security checks.


## Latest combined directives: do not rebuild care intake
Read CARE_JOURNEY.md and INFRASTRUCTURE_STATUS.md. Apply saved provider settings to the attached runtime securely;all five variables were absent despite declared configuration. Run `python -m dev_services --live` once after meaningful runtime change;never loop paid failures. Inspect the existing development cluster only,validate model384-dimensional collection compatibility,ingest a genuinely licensed reviewed corpus idempotently,and exercise retrieval/citations through actual endpoints and Chromium. Do not report unknown counts aszero or seed fabricated medical text. Preserve the tested connected care flow,mobile emergency view,explicit AI snapshot consent and revoked-stream regression. Then expand base chart authorization/proxy/identity boundaries;the new snapshot grant is intentionally not global chart consent. Third,validate real iPhone/Safari/microphone behavior using MOBILE_ADR.md. No merge,deployment,external hospital message or live billing without authorization.
