# Interrupted development recovery — 2026-10-26

Recovered in the original checkout and branch `fix/sqlite-startup-recovery`,preserving draft PR #1 and baseline946ca43. No reset,collection recreation,merge,application deployment,production patient data,billing or external clinical communication. Recovery checkpoint b98ce2e was committed and pushed before final verification. See IMPLEMENTATION_REPORT.json for final machine-readable results and ENGINEERING_LEDGER.json for all1,117requirements;all1,112prior IDs and descriptions remain intact. The complete previous master directive and recovery instruction are preserved verbatim in RAPHAEL_MASTER_SPEC.md.

## Implemented software

- Patient-scoped chart assignment replaces tenant-wide clinician access. Patient grant/revoke and bounded expiry override creator assignment;expired explicit decisions do not silently restore access. Unrelated clinicians and cross-tenant users are denied. Lists,inbox counts,cached AI history,source snapshots,later streamed tokens and async FHIR returns enforce access. Separate care-intake snapshot consent remains independent;already transmitted model context cannot be recalled. Existing approved records/audits remain durable.
- Patient AI uses the existing authenticated backend and provider adapters,explicit per-thread AI consent,optional own approved notes/reported context,adult/pediatric retrieval filtering,persistent history,cancel/revoke and owned source snapshots. Provisional text is education,not clinician-approved treatment. Phrase-based emergency notices are not validated triage. Session/consent/generation state is checked before initial transmission,each delta,before/after screening and before completion.
- Account settings add own display/language/time-zone preferences,session listing/revocation,bounded own patient export and persistent deletion-review request/cancel/admin acknowledgment. Export excludes unsigned clinician drafts,password/session secrets and document bytes. Profile preferences do not alter clinical identity. Deletion review does NOT erase accounts/medical records and does NOT satisfy Apple's deletion requirement.
- Six reviewed CDC sources are fetched from original URLs with dates,titles,content/reuse-policy hashes and public-domain-text exception review. Text-only excerpts preserve attribution. Population/specialty metadata migration retained all20original vectors without reembedding. Actual ingestion added11vectors;repeat wrote0. Original collection/data/model space were retained.
- Automated claim screening checks every deterministic paragraph,registered citations,exact server-resolved source spans and population compatibility. Strict structured output selects registered quote references rather than retyping hashes/quotes. Missing support is blocked;unknown references,malformed responses and provider failures fail closed. This is a fallible model-based screen,not independently measured entailment,medical accuracy or clinical approval. It is opt-in via RAPHAEL_VERIFY_CLAIMS=1 and exercised separately from local fixtures.
- Actual SwiftUI/WebKit iOS source project includes app/unit/UI-test targets,shared scheme,native navigation/settings,one ephemeral authenticated web session,HTTPS/origin restrictions,background cover,explicit confirmations,server/local sign-out,icons and privacy configuration. It reuses existing backend pages. Microphone/downloads are denied. No keys embedded. Linux project audit and five tests pass. Swift compilation,simulator/device tests,VoiceOver/Dynamic Type behavior,signing and submission remain unverified without macOS/Xcode. Android native source/build is absent.

## Test evidence and remaining limits

Final exact counts and real-provider outcomes are in IMPLEMENTATION_REPORT.json and RECOVERY_LIVE_RESULTS.json. Python includes original SQLite trigger failure,repeated initialization,rollback,persistence/restart,health/API operations,chart/FHIR authorization,consent/session revocation,provider failures,citation integrity,population filtering,idempotent ingestion and account isolation. Frontend tests use controller/browser-service fixtures;Chromium runs actual application processes with explicitly labeled local model fixtures. Those are not live-provider or clinical-validation results. Live checks use real authenticated OpenAI/Qdrant and temporary synthetic data separately.

Earlier live broad physician output failed closed during support validation;another narrower physician response passed. A patient response was blocked for missing support. These failures are retained in the live attempt ledger rather than hidden. The final current-code outcome is recorded explicitly;technical execution and support-screen verdicts must be read separately. No comprehensive literature/specialty coverage,diagnostic performance,guardian/legal proxy permission,real identity proofing,recovery/MFA,external EHR/PACS writes,real audio,production PostgreSQL,account erasure,security certification,clinical/regulatory approval or full-product readiness is claimed. Dependency compatibility is checked for86packages;no fresh vulnerability-database scan is claimed in this recovery.

Suitable for continued restricted synthetic local development. Shared staging requires separately authorized HTTPS infrastructure,secret management,identity/security/privacy review and operational controls;none is deployed. App Store/TestFlight readiness remains blocked by native build/device/signing,owner service/assets/legal policy and full deletion behavior. See mobile/README.md and mobile/submission/READINESS.json.

## Exact owner run instructions

In the existing cloud checkout:

```bash
cd /workspace/Raphael-AI
. /workspace/.raphael/venv/bin/activate
export UV_CACHE_DIR=/workspace/.cache/uv
export HF_HOME=/workspace/.cache/huggingface
export PYTHONDONTWRITEBYTECODE=1
export XDG_STATE_HOME=/workspace/.raphael/state
export RAPHAEL_DEV_PORTAL=1 RAPHAEL_DEV_SYNTHETIC_ONLY=1
export RAPHAEL_EMBEDDING_BACKEND=openai RAPHAEL_STREAM_EVIDENCE_BACKEND=qdrant
export RAPHAEL_VERIFY_CLAIMS=1 EMBED_DIM=384
mkdir -p /workspace/.raphael/state
# New isolated directory;does not replace existing databases/accounts.
run_dir=$(mktemp -d /workspace/.raphael/state/owner-synthetic.XXXXXX)
export DATABASE_URL="sqlite:///$run_dir/portal.db"
python -m dev_runtime migrate --database-url "$DATABASE_URL"
python -m auth.dev_bootstrap --database-url "$DATABASE_URL" --tenant synthetic --username clinician --role clinician
python -m auth.dev_bootstrap --database-url "$DATABASE_URL" --tenant synthetic --username patient --role patient
bash run_local.sh
```

Bootstrap prompts privately for passwords. Keep the new database path for later restarts;do not rerun bootstrap to overwrite existing accounts. Keep the five existing secure provider bindings;never paste keys into commands or files. The server listens on loopback8000. Open it in a browser on that machine or through an owner-configured private forwarding mechanism;this task has no published preview/staging URL. Confirm /healthz returns ok. For a local workstation,use Python3.12 and requirements.lock/requirements-dev.txt as described in DEVELOPMENT.md;cloud-only paths must be adapted.

Sign in as clinician,create a synthetic record linked to username patient,then open Clinical conversations. Sign in separately as patient,review Chart access and open Patient AI;explicitly consent and optionally select approved context,ask a CDC-supported educational question and inspect source snapshots/originals. A blocked/failed draft is not completed advice. Review persistent conversation status after reload. Use Connected intake for explicit recipient-specific snapshots;clinician approval remains a separate action before patient publication. Account page supports preferences,sessions,export and review-only deletion. Internal messages stay internal.

For reproducible actual remote checks without manual accounts:

```bash
python -m scripts.verify_live_medical --live --output /workspace/.raphael/owner-live-result.json
```

For iOS,open mobile/ios/Raphael.xcodeproj on macOS,configure ignored Local.xcconfig with the authorized synthetic HTTPS origin,matching host and owner public bundle/support values,and run mobile/scripts/native_checks.sh with an installed simulator destination. No provider secrets belong in the app. A simulator build does not require a paid developer account;device/distribution needs owner signing. Full exact commands are in mobile/README.md.

## Next three executable actions

1. Review the final live screening outcomes against original excerpts with a clinician;add a fixed adjudicated support/population/contradiction evaluation set and improve draft generation without weakening citation/consent gates.
2. On macOS,run the supplied Xcode simulator compile/tests,then authenticated synthetic device/accessibility/security tests against an explicitly authorized HTTPS staging service. Fix native failures before any signing/upload.
3. Complete owner retention/deletion/privacy policy and policy-backed account erasure,identity/session hardening and independent security/clinical review;then reassess restricted staging and TestFlight gates. Android and the broader platform backlog remain explicit open requirements.

Reusable cloud startup instructions were saved as a configuration draft. Saving does not apply/publish them:review/save settings and publish the environment separately if desired. This is not an application deployment. Work does not continue autonomously after this execution.
