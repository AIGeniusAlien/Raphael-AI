# Release-candidate audit — 2026-10-28

This is the canonical readiness report. Historical reports remain preserved and are indexed in RC_EVIDENCE_INDEX.json. PASS means the bounded behavior named in a row has evidence; it is not clinical or regulatory approval. No merge, deployment, Apple submission, production patient access or clinical communication occurred.

The audit began at PR#1 head `bbe17f264fda6ebed6b8bfa52ece478b472bdc1e`. Its CI run40434681672 failed authenticated native testing:7 passed, 1 failed, 0 skipped. The former seven-case unconfigured/unit success is not authenticated iOS proof. Existing unfinished changes were preserved and corrected. Audited application/test source:`5710cb65f8708257504994485a25b6c384418d5f`; actual run 40782065504 failed only native acceptance. Regression and dependency/source-security jobs passed. Final documentation follow-up preserves that result; RC_CI_RESULTS.json records the source and raw bounded native evidence.

## Synthetic TestFlight beta

| Gate | Status | Evidence or remaining action |
|---|---|---|
| FastAPI startup, repeat-safe SQLite schema, synthetic persistence | PASS | Consolidated Python/API/browser regressions; actual main:app startup in fresh temporary databases. |
| Tenant/patient authorization, consent revocation, draft immutability | PASS | Synthetic API/browser regressions; no real identity or clinical validation implied. |
| Physician and patient evidence-grounded browser workflows | PASS | RC_LIVE_RESULTS.json:real authenticated OpenAI/Qdrant, rendered citation links and fresh original-source checks; earlier safe failure preserved. |
| Development evidence compatibility | PASS | RC_EVIDENCE_INVENTORY.json:actual 384 dimensions, text-embedding-3-small, 9 active CDC documents/48 vectors; read-only audit. |
| Authenticated native iOS journey | FAIL | Actual current-source run 40782065504:7 passed, 1 failed, 0 skipped, total 8. Sign-in control remote AX snapshot fails before successful login; no pass inferred from compilation or server-session count. |
| Physical-device testing, signed archive, authorized HTTPS staging | BLOCKED | Owner Apple team, registered bundle, staging origin, device and signing access absent. No installable TestFlight build exists. |
| Privacy/store metadata, icon authority and distribution review | BLOCKED | Owner privacy/support URLs, approved brand assets, app/account-policy review and explicit upload authorization required. |
| Overall synthetic beta distribution | FAIL | Authenticated native acceptance is unresolved; signing/staging/privacy/device gates are additionally BLOCKED. No installable candidate exists. |

## Clinical-production launch

| Gate | Status | Evidence or remaining action |
|---|---|---|
| Medical knowledge and clinical safety validation | FAIL | Nine educational CDC documents and a tiny routing benchmark are not comprehensive medical knowledge or expert clinical evaluation.14 safety cases are deterministic protocol fixtures. |
| Production identity, MFA, guardian, erasure/retention and PHI safeguards | FAIL | Development accounts and deletion-review queue do not implement these production controls. Formal erasure policy, identity proofing, at-rest protection and independent security review remain. |
| Qualified intended-use, clinical/regulatory/privacy approval | BLOCKED | Independent clinical evaluation, risk management, jurisdiction-specific legal/regulatory assessment and accountable owners required. No HIPAA compliance or FDA clearance claimed. |
| Vendor contracts, production EHR/communications and operations | BLOCKED | Applicable BAAs/contracts, authorized vendor access, hosting/monitoring/recovery/incident response and operating ownership absent. FHIR/speech/imaging fixtures are not live clinical integrations. |
| Overall clinical-production launch | FAIL | Not safe or authorized for clinical production. Synthetic tests and green CI cannot satisfy these gates. |

## Audit fixes

- Preserved and completed unfinished native CI logging/gesture/viewport work; corrected an invalid Swift interpolation in diagnostic output. Actual gestures continue through the existing sign-in form; no authentication or TLS bypass was added.
- Split diagnostics below the observed 4096-character annotation limit, so late XCTest errors are retained. Redact the ephemeral synthetic password before console/artifact logging; three regression cases verify this and fail-closed logging.
- Run the real CI backend unprivileged on loopback8443, with only a loopback TCP forwarder listening on443. TLS terminates in the actual backend; the app still uses the bound443HTTPS origin and simulator-trusted certificate. Service startup failures record bounded status/exit metadata.
- Fixed the patient pipeline's false-completion risk:provider cleanup must succeed before completion. Added persistent, safe failure-stage codes and checker subphase diagnostics through additive SQLite migration. Consent/ownership/citation gates remain unchanged. Regression tests cover failure persistence and cleanup errors.

## Live-service findings and limitations

All five OpenAI/Qdrant bindings were present in the inherited secure environment; no values were printed. A read-only actual probe verified 384 dimensions and 9 active documents/48 vectors. No collection writes occurred in this audit.

The first physician workflow completed and passed its automated screen; the patient workflow failed safely before producing a screen. Old code discarded the precise underlying provider/protocol reason, which cannot be reconstructed from that attempt. RC_INITIAL_LIVE_FAILURE.json preserves it. After diagnostic/completion fixes, a bounded real recheck completed both workflows with exact displayed resolver links and fresh CDC source checks. No automatic retry-until-success was implemented. Model/support-screen judgments remain fallible and outputs remain provisional.

The corpus is CDC government-authored text with source dates, URLs, hashes, license-policy provenance and population scopes. Ingestion history and repeat 0 writes are in MASTER_INGESTION_RESULTS.json. Bibliographic PubMed connectivity is not licensed full-text knowledge or claim support. Broad specialty coverage, continuous source discovery/retraction relationships and expert reasoning evaluation remain incomplete.

Seven documentation formats are deterministic preparation templates from authorized transcript/chart snapshots. Version history, edits, explicit approval/publication and missing facts are verified; automated clinical synthesis, billing/prior authorization completion and legal signature validation are not complete.

## Verification and evidence

Final local suite passed 213 Python, 26 frontend, 6 Chromium and 5 mobile structural cases, plus project audit, traceability and 14 deterministic safety cases. RC_VERIFICATION_RESULTS.json preserves exact statuses and explicit native/live/advisory skips; real service proof is in separate files. Authenticated native result is FAIL: 7 passed/1 failed/0 skipped, total 8 in actual macOS CI. The expanded high-severity/high-confidence Bandit scan includes native and CI harness scripts: zero findings and zero scan errors; lower-severity findings are not a security certification (RC_SOURCE_SCAN_RESULTS.json). The affected 24-test patient/claim suite and 3 CI-redaction tests also passed. Native/source/service results remain separate. CI artifacts are synthetic only.

All 1,128 baseline requirement IDs/descriptions are preserved in the 1,136-row ledger. Presence in the ledger does not imply implementation. RC_REQUIREMENTS_AUDIT.json preserves the comparison and implementation-state totals. Stale blocked-credential/live-service rows are reconciled to real bounded evidence; the complex differential workflow remains partial and native acceptance remains partial/failed. Implementation-state counts are not release-gate counts.

A native artifact download is currently blocked at `productionresultssa9.blob.core.windows.net`. The existing allowlist was preserved and that exact domain saved for environment review/publication via cloud-environment-onboarding:setup. A saved draft is not applied runtime configuration. CI summaries remain readable through GitHub's API; actual image inspection is separately blocked until permitted. Signed artifact URLs and secret values were not exposed.

## Reproduce and release safely

1. Check out the final audited branch `fix/sqlite-startup-recovery` and exact recorded SHA. Use Python3.12. Run `python -m dev_start --setup --provision` for an isolated, synthetic, loopback development instance; provision private accounts interactively. No default/shared credentials or production DB.
2. Run `python -m dev_verify --all --require-browser --output /tmp/raphael-rc-verification`. It strips inherited provider/DB settings for ordinary regressions and distinguishes skipped native/live checks. Actual live proof requires explicit `RAPHAEL_VERIFY_CLAIMS=1 python -m scripts.verify_live_medical --live --output /tmp/raphael-rc-live.json` with approved secure bindings. Never use real patient data.
3. Reproduce the exact recorded source with `gh run rerun 40782065504` (all jobs), or use the Actions run page; inspect the actual xcresult summary and native screenshots. New PR commits trigger CI automatically. This does not require merging a workflow into main. Zero remaining server sessions alone does not verify native login/logout. Local `mobile/scripts/native_checks.sh` requires macOS/Xcode and a valid simulator destination; without the private CI fixture, the authenticated case skips and is not authentication proof.
4. Owner supplies authorized synthetic HTTPS staging, registered bundle/team, approved icon, and real support/privacy URLs. Copy `mobile/ios/Configuration/Local.xcconfig.example` to ignored `Local.xcconfig`; enter public configuration only, never OpenAI/Qdrant secrets. Complete device/accessibility/account-policy/privacy review before distribution.
5. Only after gates pass and separate explicit upload authorization, archive Release for a real iOS device in Xcode Organizer, validate signing and metadata, and distribute to the owner's App Store Connect app. Release must reject reserved `.test` hosts and placeholders. The owner then completes TestFlight processing/review and tester invitations. None of those actions was performed here; there is no build/invitation to install yet. Detailed steps:mobile/submission/REVIEW_AND_TESTFLIGHT.md.

## Authenticated native failure and remaining investigation

The normal-user backend wrapper reaches real readyz, provisions synthetic accounts/approved notes over verified HTTPS, and keeps provider credentials absent from CI. Actual Swift compilation and all 6 unit cases pass. The unconfigured UI case passes. The authenticated case still fails obtaining the labeled sign-in TextField snapshot; the remote WebKit accessibility process reports `kAXErrorServerNotFound`. Both scoped and application-level locators failed. The test remains enabled, uses the real form, and is not replaced with a mock, JavaScript credential injection or a skip. No successful authenticated native journey is claimed. Three actual simulator images were exported; their inspection here remains blocked by artifact egress. Further native diagnosis needs that artifact access or an interactive authorized Mac/device session. This is a specific failed acceptance gate, not a claim that Xcode CI was unavailable or that backend credentials were invalid.

The former CLI backend-readiness stall could not be reconstructed from the old empty log. Unprivileged startup plus the bounded actual-app wrapper reaches readiness in subsequent runs; no invented root-cause explanation is substituted for that limitation. The earlier failures are preserved. Diagnostics/redaction and fail-fast test behavior make subsequent failures reviewable without weakening authentication, TLS or assertions.

## Outstanding owner inputs versus engineering work

Owner inputs:publish the saved exact artifact-domain addition; provide authorized synthetic HTTPS staging, Apple Developer team/registered bundle and physical-device access; approve the canonical icon, privacy/support URLs, tester instructions and applicable retention/intended-use policy. These actions do not make clinical production safe.

Engineering work remains:resolve and verify the native authenticated UI failure; complete policy-backed account erasure, production identity/MFA/guardian controls, native protected export where promised, PHI security/operational hardening, broader rights-reviewed medical coverage and independent expert clinical evaluation. These are not marked complete or shifted to the owner as routine coding tasks.

The cloud onboarding skill's contract is explicit:“Saving persists configuration; it does not execute scripts, apply runtime changes, or publish.” [Skill source](skill://plugin_connector_1p_ed5feb9070a08191b08c81c47947bc16/setup/SKILL.md). The saved domain draft requires publication; this is environment configuration, not application deployment.
