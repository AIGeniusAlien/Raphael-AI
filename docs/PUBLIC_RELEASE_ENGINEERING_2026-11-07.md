# Raphael implementation and release evidence — 2026-11-07

The two new directives remain **incomplete**. This execution delivers five integrated engineering slices plus native policy integration and security repairs. It does not authorize public publication or clinical use. Existing branch `fix/sqlite-startup-recovery`,draft [PR#1](https://github.com/AIGeniusAlien/Raphael-AI/pull/1),all 1,164 prior IDs/descriptions/acceptance criteria and all prior working features are preserved. Baseline `906700fc43ede1c834ff89fff46bee58ba2fd1d9` was actually verified against live GitHub metadata: CI41390802643 passed3jobs and27native tests. That is historical evidence,not acceptance of this increment.

## Delivered working behavior

| Capability | Actual implementation and local proof | Boundary |
|---|---|---|
| Release policy and patient preview | PureASGI exact HTTP/static allowlist,version/hash manifest,patient ownership,pre-response identity fence,private operator pause;real provisioned patient records/approved notes/document download/preferences/feedback/reviewrequest;10API tests and actual Chromium | Synthetic preview only;no self-registration,external recovery,full erasure,clinical tools or publication approval |
| Durable background jobs | SQLite queue,UUIDdedup,priority/delay,leases/fencing,retries/backoff/deadletter,cancel,append-only history;separate CLI worker;10API/process/SQLCipher tests and Chromium;actual child-process crash/recovery | Two read-only inventories only;no arbitrary tool/model/network execution,clinical actions,permanent daemon or24/7claim |
| Procedure checklist coordination | Reviewed immutable source-version templates,private drafts,clinician publication,patient coordination authorization,assigned acknowledgements,clinician signoff,care-team grant checks,immutable snapshots/events,revocation and own-export withholding;15API tests and actual Chromium | No medical instructions seeded;checkboxes are not medical consent,procedure performance or independent clinical validation;comprehensive catalog still missing |
| Engineering maintenance | Read-only source/ledger/CI audit and proposals;5tests;actual authenticated GitHub metadata request;fourthCIjob;daily/event workflow definition | Scheduled workflow not yet registered on defaultbranch;no overnight/autonomous coding,automatic issue/PR changes,merge or deploy claim |
| Dependency remediation | General audit silently skipped old TorchCPUbuild;upstream2.6.0 metadata had25 advisories. Installed official2.15.0+cpu/Sympy1.14;real offline CPU/BERT/localBoWembedding adapter compatibility;4tests;general plus normalized audits | Upstream metadata does not independently attest binary contents,clinical model quality or complete supply-chain security |
| Native release-policy adoption | Same-origin bounded manifest fetch/strict validation,fail-closed policy changes,candidate dashboard-only root preview,hidden unsupported clinical/MFA controls;development/privateMFA preserved | Linux source/project checks pass;actual development28 pluscandidate1tests await two-profileMacCI;do not infer native execution from source |

Security review found and fixed actual transaction-time revocation races in procedure writes and record export,stale-role/tenant session validation,paused record-export bypass,missing pause-parent fail-open behavior,and post-handler identity changes before response bytes. Tests preserve original data and authorization invariants. No production database was opened or migrated. SQLCipher migrations/triggers are still additive and repeatable.

## Actual verification

[Final local report](NOV7_LOCAL_RESULTS.json): **373 Python,29 frontend,11 Chromium,5 mobile-source tests**;all 9 locally runnable checks PASS,0 failed. [Actual encrypted restart](NOV7_PROTECTED_RUNTIME.json) passes two real starts,health/readiness,patient/document/MFA persistence and recovery login. Python count includes two profile-summary gate tests;these validate count handling,not Xcode execution. Three checks are explicitly separate/skipped locally:Xcode,paid live services,and CI advisory gate. No skipped test is counted as passed. Actual default MiniLM download/load and new live OpenAI/Qdrant conversations were not performed in this execution;earlier9CDC/48vector/384dimension evidence retains its historical source scope. No medical corpus was changed.

[General dependency audit](NOV7_DEPENDENCY_AUDIT.json):87 installed packages,zero reported known vulnerabilities,one CPU-local Torch build skipped by general audit. [Supplementary local-build audit](NOV7_LOCAL_BUILD_ADVISORY_AUDIT.json) separately checks publicTorch2.15.0 metadata and reports no public advisories. [Initial skipped audit](NOV7_INITIAL_DEPENDENCY_AUDIT.json) and [old upstream advisories](NOV7_INITIAL_TORCH_ADVISORIES.json) remain preserved. All87 installed packages pass dependency compatibility. Existing Electron/extension prototypes lack current packaging/dependency/runtime acceptance;Python scan results are not blanket desktop security approval.

Actual new mobile/enlarged/desktop browser journeys passed;consumer-preview and patient-procedure mobile screenshots were visually inspected. Raw logs/screenshots live in `/workspace/.raphael/nov7-integrated-final`;they are local workspace evidence,not durable CI artifacts. Captured [source hashes](NOV7_SOURCE_MANIFEST.json) identify the checkpoint bytes;native hashes do not prove native execution. Current-head remote CI must be checked independently. The draft PR records exact final commit/run/profile outcomes after they exist;this pre-push report does not invent a self-referential future commit or CI pass.

The native workflow now tests all original27 cases plus1policy unit in the development profile,then restarts the actual ephemeral HTTPS backend into consumer-candidate scope and executes1 candidate UI case. Each profile has its own xcresult. The collector requires development>=28,candidate>=1,and zero failures/skips for both;it rejects missing/duplicate/inconsistent profiles. Candidate setup requires a prior real cleanup marker and zero active sessions,reuses private synthetic DB/certificate/key,never bypasses authentication and makes no provider requests. Actual Mac execution is still pending at report authoring.

## Requirements and provenance

Both uploads are preserved verbatim after CRLFnormalization. [Section reconciliation](2026-11-07_DIRECTIVE_RECONCILIATION.json) covers17 phases + 61 subsections and43 sections. Original full conversation transcripts are not available here;captured master/directives remain the source. New bounded PLATFORM-010–014 do not satisfy comprehensive clinical,enterprise,release or autonomous-engineering requirements. All1,164prior IDs/descriptions/criteria are unchanged;1,169 rows now. Legacy per-row original-upload/risk/contract adjudication remains incomplete. Broad statuses are not upgraded merely because a narrow slice passed.

History audit confirmed root Electron/preload/renderer and popup files remain tracked;they were not removed. The extension MV3manifest was replaced by the PWAmanifest in historicalc5a1dca without a product-cancellation record;desktop remains a prototype and popup targets obsolete /v1/intents. Exact recovery commits/security gaps are preserved in reconciliation. Do not restore nodeIntegration or bypass portal isolation. Android remains absent;neither desktop nor extension nor Android is claimed release-ready.

## Release gates

| Gate | Result | Evidence / remaining action |
|---|---|---|
| Isolated synthetic local web workflow | PASS | Actual373/29/11/5and9checks above |
| Consumer engineering preview | PASS for tested web scope | Authenticated provisioned-patient workflow;native execution awaitsCI |
| Controlled synthetic TestFlight | BLOCKED | Authorized HTTPSstaging,stable protected storage/keys,Apple team/bundle/signing,metadata/device/VoiceOver/OSdestination and owner acceptance absent |
| Public App Store release | FAIL | Selected public account-lifecycle/support/operations acceptance incomplete;current Apple guideline review blocked;no authorized release artifact or publication approval |
| Controlled clinical pilot | FAIL | Qualified intended-use/population/evidence/model/safety/privacy/institutional acceptance incomplete |
| Clinical production | FAIL | Pilot gates plus fullPHIlifecycle,licensed coverage,enterprise identity/integrations,monitoring/capacity/recovery/change control incomplete |

These are separate gates. Full advanced medical competence is not required for a future narrowly approved consumer release,but the selected public feature set must still be complete,secure,lawful and actually accepted. Engineering-owned missing software is not mislabeled an external blocker.

## Exact external blockers and owner decisions

Recognized release-tool bindings checked by presence only: `RAPHAEL_SERVER_URL`,`RAPHAEL_SERVER_HOST`,`RAPHAEL_BUNDLE_IDENTIFIER`,`RAPHAEL_SUPPORT_EMAIL`,`DEVELOPMENT_TEAM`,`APPLE_TEAM_ID`,`APP_STORE_CONNECT_API_KEY`,`ASC_KEY_ID`,`ASC_ISSUER_ID`,`RENDER_API_KEY`,`RENDER_SERVICE_ID` were absent. No secret values were printed. LocalXcode is absent;hostedMacCI is available for unsigned simulator testing only. Owner must provide an approved HTTPS target/matching hostname,persistent synthetic volume/private distinct keys/backup/alert ownership,and actual hosting authorization/cost budget;then authorize an integration deployment. No deployment occurred. Render configuration remains autoDeployfalse and now uses the same explicit CPU/pinned dependency installation as CI,with Python3.12.14;no service/disk was provisioned.

Apple team/registered bundle/signing/provisioning,approved support/privacy assets and disclosures,and physical-device/accessibility acceptance remain required. Do not send secrets through chat. Current official requests to `https://developer.apple.com/app-store/review/guidelines/`, `https://developer.apple.com/support/offering-account-deletion-in-your-app/` and `https://developer.apple.com/app-store/app-privacy-details/` failed with URLError/OSError transport failures through the configured proxy;no authoritative current content was retrieved or claimed reviewed. Allow HTTPS to developer.apple.com in the cloud network policy,publish it,and rerun bounded review. No submission/publication is authorized by this checkpoint.

Daily/event engineering automation requires the new workflow on GitHub's default branch and actual scheduled-run verification;it is presently in draftPR1. Protected-branch review/merge authorization is required before that schedule can activate. The real workerCLI is bounded;an operator-managed persistent worker and durable deployed storage are not installed. No overnight/24/7availability claim is made.

Clinical activation requires independent qualified reviewers,intended-use/jurisdiction decisions,rights-reviewed evidence and datasets,and actual authorized hospital/PACS/EHR/payer/notification sandboxes. No real patient data,patient contact,clinical orders,external clinical delivery,model training or live billing was used.

## Reproduce and continue

```bash
python -m dev_verify --all --require-browser --output /workspace/.raphael/NEW-results
python scripts/audit_dependency_versions.py --output /workspace/.raphael/NEW-local-build-audit.json
python -m scripts.engineering_watch --github --output /workspace/.raphael/NEW-engineering-audit.json
python -m scripts.build_engineering_ledger --check
python -m scripts.build_platform_acceptance --check
```

Create distinct new private component keys outsideGit using existing `dev_keys`;start a dedicated synthetic database using `dev_start --provision --database ... --database-key-file ... --document-key-file ... --mfa-key-file ...`. Default provider mode stays disabled. Set `RAPHAEL_RELEASE_SCOPE=consumer_candidate` to test only the gated preview;keep development scope for clinician checklists/jobs. The private pause directory must be owner-only;creating the configured pausefile blocks record/feedback access including export,while account/signout controls remain. Never point these synthetic tools at production data.

Enqueue a permitted inventory in `/static/jobs.html`,then run `python -m dev_jobs --database-url sqlite:////ABS/isolated.db --tenant TENANT --max-jobs 100 --run-seconds 300` and the matching private encryption bindings. Jobs persist across browser/worker closure;the CLI exits at its bound. Visit `/static/procedures.html` to supply and review only legally usable content,bind exact approved sources,explicitly publish,obtain patient coordination authorization,and complete role-assigned acknowledgements/signoff. These are synthetic coordination records,not medical consent or evidence of performed treatment.

Mac CI reuses existing `mobile/scripts/authenticated_ci.py` and `native_checks.sh`:developmentprepare→developmenttests→verifiedcleanup→candidateprepare→candidate-onlytests→verifiedcleanup→per-profilexcresultsummary. Release signing/packaging remains guarded by `mobile/scripts/validate_release.py`;run it only after owner supplies actual approved configuration. Do not upload to Apple or expose a clinical service without separate authorization.

Next executable engineering work: complete approved consumer identity/recovery/deletion/support flows and real operational staging under explicit policy;install/verify a supervised worker and scheduled audit after authorization;extend rights-reviewed procedure/evidence coverage and qualified feedback/evaluation;repair preserved desktop/extension prototypes with actual runtime acceptance;continue original roles,voice,radiology/interoperability,international/prearrival,Android and clinical validation backlog. No requirement was canceled and no independent medical validation was manufactured.
