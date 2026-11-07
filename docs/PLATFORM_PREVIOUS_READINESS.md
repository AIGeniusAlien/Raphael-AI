# Raphael release-candidate implementation and readiness

Current native export checkpoint, 2026-11-01 (America/New_York; raw timestamps UTC). Verified implementation: `e90c78b3174a76b2d1e235b671ff1d348e9d395e`, branch `fix/sqlite-startup-recovery`, draft [PR #1](https://github.com/AIGeniusAlien/Raphael-AI/pull/1). Starting continuation source was `ce27553d6b215c000ad43e571552e16b1f10ab87`. The following evidence-only commit is identified in the PR and final response; this report does not claim its own future CI result.

**The entire master directive is not complete.** This increment implements and demonstrates bounded native patient JSON export. It is a tested continuation checkpoint, not release approval. No merge, deployment, Apple submission, real patient data or clinical communication occurred.

## Implemented and actually verified

- Native patient settings now prepare an own-account record summary, then require explicit confirmation for JSON copy or system sharing. Each disclosure fetches fresh bounded data and rechecks the same server identity. Cancellation, backgrounding, logout and observed revocation invalidate pending transfers; no export file or persistent session cache is added. Copies already deliberately exported cannot be recalled. Clipboard local-only delivery/60-second expiry is an OS request, not forensic erasure.
- Actual clipboard JSON was deliberately pasted through the existing native Paste menu after logout and checked for format,own account/record and exclusion of an unlinked record,fixture password and password hash. The unsent field was cleared by relaunch before the separate real share workflow. No login containing JSON was submitted. There is no host clipboard endpoint or authentication bypass.
- Actual native share-container presentation/cancellation, export503, delayed-request cancellation and independent owner-session revocation pass. **No external share destination was selected.** Files/AirDrop/third-party delivery, physical-device behavior and full VoiceOver remain unverified. Document file bytes are excluded.
- The original18native cases remain. Actual full suite is23passed/0failed/0skipped (12unit+11UI), against the real authenticated synthetic HTTPS application. Backend audit reports 17logins/14logouts/0active sessions/1expired row retained and0provider requests.

[Detailed export evidence and reproducible steps](NATIVE_EXPORT_READINESS.md), [raw CI](NATIVE_EXPORT_CI_RESULTS.json), [native annotations](NATIVE_EXPORT_NATIVE_ANNOTATIONS.json), [local results](NATIVE_EXPORT_LOCAL_RESULTS.json). [CI41331977079](https://github.com/AIGeniusAlien/Raphael-AI/actions/runs/41331977079) passed regression, dependency/source security and actual native jobs on the exact implementation above.

| Verification | Actual result |
|---|---|
| Python |269passed|
| Frontend |29passed|
| Chromium |7passed|
| Mobile source/project tests |5passed;not Xcode proof|
| Actual Xcode |23passed/0failed/0skipped;12unit+11UI|
| Safety/traceability |14deterministic cases and ledger validation pass;not clinical validation|
| Fresh live provider conversation |Not rerun;unchanged pipeline;prior actual supported physician/patient OpenAI/Qdrant runs preserved on4299e13|
| Evidence inventory |Prior verified9CDCdocuments/48vectors/384dimensions retained;this increment made no corpus mutation|

## Complete directive reconciliation

All14sections remain reconciled in [the machine section record](FINAL_RC_DIRECTIVE_RECONCILIATION.json). All1,154previous requirement IDs and descriptions are preserved;one narrowly scoped RC-004 is added,for1,155total. [Preservation/count evidence](NATIVE_EXPORT_REQUIREMENTS.json). Broad capabilities are not promoted by a bounded success. The [machine acceptance](RAPHAEL_FINAL_ACCEPTANCE.json) records engineering FAIL separately from owner BLOCKED.

| Section | Status | Current evidence and remainder |
|---|---|---|
|1. Project baseline|PASS|Verified3ee4eec baseline;preserved branch,history and draftPR1.|
|2. Actual acceptance matrix|PASS|1154continuation IDs/descriptions preserved,including1151initial baseline;1155current rows;failures and owner dependencies remain explicit.|
|3. Extended native iOS acceptance|PARTIAL|23actual Xcode cases pass including bounded native JSON copy,share-container cancellation and export cancellation/revocation;full OS destination delivery,physical device,VoiceOver and independent native TOTP enrollment/entry remain.|
|4. Patient and physician experiences|PARTIAL|Existing chart,consent,documentation,observation and communication sandbox preserved;actual erasure,guardian identity,full native destination delivery and live clinical integrations incomplete.|
|5. Medical intelligence|PARTIAL|Live9document/48vector/384dimension supported-domain map verified;prior real supported conversations preserved;comprehensive coverage,retraction surveillance and independent clinical evaluation unfinished.|
|6. Longitudinal intelligence and structured extraction|PARTIAL|Authorized chronology,exact arithmetic,immutable context and reviewed tabular extraction tested;clinical synthesis,free-text/PDF/OCR and clinical identity matching unfinished.|
|7. Clinical workflow orchestration|PARTIAL|Explicit edit/approval/publication and bounded repeat-safe imports work;complete durable cross-workflow orchestration,retry and follow-up remain engineering work.|
|8. Security,privacy and lifecycle|PARTIAL|MFA/recovery,authorization,consent,session and synthetic encrypted backups tested;identity/proxy,erasure/retention,PHI controls and stable keys remain gaps.|
|9. Production infrastructure and reliability|PARTIAL|Local health/restore/HTTP probes and native outage/retry tests pass;authorized staging,production migrations,KMS,DR,monitoring and sustained load not accepted.|
|10. Premium product experience|PARTIAL|Brand preserved;actual browser layouts inspected and overflow tested;native captures generated,but artifact access and physical-device/full accessibility review remain blocked.|
|11. AI efficiency|PARTIAL|Deterministic formatting/extraction,bounded context and partial checker usage exist;complete usage/cost and evaluated multi-model routing unfinished.|
|12. Final integrated acceptance|PARTIAL|269Python29frontend7Chromium5project23native pass;bounded supported live conversation evidence preserved;account erasure/proxy and full staging/device scenarios not complete.|
|13. Separate release gates|PASS|A bounded local demoPASS;BsyntheticTestFlightFAIL;CclinicalpilotFAIL;DclinicalproductionFAIL;owner dependencies separatelyBLOCKED.|
|14. Final output and evidence|PARTIAL|Readiness/machine matrix/ledger,actual results,release instructions and owner actions prepared;no signed archive,physical-device result,Apple submission or independent clinical approval exists.|

## Release gates and exact remaining actions

| Target | Decision | Remaining acceptance |
|---|---|---|
| A. Bounded synthetic local demonstration |PASS|Supported historical live evidence and current synthetic regressions/native clipboard workflow;not clinical approval.|
| B. Controlled synthetic TestFlight beta |FAIL|Independent native TOTP enrollment/verification,full device/VoiceOver and OS destination acceptance;account/privacy acceptance remain engineering work. Staging,signing,physical device,privacy/support materials and distribution authority are owner BLOCKED.|
| C. Controlled clinical pilot |FAIL|Verified identity/proxy,policy-backed lifecycle/retention,PHI operations and independent clinical/security/privacy/legal/intended-use review incomplete.|
| D. Clinical production |FAIL|Pilot gaps plus broader evaluated clinical evidence/reasoning,operational maturity and applicable regulatory/vendor/contract acceptance unresolved.|

Next independently achievable engineering work:preserve23native cases and add actual native TOTP enrollment/verification;implement tested,fail-closed account deactivation and configurable retention/erasure/holds mechanisms;activation requires recorded policy decisions;verify full native destination delivery on authorized hardware;continue evaluated evidence lifecycle,workflow recovery,PHI operations and full usage/cost routing. Existing deletion requests remain review-only. No independently achievable gap is disguised as a missing credential.

Owner actions:provide authorized synthetic HTTPS staging;stable separate MFA and backup encryption keys via the secret manager with escrow/rotation ownership;approved retention/intended-use/guardian policy and identity service;Apple team/bundle/profiles,physical iPhone and genuine privacy/support URLs;artifact access or direct GitHub screenshot review;qualified clinical/security/privacy/legal/regulatory/vendor review before clinical use. Never place keys in chat or app configuration. Native CI's ephemeral keys do not configure staging.

Reproduce with Python3.12/Node24:run `python -m dev_verify --all --require-browser --output /tmp/raphael-verification`. The existing `development.yml` workflow provisions the actual disposable HTTPS fixture for all23native tests;local runs without private fixture inputs skip authenticated cases and are not acceptance. Development startup remains `python -m dev_start --setup --without-local-ml --provision --database /absolute/new/synthetic.sqlite`;no production database. Existing full Intel Mac/Xcode16.4,backup/restore,staging,physical-device and separately authorized archive steps are retained below. On device,include actual protected destination delivery/revocation/expiry/backgrounding and deletion of deliberate synthetic copies. Xcode16.4 compatibility does not establish current Apple upload-SDK eligibility;the official requirements page was blocked with ProxyError and must be verified before distribution. No upload or deployment is authorized by these reproduction steps.

## Preserved failure evidence and outdated reports

The first export run was22passed/1failed/0skipped;its clipboard case stalled after the copy-status check and hit240seconds. One active synthetic session remained and cleanup failed correctly. The exact OS prompt state was not captured;an OS paste prompt remains an inference. A bounded host-observation follow-up also failed22passed/1failed because its observation endpoint returned422;the exact underlying cause was not captured. That unnecessary endpoint was removed. The final real native Paste workflow passes without changing production authentication or weakening assertions. [Host-observer failure](NATIVE_EXPORT_HOST_OBSERVER_FAILURE_CI.json) and its annotations are retained. [Raw failure](NATIVE_EXPORT_FIRST_FAILURE_CI.json) and [annotations](NATIVE_EXPORT_FIRST_FAILURE_ANNOTATIONS.json) remain. [Cancelled intermediate run](NATIVE_EXPORT_CANCELLED_CI.json) and the initial/corrected local fixture logs also remain;none count as passes.

Older ce27553/4fd9116 reports with267Python/18native,original14native/257Python and earlier7native/193Python or credential-failure claims are historical source snapshots. The updated inventory,roadmap,mobile readiness and PR point here. Actual provider evidence is retained with its original source rather than presented as freshly rerun. 17native screenshot files were exported in this CI run but not manually inspected in this workspace because artifact-host access remains blocked;prior unchanged browser captures were inspected. Do not infer a signed archive,device or visual/accessibility approval from a simulator pass.

---

## Historical native/evidence checkpoint — retained original source scope

Everything below this marker belongs to the earlier4fd9116/ce27553checkpoint. Its counts,then-current gaps and wording are historical;the current checkpoint and machine acceptance above supersede them. Historical failures and exact release/reproduction instructions remain reviewable.

# Previous release-candidate checkpoint

Authoritative checkpoint for the [final RC directive](directives/FINAL_RC_COMPLETION_2026-11-01.md), 2026-11-01 (America/New_York);machine evidence uses UTC. Starting PR head: `3ee4eec89e769e8aa1c50883d0c34fb5098d143c`. Reviewed implementation: `4fd91160b5b05cc28457cd23f67a89b58fb3d5d1`, branch `fix/sqlite-startup-recovery`, draft [PR #1](https://github.com/AIGeniusAlien/Raphael-AI/pull/1). A following evidence-only commit is identified in the PR. No merge, deployment, Apple submission, real patient data or clinical communications occurred.

This increment implements and tests bounded native lifecycle/resilience controls and a supported-evidence scope panel. It does **not** complete the master platform. All directive sections were reconciled;the independently achievable remainder below is a continuation checkpoint,not a declaration that every engineering requirement is finished. The [machine acceptance matrix](RAPHAEL_FINAL_ACCEPTANCE.json) and [requirements ledger](ENGINEERING_LEDGER.json) retain incomplete engineering and external gates separately.

## Reconciliation against every directive section

The [section-by-section record](FINAL_RC_DIRECTIVE_RECONCILIATION.json) is explicit:the entire master directive is **not complete**. PASS below means that reconciliation/output requirement is satisfied,not that the product is launch-approved.

| Section | Status | Evidence / remaining work |
|---|---|---|
|1. Project baseline|PASS|Verified3ee4eec baseline;preserved branch,history and draftPR1.|
|2. Actual acceptance matrix|PASS|1151prior IDs/descriptions preserved;1154current rows;failures and owner dependencies remain explicit.|
|3. Extended native iOS acceptance|PARTIAL|18actual Xcode cases pass including visible foreground recovery;physical device,full VoiceOver,native export and independent native TOTP enrollment/entry remain unverified or incomplete.|
|4. Patient and physician experiences|PARTIAL|Existing chart,consent,documentation,observation and communication sandbox preserved;actual erasure,guardian identity,native export and live clinical integrations incomplete.|
|5. Medical intelligence|PARTIAL|Live9document/48vector/384dimension supported-domain map verified;prior real supported conversations preserved;comprehensive coverage,retraction surveillance and independent clinical evaluation unfinished.|
|6. Longitudinal intelligence and structured extraction|PARTIAL|Authorized chronology,exact arithmetic,immutable context and reviewed tabular extraction tested;clinical synthesis,free-text/PDF/OCR and clinical identity matching unfinished.|
|7. Clinical workflow orchestration|PARTIAL|Explicit edit/approval/publication and bounded repeat-safe imports work;complete durable cross-workflow orchestration,retry and follow-up remain engineering work.|
|8. Security,privacy and lifecycle|PARTIAL|MFA/recovery,authorization,consent,session and synthetic encrypted backups tested;identity/proxy,erasure/retention,PHI controls and stable keys remain gaps.|
|9. Production infrastructure and reliability|PARTIAL|Local health/restore/HTTP probes and native outage/retry tests pass;authorized staging,production migrations,KMS,DR,monitoring and sustained load not accepted.|
|10. Premium product experience|PARTIAL|Brand preserved;actual browser layouts inspected and overflow tested;native captures generated,but artifact access and physical-device/full accessibility review remain blocked.|
|11. AI efficiency|PARTIAL|Deterministic formatting/extraction,bounded context and partial checker usage exist;complete usage/cost and evaluated multi-model routing unfinished.|
|12. Final integrated acceptance|PARTIAL|267Python29frontend7Chromium5project18native pass;bounded supported live conversation evidence preserved;account erasure/proxy and full staging/device scenarios not complete.|
|13. Separate release gates|PASS|A bounded local demoPASS;BsyntheticTestFlightFAIL;CclinicalpilotFAIL;DclinicalproductionFAIL;owner dependencies separatelyBLOCKED.|
|14. Final output and evidence|PARTIAL|Readiness/machine matrix/ledger,actual results,release instructions and owner actions prepared;no signed archive,physical-device result,Apple submission or independent clinical approval exists.|

## Changes and demonstrated scope

- **Foreground blank-workspace recovery:** actual screenshot-derived flags distinguished visible native controls from an empty web surface,with no privacy cover or login form. An accessibility-only mitigation did not resolve it and was removed. After a successful same-identity server check,a read-only Boolean DOM probe has an independent8second deadline;missing/unresponsive content exposes the existing recovery action with a lost-edits warning. Only an explicit user retry replaces the stopped ephemeral context,after another actual server verification,then restores its strictly scoped HttpOnly cookie. View/auth/navigation epochs reject late callbacks;cookie handoff also has an8second deadline. No automatic reload,credential JavaScript or authentication bypass was added. The final actual MFA workflow used this visible retry and then passed owner revocation,one-use recovery rejection and clean relaunch.
- **Native session failures:** non-200 session verification now visibly reports a connection problem and clears that specific warning on healthy recovery. A real 401 still invalidates the local memory-only session. Private HTTP bodies are bounded while streaming, rather than buffering the entire response before checking64KiB. App request deadlines and authentication/consent policies were preserved.
- **Extended actual Xcode acceptance:** original14 cases retained;four new cases exercise recovery-code MFA with one-use rejection, background/foreground, authorized owner revocation, clean relaunch, real server expiry, controlled503 and16second delay/timeout/retry, largest Dynamic Type with reachable password control, and private-document/full-chart consent revocation over actual HTTPS. Actual total:18passed,0failed,0skipped (9unit+9UI-target cases). A UI-target case can exercise HTTP contract operations;that does not turn it into an app download demonstration.
- **Safe disposable fixture:** the fault-control wrapper is absent from normal `main.app`. It requires GitHub Actions,synthetic-only mode,an exact private temporary database path and a private control token. Only fixed healthy/unavailable/slow/one-account-expiry operations exist. No arbitrary SQL or auth bypass. MFA enrollment and TOTP confirmation use the real setup API;two one-use recovery factors reach only the UI-test bundle. The production app bundle receives no test credentials or provider secrets.
- **Log safety and setup:** all ephemeral password/MFA/recovery/control values are in a private redaction registry;missing registry withholds raw output. The prior logging test was corrected to supply that required registry. Provisioning allows one bounded30second request instead of10seconds;mutations are not automatically retried. This is a setup deadline change,not proof of the exact historical timeout's underlying cause.
- **Supported evidence scope:** clinician-only,on-demand read-only coverage panel exposes the indexed source links,domain,population and counts;the API retains titles and publication dates,with explicit unsupported uses. A local fixture backend says live coverage is unchecked. Provider errors are sanitized;patient access is forbidden. Withdrawn or incompatible evidence is not counted as compatible active scope. This panel never writes to Qdrant or requests an embedding.

## Actual verification

| Check | Result / evidence |
|---|---|
| Python |267passed,temporary synthetic databases;includes4CI-control,6old/new redaction,2coverage and23intelligence cases. Subsets overlap the total. |
| Frontend |29actual Node subtests passed. |
| Chromium |7actual scenarios passed;MFA/recovery,paperwork editing/approval,scope refresh,structured observation import and patient-visible trends remain integrated. |
| Mobile project |5tests and structural audit passed on Linux;not counted as simulator results. |
| Safety and requirements |14deterministic protocol cases and full ledger validation pass;no expert clinical adjudication. |
| Current implementation CI |[Run 41284154400](https://github.com/AIGeniusAlien/Raphael-AI/actions/runs/41284154400):regression,dependency/source audit and actual native jobs all PASS on `4fd91160b5b05cc28457cd23f67a89b58fb3d5d1`. [Raw CI evidence](FINAL_RC_CI_RESULTS.json). |
| Actual Xcode |18passed,0failed,0skipped;[actual xcresult counts,HTTP/audit metadata and screenshot export](FINAL_RC_NATIVE_RESULTS.json). |
| Separate backend composition |Actual Linux HTTPS login200/session200/controlled503/recovered200/expired401;[result](FINAL_RC_NATIVE_BACKEND_COMPOSITION.json). This is not native UI or provider verification. |
| Real scope-panel browser |Actual authenticated Qdrant-backed panel renders9source links;0page errors/overflow at1440/768/390/320px. [Results](FINAL_RC_LIVE_COVERAGE_BROWSER.json);actual mobile capture visually inspected. No generated model response. |
| Real indexed evidence |Read-only authenticated Qdrant:9active CDC documents/48vectors/384dimensions,`text-embedding-3-small`;0unrecognized active vectors. [Per-source inventory](FINAL_RC_EVIDENCE_COVERAGE.json). No collection mutation. |
| Real conversations |Prior source-scoped physician+patient support-checked OpenAI/Qdrant workflows preserved on4299e13;[actual responses/source verification](ULTIMATE_LIVE_RESULTS.json). Unchanged provider pipeline was not rerun in this increment. |

`FINAL_RC_VERIFICATION.json` records complete local checks and explicitly separates skipped Linux-native,live-provider and advisory checks. Skips are not passes. Current implementation CI provides separate actual native/security evidence. All native provider request counts are zero;the native backend uses synthetic records and does not inherit provider credentials.

Native session cleanup and actual audit counts: `{"audit_event_counts_including_provisioning": {"list_patients": 14, "login": 13, "logout": 11, "read_notes": 11, "read_patient": 10}, "backend": "actual_main_app_https", "expired_sessions_retained": 1, "provider_requests": 0, "synthetic_sessions_remaining": 0}`. The intentional expiry test retains its expired database row;zero **active** sessions is the cleanup assertion. Owner revocation deletes the selected session. Do not infer every row was deleted or every logout represented a UI action;counts include provisioning and HTTP contract cases.

## Native acceptance boundaries

| Requirement | Verified scope / remaining work |
|---|---|
| Patient/clinician authentication,navigation,chart,note and native/web logout |Original actual HTTPS app workflows retained,including paste/typed credentials and clean process relaunch. |
| MFA and recovery |Actual server enrollment/TOTP confirmation during provisioning;password-only denied;native UI accepts a real recovery factor;reuse denied;owner revocation clears the running app. Independent native TOTP entry/enrollment UI and external identity recovery delivery remain incomplete. |
| Expiry and invalidation |Dedicated synthetic account expires in the actual temporary DB;native poll receives authoritative401 and clears workspace. Owner session revoke uses the existing authenticated account endpoint. |
| Failure and retry |Real HTTPS503 and controlled16second delay exercise readiness timeout/retry and session warning recovery. These are deliberate server faults,not physical airplane-mode or real OpenAI/Qdrant outage proof. Broader patient-data operation recovery remains incomplete. |
| Consent and protected documents |Actual native runner HTTPS operations verify exact synthetic bytes,private consent deny/grant/revoke and full-chart revoke. Native download/export remains disabled/unimplemented. |
| Accessibility |Largest Dynamic Type changes heading height and allows bounded real scrolling to a reachable secure password control;actual attachment captured. Native control accessibility checks remain. Physical VoiceOver gestures/full audit and device acceptance are not claimed. |
| Storage/transport |Existing actual unit cases exercise ephemeral store replacement,cookie scope/expiry,HTTPS/host/navigation policy and secure text binding. Memory-only session deliberately requires sign-in after process restart;no persistent Keychain identity or forensic no-disk guarantee. |
| Release/signing |Linux structural release checks and simulator builds do not establish signed archive validation,current Apple upload eligibility or absence of secrets in an unbuilt owner-signed artifact. No signed distribution build exists. |

## Live medical context and limitations

`ULTIMATE_FIRST_LIVE_RESULTS.json` and `ULTIMATE_LIVE_RESULTS.json` preserve two actual completed supported physician/patient runs. Claim protocol2 constrains paragraph coverage and source-specific quote choices. Independent checks retain complete coverage,exact spans,population applicability,all-reference support,at least one supported claim and no unsupported/uncertain claims. No clinical screen was weakened. Source title/date/hash/excerpt checks against CDC passed;the live missing-evidence query returned0. Automated support screening is fallible and not independent clinical validation.

Actual prior physician excerpt: “Antibiotics do not work against viruses; they are only effective against certain bacterial infections.” [CDC source](https://www.cdc.gov/antibiotic-use/about/index.html),2025-09-23,registered chunk `a428f827958f1d9c18a3d44bb86cf8c4fa38f9813b98f622275528581dbfbdf2`. No clinical note was approved by that conversation. The supported domain map exposes nine small educational sources;it does not establish diagnosis,prescribing,interaction,specialty,pediatric or pregnancy competence. Indexed dates are not a fresh original-source comparison or automated retraction surveillance. New model dimensions are not inferred from configuration:the actual hosted embedding responses were checked in the preserved live runs;this increment separately reads the collection384dimension schema and point model/dimension tags.

Existing implemented bounded workflows remain:encrypted MFA/recovery,six structured paperwork formats and seven transcript preparations,explicit clinician edit/approval/publication,authorized longitudinal chronology and same-unit/source arithmetic,immutable bounded generation snapshots,reviewed CSV-in-text lab/vital extraction,content-free owner-scoped outcome checks,evidence staging/withdrawal/refresh planning,encrypted synthetic backup/restore and durable logout recovery. Their original detailed evidence is preserved in [the previous checkpoint](ULTIMATE_READINESS_CHECKPOINT.md). Free-text/PDF/OCR extraction,clinical causal synthesis,full cost accounting and evaluated multi-model routing remain unfinished.

## Evidence scope and final-head validation

The stored18case success above is explicitly the application-implementation snapshot4fd9116. Later documentation/test-only commits do not rewrite that source-scoped evidence. Current-head CI is linked from draft PR1 and must be checked independently.

Evidence head22b8863 reran all9authenticated UI cases successfully,with13logins/11logouts/0active sessions/1expired row. Its full suite was17passed/1failed because the existing bridge unit fixture allowed10seconds for a new WebKit process that actually took22.6seconds to launch. [Actual failure CI](FINAL_RC_COLD_WEBKIT_FIXTURE_FAILURE_CI.json) and [annotations](FINAL_RC_COLD_WEBKIT_FIXTURE_FAILURE_ANNOTATIONS.json) are retained. The fixture now permits one bounded30second document startup;Promise resolution/rejection assertions and all app/auth deadlines are unchanged. This is a test-host timing correction,not permission to call the failed run a pass.

The recovery log's reported content-process termination is a UI recovery label. Actual diagnostics establish blank/unresponsive content;they do not independently establish the OS termination cause. Failure-only OCR does not replace any workspace assertion.

## Failure history and stale reports

- Baseline3ee4eec CI41274334534 failed provisioning with `httpx.ReadTimeout` at the first login with10second deadline,before Xcode. [Exact annotation](FINAL_RC_BASELINE_NATIVE_SETUP_FAILURE.json). Prior14passed on4299e13 remains historical evidence;the failed follow-up is not a native pass. Exact lower-level timeout cause was not captured.
- First increment9738a87 exposed an old logging test using the obsolete redaction fixture contract. Fixed incc91830 and all266tests subsequently pass;the later progress regression increases the current total to267. [Initial failed suite](FINAL_RC_FIRST_VERIFICATION_FAILURE.json) and [exact regression failure](FINAL_RC_LOGGING_REGRESSION_FAILURE.txt) are retained. CI41276484643 was cancelled when the correction queued;its native run produced no xcresult. [Cancelled attempt](FINAL_RC_CANCELLED_NATIVE_ATTEMPT.json). No cancellation is counted as success.
- Extended runcc91830 /CI41276875857 hit the30minute job limit with no usable xcresult summary. [Timeout CI](FINAL_RC_NATIVE_TIMEOUT_CI.json) and [actual annotations](FINAL_RC_NATIVE_TIMEOUT_ANNOTATIONS.json) are preserved;8logins/7logouts/0active sessions do not establish a test count. The existing redacted artifact could not be downloaded because the network proxy forbids `productionresultssa4.blob.core.windows.net`;the exact stalled action is not known. Follow-up bounds each case at240seconds (maximum300),the compile/test step at25minutes and the job at35minutes,with cancellation-safe redacted last-action reporting. These are diagnostic bounds,not a demonstrated retrospective root cause.
- Bounded run3747c73 /CI41278996649 produced actual15passed/3failed/0skipped,not18passed. [Raw CI](FINAL_RC_BOUNDED_NATIVE_FAILURE_CI.json) and [xcresult/diagnostic annotations](FINAL_RC_BOUNDED_NATIVE_FAILURE_ANNOTATIONS.json) retain all failures. MFA failed its post-background web-identity assertion;later lines269/274 from that async method were attributed to the following native-controls/password-manager cases. This identified `continueAfterFailure=false` async continuation interference. Follow-up uses explicit Swift error propagation,waits for actual background/foreground states and terminates the app in teardown. No authentication or workspace assertion was removed;only subsequent actual full-suite results determine acceptance.
- Isolation follow-upb19c93f /CI41280240220 produced17passed/1failed/0skipped. [Raw CI](FINAL_RC_ISOLATED_LIFECYCLE_FAILURE_CI.json) and [actual diagnostics](FINAL_RC_ISOLATED_LIFECYCLE_FAILURE_ANNOTATIONS.json) show all original core cases restored;only the post-background web workspace remained inaccessible. One active synthetic session remained because the failing guard stopped before owner revocation;cleanup correctly failed rather than deleting rows to manufacture success. An accessibility-container notification mitigation was tried without resolving this failure;it was subsequently removed. The actual bounded content-health/recovery fix is described above. The test permits one actual user retry only when the existing native UI explicitly reports a terminated WebKit content process;actual web identity is still mandatory. Do not infer seamless unsaved DOM continuity after process termination.
- `PRODUCT_COMPLETION_LIVE_FAILURE_2026_11_01.json` retains the old `ClaimScreenUnavailable` failure. Original subphase/raw response was lost;the exact cause cannot be reconstructed retroactively. The later actual supported runs do not erase it.
- `ULTIMATE_BROWSER_GROUPING_FAILURE.log` preserves the earlier source-grouping browser failure byte-for-byte,including original formatting. Later working source retains document-level grouping and line provenance.
- Older14-test/257-test “current” reports and historical MASTER/PRODUCTION/INFRASTRUCTURE reports retain their source scopes. This report supersedes their current-readiness wording. `mobile/submission/READINESS.json` now explicitly labels ancient pending/failed native snapshots historical and points its latest completed field to the actual current result.

All1,151prior requirement IDs and descriptions,including1,136original master rows,are preserved. Three bounded RC rows were added:1,154total. [Preservation/count evidence](FINAL_RC_REQUIREMENTS.json). Broad requirements are not promoted to complete merely because one bounded RC case passes.

The failure-only Vision OCR helper belongs exclusively to the UI-test bundle and reports fixed UI flags;it is not clinical document OCR or a patient-facing feature. [Complete CI attempt history](FINAL_RC_RUN_HISTORY.json) also preserves the cancelled intermediate recovery run;cancelled runs never count as passes.

## Four release gates

| Target | Decision | Cause / next acceptance |
|---|---|---|
| A. Bounded synthetic local demonstration |PASS |Supported education conversations preserved;current synthetic chart,MFA,documentation,extraction,scope panel and bounded native simulator workflows pass. |
| B. Controlled synthetic TestFlight beta |FAIL |Native protected export,remaining account/privacy acceptance and complete device/accessibility acceptance unfinished. Authorized HTTPS staging,Apple signing,physical device,privacy/support metadata and distribution authorization are BLOCKED owner gates. |
| C. Controlled clinical pilot |FAIL |Verified identity/proxy/erasure/retention,PHI infrastructure,operations and independent clinical/security/privacy/legal/intended-use evaluation incomplete. |
| D. Clinical production |FAIL |Pilot gaps plus broader rights-reviewed clinical evidence,evaluated reasoning,operational maturity and applicable regulatory/vendor/contractual approval unresolved. |

No HIPAA compliance,medical accuracy,Apple approval or launch readiness is implied by tests. External owner gates are BLOCKED;separately achievable software gaps remain FAIL rather than being mislabeled credential blockers.

## Remaining implementation and exact continuation

1. **Native release:**implement reviewed protected export with authorization,memory/file handling,share confirmation and cleanup;test server revocation/expiry during transfer,interruption and cross-patient/tenant denial. Complete independent native TOTP path and device VoiceOver/navigation/keyboard/large-text acceptance. Preserve all18current tests.
2. **Account lifecycle and identity:**verified identity,recovery delivery,guardian/proxy grants and policy-backed deactivation/erasure/retention/holds. Existing account export and deletion-review requests are not actual erasure. Engineering can build mechanisms;owner must approve retention/intended-use/guardian policy and identity provider.
3. **Clinical intelligence:**rights-reviewed incremental connectors,source refresh/retraction/supersession surveillance,hybrid/rerank evaluation,medication safety and independent expert reference cases across populations. Do not treat bibliographic titles/abstracts as clinical full-text support or relax claim checks.
4. **Longitudinal/extraction/orchestration:**evaluate conflict/staleness/missing-fact relevance and clinical identity;extend reviewed formats without overwriting records. Finish durable status/idempotency/recovery across workflow stages before adding unrelated templates. Keep clinical finalization and external actions explicit.
5. **Operations:**production migration/storage encryption/KMS rotation,distributed budgets,content-safe monitoring/alerts,backup/DR/rollback/incident exercises and sustained load. Existing40operation/4worker local probe (p50 11.962ms,p95 23.889ms) and16MiB encrypted synthetic restore tests are preserved historical results,not production capacity/DR acceptance.
6. **Efficiency:**measure full generation/embedding/checker latency,usage,cost and failure rates;evaluate routing against expert quality gates. Prior two checker calls used11,387tokens excluding generation/embedding;no full monetary cost claim.

## Reproducible verification and owner release steps

1. Check out the reviewed source above. Use Python3.12 and Node24. Run `python -m dev_start --setup --without-local-ml --provision --database /absolute/new/synthetic.sqlite`;the launcher is loopback-only. Add `--live-services` only with authorized secret bindings. Intel macOS avoids unsupported optional modern Torch wheels;the exact owner hardware has not been executed here.
2. Run `python -m dev_verify --all --require-browser --output /tmp/raphael-verification` with installed Playwright Chromium. Explicit live regression,when required,is `python -m scripts.verify_live_medical --live --output /tmp/raphael-live.json`;it creates a new synthetic DB and does not approve AI notes. Do not repeatedly retry authentication errors.
3. Read-only evidence review:use clinician **Supported evidence scope → Refresh**,or `python -m curated_evidence --check-refresh` for original-source refresh planning. Any future corpus write/withdrawal is separately explicit;never recreate or retag an incompatible collection. New RC inventory made zero remote writes.
4. Configure stable separate protected `RAPHAEL_MFA_ENCRYPTION_KEY` and `RAPHAEL_BACKUP_ENCRYPTION_KEY` through the environment secret manager;both were absent in inherited development configuration. Five provider bindings were available;no values are printed. Keep keys backed up separately,define rotation and recovery,never delete/replace them blindly. Native CI's ephemeral MFA key does not configure staging.
5. Synthetic backup:with `RAPHAEL_DEV_SYNTHETIC_ONLY=1`,run `python -m dev_backup backup --database-url sqlite:////absolute/synthetic.sqlite --output /protected/new.enc`;restore via `python -m dev_backup restore --source /protected/new.enc --output /protected/new-restored.sqlite`. Existing files are never overwritten. Authenticate ciphertext,verify schema/integrity/foreign keys,start matching source and inspect authorization before cutover. Keep source+encrypted pre-upgrade snapshot;never run a pre-MFA schema against a newer DB. No live rollback/deployment was performed.
6. On owner's Intel Mac/macOSSequoia15.8.2,select full Xcode16.4 where supported,run `xcodebuild -version`,open `mobile/ios/Raphael.xcodeproj`,select an installed iPhone UDID and set `RAPHAEL_SIMULATOR_DESTINATION='platform=iOS Simulator,id=ACTUAL_UDID'`. Run `bash mobile/scripts/native_checks.sh`. Without private disposable CI inputs,authenticated tests skip and must not count as passes. The actual workflow on this commit automatically provisions HTTPS trust,random users and real MFA for CI only. Repeat through `gh workflow run development.yml --ref fix/sqlite-startup-recovery` after confirming that ref still matches the reviewed source. Owner hardware/Xcode16.4 execution remains unverified;confirm current Apple SDK upload rules separately.
7. Owner provides approved synthetic HTTPS staging,stable secrets,TLS/bound host,trusted proxy policy,encrypted persistent storage,operator access,monitoring/backup/incident ownership;do not expose a production PHI deployment using development defaults. Rehearse migration,restore and rollback against synthetic copies. No staging was deployed here.
8. Owner provides Apple team,registered bundle,profiles,physical iPhone/Developer Mode,approved icon and genuine privacy/support URLs,tester/review metadata,intended use and retention policy. Copy ignored `mobile/ios/Configuration/Local.xcconfig.example` to `Local.xcconfig` and enter only approved public host/team/bundle/support inputs. No provider keys or CI passwords belong in app config. Complete qualified clinical/security/privacy/legal/regulatory/vendor/contract review before clinical use.
9. On the physical phone,verify patient+clinician login,MFA/recovery,chart/note authorization,private consent,expiry/revoke,background/relaunch,airplane mode/recovery,Dynamic Type/VoiceOver and protected export after implementation. Record OS/device/build and actual results. Obtain CI screenshots/xcresult from the run artifact;dynamic Actions artifact-host access was blocked in this environment,no native images were visually inspected here. Browser captures are in `docs/artifacts/final-rc/`;mobile trend,account and live scope-panel captures were visually inspected and current browser overflow checks pass.
10. Only after software/device/owner gates pass,and **separate explicit submission authorization**,validate an owner-signed Release archive in Organizer,finish App Store Connect privacy/review metadata and authorize upload/tester invitations. No signed archive,upload,install link or invitation exists. Retained detailed checklist: `mobile/submission/REVIEW_AND_TESTFLIGHT.md`.
