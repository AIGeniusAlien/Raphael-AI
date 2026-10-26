# Production engineering implementation — 2026-10-26

This is a working development increment on fix/sqlite-startup-recovery and draft PR1,not a production or clinical release. The full new directive is retained in RAPHAEL_MASTER_SPEC.md. Existing clinical/portal/evidence architecture and all1,117baseline requirement IDs/descriptions are preserved. See PRODUCTION_ENGINEERING_RESULTS.json for exact verification,live and CI outcomes.

| Critical capability | Classification | Actual evidence and limit |
| --- | --- | --- |
| Portal authorization/consent/session isolation | IMPLEMENTED AND VERIFIED | Existing chart/thread/intake gates preserved;full Python and actual browser regression. Independent identity/privacy/security validation remains open. |
| Request budgets,Fetch Metadata CSRF and denial headers | IMPLEMENTED AND VERIFIED | New positive/negative expiry,spoofing,capacity,malformed-origin tests and real HTTP header checks. Single-process limits are not a distributed gateway. |
| Operator synthetic credential recovery | IMPLEMENTED AND VERIFIED | Three tests prove target-only session revocation,old-password denial,new-password login,operator audit and refusal of weak/disabled/non-synthetic resets. No email/SMS or production identity recovery. |
| One-command verification | IMPLEMENTED AND VERIFIED | Actual credential-isolated runner emits logs/results.json,critical failures nonzero,timeout/redaction tests and explicit skips. Local advisory audit is separate. |
| Cross-platform startup | PARTIALLY IMPLEMENTED | Python launcher,private provisioning,pinned setup and safe loopback mode implemented;Linux actual startup/restart passed. Windows/macOS installation/runtime not executed. |
| Clinical safety protocol automation | PARTIALLY IMPLEMENTED | Fourteen deterministic protocol cases with100%threshold;ten medical categories explicitly require adjudicated expert references. Not model medical accuracy. |
| Actual evidence-grounded AI | PARTIALLY IMPLEMENTED | Historical physician+patient live complete result retained. Current-code live attempt outcomes recorded separately;automated support can correctly block a draft. Six-source corpus is not comprehensive coverage. |
| Native readiness utility | IMPLEMENTED BUT NOT VERIFIED | Actual Swift source sends no patient data/cookies,denies redirects,bounds response/timeouts and rejects legacy readiness. Added native XCTest has not run locally. |
| iOS project structural checks | IMPLEMENTED AND VERIFIED | Actual project parser/icon/resource/target audit and five tests pass;these are not Swift compilation. |
| Native iOS build/device/signing | BLOCKED BY EXTERNAL DEPENDENCY | No local macOS/Xcode/device/signing. macos-15 simulator CI configured;only an actual completed CI result can establish remote build success. |
| CI gates | PARTIALLY IMPLEMENTED | Read-only PR/main workflows run consolidated regression,advisory/source scans and unsigned simulator tests with retained artifacts. Remote execution status is recorded,not inferred from YAML. |
| Dependency/source scanning | IMPLEMENTED AND VERIFIED | Fresh86-package advisory check:zero known vulnerabilities. High-severity/high-confidence Bandit scan:zero findings/scan errors. Lower severity/manual penetration review remains. |
| Brand consistency | PARTIALLY IMPLEMENTED | Existing web logo,palettes/layouts and native assets preserved;SHA256inventory identifies conflicting web/iOS marks. Owner's final icon authority unavailable. |
| Account erasure,public verified recovery,MFA,guardian permissions | NOT IMPLEMENTED | Review-only deletion/operator recovery are bounded substitutes for development,not completion of these release requirements. |
| Android native | NOT IMPLEMENTED | Shared backend remains;no APK/source/build claim. |
| Clinical/production/App Store release | NOT SAFE FOR RELEASE | Independent clinical/privacy/security/legal/operational assessment,BAAs/PHI controls,policy-backed deletion,native device/build/signing and owner approvals remain. No deployment/upload performed. |

## Concrete software changes

- `python -m dev_verify --all --require-browser` runs supported checks using isolated child environments:provider credentials,backend flags and inherited database paths are removed. Logs redact configured secrets;timeout cleanup terminates test process groups. Reports distinguish tests,skips,failures and supported versus external checks. Explicit --live is required for real development provider checks;no mocks can establish live success.
- `python -m dev_start --setup --provision` supplies a Python-based Windows/macOS/Linux entry point,installs in .venv,uses a new synthetic database,privately prompts for accounts and starts loopback. `--database` preserves an existing synthetic path for restart;`--live-services` explicitly uses the secure configured services. There are no shared/default passwords or inherited production databases.
- Portal budgets default to300requests/minute and12expensive-generation attempts/minute per direct peer,with10,000bounded keys. Forwarded headers cannot fabricate a new identity. Existing persistent login-failure throttling is preserved. Cross-site Fetch Metadata and malformed Origins fail closed;security headers also cover legacy API denial responses. HTTPS gets HSTS;local HTTP does not pretend to be secure transport.
- `python -m auth.dev_recover` provides local operator recovery only:private double password prompt,active tenant-matched user,atomic replacement/session revocation and explicit operator audit. It neither reactivates users nor deletes clinical records. UI identifies the absence of public self-service recovery.
- Fourteen reproducible protocol checks exercise emergency/medication refusal phrases,age handling,citation span integrity,unknown references and conflicting verdicts. Labels state the synthetic fixtures do not evaluate diagnosis or medical truth. Pregnancy/rare-disease/dose/real emergency accuracy remain expert-validation gaps.
- Native Settings has a separate ephemeral,no-cookie readiness utility with10second timeouts,64KiB bound,redirect denial and explicit synthetic-only response validation. Native test results use unique bundles so reruns preserve earlier evidence. No new logo/design replacement,provider key or medical claim is introduced.
- GitHub Actions now includes actual macOS simulator build/test commands and uploads result bundles. No deployment,merge,App Store or clinical communication job exists. Privacy/HIPAA evidence maps controls to code,tests,owner responsibilities and unresolved operational/legal gaps.

## Run and inspect

Exact Windows/macOS/cloud commands and browser walkthrough:AUTOMATED_DEVELOPMENT.md. In this existing cloud checkout:

```bash
cd /workspace/Raphael-AI
/workspace/.raphael/venv/bin/python -m dev_verify --all --require-browser --output /workspace/.raphael/owner-verification
/workspace/.raphael/venv/bin/python -m dev_start --provision
```

Use the printed synthetic database path for restart;do not create production records. The server is loopback8000;open locally or via an owner-configured private forwarding mechanism. No hosted preview/staging URL exists. Accounts are privately prompted,tenant synthetic. Clinician creates a linked synthetic patient;patient uses a separate browser session. For live AI,add --live-services only when the existing secure development bindings are available. Source-linked AI drafts still require review;blocked drafts are not approved advice.

On macOS,use mobile/README.md for simulator commands,configure an authorized synthetic HTTPS origin and run unsigned native tests. Owner bundle/signing/support/privacy/brand approvals remain required before device distribution;the workflow has not uploaded anything. Windows/macOS Python setup still needs actual target-host verification. Draft App Store materials and review-only deletion do not satisfy submission readiness.

## Next three actions

1. Inspect actual PR CI/native results;fix build/test failures before claiming any iOS compilation success. Then exercise authenticated synthetic native device/accessibility/offline/privacy behavior.
2. Build clinician/pharmacist-adjudicated medical reference sets and evaluate support,abstention,age/pregnancy/medication/emergency risks across versioned models;do not relax failed support checks to improve completion rates.
3. Resolve owner retention/deletion,identity/guardian/MFA,HTTPS staging/at-rest encryption,tamper-evident audit,incident response and vendor BAAs/privacy/security/clinical review before shared PHI or TestFlight release. No production deployment is authorized.
