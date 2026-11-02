# RAPHAEL AI — DEFINITIVE PRODUCT COMPLETION & WORLD-CLASS LAUNCH ENGINEERING DIRECTIVE

## MISSION: FINISH THE APPLICATION, NOT ANOTHER AUDIT

You are the lead engineering organization responsible for bringing Raphael AI from its current development state to a world-class, secure, reliable, polished, integrated healthcare application.

This is a major implementation and product-completion assignment.

The previous engineering and release-audit runs are finished. Their findings must now drive actual implementation, not another cycle of diagnostics and status reports.

Your responsibility is to implement every independently achievable missing requirement, resolve critical failures, validate complete patient and clinician journeys, strengthen the medical intelligence system, and prepare the software for a controlled release.

**Do not stop because regression tests pass. Do not stop because documentation is complete. Do not stop after fixing only one failure.**

A completed Codex run is not equivalent to a completed Raphael application.

The target is the highest-quality release candidate achievable within the available environment, with no hidden critical defects and no unsupported release claims.

No one can guarantee zero bugs or perfect clinical accuracy. Therefore, use explicit acceptance criteria, adversarial testing, documented residual risk, and independent review requirements rather than claims of perfection.

---

# PART I — AUTHORITATIVE PROJECT CONTINUITY

## 1. Preserve the entire Raphael project

Repository: `AIGeniusAlien/Raphael-AI`

Existing development branch: `fix/sqlite-startup-recovery`

Existing draft PR: #1

Latest owner-reported completion commit:

`a7e79cb792121b794af6aca110816cd6d123843c`

Verify the actual repository HEAD, PR state, and commit ancestry before making changes. Do not assume this commit is still HEAD.

Read the full existing master specification and current release evidence, including:

- `RAPHAEL_MASTER_SPEC.md`
- `docs/RC_RELEASE_AUDIT.md`
- `docs/RC_VERIFICATION_RESULTS.json`
- `docs/RC_CI_RESULTS.json`
- `docs/RC_REQUIREMENTS_AUDIT.json`
- `docs/MASTER_ENGINEERING_REPORT.md`
- `docs/RELEASE_READINESS.md`
- `docs/SECURITY_PRIVACY_READINESS.md`
- `docs/REQUIREMENTS_MATRIX.md`
- `mobile/submission/READINESS.json`
- All relevant medical evidence, native iOS, integration, security, and release reports.

Preserve the existing requirements ledger, including all original requirement identifiers and descriptions.

Preserve Raphael's established design, brand assets, working clinical features, database migrations, evidence provenance, consent rules, authorization boundaries, and historical failure artifacts.

Never reset existing work, silently replace newer code, or weaken existing safeguards.

Do not merge PR #1, deploy production services, publish to Apple, or process real patient data without explicit owner authorization.

## 2. Current verified baseline

The latest reported engineering run established:

- 213 Python tests passed.
- 26 frontend tests passed.
- 6 Chromium scenarios passed.
- 5 mobile structural tests passed.
- Real OpenAI/Qdrant physician and patient browser workflows passed.
- Nine active CDC evidence documents and 48 vectors at 384 dimensions.
- Regression and source-security CI passed.
- Authenticated native iOS testing failed: seven passed, one failed, zero skipped.
- iOS distribution prerequisites remain blocked.
- Comprehensive medical knowledge and independent clinical validation remain incomplete.
- Production MFA, identity, guardian permissions, account erasure, and PHI safeguards remain incomplete.
- Clinical-production readiness remains failed.

Treat these as the starting point, not final acceptance.

Verify all claims against the current source and recorded test artifacts.

---

# PART II — BREAK THE REPEATED NATIVE iOS FAILURE CYCLE

## 3. Resolve the authenticated iOS workflow as the first critical engineering priority

The authenticated iOS test has repeatedly failed with a WebKit accessibility error involving remote accessibility element resolution and a 60-second timeout.

Prior attempts added diagnostics, changed navigation behavior, and adjusted UI-test gestures, but did not establish successful authenticated native operation.

Do not repeat the same strategy indefinitely.

Perform a root-cause investigation that distinguishes:

1. Actual app authentication failure.
2. WebKit accessibility infrastructure failure.
3. UI-test automation failure.
4. Native/WebView lifecycle problems.
5. Backend readiness or HTTPS configuration problems.
6. Simulator-specific failures.

Inspect the real app and actual backend behavior independently of the failing accessibility locator.

Use available simulator screenshots, xcresult artifacts, app logs, HTTP request traces, and server-side session evidence without exposing credentials.

Create a minimal reproducible test for the failing interaction.

Establish whether the user-facing sign-in flow actually works even when the accessibility test infrastructure fails.

### Required architecture decision

If repeated evidence shows that WebKit-based authentication is the source of fragility, evaluate implementing a secure native SwiftUI authentication experience backed by Raphael's existing FastAPI authentication/session system.

Do not replace the entire application unnecessarily.

Prefer the smallest maintainable architecture that delivers:

- Native, accessible sign-in controls.
- Secure session establishment.
- Correct cookie/token lifecycle.
- CSRF and origin protection.
- Session invalidation.
- Role-specific navigation.
- Consistent native/WebView session behavior.
- Correct error and expiry handling.

Do not bypass login, inject credentials through JavaScript, weaken TLS, or manufacture successful tests.

The authenticated iOS acceptance test must exercise actual authentication and protected application functionality.

### Required native acceptance

Demonstrate:

- Patient sign-in.
- Clinician sign-in.
- Protected chart access.
- Clinical note access.
- Role isolation.
- Session persistence.
- Logout.
- Server-side session invalidation.
- Session expiry.
- Network failure and recovery.
- Accessibility.
- Secure HTTPS communication.
- App relaunch behavior.

Execute actual Xcode simulator tests.

If the remote accessibility service remains broken, use independent, appropriately strong integration evidence to isolate the infrastructure defect while keeping the unresolved UI test visible as failed. Do not relabel the entire native gate PASS without sufficient equivalent end-to-end evidence and documented acceptance criteria.

A source-code review or successful compilation alone is not enough.

---

# PART III — COMPLETE THE PATIENT AND PHYSICIAN APPLICATIONS

## 4. Deliver complete integrated patient workflows

The patient experience must be intuitive, secure, visually refined, and functional.

Complete the supported workflows for:

- Account registration or authorized provisioning.
- Verified authentication and recovery.
- Patient profile.
- Medical history.
- Allergies and medications.
- Conditions and diagnoses as documented.
- Laboratory results and vital trends.
- Medical documents.
- Appointments and care information.
- Authorized clinician connections.
- Consent and revocation.
- Secure clinical communications.
- Reviewed care summaries.
- Patient education.
- Data access and export.
- Account lifecycle controls.

Do not present unsupported features as operational.

All workflows must persist correctly, enforce patient-level authorization, and survive application restarts.

## 5. Deliver complete physician workflows

Complete a professional clinical workspace with:

- Patient selection and authorized chart access.
- Encounter management.
- Longitudinal clinical review.
- Structured history and examination documentation.
- Clinical decision-support preparation.
- Evidence-grounded medical questions.
- Documentation drafting and editing.
- Versioned review and approval.
- Referrals and discharge preparation.
- Care plans.
- Secure communications.
- Patient education preparation.
- Clinical source inspection.
- Auditability and provenance.

Clinical actions must require appropriate human authorization.

Never allow AI output to become a signed clinical document, prescription, order, diagnosis, or external communication automatically.

## 6. Complete the entire user journey

Test workflows across the real system rather than isolated endpoints.

A patient must be able to enter information, share authorized information with a clinician, have the clinician review it, receive an approved response or document, and see the appropriate result in the patient application.

Verify correct persistence, notifications or status indicators, consent enforcement, revocation, audit trails, and error recovery.

---

# PART IV — DEVELOP A SERIOUS MEDICAL INTELLIGENCE PLATFORM

## 7. Move beyond the tiny development corpus

The existing nine CDC documents and 48 vectors establish only a small proof of evidence retrieval.

This is not comprehensive clinical coverage.

Build an extensible, production-oriented medical evidence architecture.

Implement robust source connectors and ingestion pipelines for authorized and legally reusable content from relevant sources such as:

- PubMed/MEDLINE.
- Permitted PubMed Central content.
- CDC.
- NIH.
- FDA/DailyMed.
- AHRQ.
- USPSTF.
- WHO.
- Licensed or permitted specialty guidelines.
- Authorized institutional and EHR resources.

Do not claim access or rights that have not been established.

Do not ingest copyrighted full text without appropriate permission.

Maintain exact source metadata, provenance, publication dates, review dates, applicable populations, licensing, hashes, supersession, and retraction status.

Support repeat-safe incremental ingestion, scheduled refresh mechanisms, source withdrawal, and embedding compatibility.

Preserve the existing Qdrant collection and prevent incompatible model writes.

### Clinical retrieval and reasoning

Improve:

- Hybrid retrieval and reranking.
- Source relevance.
- Clinical question decomposition.
- Patient-population matching.
- Temporal relevance.
- Source disagreement handling.
- Claim-level citations.
- Evidence sufficiency assessment.
- Unsupported-claim rejection.
- Medication safety.
- Emergency recognition.
- Clinical uncertainty.
- Differential diagnosis assistance within approved intended use.
- Separation of general medical evidence from individual patient facts.

Do not invent sources or medical findings.

Do not reduce citation thresholds merely to improve pass rates.

If comprehensive coverage cannot be achieved with authorized sources, clearly scope the supported medical domains and block unsupported uses.

### Medical evaluation

Create a clinically meaningful evaluation framework using independently reviewed reference cases where available.

Cover:

- Common outpatient conditions.
- Emergency presentations.
- Cardiovascular disease.
- Infectious disease.
- Medication safety.
- Pediatrics.
- Pregnancy.
- Geriatrics.
- Chronic disease.
- Complex differential reasoning.
- Rare or low-evidence presentations.
- Contraindications and uncertainty.

Separate automated protocol compliance from expert-adjudicated medical accuracy.

Measure evidence correctness, factuality, unsupported assertions, unsafe recommendations, appropriate abstention, and performance by population.

Do not declare medical reasoning clinically validated without qualified independent review.

---

# PART V — INTELLIGENT DOCUMENTATION AND PAPERWORK

## 8. Complete clinical documentation assistance

Build beyond the existing seven deterministic preparation templates.

Deliver practical, editable, evidence-aware documentation workflows for:

- SOAP notes.
- H&P.
- Progress notes.
- Consultation notes.
- Discharge summaries.
- Referrals.
- After-visit summaries.
- Care plans.
- Medical necessity letters.
- Prior authorization preparation.
- Insurance appeals.
- Structured forms.
- Coding and billing suggestions requiring verification.

Ground every patient-specific statement in authorized patient information, clinical input, or documented evidence.

Missing information must be visible rather than fabricated.

Implement structured field extraction, provenance, draft validation, clinician editing, version control, explicit approval, and safe publication.

Do not automatically submit orders, claims, prescriptions, or external forms without authorized integrations and clinician approval.

Test every implemented document workflow end to end.

---

# PART VI — PREMIUM DESIGN AND MOBILE PRODUCT QUALITY

## 9. Deliver a world-class interface

Preserve Raphael's established branding and visual identity.

Do not replace the existing design with a generic dashboard or an unrelated theme.

Create a cohesive design system across patient web, clinician web, and iOS.

Refine:

- Typography.
- Navigation.
- Layout consistency.
- Information hierarchy.
- Dashboard clarity.
- Medical chart readability.
- Forms.
- Error messages.
- Loading states.
- Empty states.
- Accessibility.
- Responsive behavior.
- Dynamic Type.
- Screen-reader support.
- Keyboard navigation.
- Color contrast.
- Performance.
- Mobile interaction quality.

Run real rendered browser checks at desktop, tablet, and mobile sizes, including enlarged text.

Capture and inspect screenshots.

Fix overflow, clipping, confusing navigation, inaccessible controls, and inconsistent styling.

Ensure patient and clinician experiences feel like one polished product.

## 10. Finish the native iOS product

Existing project:

`mobile/ios/Raphael.xcodeproj`

Target owner development hardware:

- Intel MacBook Pro, 2020.
- macOS Sequoia 15.8.2.
- Xcode 16.4.

Maintain compatibility with that development environment where technically feasible.

Complete native navigation, authentication, secure backend configuration, session handling, role-specific experiences, accessibility, offline/error handling, privacy-sensitive storage, and application lifecycle behavior.

Use approved Raphael icons and assets.

Ensure Release builds contain no secrets, test credentials, reserved `.test` hosts, or development-only configurations.

Prepare a reproducible iOS build and signing procedure.

Do not claim a working physical-device build unless it has actually been executed.

---

# PART VII — PRODUCTION SECURITY AND PRIVACY

## 11. Implement the missing security controls

The current release audit explicitly identifies production identity, MFA, guardian permissions, account erasure, and PHI safeguards as incomplete.

These are engineering tasks requiring implementation, not merely documentation.

Build and test the technical controls needed for:

- Strong authentication.
- MFA.
- Secure account recovery.
- Verified identity workflows.
- Role and tenant isolation.
- Patient-level authorization.
- Guardian/proxy permissions.
- Consent management.
- Session revocation.
- Data access and export.
- Account deactivation and deletion.
- Clinical-record retention exceptions.
- Legal holds.
- Encryption in transit and at rest.
- Secure backups.
- Audit logging.
- Key and secret management.
- Safe document processing.
- Abuse prevention.
- Security monitoring.

Implement configurable policy enforcement where legal or institutional decisions are required.

Do not invent those policy decisions.

Account deletion must accurately distinguish deletion requests, deactivation, legally retained clinical records, and actual erasure.

Run negative security tests, dependency scans, static analysis, authorization tests, and threat modeling.

Prepare evidence for independent penetration testing and privacy review.

Do not claim HIPAA compliance solely because technical safeguards exist.

---

# PART VIII — PRODUCTION INFRASTRUCTURE AND INTEGRATIONS

## 12. Engineer a reliable production architecture

Evaluate and implement the necessary production path for:

- Database migrations.
- Concurrent transactions.
- Persistent storage.
- Service deployment.
- HTTPS.
- Monitoring.
- Health checks.
- Audit retention.
- Backup and restoration.
- Disaster recovery.
- Secret rotation.
- Dependency failure handling.
- Rate limiting.
- Capacity management.
- Performance testing.
- Secure staging.
- Deployment rollback.
- Operational documentation.

Preserve all existing synthetic development data and historical evidence.

Never silently destroy databases or replace production configuration.

Use external infrastructure only when authorized.

Where cloud accounts, DNS, contracts, or credentials are unavailable, complete the application-side engineering and provide exact integration requirements.

## 13. Integrations

Review and advance existing:

- FHIR R4.
- SMART-on-FHIR.
- EHR sandbox integration.
- Voice and ambient documentation.
- DICOM and medical imaging.
- Multimodal clinical input.
- Secure clinician communication.
- External clinical services.

Use real authorized integrations when available.

Otherwise, implement maintainable adapters, contract tests, and accurately labeled sandbox capabilities.

Never represent fixtures as live hospital integrations.

---

# PART IX — ENGINEERING EXECUTION RULES

## 14. Do not repeat unsuccessful approaches

For any critical failure that has survived multiple prior runs:

1. Identify the previous attempted fixes.
2. Establish what each attempt actually proved.
3. Form competing root-cause hypotheses.
4. Create a minimal reproducible failure.
5. Test hypotheses with observable evidence.
6. Choose a durable architectural or implementation fix.
7. Execute regression and end-to-end verification.
8. Preserve unsuccessful evidence.

If a strategy fails repeatedly, change the strategy rather than adding more logging indefinitely.

This specifically applies to authenticated iOS testing.

## 15. Do not substitute reports for engineering

The primary deliverable is working, verified software.

Reports are required only to document implementation and evidence.

Do not spend the entire run rewriting readiness documents.

Do not mark a requirement complete because a placeholder, interface, test stub, or plan exists.

Do not claim a workflow is implemented unless it is usable and integrated.

## 16. Execute in implementation milestones

### Milestone A — Critical native authentication

Resolve or conclusively isolate the repeated iOS failure, implement the durable fix, and rerun native acceptance.

### Milestone B — Complete product workflows

Finish patient, clinician, consent, documentation, and account-management journeys.

### Milestone C — Medical intelligence

Expand authorized knowledge infrastructure, improve retrieval and reasoning safeguards, and establish meaningful clinical evaluation.

### Milestone D — Security and operations

Complete independently achievable production identity, privacy, account lifecycle, infrastructure, and security work.

### Milestone E — Premium experience

Refine web and iOS interfaces and inspect actual rendered behavior.

### Milestone F — Final verification

Execute full end-to-end, native, live-service, security, performance, and release checks.

Commit verified work incrementally to the existing development branch.

Continue across milestones rather than terminating after the first successful test.

If an external dependency blocks a milestone, continue other independent work and record the blocker precisely.

---

# PART X — FINAL ACCEPTANCE CRITERIA

## 17. Complete-system acceptance

The final verification must demonstrate:

- Patient onboarding and authentication.
- Clinician authentication.
- Correct tenant and patient authorization.
- Chart creation and review.
- Consent grant and revocation.
- Evidence-grounded physician and patient interactions.
- Safe rejection of unsupported medical claims.
- Documentation generation, editing, and approval.
- Patient-visible approved results.
- MFA and secure recovery.
- Guardian/proxy access where supported.
- Account export and policy-backed deletion.
- Native authenticated iOS workflows.
- Session expiry and logout.
- Accessibility.
- Responsive layouts.
- Error and outage recovery.
- Secure configuration.
- Production deployment readiness checks.

All tests must use synthetic data unless separately authorized under an approved clinical evaluation protocol.

Distinguish actual service calls from mocks.

A skipped or failed critical test cannot be counted as passed.

## 18. Separate release decisions

Produce independent readiness decisions for:

**A. Synthetic local development demonstration**

**B. Controlled synthetic TestFlight beta**

**C. Controlled clinical pilot**

**D. Clinical production**

Each must be marked PASS, FAIL, or BLOCKED with supporting evidence.

For synthetic TestFlight distribution, require actual authenticated native acceptance, authorized HTTPS staging, appropriate account/privacy controls, signing, device verification, approved metadata, and owner authorization.

For clinical production, require independent clinical validation, appropriate regulatory/legal analysis, applicable BAAs, security and privacy assessment, operational readiness, and accountable clinical governance.

Do not claim clinical or regulatory certification based on software tests.

---

# PART XI — FINAL DELIVERABLES

## 19. Required evidence

Provide:

1. Exact final commit SHA and branch.
2. Summary of implemented functionality.
3. Requirement reconciliation.
4. Full backend test results.
5. Frontend and browser acceptance results.
6. Actual authenticated native iOS results.
7. Screenshots and UI inspection evidence.
8. Real OpenAI/Qdrant integration results.
9. Evidence corpus inventory.
10. Clinical documentation verification.
11. Security and privacy implementation results.
12. Performance and reliability results.
13. Production deployment and rollback procedures.
14. Intel Mac/Xcode 16.4 setup instructions.
15. Physical iPhone testing instructions.
16. TestFlight preparation steps.
17. Exact owner-only external actions.
18. Separate release-readiness decisions.

Maintain one authoritative report:

`docs/RAPHAEL_FINAL_LAUNCH_READINESS.md`

Include a machine-readable acceptance matrix.

For every failed or blocked requirement, identify the precise cause, attempted remediation, responsible party, and next verification step.

Do not overwrite historical failures.

## 20. Final execution instruction

Start from the verified latest repository state.

Do not reset, merge, deploy, or publish.

Do not stop at planning.

Do not stop after passing existing regression tests.

Do not stop after producing another audit.

Implement missing features, resolve recurring failures, strengthen the medical intelligence engine, complete the patient and physician applications, finish the native iOS experience, harden security, and verify the complete integrated system.

Continue until all independently achievable engineering work is complete within the available execution budget, or document the exact remaining work and blockers without misrepresenting them as complete.

The final result must demonstrate substantial new working functionality, not merely additional documentation or diagnostics.

**Raphael must be engineered to a world-class standard, with objective release gates and no unsupported claims of readiness.**
