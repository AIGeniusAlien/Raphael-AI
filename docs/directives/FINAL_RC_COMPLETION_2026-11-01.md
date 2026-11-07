# RAPHAEL AI — FINAL RELEASE-CANDIDATE COMPLETION & ENGINEERING EXECUTION

## PRIMARY OBJECTIVE

Complete Raphael AI's remaining engineering requirements and deliver a polished, secure, reliable, fully tested release candidate suitable for a controlled synthetic-data iPhone beta, subject to required Apple distribution and owner approvals.

Continue developing the broader clinical platform toward the original world-class product vision.

**This is an implementation assignment. Do not stop after another audit, documentation update, or passing the existing tests.**

Work directly from the latest acceptance matrix and the complete Raphael master specification.

Do not repeat work already verified unless necessary for regression testing.

Do not claim the application is launch-ready until the applicable release gates have passed.

---

## 1. PROJECT BASELINE

Repository: `AIGeniusAlien/Raphael-AI`

Development branch: `fix/sqlite-startup-recovery`

Draft PR: #1

Latest owner-reported commit:

`3ee4eec89e769e8aa1c50883d0c34fb5098d143c`

Verify the actual current HEAD before making changes.

The latest engineering run reported:

- 257 Python tests passed.
- 29 frontend tests passed.
- Seven Chromium scenarios passed.
- Five mobile structural tests passed.
- Fourteen actual Xcode tests passed with zero failures or skips.
- Real OpenAI/Qdrant physician and patient conversations passed support verification.
- Longitudinal patient review was implemented.
- Inspectable context snapshots were implemented.
- Clinician-approved structured observation extraction was implemented.
- Nine medical evidence documents and 48 vectors were preserved.
- Synthetic development demonstration passed.
- Synthetic TestFlight beta failed its remaining acceptance requirements.
- Clinical pilot and production remain unapproved.

Preserve all working code, existing requirements, branding, clinical safety checks, and historical test evidence.

Do not reset the branch, merge the PR, deploy, or submit to Apple without owner authorization.

---

## 2. USE THE ACTUAL ACCEPTANCE MATRIX

Read the current authoritative readiness report, acceptance matrix, master specification, and relevant test artifacts.

Identify every outstanding requirement marked FAIL, BLOCKED, incomplete, or unverified.

For each requirement, distinguish:

- Engineering work that can be completed now.
- Work requiring unavailable external services or credentials.
- Work requiring physical hardware.
- Work requiring qualified clinical, legal, security, or regulatory review.

Prioritize independently achievable critical engineering work.

Do not classify a feature as complete merely because its code exists.

A requirement passes only when its intended behavior is demonstrated with appropriate tests and reproducible evidence.

Preserve original requirement identifiers and descriptions.

---

## 3. COMPLETE EXTENDED NATIVE iOS ACCEPTANCE

Preserve the 14 successful native Xcode tests.

Expand actual simulator coverage to include:

- Patient sign-in and logout.
- Clinician sign-in and logout.
- MFA enrollment and verification.
- Account recovery.
- Session expiry and invalidation.
- Backgrounding and foregrounding.
- App termination and relaunch.
- Role-specific navigation.
- Authorized chart access.
- Clinical note access.
- Consent revocation.
- Protected document access.
- Network interruption and recovery.
- Backend failure and retry.
- Accessibility and Dynamic Type.
- VoiceOver navigation where testable.
- Secure storage behavior.
- HTTPS and ATS enforcement.
- No embedded production secrets.
- Release configuration validation.

Use actual authenticated application workflows against a synthetic HTTPS backend.

Do not bypass authentication or weaken security to make tests pass.

Fix any failures and rerun the complete native suite.

Produce actual Xcode test results, not only structural source checks.

Maintain compatibility with the owner's Intel MacBook Pro running macOS Sequoia 15.8.2 and Xcode 16.4 where technically feasible.

---

## 4. FINISH THE PATIENT AND PHYSICIAN EXPERIENCES

Complete all independently achievable missing workflows in the existing product scope.

### Patient application

Verify and finish:

- Account access and recovery.
- Patient profile and history.
- Medications and allergies.
- Medical conditions.
- Laboratory results and trends.
- Documents.
- Care plans.
- Appointments.
- Clinician communication.
- Consent and revocation.
- Longitudinal health review.
- Reviewed medical summaries.
- Medical education.
- Data export.
- Policy-backed account deletion.

### Physician application

Verify and finish:

- Authorized patient chart access.
- Encounter review.
- Clinical evidence retrieval.
- Longitudinal patient context.
- Clinical documentation.
- Structured observation extraction.
- Missing-fact detection.
- Editing and version control.
- Clinician approval.
- Patient publication.
- Referrals.
- Discharge summaries.
- Prior authorizations.
- Insurance appeals.
- Medical necessity documentation.
- Secure patient communication.
- Audit history.

All patient-specific information must be grounded in authorized sources.

Never invent clinical findings.

Never automatically finalize clinically consequential documents or actions.

---

## 5. ADVANCE MEDICAL INTELLIGENCE

Preserve the now-working real physician and patient conversation pipelines.

Continue building the evidence architecture toward broader clinical coverage.

Implement and verify authorized, rights-aware ingestion from appropriate sources, including permitted government medical publications, bibliographic databases, licensed clinical references, and approved institutional resources.

Do not treat bibliographic abstracts or titles as sufficient evidence for unsupported clinical claims.

Implement:

- Incremental evidence ingestion.
- Provenance and licensing metadata.
- Retraction and supersession handling.
- Evidence freshness checks.
- Hybrid retrieval and reranking.
- Claim-level citation verification.
- Population applicability.
- Clinical uncertainty.
- Safe abstention.
- Emergency escalation.
- Medication safety checks.
- Clinically relevant evaluation suites.

Preserve embedding compatibility and existing Qdrant data.

Do not weaken the medical claim screen to manufacture successful conversations.

Develop a clear supported-domain coverage map rather than claiming comprehensive medical knowledge from a small corpus.

Prepare expert-reviewable clinical evaluation cases and clearly separate automated safety checks from independently validated clinical accuracy.

---

## 6. COMPLETE LONGITUDINAL INTELLIGENCE AND STRUCTURED EXTRACTION

Extend the existing longitudinal patient review and structured observation functionality.

Support authorized, provenance-preserving review of:

- Medical history.
- Encounters.
- Medications.
- Allergies.
- Diagnoses.
- Laboratory trends.
- Vital signs.
- Clinical notes.
- Patient-reported symptoms.
- Approved documents.

Clearly distinguish patient-reported information, verified records, and AI-generated suggestions.

Require review for ambiguous or clinically consequential extracted information.

Test wrong-patient documents, conflicting records, missing information, stale data, and revoked consent.

Do not silently overwrite verified chart information.

---

## 7. COMPLETE CLINICAL WORKFLOW ORCHESTRATION

Create or finish durable workflows for:

1. Authorized data collection.
2. Structured extraction.
3. Evidence retrieval.
4. Document preparation.
5. Missing-information validation.
6. Clinician editing.
7. Explicit approval.
8. Authorized publication.
9. Patient notification or status update.
10. Audit history and follow-up.

Support retries, idempotency, failure recovery, and accurate workflow status.

Finish existing documentation and paperwork features before adding unrelated new templates.

Do not automatically submit claims, prescriptions, orders, or external communications without appropriate authorization.

---

## 8. COMPLETE SECURITY, PRIVACY, AND ACCOUNT LIFECYCLE

Preserve the implemented MFA, recovery, session, and encrypted-backup improvements.

Verify and finish:

- MFA lifecycle.
- Secure recovery.
- Verified identity controls.
- Guardian and proxy access.
- Patient-level authorization.
- Tenant isolation.
- Consent enforcement.
- Session revocation.
- Data export.
- Account deactivation.
- Policy-backed erasure.
- Clinical-record retention exceptions.
- Audit logging.
- Secure uploads.
- Backup restoration.
- Secret rotation.
- Rate limiting.
- Security monitoring.

Implement technical controls where feasible.

Where legal or institutional policy is required, use safe configurable mechanisms and document the exact external decision needed.

Run negative authorization tests, security scans, and backup-recovery tests.

Do not claim HIPAA compliance without appropriate independent organizational, legal, technical, and contractual review.

---

## 9. PRODUCTION INFRASTRUCTURE AND RELIABILITY

Prepare and verify:

- Secure HTTPS staging configuration.
- Production database migration path.
- Deployment reproducibility.
- Health checks.
- Monitoring and alerting.
- Backup and restore.
- Disaster recovery.
- Secrets management.
- Provider timeout and retry behavior.
- Audit retention.
- Performance testing.
- Concurrency testing.
- Rollback procedures.
- Incident-response preparation.

Use synthetic data and authorized services.

Do not deploy production infrastructure without approval.

Where owner-controlled accounts or credentials are missing, provide exact setup instructions and continue all independent engineering work.

---

## 10. PREMIUM PRODUCT EXPERIENCE

Preserve Raphael's existing branding.

Improve patient web, clinician web, and native iOS consistency.

Inspect real rendered interfaces for:

- Navigation.
- Typography.
- Layout.
- Medical readability.
- Form usability.
- Error handling.
- Loading states.
- Accessibility.
- Dynamic Type.
- Responsive behavior.
- Performance.

Capture screenshots and fix observed defects.

Do not replace approved branding or introduce unrelated redesigns.

---

## 11. OPTIMIZE AI EFFICIENCY

Implement an evaluated AI task-routing and performance strategy.

Use deterministic processing where appropriate.

Route tasks to suitable models based on verified quality, safety, latency, and cost.

Measure:

- Response latency.
- Retrieval quality.
- Citation accuracy.
- Workflow completion.
- Token consumption.
- Cost per successful task.
- Provider failure rates.
- Unsafe-output rejection.
- Appropriate abstention.

Never sacrifice medical safety or privacy to reduce cost.

Prevent cross-user or cross-tenant leakage through caching.

Use bounded context, safe retries, and configurable budgets.

---

## 12. FINAL INTEGRATED ACCEPTANCE

Run complete synthetic end-to-end scenarios covering:

- Patient onboarding and authentication.
- MFA and recovery.
- Clinician authentication.
- Chart creation and review.
- Consent and revocation.
- Longitudinal history.
- Structured extraction.
- Real evidence-grounded conversations.
- Unsupported medical claim rejection.
- Clinical documentation preparation.
- Clinician editing and approval.
- Patient publication.
- Account export and deletion.
- Guardian/proxy permissions.
- Cross-patient and cross-tenant denial.
- Native iPhone authentication and navigation.
- Network failure and recovery.
- Backup restoration.
- Accessibility.
- Secure staging configuration.
- Performance and concurrency.

Clearly separate actual external-service verification from mocks.

Preserve all failures and artifacts.

Do not count skipped tests as passed.

---

## 13. RELEASE GATES

Produce separate evidence-based decisions for:

**A. Synthetic local demonstration**

**B. Controlled synthetic TestFlight beta**

**C. Controlled clinical pilot**

**D. Clinical production**

For each gate, report PASS, FAIL, or BLOCKED.

Do not declare TestFlight distribution ready without the required native tests, staging, privacy materials, signing, and physical-device verification.

Do not declare clinical production ready without appropriate independent clinical, security, privacy, legal, regulatory, contractual, and operational reviews.

---

## 14. REQUIRED FINAL OUTPUT

Provide:

1. Final commit SHA and branch.
2. Summary of implemented changes.
3. Requirement-by-requirement completion evidence.
4. Python, frontend, browser, and native test results.
5. Real OpenAI/Qdrant conversation results.
6. Clinical evidence coverage inventory.
7. Longitudinal and documentation workflow results.
8. Security and privacy test evidence.
9. Performance and reliability measurements.
10. Actual screenshots and accessibility results.
11. Intel Mac/Xcode 16.4 setup instructions.
12. Physical-device verification steps.
13. Secure staging and deployment instructions.
14. TestFlight prerequisites.
15. Exact owner actions.
16. Remaining failed or blocked requirements.
17. Separate release-gate decisions.

Update:

`docs/RAPHAEL_FINAL_LAUNCH_READINESS.md`

Update the authoritative machine-readable acceptance matrix.

Do not erase historical failures or overwrite newer results with stale findings.

---

## FINAL EXECUTION INSTRUCTION

Begin immediately from the verified latest repository state.

**Implement the remaining engineering work, not merely another audit.**

Preserve successful medical conversations and native authentication.

Close outstanding software defects.

Complete integrated patient and physician workflows.

Finish extended native acceptance.

Strengthen medical intelligence, longitudinal review, clinical documentation, security, privacy, and production reliability.

Optimize performance and refine the user experience.

Continue through all independently achievable requirements within the available execution environment.

If execution limits prevent completion, commit tested work, preserve an exact continuation checkpoint, and identify the next implementation tasks without claiming release readiness.

Do not merge, deploy, publish, or use real patient data without explicit owner authorization.

**Deliver a demonstrably functional, secure, polished Raphael release candidate with objective evidence for every claimed capability.**
