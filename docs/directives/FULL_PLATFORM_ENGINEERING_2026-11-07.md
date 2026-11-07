# RAPHAEL-AI — DEFINITIVE FULL-PLATFORM ENGINEERING, AUTONOMOUS DEVELOPMENT, AND DEPLOYMENT DIRECTIVE

## EXECUTIVE MANDATE

You are the principal engineering organization responsible for completing Raphael-AI.

Operate with the combined expertise of exceptional software architects, full-stack engineers, clinical AI researchers, radiologists, physicians across specialties, medical informaticians, interoperability specialists, cybersecurity engineers, DevOps engineers, mobile engineers, product designers, accessibility experts, clinical safety officers, and quality-assurance engineers.

Your mandate is to transform the EXISTING Raphael repository into the most complete, reliable, sophisticated, and clinically responsible implementation achievable with the available infrastructure, resources, and authorized access.

**This is an implementation and deployment-readiness assignment, not a request for a roadmap.**

The founder has repeatedly described Raphael across extensive historical conversations. Do not require the founder to repeat requirements already captured in the repository or accessible conversation records.

The founder's vision is cumulative. Every requirement remains in scope unless explicitly superseded.

Your goal is to minimize further founder prompts by implementing, testing, repairing, integrating, documenting, and continuing work autonomously within authorized boundaries.

Do not stop after implementing one feature. Do not stop after generating documents. Do not stop because the existing test suite passes. Continue working through dependency-ready requirements for as long as the execution environment permits.

Never claim functionality, safety, deployment, or clinical validation that has not been demonstrated.

---

# PART I — REPOSITORY AND HISTORICAL CONTINUITY

## 1. Establish the actual current state

Repository:

`AIGeniusAlien/Raphael-AI`

Previously recorded development:

- Draft PR #1.
- Last previously verified development commit: `906700fc43ede1c834ff89fff46bee58ba2fd1d9`.
- Previously reported successful automated tests: 326 Python, 29 frontend, 8 Chromium, 5 mobile-source, and 27 native iOS tests.
- Previously reported release gates: synthetic local PASS; synthetic TestFlight BLOCKED; clinical production FAIL.
- Previously tracked engineering requirements: 1,164.

These are historical checkpoints, not assumptions about current repository state.

Verify everything against the live repository.

Inspect:

- All branches and open pull requests.
- Current commit history.
- Existing architecture.
- Application entry points.
- Backend and frontend source.
- Native mobile applications.
- Existing databases and migrations.
- Clinical knowledge infrastructure.
- Radiology modules.
- Voice and documentation systems.
- Authentication and authorization.
- Hospital interoperability modules.
- Existing deployment configurations.
- Automated tests and CI/CD.
- Security scans and dependencies.
- Existing implementation reports.

Read the full contents of applicable authoritative documents, including:

- `AGENTS.md`
- `docs/RAPHAEL_MASTER_SPEC.md`
- `docs/directives/DEFINITIVE_HEALTHCARE_PLATFORM_2026-11-01.md`
- `docs/PLATFORM_DIRECTIVE_RECONCILIATION.json`
- `docs/FOUNDATION_IMPLEMENTATION_CHECKPOINT.md`
- `docs/RAPHAEL_FINAL_ACCEPTANCE.json`
- `docs/ENGINEERING_LEDGER.json`
- `docs/REQUIREMENTS_MATRIX.md`
- All other applicable architecture, workflow, clinical safety, and release documents.

Do not rely solely on executive summaries.

Read relevant original conversation transcripts if available, especially:

1. The original Raphael medical-industry vision.
2. Initial Raphael architecture and setup.
3. Raphael AI Copilot Overview.
4. Building Raphael's Medical System.
5. Historical Codex development conversations.

If complete transcripts are unavailable, do not fabricate their contents. Preserve existing requirements, identify provenance gaps, and proceed with implementation.

## 2. Create the definitive requirements authority

Consolidate all known requirements into a canonical, machine-readable system.

For every requirement record:

- Stable ID.
- Original source and founder intent.
- Exact expected behavior.
- Clinical or operational use case.
- Dependencies.
- Risk classification.
- Implementation location.
- Backend/frontend/mobile applicability.
- Acceptance tests.
- Validation evidence.
- Status.
- Remaining work.
- Blocking external dependencies.

Use the existing engineering ledger as the starting point.

Preserve all existing requirement identifiers and history.

Prevent duplicate requirements and conflicting interpretations.

Distinguish:

- Not started.
- Infrastructure only.
- Prototype.
- Implemented but unverified.
- Engineering verified.
- Integrated and operational.
- Externally validated.
- Approved for the intended release.

A file, interface, mock, or documentation entry does not establish completion.

---

# PART II — RAPHAEL'S PRODUCT IDENTITY

## 3. The complete Raphael vision

Raphael is an interconnected, multimodal medical intelligence and healthcare operating platform.

It must support the complete patient journey and the workflows of authorized healthcare professionals and institutions.

It must be useful in:

- Hospitals.
- Emergency departments.
- Ambulance and prehospital workflows.
- Primary care.
- Pediatrics.
- Urgent care.
- Specialist practices.
- Radiology departments.
- Outpatient clinics.
- Rehabilitation and follow-up.
- Care coordination.
- International and cross-border healthcare.

Raphael must provide an accessible, intelligent interface through which authorized people can communicate naturally, retrieve relevant information, coordinate care, and complete approved workflows.

It must not merely answer questions.

It must maintain context, understand pending work, prepare actions, detect relevant changes, and assist proactively.

Raphael should reduce repetitive administrative work and cognitive burden while maintaining clear human accountability.

## 4. One intelligence system, multiple experiences

Build a shared intelligence and workflow platform serving:

- Patients.
- Physicians.
- Radiologists.
- Nurses.
- Specialists.
- Authorized caregivers.
- Clinical administrators.
- Hospital departments.
- Researchers.
- Approved external healthcare partners.

Each role must receive an appropriate interface and permissions.

Avoid building unrelated applications with inconsistent patient data or duplicated business logic.

All experiences must share a coherent identity, consent, provenance, audit, and workflow architecture.

---

# PART III — INTELLIGENCE ARCHITECTURE

## 5. Build Raphael's central intelligence system

Implement a modular intelligence architecture consisting of:

1. Conversation and voice interaction.
2. Clinical context assembly.
3. Longitudinal patient memory.
4. Evidence retrieval.
5. Specialty-specific reasoning.
6. Medical image processing.
7. Clinical workflow planning.
8. Authorized tool execution.
9. Clinical safety and uncertainty controls.
10. Task persistence.
11. Notification and escalation.
12. Audit and observability.
13. Knowledge maintenance.
14. Evaluation and continuous improvement.

The architecture must support multiple model providers and appropriately selected models without unnecessary vendor lock-in.

Build robust provider adapters, timeouts, retries, circuit breakers, cost monitoring, rate limiting, and graceful degradation.

Never allow a model to bypass the authorization layer.

## 6. Persistent, context-aware assistance

Raphael must maintain durable, authorized context across sessions.

Implement:

- Patient-specific clinical context.
- Encounter context.
- User preferences where appropriate.
- Authorized task memory.
- Pending workflow state.
- Clinical document history.
- Prior recommendations and their status.
- Consent-aware retrieval.
- Context expiration and correction.
- Provenance and versioning.

Prevent cross-patient, cross-user, and cross-tenant information leakage.

A clinician should be able to resume an encounter or pending workflow without unnecessarily repeating information already available and authorized.

A patient should be able to continue a care journey across encounters and institutions when access is permitted.

## 7. Proactive, event-driven intelligence

Develop an event-driven system that can respond to authorized events such as:

- New clinical documents.
- New laboratory results.
- New imaging reports.
- Upcoming appointments.
- Unfinished documentation.
- Referral status changes.
- Follow-up deadlines.
- Updated clinical guidelines.
- New patient-provided information.
- Workflow failures.
- Approved hospital integration events.

Agents may prepare summaries, identify missing information, propose tasks, and notify authorized users.

Use durable queues, idempotent workers, retry policies, dead-letter handling, and crash recovery.

Provide explicit configuration for what Raphael may perform automatically, what requires confirmation, and what is prohibited.

Do not enable unapproved autonomous clinical actions.

---

# PART IV — MEDICAL KNOWLEDGE AND CLINICAL REASONING

## 8. Comprehensive medical knowledge

Develop an extensible, structured medical knowledge system intended to support all major medical specialties and subspecialties.

Include:

- Anatomy and physiology.
- Diseases and conditions.
- Symptoms and signs.
- Diagnostic evaluation.
- Laboratory interpretation support.
- Medical imaging.
- Pharmacology and medication safety.
- Medical procedures.
- Surgical procedures.
- Clinical pathways.
- Preventive medicine.
- Emergency medicine.
- Pediatrics.
- Obstetrics and gynecology.
- Internal medicine.
- Cardiology.
- Neurology.
- Oncology.
- Psychiatry.
- Infectious diseases.
- Orthopedics.
- Rehabilitation.
- Geriatrics.
- Additional specialties defined by the canonical requirements.

Implement a formal coverage taxonomy and report actual coverage.

Do not claim mastery of every condition or procedure without evidence.

## 9. Evidence and retrieval

Build on the existing retrieval and terminology architecture.

Support:

- Approved guideline sources.
- Institutional protocols.
- Peer-reviewed evidence.
- Authorized medical reference content.
- Terminology systems.
- Source licensing and attribution.
- Versioning.
- Publication and review dates.
- Source withdrawal and expiration.
- Evidence-quality classification.
- Conflicting evidence.
- Citation-grounded answers.
- Context-specific retrieval.
- Insufficient-evidence abstention.

Use properly licensed or publicly authorized resources.

Do not indiscriminately scrape copyrighted medical references.

Do not fabricate clinical references or citations.

## 10. Clinical reasoning engine

Develop clinician-facing reasoning support that combines available:

- Symptoms.
- Medical history.
- Physical examination findings.
- Laboratory results.
- Vital signs.
- Imaging.
- Medications.
- Allergies.
- Prior encounters.
- Existing diagnoses.
- Evidence-based guidelines.

Provide:

- Differential diagnosis support.
- Evidence for and against possibilities.
- Missing-information identification.
- Contradiction detection.
- Relevant risk factors.
- Suggested questions for clinician consideration.
- Guideline-linked evaluation pathways.
- Longitudinal trend analysis.
- Clear uncertainty.
- Explicit clinician review.

Distinguish facts from hypotheses.

Do not manufacture confidence percentages.

Calibrate quantitative confidence only when supported by appropriate evaluation.

Do not autonomously establish diagnoses or prescribe treatment.

## 11. Medical procedures and care pathways

Implement a structured, specialty-extensible procedure intelligence system.

For each supported procedure or pathway, represent:

- Indications.
- Contraindications.
- Patient selection considerations.
- Preparation.
- Relevant tests and imaging.
- Consent requirements.
- Safety checks.
- Equipment and resource considerations.
- Procedural stages at the appropriate clinical reference level.
- Potential complications.
- Post-procedure monitoring.
- Follow-up.
- Escalation criteria.
- Evidence sources.
- Review status.
- Version history.

Build real interfaces and workflow integration, not only static documents.

Require clinical review for content used in consequential care.

Track which procedures are genuinely covered and which remain incomplete.

---

# PART V — RADIOLOGY AND MULTIMODAL MEDICINE

## 12. Advanced radiology intelligence

Radiology is a major design anchor, not Raphael's only purpose.

Implement an end-to-end radiology architecture supporting:

- DICOM study ingestion.
- DICOM metadata.
- Study and series organization.
- Imaging viewer integration.
- PACS connectivity.
- Prior-study retrieval.
- Longitudinal comparison.
- Relevant patient context.
- Structured report templates.
- Report drafting.
- Report discrepancy detection.
- Missing-history alerts.
- Follow-up recommendation consistency.
- Evidence-supported differential assistance.
- Clinician editing and sign-off.
- Versioned reports.
- Audit trails.

Implement secure DICOMweb integration where applicable.

Support synthetic imaging workflows and authorized external test systems.

Separate image display, metadata analysis, report-text reasoning, and genuine image-model interpretation.

Do not claim automated diagnostic image interpretation unless an actual model has been integrated and appropriately evaluated.

## 13. Multimodal model infrastructure

Develop interfaces for clinically appropriate models capable of processing supported combinations of:

- Medical images.
- Clinical text.
- Structured records.
- Voice transcripts.
- Time-series information.

Implement model registry, versioning, input validation, preprocessing, output normalization, provenance, latency measurement, evaluation, and rollback.

Build controlled evaluation pipelines for clinically relevant datasets.

Unsupported modalities must be explicitly labeled rather than simulated.

---

# PART VI — VOICE, LISTENING, AND DOCUMENTATION

## 14. Raphael's voice experience

Build an intuitive, low-latency, conversational interface.

Support:

- Voice activation where platform permissions allow.
- Clinician-directed listening.
- Explicit session controls.
- Natural spoken questions.
- Streaming transcription.
- Streaming spoken responses where appropriate.
- Interruption handling.
- Contextual follow-up.
- Voice navigation.
- Accessibility features.
- Clear microphone status.
- Privacy and consent controls.

The interface should feel responsive and natural, not like a slow sequence of disconnected chatbot requests.

Measure time to first response, transcription delay, and end-to-end interaction latency.

Do not promise always-on listening where operating-system permissions or privacy requirements prohibit it.

## 15. Ambient clinical documentation

Implement authorized encounter workflows:

1. Establish identity, patient context, and consent.
2. Start a clearly indicated listening session.
3. Capture permitted audio.
4. Transcribe speech.
5. Extract structured clinical information.
6. Distinguish speaker statements and uncertain attribution.
7. Generate appropriate draft notes.
8. Identify missing or conflicting information.
9. Present drafts for clinician review.
10. Apply clinician corrections.
11. Save or export through approved workflows.
12. Record audit and provenance information.

Support common note formats and specialty-specific templates.

Build robust controls for transcription errors, omissions, unsupported statements, and incorrect patient attribution.

No clinical note may be silently finalized by an unapproved agent.

## 16. Comprehensive documentation workflows

Support:

- History and physical.
- Progress notes.
- Consultation notes.
- Emergency encounter notes.
- Radiology reports.
- Procedure notes.
- Discharge summaries.
- Referral letters.
- Patient instructions.
- Follow-up documentation.
- Medication reconciliation drafts.
- Administrative forms where authorized.

Use reusable structured data and templates.

Minimize duplicate data entry.

Maintain clinician review and authorization requirements.

---

# PART VII — COMPLETE PATIENT JOURNEY

## 17. Patient application

Develop a polished, accessible patient experience supporting:

- Secure registration.
- Identity verification.
- Patient profiles.
- Medical history.
- Allergies.
- Medication lists.
- Medical record uploads.
- Document organization.
- Appointment preparation.
- Authorized messaging.
- Care plans.
- Follow-up reminders.
- Referral tracking.
- Clinician-approved summaries.
- Access permissions.
- Authorized caregiver management.
- Emergency information access under appropriate controls.
- Language and accessibility preferences.

Patients should be able to provide information once and reuse it across authorized care workflows.

## 18. Pre-arrival and emergency coordination

Implement the infrastructure for authorized patient-to-provider pre-arrival communication.

Potential workflows include:

- Patient-initiated emergency assistance.
- Collection of relevant medical history.
- Authorized transmission of information to a participating receiving facility.
- Pre-arrival intake preparation.
- Relevant allergies and medication summaries.
- Patient location sharing with explicit consent.
- Facility acknowledgment and communication status.
- Arrival and handoff tracking.
- Escalation when transmission fails.

Do not imply that hospitals, ambulances, or emergency services receive information unless an actual integration exists and delivery has been confirmed.

Do not substitute Raphael for emergency dispatch.

Provide appropriate emergency-service guidance.

## 19. Complete encounter-to-follow-up workflows

Build integrated journeys for:

- Primary care.
- Urgent care.
- Pediatric visits.
- Specialist consultation.
- Hospital admission.
- Emergency department encounters.
- Inpatient care.
- Discharge.
- Rehabilitation.
- Follow-up.

Each workflow should support the applicable steps:

Intake → history → information gathering → clinician assessment support → documentation → authorized care coordination → discharge or follow-up → closed-loop tracking.

Implement reusable workflow components instead of duplicating the same functionality for every specialty.

## 20. International and cross-border care

Build architecture for authorized medical record exchange across institutions and jurisdictions.

Support:

- Patient-directed sharing.
- Time-limited access.
- Identity verification.
- Provider authentication.
- Consent tracking.
- Record provenance.
- Structured data export.
- Language localization.
- Translation with original-text preservation.
- Terminology mapping.
- Access revocation.
- Audit logs.
- Jurisdiction-specific privacy controls.

Do not assume that foreign healthcare systems are connected or that international data transfer is automatically lawful.

Build testable interoperability adapters and identify jurisdictional requirements.

---

# PART VIII — HOSPITAL OPERATIONS AND AUTOMATION

## 21. Clinical and administrative workflows

Implement authorized automation for:

- Intake.
- Scheduling integrations.
- Documentation preparation.
- Referral coordination.
- Care-team communication.
- Task assignment.
- Discharge preparation.
- Follow-up tracking.
- Record reconciliation.
- Patient education.
- Worklist organization.
- Operational notifications.
- Quality improvement reporting.
- Administrative document preparation.

Where appropriate, provide human-in-the-loop approval.

Implement reliable task state and escalation.

Do not represent unconnected external systems as automated.

## 22. Hospital interoperability

Implement and verify the appropriate standards and integration infrastructure:

- FHIR R4.
- SMART-on-FHIR.
- OAuth/OIDC.
- HL7 v2 where applicable.
- DICOM.
- DICOMweb.
- PACS.
- EHR integration.
- Terminology services.
- Clinical document exchange.
- Webhooks and event subscriptions.
- Secure messaging adapters.

Test with authorized synthetic sandboxes.

Build error handling for unavailable systems, inconsistent identifiers, partial records, duplicate messages, and failed synchronization.

Maintain patient identity integrity and provenance.

---

# PART IX — WELLNESS, PREVENTION, AND COMPLEMENTARY APPROACHES

## 23. Evidence-aware whole-person care

Preserve the founder's interest in preventive medicine, lifestyle interventions, and appropriately supported complementary practices.

Include evidence-aware support for:

- Nutrition.
- Exercise.
- Sleep.
- Stress management.
- Rehabilitation.
- Behavioral health.
- Preventive care.
- Patient education.
- Relevant breathing and cold-exposure practices, including discussion of Wim Hof-style methods where appropriate.

Clearly distinguish established clinical evidence, emerging research, limited evidence, and unsupported claims.

Evaluate contraindications and patient-specific risks.

Never recommend replacing necessary medical treatment with an unproven alternative.

Do not present potentially dangerous breathing or cold-exposure practices as universally safe.

---

# PART X — SECURITY, PRIVACY, AND CLINICAL GOVERNANCE

## 24. Security architecture

Implement and verify:

- Strong authentication.
- MFA.
- Role-based and attribute-based authorization.
- Tenant isolation.
- Consent enforcement.
- Encryption in transit and at rest.
- Secure secret storage.
- SQLCipher where appropriate.
- Database migration and key rotation.
- Audit logging.
- Session management.
- Secure file handling.
- Data retention and deletion.
- Backup and restoration.
- Disaster recovery.
- Dependency scanning.
- Vulnerability management.
- Prompt-injection defenses.
- Tool authorization.
- Rate limiting.
- Abuse protection.
- Monitoring and incident response.

Perform negative-path tests for privilege escalation, cross-tenant access, record leakage, unauthorized tools, malicious documents, and unsafe workflow execution.

Never place secrets or real patient information in test fixtures, logs, screenshots, or pull-request descriptions.

## 25. Clinical governance

Define and enforce:

- Intended-use boundaries.
- Clinical risk classification.
- Human oversight.
- Review and approval workflows.
- Evidence provenance.
- Model versioning.
- Knowledge versioning.
- Uncertainty and abstention.
- Error reporting.
- Clinical evaluation.
- Change management.
- Safety monitoring.
- Incident response.
- Rollback procedures.

Keep clinical production disabled until the intended use has passed applicable institutional, legal, regulatory, and clinical validation requirements.

Do not claim HIPAA compliance, FDA clearance, or medical-device approval solely because security features or documentation exist.

---

# PART XI — PRODUCT QUALITY AND USER EXPERIENCE

## 26. World-class interface

Develop a cohesive, professional, accessible product across supported platforms.

Prioritize:

- Clear information hierarchy.
- Fast interaction.
- Responsive layouts.
- Readable clinical information.
- Consistent navigation.
- Role-specific dashboards.
- Efficient chart review.
- Excellent search.
- Minimal unnecessary clicks.
- Reliable forms.
- Useful loading and error states.
- Accessible controls.
- Mobile-friendly workflows.
- Keyboard and screen-reader support.
- Appropriate dark and light modes.

Avoid decorative complexity that slows clinical work.

Test small screens, enlarged text, poor connectivity, interrupted sessions, and accessibility settings.

## 27. Platform coverage

Continue and integrate the existing:

- Backend.
- Web application.
- Native iOS application.
- API layer.
- Authentication.
- Clinical data infrastructure.

Develop Android and desktop support according to the existing requirements and realistic platform dependencies.

Share domain logic and contracts wherever possible.

Do not create platform-specific implementations that silently diverge in safety or authorization behavior.

---

# PART XII — CONTINUOUS MEDICAL AND SOFTWARE EVOLUTION

## 28. Medical knowledge update engine

Raphael must evolve as medicine evolves.

Implement a controlled maintenance pipeline:

1. Monitor approved medical sources.
2. Detect updates.
3. Identify affected knowledge entries and pathways.
4. Prepare proposed revisions.
5. Verify evidence and licensing.
6. Identify conflicting guidance.
7. Run automated regression evaluations.
8. Route clinically consequential changes for expert approval.
9. Publish approved versions.
10. Monitor post-release behavior.
11. Roll back defective changes.

Track publication dates, approval dates, and source expiration.

Never permit unreviewed information to silently alter consequential clinical recommendations.

## 29. Feedback and product improvement

Implement structured feedback mechanisms for:

- Patients.
- Clinicians.
- Administrators.
- Internal testers.
- Approved research collaborators.

Allow authorized users to report:

- Incorrect information.
- Missing capabilities.
- Workflow friction.
- Documentation problems.
- Integration failures.
- Accessibility issues.
- Potential clinical safety concerns.

Classify and prioritize feedback.

Link accepted improvements to the engineering requirements system.

Do not automatically retrain or modify clinical models using identifiable patient feedback without the required consent, governance, and validation.

## 30. Persistent autonomous engineering

Build or configure an engineering automation system capable of:

- Scheduled repository audits.
- Dependency monitoring.
- CI execution.
- Regression detection.
- Security scanning.
- Requirements reconciliation.
- Issue creation.
- Development task selection.
- Bounded implementation jobs.
- Automated code review.
- Draft pull-request preparation.
- Release-candidate preparation.
- Development status summaries.

Use durable state so work survives interrupted sessions.

Where supported, configure scheduled runners and event-triggered jobs.

If infrastructure or permissions are missing, implement the configuration and document the precise authorization needed.

Never falsely claim overnight or continuous execution without a verified scheduler and runner.

Protect production branches and clinical deployments with explicit approval gates.

---

# PART XIII — REAL DEPLOYMENT ENGINEERING

## 31. Build deployable infrastructure

Do not stop at localhost functionality.

Inspect existing deployment infrastructure and implement the most appropriate supported architecture for:

- Application hosting.
- Backend services.
- Database services.
- Persistent storage.
- Background workers.
- Task queues.
- Secrets.
- TLS.
- Domain configuration.
- Health checks.
- Monitoring.
- Logging.
- Backups.
- Recovery.
- Database migrations.
- Staging environments.
- Release rollback.
- CI/CD.
- Cost controls.

Prefer existing hosting and infrastructure choices unless there is a documented reason to change them.

Avoid unnecessary paid services.

Use infrastructure-as-code where practical.

Do not expose a synthetic development backend to the public as a clinical production service.

## 32. Real working environments

Establish distinct environments:

### Local development

Synthetic data, complete developer setup, automated tests, reproducible startup.

### Integrated staging

Deployed application, persistent infrastructure, synthetic identities and records, working background jobs, realistic integration testing.

### Research and clinical evaluation

Controlled datasets, documented protocols, appropriate approvals, reproducible evaluation, and strict access controls.

### Clinical production

Permitted only after engineering readiness, security review, clinical validation, regulatory assessment where applicable, institutional agreements, and operational approval.

Implement the environment separation in code and configuration.

## 33. Mobile distribution readiness

Prepare the iOS application for a genuine TestFlight release.

Implement or verify:

- Correct bundle configuration.
- Signing and provisioning requirements.
- Application capabilities.
- Privacy disclosures.
- Permission descriptions.
- Backend configuration.
- Crash reporting.
- Secure authentication.
- Account deletion where applicable.
- Data handling disclosures.
- Accessibility.
- App Store metadata.
- Release build automation.
- TestFlight acceptance testing.

Prepare Android release infrastructure as required.

Complete all work possible without the founder's Apple, Google, or hosting credentials.

Do not fabricate successful app-store submissions.

Do not publish publicly without explicit approval.

## 34. Operational reliability

Implement and verify:

- Automatic process recovery.
- Persistent task recovery.
- Database backup and restoration.
- Monitoring alerts.
- Error budgets.
- Graceful degradation.
- Queue failure recovery.
- Data consistency checks.
- Audit integrity.
- Safe schema migrations.
- Release rollback.
- Capacity and load testing.
- Recovery procedures.

Measure actual operational behavior.

Do not claim 24/7 reliability without evidence.

---

# PART XIV — AUTOMATED TESTING AND ACCEPTANCE

## 35. Eliminate the need for repetitive founder testing

The founder should not have to manually verify every routine implementation detail.

You must perform the engineering verification.

For each relevant feature, add and execute:

- Unit tests.
- Integration tests.
- API contract tests.
- Database migration tests.
- Authorization tests.
- Negative-path tests.
- Security tests.
- Browser end-to-end tests.
- Mobile tests.
- Accessibility tests.
- Performance tests.
- Recovery tests.
- Regression tests.
- Synthetic clinical scenario tests.

Repair failures before claiming completion.

Record exact test commands, environments, results, and evidence.

Do not fabricate results or treat skipped tests as passed.

## 36. Clinical evaluation

For clinical capabilities, design evaluation against appropriate reference standards.

Include, as applicable:

- Accuracy.
- Sensitivity and specificity.
- False-negative analysis.
- False-positive analysis.
- Subgroup performance.
- Calibration.
- Abstention behavior.
- Source faithfulness.
- Documentation completeness.
- Imaging-model performance.
- Emergency escalation performance.
- Human factors.
- Workflow impact.
- Failure-mode analysis.

Use independent clinical review where required.

Synthetic test success is not proof of real-world clinical safety.

## 37. End-to-end acceptance scenarios

Create and automate comprehensive scenarios including:

1. New patient registration and consent.
2. Patient medical-record upload.
3. Authorized clinician chart review.
4. Voice-enabled clinical encounter.
5. Draft note generation and clinician correction.
6. Differential support with conflicting evidence.
7. Radiology study ingestion and report workflow.
8. Specialist referral and follow-up.
9. Hospital intake and discharge.
10. Patient-to-provider pre-arrival communication through a verified test integration.
11. International record-sharing authorization.
12. Guideline update and approval.
13. Background task interruption and recovery.
14. Unauthorized access attempts.
15. Emergency-risk scenario with safe escalation.
16. Mobile authentication and workflow continuity.
17. Deployment and rollback.
18. Backup restoration.
19. Offline or degraded connectivity.
20. Cross-platform data consistency.

Mark each scenario as passed, failed, blocked, or not applicable with evidence.

---

# PART XV — EXECUTION STRATEGY

## 38. Do not waste execution time

Avoid repeated high-level planning.

Perform a concise dependency analysis, then implement.

Prioritize shared foundations that unlock multiple capabilities.

Favor integrated, functioning vertical workflows over superficial coverage.

Use existing modules wherever technically sound.

Do not rebuild working systems without evidence of necessity.

Avoid unnecessary new frameworks, services, and dependencies.

Use parallel agents when available, with explicit ownership of files and interfaces.

Keep the application buildable and tests runnable throughout development.

## 39. Autonomous decision authority

Make routine engineering decisions without repeatedly consulting the founder.

You may:

- Refactor code.
- Add tests.
- Repair defects.
- Implement documented requirements.
- Improve architecture.
- Update development documentation.
- Prepare migrations.
- Configure synthetic staging.
- Create development branches.
- Open draft pull requests.
- Prepare release artifacts.

You must request authorization for:

- Material spending.
- New paid external services.
- Production data access.
- Live patient communication.
- Public clinical deployment.
- Clinical model activation.
- Regulatory representations.
- Destructive migrations.
- Merging protected branches when approval is required.
- App Store publication.
- Changes to the founder's explicit product intent.

If authorization is unavailable, record the blocker and continue independent work.

## 40. Work continuously within execution limits

Execute this loop:

DISCOVER → IMPLEMENT → INTEGRATE → VERIFY → REPAIR → RECORD → CONTINUE

Do not arbitrarily end after a single implementation increment.

If the execution environment imposes a time or context limit, preserve durable continuation state.

At the next execution, resume from verified state rather than restarting the audit.

Create or update a concise machine-readable continuation manifest containing:

- Current branch.
- Current commit.
- Active work.
- Completed requirements.
- Remaining requirements.
- Dependencies.
- Test evidence.
- Known defects.
- External blockers.
- Next executable tasks.

Ensure this manifest reflects the actual repository.

---

# PART XVI — FINAL RELEASE OBJECTIVES

## 41. Define completion correctly

Do not declare Raphael complete merely because:

- The application launches.
- A chatbot responds.
- A note can be generated.
- A DICOM file can be displayed.
- A FHIR endpoint exists.
- A test suite passes.
- A requirements document is complete.
- A deployment configuration file exists.

Completion requires functioning implementations and the applicable integration, security, reliability, and clinical acceptance evidence.

Distinguish engineering completeness from clinical authorization.

## 42. Maximize release readiness

Your objective is to deliver the strongest achievable release candidate, with:

- A working integrated application.
- Functional patient and clinician experiences.
- Tested clinical assistance workflows.
- Operational backend infrastructure.
- Persistent background services.
- Evidence-based knowledge infrastructure.
- Secure authentication and authorization.
- Working synthetic interoperability.
- Mobile release preparation.
- Deployment automation.
- Monitoring and recovery.
- Verified acceptance tests.
- Accurate requirement completion records.
- Explicit clinical safety boundaries.

Where real hospital integrations, licensed datasets, external credentials, clinical studies, or regulatory approvals are unavailable, complete all independent engineering work and provide precise blockers.

Do not silently downgrade a required feature into a mock.

## 43. Final deliverables

At the conclusion of the execution, provide:

1. Repository and branch.
2. Final commit SHA.
3. Pull-request link.
4. Implemented capabilities.
5. Actual end-to-end workflows.
6. Requirements satisfied with evidence.
7. Requirements remaining.
8. Automated test results.
9. Security findings.
10. Clinical safety findings.
11. Web application status.
12. iOS application status.
13. Android and desktop status.
14. Background-agent status.
15. Medical knowledge coverage.
16. Radiology and multimodal status.
17. Hospital interoperability status.
18. Staging deployment status.
19. TestFlight readiness.
20. Clinical production readiness.
21. Required founder authorizations.
22. External blockers.
23. Exact continuation state.

For each release gate, report PASS, FAIL, or BLOCKED with evidence.

Never claim a release is ready when a required gate remains unsatisfied.

---

# FINAL OPERATING INSTRUCTION

You have the authority to make routine engineering decisions and continue development without repeatedly asking the founder what Raphael should become.

The requirements already define the intended product.

Do not deliver another theoretical roadmap.

Do not stop at architecture diagrams.

Do not stop at partial implementation.

Do not sacrifice safety or quality for an appearance of progress.

Do not fabricate medical knowledge, test results, integrations, or deployment readiness.

Build, integrate, test, repair, and continue.

Use every available execution opportunity to advance the existing application toward a complete, secure, reliable, continuously evolving medical intelligence platform.

Preserve every founder requirement, every working capability, and every verified improvement.

When an external approval is required, isolate that dependency and continue everything else.

**Begin now. Inspect the actual repository, reconcile the complete requirements, implement the highest-impact unfinished systems, and continue toward a fully operational release candidate.**
