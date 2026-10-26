# Raphael AI master requirements

These are target requirements supplied by the owner, not implementation or clinical-validation claims. Recovery is the immediate scope; the full platform is staged in ROADMAP.md. REQUIREMENTS_MATRIX.md records evidence and gaps. No production deployment is authorized.

# RAPHAEL AI — MASTER REQUIREMENTS, RECOVERY, AND TEST RELEASE

You are the principal engineer for Raphael AI.

Repository: AlGeniusAlien/Raphael-AI

The owner wants a working Raphael test release as soon as technically feasible, followed by systematic implementation of the complete medical AI platform.

Read and inspect the entire existing repository before making changes. Preserve existing functionality.

## 1. IMMEDIATE APPLICATION RECOVERY

Codex previously identified a SQLite initialization defect in main.py around line 425 involving incorrectly split SQL triggers and an "incomplete input" error.

Fix this defect safely and permanently.

Then:
- Verify Python dependencies and application startup.
- Verify FastAPI routes and /healthz.
- Test patient creation, conversation management, and persistence.
- Test the frontend and backend together.
- Verify authentication and authorization behavior.
- Test OpenAI and Qdrant integrations using mocks or dedicated development credentials.
- Add regression tests for every startup defect.
- Fix additional reproducible errors.
- Prepare a working development build.

Do not delete existing patient records or production databases. Use temporary databases and synthetic data.

## 2. CREATE THE PERMANENT MASTER REQUIREMENTS

Create docs/RAPHAEL_MASTER_SPEC.md containing the complete requirements below.

Do not treat the list as implemented merely because it appears in documentation.

### Clinical intelligence
- Evidence-grounded medical question answering
- Working diagnoses and ranked differential diagnoses
- Supporting and contradictory evidence
- Missing information and recommended diagnostic tests
- Emergency red flags and escalation
- Clinical uncertainty and appropriate abstention
- Treatment considerations with evidence
- Multispecialty reasoning and cross-specialty correlation
- Medical education and reasoning explanations
- Human clinician confirmation and oversight

### Medical knowledge
- PubMed and PubMed Central
- ClinicalTrials.gov
- NIH, CDC, WHO, FDA, NICE, USPSTF
- Publicly reusable specialty-society guidelines
- Legally licensed educational and reference material
- Evidence classification and quality appraisal
- Source metadata and provenance
- Hybrid retrieval, reranking, and Qdrant indexing
- Verified citations and links
- Source versioning and update monitoring
- Contradiction detection and guideline currency
- Automated evidence ingestion with review safeguards

### Radiology and imaging
- DICOM ingestion and validation
- X-ray, CT, MRI, and ultrasound workflows
- Medical image viewing
- Validated image-model integrations
- Structured radiology reporting
- Imaging findings correlated with clinical context
- Prior-study and prior-report comparison
- Missing findings, discordance, and ambiguity detection
- PACS integration
- Radiology-specific clinical safety evaluation

### Patient intelligence
- Patient records and longitudinal timelines
- Medical histories and encounter notes
- Laboratory values, trends, and abnormalities
- Medications, allergies, and contraindications
- Vital signs and clinical observations
- Clinical summarization and case review
- FHIR resource ingestion and provenance
- Secure organization and patient separation

### Voice and documentation
- Clinical speech transcription
- Natural spoken interaction
- Encounter note drafting
- Clinical documentation templates
- Structured reports
- Clinician editing, approval, and export
- Documentation quality evaluation

### Clinical safety and workflow
- Drug-drug interaction checks
- Allergy and contraindication checks
- Validated dosing support where available
- Deterministic clinical safety rules
- Clinician approval and review
- Audit logging and traceability
- Appropriate refusal, abstention, and escalation
- Clinical workflow integration

### Enterprise platform
- Secure authentication and role-based access
- Multi-tenant isolation
- PostgreSQL persistence and migrations
- Encryption and secrets management
- SMART-on-FHIR and EHR interoperability
- PACS connectivity
- Monitoring, logging, backups, and recovery
- Admin controls and usage analytics
- Billing and subscriptions
- Deployment environments and CI/CD
- Regulatory documentation and risk management

### Clinical evaluation and research
- Expert-reviewed clinical case benchmarks
- Diagnostic and differential accuracy
- Evidence factuality and citation verification
- Guideline adherence
- ICD/CPT coding evaluation
- Clinical note quality
- Clinician usefulness, safety, and time-saved ratings
- Model comparisons and ablation studies
- Reproducible evaluation configurations
- Red-team and regression tests
- Controlled clinician-feedback loops
- Research-ready documentation and reporting

## 3. CREATE A REQUIREMENTS TRACEABILITY MATRIX

Create docs/REQUIREMENTS_MATRIX.md.

Assign every requirement a unique ID.

For each requirement record:
- Requirement description
- Priority
- Current status: verified, partial, missing, blocked, or untested
- Existing implementation files
- Required dependencies
- Tests that verify it
- Completion criteria
- Remaining work
- Target release milestone

Do not mark any feature complete without evidence and passing tests.

## 4. CREATE A DEVELOPMENT ROADMAP

Create docs/ROADMAP.md with milestones:

M1: Working development application
M2: Evidence-grounded medical intelligence
M3: Advanced diagnostic reasoning
M4: Multimodal imaging, voice, and clinical integrations
M5: Clinical evaluation and security hardening
M6: Regulated clinical deployment readiness

Features may be developed in parallel where dependencies permit.

Clearly distinguish the immediate test release from later clinical deployment.

## 5. CREATE PERMANENT CODEX INSTRUCTIONS

Create or update AGENTS.md to require:
- Reading the master specification before relevant changes
- Maintaining the requirements matrix
- Preserving existing working functionality
- Implementing complete features rather than misleading placeholders
- Automated testing and regression checks
- No fabricated citations or clinical validation claims
- No real patient data in development
- No production deployment without approval
- No autonomous clinical decisions
- No production secrets in source code
- Clinician oversight for diagnostic outputs

## 6. DELIVER THE FIRST WORKING TEST RELEASE

Implement the highest-priority missing functionality necessary for a usable development application.

Acceptance criteria:
- Application starts successfully
- Frontend is accessible
- Authentication works for development users
- Synthetic patients can be created and managed
- Clinical conversations persist
- AI inference works with development credentials
- Medical evidence retrieval works against verified sources
- Structured clinical answers include citations and uncertainty
- Automated tests pass
- Known limitations are documented
- Development deployment instructions are provided

Do not represent mocked integrations as live functionality.

## 7. DELIVERY RULES

Work in a development branch.

Implement code and run tests rather than stopping at recommendations.

Prepare reviewable pull requests.

Do not merge, deploy, delete production data, or activate billing without approval.

If the complete scope cannot fit into one task, finish the application recovery and master documentation first, then create clearly scoped implementation tasks for the remaining work.

At completion, provide:
1. A working-build status report
2. Tests run and their results
3. A complete feature inventory
4. A requirements traceability matrix
5. A prioritized development roadmap
6. Exact remaining steps for a development deployment
7. A pull request for review

The goal is a working Raphael test release now, with a permanent engineering system that systematically delivers the entire long-term vision.

# Canonical reconciliation: expanded platform directive

This section adds to every requirement above. No earlier requirement or ID is retired. The matrix remains the authoritative implementation checklist; a source registry, scaffold, mocked service, historical route or documentation entry does not prove a feature works. The canonical source set is the two owner-supplied directives, tracked current source/configuration/tests/docs, and locally available Git history. No access to unprovided ChatGPT conversation history is claimed.

The sanitized repository origin is `AIGeniusAlien/Raphael-AI`. The owner text uses a visually similar spelling; repository identity is taken from Git, not guessed. Continue on `fix/sqlite-startup-recovery`; do not merge or deploy automatically.

Status vocabulary is defined in REQUIREMENTS_MATRIX.md. Every new ID below has implementation status, dependencies, test coverage, acceptance criteria and milestone there. IDs CI/MK/RI/PI/VD/CS/EP/ER/REC/REL retain their prior meanings.

## Complete patient portal

- **PP-001** — Secure patient registration and authentication
- **PP-002** — Identity verification and account recovery
- **PP-003** — Patient profiles and authorized chart access
- **PP-004** — Medical histories and longitudinal records
- **PP-005** — Laboratory results and trends
- **PP-006** — Imaging results and reports
- **PP-007** — Medication lists and allergies
- **PP-008** — Clinical notes and visit summaries
- **PP-009** — Document upload and authorized sharing
- **PP-010** — Patient-friendly explanations of approved medical information
- **PP-011** — Care plans and follow-up instructions
- **PP-012** — Appointment-related workflows and reminders
- **PP-013** — Patient consent and information-sharing permissions
- **PP-014** — Secure communication with physicians and care teams
- **PP-015** — Message history and notification preferences
- **PP-016** — Accessibility and responsive mobile/desktop design
- **PP-017** — Proxy/caregiver access with appropriate authorization
- **PP-018** — Auditable access and privacy controls
- **PP-019** — Patient-facing AI explanations distinguish general information from clinician-reviewed advice

## Complete physician portal

- **PH-001** — Secure clinician authentication and role verification
- **PH-002** — Patient worklists and search
- **PH-003** — Clinical chart review
- **PH-004** — Longitudinal patient timelines
- **PH-005** — Clinical summaries
- **PH-006** — Laboratory and medication review
- **PH-007** — Clinical AI chat with patient context
- **PH-008** — Ranked differential diagnoses and working diagnoses
- **PH-009** — Supporting and contradicting evidence
- **PH-010** — Clinical red flags and diagnostic recommendations
- **PH-011** — Evidence citations and source verification
- **PH-012** — Medical note drafting and editing
- **PH-013** — Clinical coding assistance
- **PH-014** — Draft order sets and care plans
- **PH-015** — Clinician approval and electronic sign-off workflows
- **PH-016** — Radiology interpretation support
- **PH-017** — Prior imaging comparison
- **PH-018** — Report generation and review
- **PH-019** — Follow-up tracking
- **PH-020** — Secure patient and care-team messaging
- **PH-021** — Critical finding communication workflows
- **PH-022** — Institutional protocols
- **PH-023** — Audit trails and user activity records
- **PH-024** — No autonomous clinical orders or unauthorized clinical decisions

## Complete secure messaging

- **MSG-001** — Two-way secure messaging
- **MSG-002** — Conversation threads
- **MSG-003** — Patient-to-doctor messages
- **MSG-004** — Doctor-to-patient messages
- **MSG-005** — Authorized clinician-to-clinician communication
- **MSG-006** — Attachments and clinical documents
- **MSG-007** — Message routing and inbox management
- **MSG-008** — Read, delivery, and response status
- **MSG-009** — Notifications
- **MSG-010** — Escalation and urgent-message policies
- **MSG-011** — Communication audit logs
- **MSG-012** — Consent and recipient authorization
- **MSG-013** — Clinician-reviewed AI-assisted drafting
- **MSG-014** — Explicit approval before AI-assisted messages are sent
- **MSG-015** — EHR messaging integration where supported
- **MSG-016** — Clear distinction between internal portal messages and externally delivered messages
- **MSG-017** — Do not claim delivery unless the delivery service confirms it

## Comprehensive medical knowledge

- **KNOW-001** — PubMed and PubMed Central
- **KNOW-002** — ClinicalTrials.gov
- **KNOW-003** — FDA labeling and safety communications
- **KNOW-004** — NIH and CDC
- **KNOW-005** — WHO
- **KNOW-006** — NICE
- **KNOW-007** — USPSTF
- **KNOW-008** — Medical specialty societies
- **KNOW-009** — Clinical guidelines and consensus statements
- **KNOW-010** — Peer-reviewed journals
- **KNOW-011** — Systematic reviews and meta-analyses
- **KNOW-012** — Medical education and teaching cases
- **KNOW-013** — Institutional clinical protocols
- **KNOW-014** — Licensed medical textbooks and references
- **KNOW-015** — Drug and medication terminology databases
- **KNOW-016** — ICD-10-CM, SNOMED CT, and CPT resources where licensing permits
- **KNOW-017** — Evidence provenance
- **KNOW-018** — Licensing checks before ingestion
- **KNOW-019** — Source versioning
- **KNOW-020** — Evidence quality appraisal
- **KNOW-021** — Citation verification
- **KNOW-022** — Evidence deduplication
- **KNOW-023** — Evidence freshness checks
- **KNOW-024** — Evidence contradiction detection
- **KNOW-025** — Retrieval evaluation
- **KNOW-026** — Never fabricate citations or claim comprehensive coverage without verification

## Advanced clinical intelligence

- **ADV-001** — Multispecialty diagnostic reasoning
- **ADV-002** — Working and differential diagnoses
- **ADV-003** — Emergency red flags
- **ADV-004** — Clinical uncertainty and abstention
- **ADV-005** — Treatment decision support
- **ADV-006** — Medication safety
- **ADV-007** — Contraindication and interaction detection
- **ADV-008** — Laboratory and vital-sign analysis
- **ADV-009** — Longitudinal clinical reasoning
- **ADV-010** — Patient-specific evidence applicability
- **ADV-011** — Guideline comparison
- **ADV-012** — Clinician-reviewed order sets
- **ADV-013** — Clinical coding
- **ADV-014** — Medical education
- **ADV-015** — Clinical documentation

## Multimodal capabilities

- **MM-001** — DICOM processing
- **MM-002** — CT, MRI, X-ray, and ultrasound workflows
- **MM-003** — Validated imaging-model integration
- **MM-004** — PACS connectivity
- **MM-005** — Prior imaging comparison
- **MM-006** — Structured radiology reports
- **MM-007** — Voice transcription
- **MM-008** — Speech synthesis
- **MM-009** — Conversational voice interaction
- **MM-010** — Optional wake-phrase activation
- **MM-011** — SMART-on-FHIR and EHR interoperability
- **MM-012** — Clinical time-series data
- **MM-013** — Multimodal evidence synthesis

## Security, regulation and quality

- **SEC-001** — Role-based and attribute-based access controls
- **SEC-002** — Multi-tenant data isolation
- **SEC-003** — Patient consent management
- **SEC-004** — Authentication and authorization testing
- **SEC-005** — Encryption and secure secrets management
- **SEC-006** — Audit logs
- **SEC-007** — PostgreSQL migrations
- **SEC-008** — Backups and disaster recovery
- **SEC-009** — Secure message delivery
- **SEC-010** — Clinical risk management
- **SEC-011** — HIPAA and BAA readiness assessment
- **SEC-012** — FDA intended-use and regulatory pathway assessment
- **SEC-013** — IRB requirements where applicable
- **SEC-014** — Independent security and clinical validation
- **SEC-015** — Bias and subgroup performance evaluations
- **SEC-016** — Clinical regression testing
- **SEC-017** — Model and dataset versioning
- **SEC-018** — Clinician-feedback review
- **SEC-019** — Controlled releases

## Immediate synthetic-data test release

- **TEST-001** — Working backend and frontend
- **TEST-002** — Development authentication
- **TEST-003** — Synthetic patient records
- **TEST-004** — Persistent clinical conversations
- **TEST-005** — OpenAI development inference
- **TEST-006** — Evidence retrieval with verified citations
- **TEST-007** — Structured clinical assessments
- **TEST-008** — Basic patient and physician portal interfaces
- **TEST-009** — Internal test messaging between synthetic patient and clinician accounts
- **TEST-010** — Clinical note drafting
- **TEST-011** — Automated tests
- **TEST-012** — Explicitly labeled incomplete capabilities
- **TEST-013** — Use synthetic data only in the immediate release
- **TEST-014** — No real patient accounts, real clinical message delivery or real-patient care workflows

## Delivery and governance

- **GOV-001** — Reconcile all available repository, documentation, tests, configuration and historical artifacts without claiming unavailable conversation history
- **GOV-002** — Keep one canonical additive specification and preserve incomplete, blocked and unimplemented requirements
- **GOV-003** — Maintain the authoritative requirements matrix with status, dependencies, tests, criteria and milestones
- **GOV-004** — Maintain a complete feature inventory and explicit gap analysis
- **GOV-005** — Continue on fix/sqlite-startup-recovery without discarding existing fixes
- **GOV-006** — Do not deploy, merge or modify production data without approval
- **GOV-007** — Resolve pull-request access only through documented authorized settings and review
- **GOV-008** — Implement the immediate release in manageable, tested changes
- **GOV-009** — Classify each feature as tested, unvalidated, partial, missing, externally blocked or awaiting independent validation
- **GOV-010** — Do not report Raphael complete until every requirement independently meets its acceptance criteria
- **GOV-011** — Preserve no-real-data, no-secrets-in-source, clinician-oversight and regression-test engineering instructions
- **GOV-012** — Maintain the required five canonical documents

## Historical and repository artifact requirements

- **HIST-001** — Safe public application/model/voice capability configuration
- **HIST-002** — Authorized patient removal with safe dependent-record policy
- **HIST-003** — Direct uploaded-PDF evidence ingestion
- **HIST-004** — URL evidence ingestion compatibility and safe destinations
- **HIST-005** — SMART-on-FHIR authorization and callback workflow
- **HIST-006** — FHIR patient create/read adapters
- **HIST-007** — FHIR observations retrieval
- **HIST-008** — FHIR DocumentReference upload
- **HIST-009** — FHIR condition/medication/allergy/vitals summary
- **HIST-010** — FHIR note drafting without automatic clinical submission
- **HIST-011** — FHIR order drafting without autonomous ordering
- **HIST-012** — Authorized local DICOM metadata ingestion adapter
- **HIST-013** — DICOMweb study search and instance retrieval adapter
- **HIST-014** — Reviewed subscription checkout integration
- **HIST-015** — Realtime voice session negotiation
- **HIST-016** — Voice intent parsing with explicit confirmation before actions
- **HIST-017** — Voice WebSocket transport and streaming protocol
- **HIST-018** — Frontend discovery and legacy handsfree/FHIR-demo navigation
- **HIST-019** — Electron desktop shell and isolated preload API bridge
- **HIST-020** — Browser extension and optional wake phrase
- **HIST-021** — Kubernetes deployment architecture planning
- **HIST-022** — Source registry update cadence and curated PubMed query packs
- **HIST-023** — Reproducible case/artifact-based evaluation workflow
- **HIST-024** — OIDC, restricted CORS and licensed reference/drug integration for pilots

## Clinical and release boundaries

The immediate deliverable is a synthetic-data development test application, with distinct patient and physician roles, authorized internal messaging and clinician-reviewed drafts. It must not connect real patient identities, deliver clinical messages externally, place autonomous orders or imply that AI-generated text is clinician-approved. Live development inference and verified evidence retrieval are separate gates from mocked protocol tests.

The complete patient portal, physician portal, messaging system and clinical platform remain required after the test release. Clinical, security, regulatory and operational acceptance require independent evaluation, not merely passing developer tests. M1 is not M6. Historical dangerous shortcuts are recorded as requirements for secure replacement, not instructions to restore unsafe code verbatim.

## Source-to-ID reconciliation

| Source | Preserved IDs |
| --- | --- |
| Previous owner master directive, original eight feature categories | CI, MK, RI, PI, VD, CS, EP, ER |
| Prior recovery and first-release gates | REC, REL |
| New patient/physician portals and secure messaging | PP, PH, MSG |
| Expanded knowledge, reasoning, multimodal and quality requirements | KNOW, ADV, MM, SEC |
| New immediate release and delivery rules | TEST, GOV |
| Current repository artifacts and historical routes | HIST |

The explicit gap analysis is in FEATURE_INVENTORY.md and RELEASE_READINESS.md; sequenced implementation and release dependencies are in ROADMAP.md.


# Complete-platform implementation directive (additive)

The owner supplied the following directive after the portal increment. It extends the canonical specification; it does not remove any preceding requirement. The exact source is retained below, followed by a source-to-ID index. Implementation and tests, not this directive, determine completion. Only repository artifacts and supplied documents were inspected; no unprovided conversation history is claimed.

# RAPHAEL AI — COMPLETE PLATFORM IMPLEMENTATION DIRECTIVE

## YOUR ROLE

You are the principal software architect, senior full-stack engineer, AI systems engineer, clinical informatics engineer, medical imaging engineer, security engineer, and quality assurance lead responsible for building Raphael AI.

Repository: AlGeniusAlien/Raphael-AI

Existing development branch: fix/sqlite-startup-recovery

**PRIMARY OBJECTIVE: IMPLEMENT THE COMPLETE RAPHAEL PLATFORM, NOT JUST DOCUMENT OR PLAN IT.**

The owner wants a fully functional, comprehensive, multimodal medical AI platform that combines medical knowledge, clinical reasoning, diagnostic support, radiology, patient and physician portals, secure communications, EHR interoperability, and continuous improvement.

The immediate deliverable is an integrated, deployable application that operates end-to-end with synthetic medical data and development integrations.

The longer-term objective is a clinically validated platform suitable for appropriately authorized medical environments.

Do not confuse a working development platform with a clinically validated or regulator-authorized medical device.

## NON-NEGOTIABLE INSTRUCTIONS

1. Inspect the entire repository before making changes.
2. Read AGENTS.md, RAPHAEL_MASTER_SPEC.md, REQUIREMENTS_MATRIX.md, ROADMAP.md, and all other project documentation.
3. Preserve all existing functional code, designs, integrations, and features.
4. Fix broken or incomplete functionality.
5. Implement the full scope below, not a generic chatbot.
6. Do not stop after writing plans, specifications, or TODO lists.
7. Work through implementation, integration, testing, debugging, and documentation in that order as appropriate.
8. Continue making progress through the requirements backlog within the available execution context.
9. If a task exceeds execution limits, leave the repository in a tested state, document the exact next executable steps, and provide a continuation instruction that resumes without repeating completed work.
10. Never claim functionality is complete unless it has passed meaningful tests.
11. Use mock services only in explicitly identified development/test modes.
12. Do not represent mock results as real clinical findings or live integrations.
13. Never fabricate medical evidence, citations, test results, or regulatory compliance.
14. Do not use real patient information.
15. Do not deploy to production, merge protected branches, send real clinical communications, or activate live billing without explicit approval.

## SYSTEM 1 — COMPLETE USER INTERFACE

Build a polished, responsive Raphael application with separate experiences for:

- Patients
- Physicians
- Radiologists
- Clinical staff
- Organization administrators
- Platform administrators

Implement account management, secure login, role-aware navigation, accessibility, responsive layouts, and useful error states.

The interface must support desktop and mobile devices.

## SYSTEM 2 — PATIENT PORTAL

Implement:

- Patient profiles
- Authorized medical records
- Medical history
- Medications and allergies
- Laboratory results and trends
- Imaging reports
- Visit summaries
- Clinical documents
- Care plans
- Appointments and follow-up workflows
- Secure doctor messaging
- Notifications
- Document upload
- Consent management
- Proxy and caregiver authorization
- Patient-friendly explanations of clinician-approved information

Build working frontend components, backend APIs, database models, authorization rules, and automated tests.

## SYSTEM 3 — PHYSICIAN PORTAL

Implement:

- Patient worklists
- Patient search and chart review
- Longitudinal medical histories
- Clinical encounter management
- AI-assisted clinical chat
- Differential diagnosis generation
- Working diagnosis generation
- Diagnostic test suggestions
- Laboratory interpretation
- Medication review
- Clinical summaries
- Note generation
- Clinical documentation editing
- Coding assistance
- Draft orders and order sets
- Clinical guideline references
- Evidence-backed recommendations
- Radiology report review
- Patient messaging
- Care-team communication
- Physician approval and sign-off
- Audit trails

The physician must remain in control of consequential clinical decisions.

## SYSTEM 4 — SECURE CLINICAL MESSAGING

Build a complete internal communication system with:

- Patient-to-physician messaging
- Physician-to-patient messaging
- Clinician-to-clinician messaging
- Message threads
- Inbox management
- Recipient authorization
- Attachments
- Message status
- Read receipts where appropriate
- Notifications
- Clinician-approved AI-assisted message drafting
- Audit records
- Message routing
- Follow-up tracking
- Urgent-message escalation policies

Internal synthetic-account messaging must function end-to-end.

External email, SMS, and EHR messaging integrations should use modular adapters and remain disabled until appropriately configured and authorized.

Never autonomously send clinical advice to real patients.

## SYSTEM 5 — ADVANCED MEDICAL INTELLIGENCE

Build an evidence-grounded clinical reasoning system capable of:

- Working diagnoses
- Ranked differential diagnoses
- Supporting clinical findings
- Contradictory evidence
- Dangerous alternative diagnoses
- Missing clinical information
- Recommended diagnostic evaluation
- Treatment considerations
- Medication safety analysis
- Clinical guideline comparisons
- Multispecialty reasoning
- Longitudinal patient analysis
- Clinical uncertainty
- Evidence verification
- Clear medical explanations

Implement structured outputs, input validation, safety checks, and source attribution.

Clinical responses must distinguish evidence, inference, and uncertainty.

## SYSTEM 6 — COMPREHENSIVE MEDICAL KNOWLEDGE

Implement source-aware retrieval and ingestion for legally accessible resources, including:

- PubMed
- PubMed Central
- ClinicalTrials.gov
- FDA drug labeling
- FDA safety communications
- NIH
- CDC
- WHO
- NICE
- USPSTF
- Publicly reusable specialty-society guidelines
- Licensed medical references when credentials and permissions are supplied
- Open-access medical educational materials
- Institutional guidelines where authorized

Build:

- Source connectors
- Document ingestion
- Metadata extraction
- Licensing checks
- Evidence quality assessment
- Publication and update dates
- Version tracking
- Deduplication
- Document chunking
- Embeddings
- Qdrant integration
- Hybrid search
- Reranking
- Citation verification
- Source freshness checks
- Contradiction identification
- Scheduled ingestion jobs
- Retrieval evaluation
- Evidence coverage reporting

Do not attempt to scrape copyrighted or restricted medical resources without permission.

Do not claim universal medical knowledge coverage.

## SYSTEM 7 — RADIOLOGY AND MEDICAL IMAGING

Implement:

- DICOM upload
- DICOM validation
- DICOM metadata extraction
- Medical image viewing
- Imaging study management
- X-ray workflows
- CT workflows
- MRI workflows
- Ultrasound workflows
- Prior-study comparison
- Structured radiology reports
- Radiology findings review
- Clinical context integration
- Critical finding communication
- PACS integration adapters
- Imaging audit logs

Support validated imaging models through modular integrations.

Do not generate unsupported diagnostic image findings from text-only models.

## SYSTEM 8 — VOICE AND CLINICAL DOCUMENTATION

Implement:

- Speech-to-text
- Clinical dictation
- Encounter transcription
- Voice-driven clinical interaction
- Text-to-speech where supported
- Optional wake-phrase architecture
- Clinical note drafting
- Structured documentation
- Report generation
- Physician editing and approval
- Documentation export

Use modular providers and testable interfaces.

## SYSTEM 9 — EHR AND INTEROPERABILITY

Implement:

- SMART-on-FHIR architecture
- OAuth authorization flows
- FHIR patient resources
- Observations
- Conditions
- Medication records
- Allergies
- Diagnostic reports
- Imaging study references
- Encounters
- Clinical document resources
- Terminology mapping
- SNOMED CT support where licensed
- ICD-10-CM coding assistance
- CPT assistance where licensed
- EHR integration adapters
- PACS interoperability
- Data provenance

Use sandbox FHIR servers and synthetic data for integration testing.

## SYSTEM 10 — DATABASE AND SECURITY

Implement or complete:

- PostgreSQL support
- Database migrations
- Role-based access control
- Tenant isolation
- Patient-level authorization
- Secure sessions
- Password hashing
- Account recovery
- Audit logging
- Encryption
- Secrets management
- Input validation
- Rate limiting
- Secure file handling
- Backup and recovery procedures
- Health checks
- Readiness checks
- Structured monitoring
- Error reporting
- Security regression tests

Do not weaken security to make a feature work.

## SYSTEM 11 — ADMINISTRATION AND BILLING

Implement:

- Organization management
- User invitations
- Role assignment
- Account status controls
- Usage tracking
- Administrative audit logs
- Subscription management
- Stripe test-mode billing
- Billing history
- Organization settings
- Feature access management

Keep live charges disabled until explicit approval.

## SYSTEM 12 — CLINICAL SAFETY

Implement:

- Emergency red-flag identification
- Evidence-supported safety checks
- Medication interaction validation
- Allergy and contraindication detection
- Appropriate uncertainty reporting
- Abstention when evidence is insufficient
- Clinician confirmation for consequential actions
- Clinical review workflows
- Traceable clinical outputs
- Source verification
- Model failure logging
- Safe handling of incomplete or contradictory data

Document residual clinical risks and validation requirements.

## SYSTEM 13 — CONTINUOUS IMPROVEMENT

Build:

- Clinician feedback collection
- Error reporting
- Correction review
- Evidence gap tracking
- Evaluation datasets
- Model comparison tools
- Retrieval evaluation
- Clinical benchmark testing
- Citation accuracy measurement
- Diagnostic accuracy measurement
- Safety regression testing
- Performance dashboards
- Model and prompt version tracking
- Controlled release procedures

Do not automatically train or deploy models based on unverified clinical feedback.

## SYSTEM 14 — CLINICAL RESEARCH AND VALIDATION

Prepare reproducible evaluation workflows using appropriately authorized datasets and expert-reviewed cases.

Include evaluation of:

- Diagnostic accuracy
- Differential diagnosis quality
- Dangerous-condition detection
- Medical factuality
- Evidence citation accuracy
- Guideline adherence
- Imaging performance
- Medication safety
- Clinical note quality
- Speech transcription accuracy
- Clinician usability
- Bias and subgroup performance
- Model uncertainty

Create research-ready evaluation documentation.

Do not claim independent clinical validation unless it has occurred.

## SYSTEM 15 — DEPLOYMENT AND OPERATIONS

Prepare:

- Reproducible development environment
- Automated tests
- GitHub Actions CI
- Development and staging configurations
- Render-compatible deployment
- Environment-variable validation
- Database migration procedures
- Logging and monitoring
- Dependency vulnerability checks
- Startup and readiness checks
- Deployment rollback procedures
- Operational documentation

Maintain a reproducible, deployable application throughout development.

## REQUIRED IMPLEMENTATION PROCESS

Work through the systems in dependency-aware order.

Prioritize the core platform, authentication, patient and physician portals, secure internal messaging, clinical AI, and medical evidence retrieval so a complete synthetic-data demonstration can operate as soon as possible.

Then implement imaging, voice, interoperability, administration, and advanced evaluation systems.

For each implemented capability:

1. Write the necessary backend code.
2. Write the necessary frontend code.
3. Add database models and migrations where needed.
4. Implement authorization and security.
5. Add automated tests.
6. Execute tests.
7. Fix failures.
8. Verify integration with existing functionality.
9. Update the requirements matrix.
10. Document remaining limitations.

Do not substitute documentation for implementation.

## REQUIRED END-TO-END TEST SCENARIOS

The application must support these synthetic-data workflows:

A. A patient signs in, views records, and sends a message to a physician.

B. A physician signs in, receives the message, reviews the synthetic patient chart, drafts a response, approves it, and sends it internally.

C. A physician enters a complex synthetic clinical case and receives a structured differential diagnosis with supporting and contradicting evidence.

D. Raphael retrieves verifiable medical evidence and provides citations that resolve to the actual sources.

E. A physician creates and edits an AI-assisted clinical note.

F. A user uploads a valid test DICOM study and views the image and metadata.

G. A development FHIR integration retrieves synthetic patient resources.

H. An administrator manages development users and permissions.

I. Unauthorized users cannot access another patient's records or messages.

J. The application restarts without losing persisted test data.

K. The application reports integration failures without fabricating successful results.

## RELEASE ACCEPTANCE CRITERIA

A development release is ready only when:

- The application starts successfully.
- Critical backend tests pass.
- Frontend tests pass.
- Core end-to-end workflows pass.
- Development authentication works.
- Patient and clinician portals function.
- Internal secure messaging works.
- AI inference is tested.
- Evidence retrieval and citations are verified.
- Synthetic patient records persist.
- Authorization and tenant-isolation tests pass.
- Known limitations are clearly displayed.
- No production secrets are exposed.
- No real patient data is required.

For optional external integrations, provide working adapters, tests, configuration documentation, and explicit status reporting.

Do not describe an integration as operational until it has been tested against the relevant service.

## FINAL DELIVERY

Deliver the maximum complete, tested implementation achievable within the available execution environment.

Create or update:

- AGENTS.md
- docs/RAPHAEL_MASTER_SPEC.md
- docs/REQUIREMENTS_MATRIX.md
- docs/FEATURE_INVENTORY.md
- docs/ROADMAP.md
- docs/RELEASE_READINESS.md
- docs/DEPLOYMENT.md
- docs/TEST_RESULTS.md

Provide:

1. Actual working application status.
2. Implemented features and file locations.
3. Test results.
4. Remaining external dependencies.
5. Known limitations and security risks.
6. A deployable development build.
7. Reviewable GitHub changes.
8. The exact continuation task for any work that could not fit into this execution.

Do not stop at a roadmap.

Do not mark incomplete systems as finished.

Do not merge, deploy to production, activate live billing, use real patient data, or enable real clinical messaging without explicit authorization.

**BEGIN IMPLEMENTATION NOW. PRIORITIZE A WORKING INTEGRATED RAPHAEL APPLICATION AND CONTINUE THROUGH THE FULL REQUIREMENTS BACKLOG AS FAR AS THE EXECUTION ENVIRONMENT PERMITS.**

## Latest directive source-to-ID index

| Directive section | Requirement ID | Requirement |
| --- | --- | --- |
| NON-NEGOTIABLE INSTRUCTIONS | DIRECT-001 | Inspect the entire repository before making changes. |
| NON-NEGOTIABLE INSTRUCTIONS | DIRECT-002 | Read AGENTS.md, RAPHAEL_MASTER_SPEC.md, REQUIREMENTS_MATRIX.md, ROADMAP.md, and all other project documentation. |
| NON-NEGOTIABLE INSTRUCTIONS | DIRECT-003 | Preserve all existing functional code, designs, integrations, and features. |
| NON-NEGOTIABLE INSTRUCTIONS | DIRECT-004 | Fix broken or incomplete functionality. |
| NON-NEGOTIABLE INSTRUCTIONS | DIRECT-005 | Implement the full scope below, not a generic chatbot. |
| NON-NEGOTIABLE INSTRUCTIONS | DIRECT-006 | Do not stop after writing plans, specifications, or TODO lists. |
| NON-NEGOTIABLE INSTRUCTIONS | DIRECT-007 | Work through implementation, integration, testing, debugging, and documentation in that order as appropriate. |
| NON-NEGOTIABLE INSTRUCTIONS | DIRECT-008 | Continue making progress through the requirements backlog within the available execution context. |
| NON-NEGOTIABLE INSTRUCTIONS | DIRECT-009 | If a task exceeds execution limits, leave the repository in a tested state, document the exact next executable steps, and provide a continuation instruction that resumes without repeating completed work. |
| NON-NEGOTIABLE INSTRUCTIONS | DIRECT-010 | Never claim functionality is complete unless it has passed meaningful tests. |
| NON-NEGOTIABLE INSTRUCTIONS | DIRECT-011 | Use mock services only in explicitly identified development/test modes. |
| NON-NEGOTIABLE INSTRUCTIONS | DIRECT-012 | Do not represent mock results as real clinical findings or live integrations. |
| NON-NEGOTIABLE INSTRUCTIONS | DIRECT-013 | Never fabricate medical evidence, citations, test results, or regulatory compliance. |
| NON-NEGOTIABLE INSTRUCTIONS | DIRECT-014 | Do not use real patient information. |
| NON-NEGOTIABLE INSTRUCTIONS | DIRECT-015 | Do not deploy to production, merge protected branches, send real clinical communications, or activate live billing without explicit approval. |
| SYSTEM 1 — COMPLETE USER INTERFACE | UIX-001 | Patients |
| SYSTEM 1 — COMPLETE USER INTERFACE | UIX-002 | Physicians |
| SYSTEM 1 — COMPLETE USER INTERFACE | UIX-003 | Radiologists |
| SYSTEM 1 — COMPLETE USER INTERFACE | UIX-004 | Clinical staff |
| SYSTEM 1 — COMPLETE USER INTERFACE | UIX-005 | Organization administrators |
| SYSTEM 1 — COMPLETE USER INTERFACE | UIX-006 | Platform administrators |
| SYSTEM 2 — PATIENT PORTAL | PTX-001 | Patient profiles |
| SYSTEM 2 — PATIENT PORTAL | PTX-002 | Authorized medical records |
| SYSTEM 2 — PATIENT PORTAL | PTX-003 | Medical history |
| SYSTEM 2 — PATIENT PORTAL | PTX-004 | Medications and allergies |
| SYSTEM 2 — PATIENT PORTAL | PP-005 | Laboratory results and trends |
| SYSTEM 2 — PATIENT PORTAL | PTX-005 | Imaging reports |
| SYSTEM 2 — PATIENT PORTAL | PTX-006 | Visit summaries |
| SYSTEM 2 — PATIENT PORTAL | PTX-007 | Clinical documents |
| SYSTEM 2 — PATIENT PORTAL | PTX-008 | Care plans |
| SYSTEM 2 — PATIENT PORTAL | PTX-009 | Appointments and follow-up workflows |
| SYSTEM 2 — PATIENT PORTAL | PTX-010 | Secure doctor messaging |
| SYSTEM 2 — PATIENT PORTAL | MSG-009 | Notifications |
| SYSTEM 2 — PATIENT PORTAL | PTX-011 | Document upload |
| SYSTEM 2 — PATIENT PORTAL | PTX-012 | Consent management |
| SYSTEM 2 — PATIENT PORTAL | PTX-013 | Proxy and caregiver authorization |
| SYSTEM 2 — PATIENT PORTAL | PTX-014 | Patient-friendly explanations of clinician-approved information |
| SYSTEM 2 — PATIENT PORTAL | SAFE-001 | Build working frontend components, backend APIs, database models, authorization rules, and automated tests. |
| SYSTEM 3 — PHYSICIAN PORTAL | PHX-001 | Patient worklists |
| SYSTEM 3 — PHYSICIAN PORTAL | PHX-002 | Patient search and chart review |
| SYSTEM 3 — PHYSICIAN PORTAL | PHX-003 | Longitudinal medical histories |
| SYSTEM 3 — PHYSICIAN PORTAL | PHX-004 | Clinical encounter management |
| SYSTEM 3 — PHYSICIAN PORTAL | PHX-005 | AI-assisted clinical chat |
| SYSTEM 3 — PHYSICIAN PORTAL | PHX-006 | Differential diagnosis generation |
| SYSTEM 3 — PHYSICIAN PORTAL | PHX-007 | Working diagnosis generation |
| SYSTEM 3 — PHYSICIAN PORTAL | PHX-008 | Diagnostic test suggestions |
| SYSTEM 3 — PHYSICIAN PORTAL | PHX-009 | Laboratory interpretation |
| SYSTEM 3 — PHYSICIAN PORTAL | PHX-010 | Medication review |
| SYSTEM 3 — PHYSICIAN PORTAL | PH-005 | Clinical summaries |
| SYSTEM 3 — PHYSICIAN PORTAL | PHX-011 | Note generation |
| SYSTEM 3 — PHYSICIAN PORTAL | PHX-012 | Clinical documentation editing |
| SYSTEM 3 — PHYSICIAN PORTAL | PHX-013 | Coding assistance |
| SYSTEM 3 — PHYSICIAN PORTAL | PHX-014 | Draft orders and order sets |
| SYSTEM 3 — PHYSICIAN PORTAL | PHX-015 | Clinical guideline references |
| SYSTEM 3 — PHYSICIAN PORTAL | PHX-016 | Evidence-backed recommendations |
| SYSTEM 3 — PHYSICIAN PORTAL | PHX-017 | Radiology report review |
| SYSTEM 3 — PHYSICIAN PORTAL | PHX-018 | Patient messaging |
| SYSTEM 3 — PHYSICIAN PORTAL | PHX-019 | Care-team communication |
| SYSTEM 3 — PHYSICIAN PORTAL | PHX-020 | Physician approval and sign-off |
| SYSTEM 3 — PHYSICIAN PORTAL | PHX-021 | Audit trails |
| SYSTEM 3 — PHYSICIAN PORTAL | SAFE-002 | The physician must remain in control of consequential clinical decisions. |
| SYSTEM 4 — SECURE CLINICAL MESSAGING | MSX-001 | Patient-to-physician messaging |
| SYSTEM 4 — SECURE CLINICAL MESSAGING | MSX-002 | Physician-to-patient messaging |
| SYSTEM 4 — SECURE CLINICAL MESSAGING | MSX-003 | Clinician-to-clinician messaging |
| SYSTEM 4 — SECURE CLINICAL MESSAGING | MSX-004 | Message threads |
| SYSTEM 4 — SECURE CLINICAL MESSAGING | MSX-005 | Inbox management |
| SYSTEM 4 — SECURE CLINICAL MESSAGING | MSX-006 | Recipient authorization |
| SYSTEM 4 — SECURE CLINICAL MESSAGING | MSX-007 | Attachments |
| SYSTEM 4 — SECURE CLINICAL MESSAGING | MSX-008 | Message status |
| SYSTEM 4 — SECURE CLINICAL MESSAGING | MSX-009 | Read receipts where appropriate |
| SYSTEM 4 — SECURE CLINICAL MESSAGING | MSG-009 | Notifications |
| SYSTEM 4 — SECURE CLINICAL MESSAGING | MSX-010 | Clinician-approved AI-assisted message drafting |
| SYSTEM 4 — SECURE CLINICAL MESSAGING | MSX-011 | Audit records |
| SYSTEM 4 — SECURE CLINICAL MESSAGING | MSX-012 | Message routing |
| SYSTEM 4 — SECURE CLINICAL MESSAGING | PH-019 | Follow-up tracking |
| SYSTEM 4 — SECURE CLINICAL MESSAGING | MSX-013 | Urgent-message escalation policies |
| SYSTEM 4 — SECURE CLINICAL MESSAGING | SAFE-003 | Internal synthetic-account messaging must function end-to-end. |
| SYSTEM 4 — SECURE CLINICAL MESSAGING | SAFE-004 | External email, SMS, and EHR messaging integrations should use modular adapters and remain disabled until appropriately configured and authorized. |
| SYSTEM 4 — SECURE CLINICAL MESSAGING | SAFE-005 | Never autonomously send clinical advice to real patients. |
| SYSTEM 5 — ADVANCED MEDICAL INTELLIGENCE | AIX-001 | Working diagnoses |
| SYSTEM 5 — ADVANCED MEDICAL INTELLIGENCE | AIX-002 | Ranked differential diagnoses |
| SYSTEM 5 — ADVANCED MEDICAL INTELLIGENCE | AIX-003 | Supporting clinical findings |
| SYSTEM 5 — ADVANCED MEDICAL INTELLIGENCE | AIX-004 | Contradictory evidence |
| SYSTEM 5 — ADVANCED MEDICAL INTELLIGENCE | AIX-005 | Dangerous alternative diagnoses |
| SYSTEM 5 — ADVANCED MEDICAL INTELLIGENCE | AIX-006 | Missing clinical information |
| SYSTEM 5 — ADVANCED MEDICAL INTELLIGENCE | AIX-007 | Recommended diagnostic evaluation |
| SYSTEM 5 — ADVANCED MEDICAL INTELLIGENCE | AIX-008 | Treatment considerations |
| SYSTEM 5 — ADVANCED MEDICAL INTELLIGENCE | AIX-009 | Medication safety analysis |
| SYSTEM 5 — ADVANCED MEDICAL INTELLIGENCE | AIX-010 | Clinical guideline comparisons |
| SYSTEM 5 — ADVANCED MEDICAL INTELLIGENCE | AIX-011 | Multispecialty reasoning |
| SYSTEM 5 — ADVANCED MEDICAL INTELLIGENCE | AIX-012 | Longitudinal patient analysis |
| SYSTEM 5 — ADVANCED MEDICAL INTELLIGENCE | AIX-013 | Clinical uncertainty |
| SYSTEM 5 — ADVANCED MEDICAL INTELLIGENCE | AIX-014 | Evidence verification |
| SYSTEM 5 — ADVANCED MEDICAL INTELLIGENCE | AIX-015 | Clear medical explanations |
| SYSTEM 5 — ADVANCED MEDICAL INTELLIGENCE | SAFE-006 | Implement structured outputs, input validation, safety checks, and source attribution. |
| SYSTEM 5 — ADVANCED MEDICAL INTELLIGENCE | SAFE-007 | Clinical responses must distinguish evidence, inference, and uncertainty. |
| SYSTEM 6 — COMPREHENSIVE MEDICAL KNOWLEDGE | KNX-001 | PubMed |
| SYSTEM 6 — COMPREHENSIVE MEDICAL KNOWLEDGE | KNX-002 | PubMed Central |
| SYSTEM 6 — COMPREHENSIVE MEDICAL KNOWLEDGE | KNOW-002 | ClinicalTrials.gov |
| SYSTEM 6 — COMPREHENSIVE MEDICAL KNOWLEDGE | KNX-003 | FDA drug labeling |
| SYSTEM 6 — COMPREHENSIVE MEDICAL KNOWLEDGE | KNX-004 | FDA safety communications |
| SYSTEM 6 — COMPREHENSIVE MEDICAL KNOWLEDGE | KNX-005 | NIH |
| SYSTEM 6 — COMPREHENSIVE MEDICAL KNOWLEDGE | KNX-006 | CDC |
| SYSTEM 6 — COMPREHENSIVE MEDICAL KNOWLEDGE | KNOW-005 | WHO |
| SYSTEM 6 — COMPREHENSIVE MEDICAL KNOWLEDGE | KNOW-006 | NICE |
| SYSTEM 6 — COMPREHENSIVE MEDICAL KNOWLEDGE | KNOW-007 | USPSTF |
| SYSTEM 6 — COMPREHENSIVE MEDICAL KNOWLEDGE | MK-004 | Publicly reusable specialty-society guidelines |
| SYSTEM 6 — COMPREHENSIVE MEDICAL KNOWLEDGE | KNX-007 | Licensed medical references when credentials and permissions are supplied |
| SYSTEM 6 — COMPREHENSIVE MEDICAL KNOWLEDGE | KNX-008 | Open-access medical educational materials |
| SYSTEM 6 — COMPREHENSIVE MEDICAL KNOWLEDGE | KNX-009 | Institutional guidelines where authorized |
| SYSTEM 6 — COMPREHENSIVE MEDICAL KNOWLEDGE | KNX-010 | Source connectors |
| SYSTEM 6 — COMPREHENSIVE MEDICAL KNOWLEDGE | KNX-011 | Document ingestion |
| SYSTEM 6 — COMPREHENSIVE MEDICAL KNOWLEDGE | KNX-012 | Metadata extraction |
| SYSTEM 6 — COMPREHENSIVE MEDICAL KNOWLEDGE | KNX-013 | Licensing checks |
| SYSTEM 6 — COMPREHENSIVE MEDICAL KNOWLEDGE | KNX-014 | Evidence quality assessment |
| SYSTEM 6 — COMPREHENSIVE MEDICAL KNOWLEDGE | KNX-015 | Publication and update dates |
| SYSTEM 6 — COMPREHENSIVE MEDICAL KNOWLEDGE | KNX-016 | Version tracking |
| SYSTEM 6 — COMPREHENSIVE MEDICAL KNOWLEDGE | KNX-017 | Deduplication |
| SYSTEM 6 — COMPREHENSIVE MEDICAL KNOWLEDGE | KNX-018 | Document chunking |
| SYSTEM 6 — COMPREHENSIVE MEDICAL KNOWLEDGE | KNX-019 | Embeddings |
| SYSTEM 6 — COMPREHENSIVE MEDICAL KNOWLEDGE | KNX-020 | Qdrant integration |
| SYSTEM 6 — COMPREHENSIVE MEDICAL KNOWLEDGE | KNX-021 | Hybrid search |
| SYSTEM 6 — COMPREHENSIVE MEDICAL KNOWLEDGE | KNX-022 | Reranking |
| SYSTEM 6 — COMPREHENSIVE MEDICAL KNOWLEDGE | KNOW-021 | Citation verification |
| SYSTEM 6 — COMPREHENSIVE MEDICAL KNOWLEDGE | KNX-023 | Source freshness checks |
| SYSTEM 6 — COMPREHENSIVE MEDICAL KNOWLEDGE | KNX-024 | Contradiction identification |
| SYSTEM 6 — COMPREHENSIVE MEDICAL KNOWLEDGE | KNX-025 | Scheduled ingestion jobs |
| SYSTEM 6 — COMPREHENSIVE MEDICAL KNOWLEDGE | KNOW-025 | Retrieval evaluation |
| SYSTEM 6 — COMPREHENSIVE MEDICAL KNOWLEDGE | KNX-026 | Evidence coverage reporting |
| SYSTEM 6 — COMPREHENSIVE MEDICAL KNOWLEDGE | SAFE-008 | Do not attempt to scrape copyrighted or restricted medical resources without permission. |
| SYSTEM 6 — COMPREHENSIVE MEDICAL KNOWLEDGE | SAFE-009 | Do not claim universal medical knowledge coverage. |
| SYSTEM 7 — RADIOLOGY AND MEDICAL IMAGING | IMX-001 | DICOM upload |
| SYSTEM 7 — RADIOLOGY AND MEDICAL IMAGING | IMX-002 | DICOM validation |
| SYSTEM 7 — RADIOLOGY AND MEDICAL IMAGING | IMX-003 | DICOM metadata extraction |
| SYSTEM 7 — RADIOLOGY AND MEDICAL IMAGING | RI-003 | Medical image viewing |
| SYSTEM 7 — RADIOLOGY AND MEDICAL IMAGING | IMX-004 | Imaging study management |
| SYSTEM 7 — RADIOLOGY AND MEDICAL IMAGING | IMX-005 | X-ray workflows |
| SYSTEM 7 — RADIOLOGY AND MEDICAL IMAGING | IMX-006 | CT workflows |
| SYSTEM 7 — RADIOLOGY AND MEDICAL IMAGING | IMX-007 | MRI workflows |
| SYSTEM 7 — RADIOLOGY AND MEDICAL IMAGING | IMX-008 | Ultrasound workflows |
| SYSTEM 7 — RADIOLOGY AND MEDICAL IMAGING | IMX-009 | Prior-study comparison |
| SYSTEM 7 — RADIOLOGY AND MEDICAL IMAGING | MM-006 | Structured radiology reports |
| SYSTEM 7 — RADIOLOGY AND MEDICAL IMAGING | IMX-010 | Radiology findings review |
| SYSTEM 7 — RADIOLOGY AND MEDICAL IMAGING | IMX-011 | Clinical context integration |
| SYSTEM 7 — RADIOLOGY AND MEDICAL IMAGING | IMX-012 | Critical finding communication |
| SYSTEM 7 — RADIOLOGY AND MEDICAL IMAGING | IMX-013 | PACS integration adapters |
| SYSTEM 7 — RADIOLOGY AND MEDICAL IMAGING | IMX-014 | Imaging audit logs |
| SYSTEM 7 — RADIOLOGY AND MEDICAL IMAGING | SAFE-010 | Support validated imaging models through modular integrations. |
| SYSTEM 7 — RADIOLOGY AND MEDICAL IMAGING | SAFE-011 | Do not generate unsupported diagnostic image findings from text-only models. |
| SYSTEM 8 — VOICE AND CLINICAL DOCUMENTATION | VOX-001 | Speech-to-text |
| SYSTEM 8 — VOICE AND CLINICAL DOCUMENTATION | VOX-002 | Clinical dictation |
| SYSTEM 8 — VOICE AND CLINICAL DOCUMENTATION | VOX-003 | Encounter transcription |
| SYSTEM 8 — VOICE AND CLINICAL DOCUMENTATION | VOX-004 | Voice-driven clinical interaction |
| SYSTEM 8 — VOICE AND CLINICAL DOCUMENTATION | VOX-005 | Text-to-speech where supported |
| SYSTEM 8 — VOICE AND CLINICAL DOCUMENTATION | VOX-006 | Optional wake-phrase architecture |
| SYSTEM 8 — VOICE AND CLINICAL DOCUMENTATION | TEST-010 | Clinical note drafting |
| SYSTEM 8 — VOICE AND CLINICAL DOCUMENTATION | VOX-007 | Structured documentation |
| SYSTEM 8 — VOICE AND CLINICAL DOCUMENTATION | VOX-008 | Report generation |
| SYSTEM 8 — VOICE AND CLINICAL DOCUMENTATION | VOX-009 | Physician editing and approval |
| SYSTEM 8 — VOICE AND CLINICAL DOCUMENTATION | VOX-010 | Documentation export |
| SYSTEM 8 — VOICE AND CLINICAL DOCUMENTATION | SAFE-012 | Use modular providers and testable interfaces. |
| SYSTEM 9 — EHR AND INTEROPERABILITY | FHX-001 | SMART-on-FHIR architecture |
| SYSTEM 9 — EHR AND INTEROPERABILITY | FHX-002 | OAuth authorization flows |
| SYSTEM 9 — EHR AND INTEROPERABILITY | FHX-003 | FHIR patient resources |
| SYSTEM 9 — EHR AND INTEROPERABILITY | FHX-004 | Observations |
| SYSTEM 9 — EHR AND INTEROPERABILITY | FHX-005 | Conditions |
| SYSTEM 9 — EHR AND INTEROPERABILITY | FHX-006 | Medication records |
| SYSTEM 9 — EHR AND INTEROPERABILITY | FHX-007 | Allergies |
| SYSTEM 9 — EHR AND INTEROPERABILITY | FHX-008 | Diagnostic reports |
| SYSTEM 9 — EHR AND INTEROPERABILITY | FHX-009 | Imaging study references |
| SYSTEM 9 — EHR AND INTEROPERABILITY | FHX-010 | Encounters |
| SYSTEM 9 — EHR AND INTEROPERABILITY | FHX-011 | Clinical document resources |
| SYSTEM 9 — EHR AND INTEROPERABILITY | FHX-012 | Terminology mapping |
| SYSTEM 9 — EHR AND INTEROPERABILITY | FHX-013 | SNOMED CT support where licensed |
| SYSTEM 9 — EHR AND INTEROPERABILITY | FHX-014 | ICD-10-CM coding assistance |
| SYSTEM 9 — EHR AND INTEROPERABILITY | FHX-015 | CPT assistance where licensed |
| SYSTEM 9 — EHR AND INTEROPERABILITY | FHX-016 | EHR integration adapters |
| SYSTEM 9 — EHR AND INTEROPERABILITY | FHX-017 | PACS interoperability |
| SYSTEM 9 — EHR AND INTEROPERABILITY | FHX-018 | Data provenance |
| SYSTEM 9 — EHR AND INTEROPERABILITY | SAFE-013 | Use sandbox FHIR servers and synthetic data for integration testing. |
| SYSTEM 10 — DATABASE AND SECURITY | DBX-001 | PostgreSQL support |
| SYSTEM 10 — DATABASE AND SECURITY | DBX-002 | Database migrations |
| SYSTEM 10 — DATABASE AND SECURITY | DBX-003 | Role-based access control |
| SYSTEM 10 — DATABASE AND SECURITY | DBX-004 | Tenant isolation |
| SYSTEM 10 — DATABASE AND SECURITY | DBX-005 | Patient-level authorization |
| SYSTEM 10 — DATABASE AND SECURITY | DBX-006 | Secure sessions |
| SYSTEM 10 — DATABASE AND SECURITY | DBX-007 | Password hashing |
| SYSTEM 10 — DATABASE AND SECURITY | DBX-008 | Account recovery |
| SYSTEM 10 — DATABASE AND SECURITY | DBX-009 | Audit logging |
| SYSTEM 10 — DATABASE AND SECURITY | DBX-010 | Encryption |
| SYSTEM 10 — DATABASE AND SECURITY | DBX-011 | Secrets management |
| SYSTEM 10 — DATABASE AND SECURITY | DBX-012 | Input validation |
| SYSTEM 10 — DATABASE AND SECURITY | DBX-013 | Rate limiting |
| SYSTEM 10 — DATABASE AND SECURITY | DBX-014 | Secure file handling |
| SYSTEM 10 — DATABASE AND SECURITY | DBX-015 | Backup and recovery procedures |
| SYSTEM 10 — DATABASE AND SECURITY | DBX-016 | Health checks |
| SYSTEM 10 — DATABASE AND SECURITY | DBX-017 | Readiness checks |
| SYSTEM 10 — DATABASE AND SECURITY | DBX-018 | Structured monitoring |
| SYSTEM 10 — DATABASE AND SECURITY | DBX-019 | Error reporting |
| SYSTEM 10 — DATABASE AND SECURITY | DBX-020 | Security regression tests |
| SYSTEM 10 — DATABASE AND SECURITY | SAFE-014 | Do not weaken security to make a feature work. |
| SYSTEM 11 — ADMINISTRATION AND BILLING | ADX-001 | Organization management |
| SYSTEM 11 — ADMINISTRATION AND BILLING | ADX-002 | User invitations |
| SYSTEM 11 — ADMINISTRATION AND BILLING | ADX-003 | Role assignment |
| SYSTEM 11 — ADMINISTRATION AND BILLING | ADX-004 | Account status controls |
| SYSTEM 11 — ADMINISTRATION AND BILLING | ADX-005 | Usage tracking |
| SYSTEM 11 — ADMINISTRATION AND BILLING | ADX-006 | Administrative audit logs |
| SYSTEM 11 — ADMINISTRATION AND BILLING | ADX-007 | Subscription management |
| SYSTEM 11 — ADMINISTRATION AND BILLING | ADX-008 | Stripe test-mode billing |
| SYSTEM 11 — ADMINISTRATION AND BILLING | ADX-009 | Billing history |
| SYSTEM 11 — ADMINISTRATION AND BILLING | ADX-010 | Organization settings |
| SYSTEM 11 — ADMINISTRATION AND BILLING | ADX-011 | Feature access management |
| SYSTEM 12 — CLINICAL SAFETY | SFX-001 | Emergency red-flag identification |
| SYSTEM 12 — CLINICAL SAFETY | SFX-002 | Evidence-supported safety checks |
| SYSTEM 12 — CLINICAL SAFETY | SFX-003 | Medication interaction validation |
| SYSTEM 12 — CLINICAL SAFETY | SFX-004 | Allergy and contraindication detection |
| SYSTEM 12 — CLINICAL SAFETY | SFX-005 | Appropriate uncertainty reporting |
| SYSTEM 12 — CLINICAL SAFETY | SFX-006 | Abstention when evidence is insufficient |
| SYSTEM 12 — CLINICAL SAFETY | SFX-007 | Clinician confirmation for consequential actions |
| SYSTEM 12 — CLINICAL SAFETY | SFX-008 | Clinical review workflows |
| SYSTEM 12 — CLINICAL SAFETY | SFX-009 | Traceable clinical outputs |
| SYSTEM 12 — CLINICAL SAFETY | SFX-010 | Source verification |
| SYSTEM 12 — CLINICAL SAFETY | SFX-011 | Model failure logging |
| SYSTEM 12 — CLINICAL SAFETY | SFX-012 | Safe handling of incomplete or contradictory data |
| SYSTEM 13 — CONTINUOUS IMPROVEMENT | COX-001 | Clinician feedback collection |
| SYSTEM 13 — CONTINUOUS IMPROVEMENT | DBX-019 | Error reporting |
| SYSTEM 13 — CONTINUOUS IMPROVEMENT | COX-002 | Correction review |
| SYSTEM 13 — CONTINUOUS IMPROVEMENT | COX-003 | Evidence gap tracking |
| SYSTEM 13 — CONTINUOUS IMPROVEMENT | COX-004 | Evaluation datasets |
| SYSTEM 13 — CONTINUOUS IMPROVEMENT | COX-005 | Model comparison tools |
| SYSTEM 13 — CONTINUOUS IMPROVEMENT | KNOW-025 | Retrieval evaluation |
| SYSTEM 13 — CONTINUOUS IMPROVEMENT | COX-006 | Clinical benchmark testing |
| SYSTEM 13 — CONTINUOUS IMPROVEMENT | COX-007 | Citation accuracy measurement |
| SYSTEM 13 — CONTINUOUS IMPROVEMENT | COX-008 | Diagnostic accuracy measurement |
| SYSTEM 13 — CONTINUOUS IMPROVEMENT | COX-009 | Safety regression testing |
| SYSTEM 13 — CONTINUOUS IMPROVEMENT | COX-010 | Performance dashboards |
| SYSTEM 13 — CONTINUOUS IMPROVEMENT | COX-011 | Model and prompt version tracking |
| SYSTEM 13 — CONTINUOUS IMPROVEMENT | COX-012 | Controlled release procedures |
| SYSTEM 13 — CONTINUOUS IMPROVEMENT | SAFE-015 | Do not automatically train or deploy models based on unverified clinical feedback. |
| SYSTEM 14 — CLINICAL RESEARCH AND VALIDATION | SAFE-016 | Prepare reproducible evaluation workflows using appropriately authorized datasets and expert-reviewed cases. |
| SYSTEM 14 — CLINICAL RESEARCH AND VALIDATION | RSX-001 | Diagnostic accuracy |
| SYSTEM 14 — CLINICAL RESEARCH AND VALIDATION | RSX-002 | Differential diagnosis quality |
| SYSTEM 14 — CLINICAL RESEARCH AND VALIDATION | RSX-003 | Dangerous-condition detection |
| SYSTEM 14 — CLINICAL RESEARCH AND VALIDATION | RSX-004 | Medical factuality |
| SYSTEM 14 — CLINICAL RESEARCH AND VALIDATION | RSX-005 | Evidence citation accuracy |
| SYSTEM 14 — CLINICAL RESEARCH AND VALIDATION | ER-004 | Guideline adherence |
| SYSTEM 14 — CLINICAL RESEARCH AND VALIDATION | RSX-006 | Imaging performance |
| SYSTEM 14 — CLINICAL RESEARCH AND VALIDATION | ADV-006 | Medication safety |
| SYSTEM 14 — CLINICAL RESEARCH AND VALIDATION | ER-006 | Clinical note quality |
| SYSTEM 14 — CLINICAL RESEARCH AND VALIDATION | RSX-007 | Speech transcription accuracy |
| SYSTEM 14 — CLINICAL RESEARCH AND VALIDATION | RSX-008 | Clinician usability |
| SYSTEM 14 — CLINICAL RESEARCH AND VALIDATION | RSX-009 | Bias and subgroup performance |
| SYSTEM 14 — CLINICAL RESEARCH AND VALIDATION | RSX-010 | Model uncertainty |
| SYSTEM 14 — CLINICAL RESEARCH AND VALIDATION | SAFE-017 | Do not claim independent clinical validation unless it has occurred. |
| SYSTEM 15 — DEPLOYMENT AND OPERATIONS | OPX-001 | Reproducible development environment |
| SYSTEM 15 — DEPLOYMENT AND OPERATIONS | TEST-011 | Automated tests |
| SYSTEM 15 — DEPLOYMENT AND OPERATIONS | OPX-002 | GitHub Actions CI |
| SYSTEM 15 — DEPLOYMENT AND OPERATIONS | OPX-003 | Development and staging configurations |
| SYSTEM 15 — DEPLOYMENT AND OPERATIONS | OPX-004 | Render-compatible deployment |
| SYSTEM 15 — DEPLOYMENT AND OPERATIONS | OPX-005 | Environment-variable validation |
| SYSTEM 15 — DEPLOYMENT AND OPERATIONS | OPX-006 | Database migration procedures |
| SYSTEM 15 — DEPLOYMENT AND OPERATIONS | OPX-007 | Logging and monitoring |
| SYSTEM 15 — DEPLOYMENT AND OPERATIONS | OPX-008 | Dependency vulnerability checks |
| SYSTEM 15 — DEPLOYMENT AND OPERATIONS | OPX-009 | Startup and readiness checks |
| SYSTEM 15 — DEPLOYMENT AND OPERATIONS | OPX-010 | Deployment rollback procedures |
| SYSTEM 15 — DEPLOYMENT AND OPERATIONS | OPX-011 | Operational documentation |
| SYSTEM 15 — DEPLOYMENT AND OPERATIONS | SAFE-018 | Maintain a reproducible, deployable application throughout development. |
| REQUIRED IMPLEMENTATION PROCESS | PROCESS-001 | Write the necessary backend code. |
| REQUIRED IMPLEMENTATION PROCESS | PROCESS-002 | Write the necessary frontend code. |
| REQUIRED IMPLEMENTATION PROCESS | PROCESS-003 | Add database models and migrations where needed. |
| REQUIRED IMPLEMENTATION PROCESS | PROCESS-004 | Implement authorization and security. |
| REQUIRED IMPLEMENTATION PROCESS | PROCESS-005 | Add automated tests. |
| REQUIRED IMPLEMENTATION PROCESS | PROCESS-006 | Execute tests. |
| REQUIRED IMPLEMENTATION PROCESS | PROCESS-007 | Fix failures. |
| REQUIRED IMPLEMENTATION PROCESS | PROCESS-008 | Verify integration with existing functionality. |
| REQUIRED IMPLEMENTATION PROCESS | PROCESS-009 | Update the requirements matrix. |
| REQUIRED IMPLEMENTATION PROCESS | PROCESS-010 | Document remaining limitations. |
| REQUIRED IMPLEMENTATION PROCESS | SAFE-019 | Do not substitute documentation for implementation. |
| REQUIRED END-TO-END TEST SCENARIOS | FLOW-001 | A. A patient signs in, views records, and sends a message to a physician. |
| REQUIRED END-TO-END TEST SCENARIOS | FLOW-002 | B. A physician signs in, receives the message, reviews the synthetic patient chart, drafts a response, approves it, and sends it internally. |
| REQUIRED END-TO-END TEST SCENARIOS | FLOW-003 | C. A physician enters a complex synthetic clinical case and receives a structured differential diagnosis with supporting and contradicting evidence. |
| REQUIRED END-TO-END TEST SCENARIOS | FLOW-004 | D. Raphael retrieves verifiable medical evidence and provides citations that resolve to the actual sources. |
| REQUIRED END-TO-END TEST SCENARIOS | FLOW-005 | E. A physician creates and edits an AI-assisted clinical note. |
| REQUIRED END-TO-END TEST SCENARIOS | FLOW-006 | F. A user uploads a valid test DICOM study and views the image and metadata. |
| REQUIRED END-TO-END TEST SCENARIOS | FLOW-007 | G. A development FHIR integration retrieves synthetic patient resources. |
| REQUIRED END-TO-END TEST SCENARIOS | FLOW-008 | H. An administrator manages development users and permissions. |
| REQUIRED END-TO-END TEST SCENARIOS | FLOW-009 | I. Unauthorized users cannot access another patient's records or messages. |
| REQUIRED END-TO-END TEST SCENARIOS | FLOW-010 | J. The application restarts without losing persisted test data. |
| REQUIRED END-TO-END TEST SCENARIOS | FLOW-011 | K. The application reports integration failures without fabricating successful results. |
| RELEASE ACCEPTANCE CRITERIA | ACCEPT-001 | The application starts successfully. |
| RELEASE ACCEPTANCE CRITERIA | ACCEPT-002 | Critical backend tests pass. |
| RELEASE ACCEPTANCE CRITERIA | ACCEPT-003 | Frontend tests pass. |
| RELEASE ACCEPTANCE CRITERIA | ACCEPT-004 | Core end-to-end workflows pass. |
| RELEASE ACCEPTANCE CRITERIA | ACCEPT-005 | Development authentication works. |
| RELEASE ACCEPTANCE CRITERIA | ACCEPT-006 | Patient and clinician portals function. |
| RELEASE ACCEPTANCE CRITERIA | ACCEPT-007 | Internal secure messaging works. |
| RELEASE ACCEPTANCE CRITERIA | ACCEPT-008 | AI inference is tested. |
| RELEASE ACCEPTANCE CRITERIA | ACCEPT-009 | Evidence retrieval and citations are verified. |
| RELEASE ACCEPTANCE CRITERIA | ACCEPT-010 | Synthetic patient records persist. |
| RELEASE ACCEPTANCE CRITERIA | ACCEPT-011 | Authorization and tenant-isolation tests pass. |
| RELEASE ACCEPTANCE CRITERIA | ACCEPT-012 | Known limitations are clearly displayed. |
| RELEASE ACCEPTANCE CRITERIA | ACCEPT-013 | No production secrets are exposed. |
| RELEASE ACCEPTANCE CRITERIA | ACCEPT-014 | No real patient data is required. |
| RELEASE ACCEPTANCE CRITERIA | SAFE-020 | For optional external integrations, provide working adapters, tests, configuration documentation, and explicit status reporting. |
| RELEASE ACCEPTANCE CRITERIA | SAFE-021 | Do not describe an integration as operational until it has been tested against the relevant service. |
| FINAL DELIVERY | DELIVER-001 | AGENTS.md |
| FINAL DELIVERY | DELIVER-002 | docs/RAPHAEL_MASTER_SPEC.md |
| FINAL DELIVERY | DELIVER-003 | docs/REQUIREMENTS_MATRIX.md |
| FINAL DELIVERY | DELIVER-004 | docs/FEATURE_INVENTORY.md |
| FINAL DELIVERY | DELIVER-005 | docs/ROADMAP.md |
| FINAL DELIVERY | DELIVER-006 | docs/RELEASE_READINESS.md |
| FINAL DELIVERY | DELIVER-007 | docs/DEPLOYMENT.md |
| FINAL DELIVERY | DELIVER-008 | docs/TEST_RESULTS.md |
| FINAL DELIVERY | DELIVER-009 | Actual working application status. |
| FINAL DELIVERY | DELIVER-010 | Implemented features and file locations. |
| FINAL DELIVERY | DELIVER-011 | Test results. |
| FINAL DELIVERY | DELIVER-012 | Remaining external dependencies. |
| FINAL DELIVERY | DELIVER-013 | Known limitations and security risks. |
| FINAL DELIVERY | DELIVER-014 | A deployable development build. |
| FINAL DELIVERY | DELIVER-015 | Reviewable GitHub changes. |
| FINAL DELIVERY | DELIVER-016 | The exact continuation task for any work that could not fit into this execution. |
| FINAL DELIVERY | SAFE-022 | Do not stop at a roadmap. |
| FINAL DELIVERY | SAFE-023 | Do not mark incomplete systems as finished. |
| FINAL DELIVERY | SAFE-024 | Do not merge, deploy to production, activate live billing, use real patient data, or enable real clinical messaging without explicit authorization. |


## Integrated release directive (latest owner input)

# RAPHAEL AI — INTEGRATED DEVELOPMENT RELEASE BUILD

You are the principal full-stack engineer, clinical informatics architect, medical AI engineer, security engineer, and quality assurance lead for Raphael AI.

Repository: AlGeniusAlien/Raphael-AI

Continue from the latest existing Raphael development work. Do not start over, discard working features, or duplicate previous implementations.

## PRIMARY MISSION

Deliver the most complete, functional, professionally designed, end-to-end Raphael development application achievable in this execution.

The current web preview loads a basic "Raphael development portal" sign-in screen. We must advance beyond this scaffolding into a usable clinical AI application.

The project already contains a master specification and requirements matrix with approximately 278 documented requirements. Preserve every requirement and its identifier.

THIS IS AN IMPLEMENTATION TASK, NOT A DOCUMENTATION OR PLANNING TASK.

Prioritize working code, integrated functionality, browser verification, and a reviewable release candidate.

## 1. REPOSITORY RECONCILIATION

First:
- Inspect the current working branch and Git status.
- Inspect existing pull requests and uncommitted changes.
- Read AGENTS.md, the master specification, requirements matrix, feature inventory, and release readiness documentation.
- Identify the latest tested code containing the SQLite repair, portal features, and development runtime.
- Preserve all previously implemented functionality.
- Avoid overwriting changes from other branches.
- Do not merge protected branches automatically.

Identify the current startup procedure, database architecture, frontend structure, and authentication implementation.

## 2. COMPLETE DEVELOPMENT AUTHENTICATION

Create a safe, repeatable development-only provisioning procedure for:

- One synthetic healthcare organization
- One synthetic administrator
- One synthetic physician
- One synthetic patient

Use unique test accounts and strong authentication.

Implement:
- Login and logout
- Secure password handling
- Session expiration
- Role-based authorization
- Tenant isolation
- Patient-specific record permissions
- Unauthorized-access rejection
- Authentication error handling

Do not hardcode passwords into source code.

Provide a secure procedure for setting temporary test credentials.

Development accounts must never automatically appear in production.

## 3. COMPLETE THE PATIENT PORTAL

Build a polished, responsive patient experience with working functionality for:

- Dashboard
- Medical profile
- Medical history
- Laboratory results
- Medications
- Allergies
- Imaging reports
- Clinical documents
- Visit summaries
- Care plans
- Appointments and follow-up information
- Secure internal messaging
- Notifications
- Document uploads
- Patient-friendly clinical explanations where appropriate

Use realistic synthetic medical records.

Every functional interface element must connect to a working backend API or be clearly identified as unavailable.

Do not create decorative buttons that do nothing.

## 4. COMPLETE THE PHYSICIAN PORTAL

Build a professional clinical workspace supporting:

- Physician dashboard
- Patient worklist
- Patient search
- Chart review
- Longitudinal patient timeline
- Clinical encounters
- Laboratory and medication review
- Clinical AI assistant
- Differential diagnosis review
- Medical evidence review
- Clinical note drafting
- Clinical note editing
- Clinician approval
- Patient messaging
- Care-team communication
- Imaging report review
- Follow-up tracking

Implement real data flow between frontend, API, and database.

## 5. COMPLETE INTERNAL MESSAGING

Implement and test:

- Patient-to-physician messages
- Physician-to-patient messages
- Clinician-to-clinician messages where supported
- Conversation threads
- Inbox and sent messages
- Message timestamps
- Read status
- Authorized attachments
- Recipient validation
- Audit records
- Clinician-reviewed AI-assisted draft messages
- Explicit approval before sending AI-assisted clinical advice

All messaging must remain internal to synthetic development accounts.

Do not send real emails, SMS messages, or EHR messages.

## 6. INTEGRATE RAPHAEL'S MEDICAL AI

Inspect and improve the existing medical intelligence implementation.

Support:
- Clinical question answering
- Structured clinical assessments
- Ranked differential diagnoses
- Supporting and contradictory findings
- Missing clinical information
- Dangerous alternative diagnoses
- Diagnostic evaluation suggestions
- Treatment considerations
- Clinical uncertainty
- Evidence-grounded explanations
- Clear separation of evidence and inference

Use the existing OpenAI integration when dedicated development credentials are configured.

When credentials are unavailable, implement and run deterministic integration tests with mocks and clearly document the unverified live integration.

Never fabricate medical evidence or cite nonexistent sources.

## 7. COMPLETE MEDICAL EVIDENCE INTEGRATION

Inspect the existing RAG, Qdrant, evidence ingestion, and retrieval modules.

Implement or repair:
- PubMed integration
- Source metadata
- Citation verification
- Evidence retrieval
- Evidence relevance ranking
- Guideline and publication dates
- Qdrant development collection support
- Safe document ingestion
- Retrieval error handling
- Source freshness indicators
- Evidence provenance

Use legally accessible sources.

Do not claim comprehensive medical knowledge coverage without evidence.

Verify live retrieval when permitted network access and development credentials are available.

## 8. CLINICAL SAFETY

Implement:
- Emergency red-flag handling
- Uncertainty and abstention
- Clinician approval for consequential actions
- Citation verification
- Clear medical limitations
- Prevention of unauthorized patient-data access
- Safe handling of incomplete medical histories
- Medication and allergy information visibility
- Clinical output auditability

Do not represent heuristic medication checks or unvalidated AI output as clinically validated decision support.

## 9. PROFESSIONAL VISUAL DESIGN

Replace the plain development sign-in page with a polished medical software interface.

Use:
- Professional typography
- Consistent spacing
- Responsive navigation
- Accessible contrast
- Clean dashboards
- Patient summary cards
- Clear clinical information hierarchy
- Well-designed forms
- Loading states
- Empty states
- Error messages
- Confirmation states
- Mobile-friendly layouts

Raphael should look like a credible, modern clinical application rather than a raw HTML prototype.

Preserve accessibility and functional usability over visual decoration.

## 10. PREPARE IMAGING, VOICE, AND FHIR WORKFLOWS

Integrate existing implementations where possible.

At minimum, provide tested development workflows for:

- DICOM upload, parsing, and metadata viewing
- Medical imaging study records
- Voice transcription adapter architecture
- Clinical note dictation integration
- FHIR sandbox patient retrieval
- Clinical terminology mapping

If live services or specialized models are unavailable, implement clearly separated development adapters and test them.

Do not claim validated image diagnosis, production EHR connectivity, or live voice functionality unless actually verified.

## 11. DATABASE AND APPLICATION RELIABILITY

Verify:
- Safe database initialization
- Existing SQLite startup fix
- Repeatable migrations
- Persistence after restart
- Authentication and authorization
- Tenant isolation
- Patient-level access restrictions
- Secure document storage
- Structured error handling
- Health checks
- Readiness checks
- Environment configuration
- Dependency compatibility

Add regression tests for every corrected defect.

## 12. AUTOMATED AND BROWSER TESTING

Run:
- Python unit tests
- Backend integration tests
- Frontend tests
- Browser end-to-end tests
- Authentication tests
- Tenant isolation tests
- Patient authorization tests
- Internal messaging tests
- Clinical note tests
- Medical assessment schema tests
- Citation verification tests
- Persistence and restart tests
- Relevant security regression tests

Perform an actual browser walkthrough:

1. Open Raphael.
2. Sign in as a synthetic patient.
3. View the patient dashboard and chart.
4. Send a message to a physician.
5. Sign out.
6. Sign in as the physician.
7. Review the patient message and chart.
8. Send an approved internal reply.
9. Generate and edit a clinical note.
10. Open the medical AI assistant.
11. Test a synthetic clinical case.
12. Verify available evidence citations.
13. Sign in as an administrator.
14. Verify account and permission controls.
15. Verify unauthorized access is rejected.

Capture actual test results. Do not claim success for tests that were not executed.

## 13. DEVELOPMENT DEPLOYMENT

Prepare a reproducible deployment configuration for Raphael's development or staging environment.

Include:
- Installation instructions
- Startup commands
- Environment variable requirements
- Database setup
- Test-account provisioning
- Health and readiness endpoints
- Dependency checks
- Security settings
- Deployment verification
- Rollback instructions

Do not deploy to production or use real patient information.

## 14. PRESERVE THE COMPLETE RAPHAEL VISION

The permanent requirements matrix must continue tracking all previously documented capabilities, including:

- Full medical knowledge integration
- Advanced clinical reasoning
- Radiology and multimodal imaging
- Patient and physician portals
- Secure doctor messaging
- Voice interaction
- Clinical documentation
- Medication safety
- EHR and SMART-on-FHIR
- PACS
- Clinical terminology
- Organization administration
- Billing
- Clinical research
- Benchmarking
- Continuous improvement
- Privacy, security, and regulatory readiness

Do not delete or silently downgrade existing requirements.

## 15. EXECUTION AND DELIVERY

Spend the available execution time implementing and testing, not repeatedly rewriting plans.

Work in dependency-aware order.

Prioritize:
1. Secure synthetic authentication
2. Functional patient and physician portals
3. Internal messaging
4. Clinical AI integration
5. Evidence retrieval
6. Professional frontend
7. End-to-end verification
8. Deployment readiness

Continue implementing until the execution limit is reached or the defined release acceptance criteria are satisfied.

Before stopping:
- Run relevant tests.
- Fix reproducible failures.
- Save changes to a development branch.
- Prepare or update a pull request.
- Update the requirements matrix.
- Document actual working functionality.
- Document blocked external integrations.
- Provide exact steps to run Raphael.

Do not merge or deploy automatically.

## REQUIRED FINAL RESPONSE

Report:

1. Features implemented and verified
2. Files changed
3. Tests passed and failed
4. Browser preview status
5. Synthetic test-account setup instructions
6. OpenAI and Qdrant integration status
7. Patient and physician portal status
8. Remaining security issues
9. Development deployment readiness
10. Exact blockers preventing full functionality

Do not respond with another roadmap alone.

DELIVER WORKING SOFTWARE, VERIFY IT, AND PRESERVE THE COMPLETE RAPHAEL REQUIREMENTS.