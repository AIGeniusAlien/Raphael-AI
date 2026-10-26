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

## Definitive full-product owner directive

# RAPHAEL AI — DEFINITIVE FULL-PRODUCT ENGINEERING DIRECTIVE

## ROLE AND AUTHORITY

Act as the principal software architect, senior full-stack engineering team, medical AI engineer, clinical informatics architect, radiology AI engineer, security engineer, infrastructure engineer, product designer, and quality assurance lead for Raphael AI.

Repository: AlGeniusAlien/Raphael-AI

Existing development work: Draft Pull Request #1 and its associated development branch.

Your responsibility is to implement, integrate, test, stabilize, and prepare the most complete Raphael AI platform possible.

THIS IS NOT A REQUEST FOR ANOTHER ROADMAP.

THIS IS AN EXECUTION DIRECTIVE TO BUILD WORKING SOFTWARE.

The owner wants the complete Raphael platform, not a limited chatbot, a mock clinical application, or a collection of disconnected prototypes.

## THE RAPHAEL VISION

Raphael is an advanced, multimodal, evidence-grounded clinical intelligence platform intended to augment physicians, radiologists, healthcare organizations, patients, and medical researchers.

Raphael must bring together medical knowledge, diagnostic reasoning, patient records, imaging, voice, documentation, secure communications, and clinical workflow assistance.

The system must help clinicians identify important possibilities, detect inconsistencies, reduce documentation burdens, and access verifiable medical evidence.

Raphael must never substitute unsupported AI output for authorized clinical judgment.

The immediate engineering objective is a complete, integrated synthetic-data development product.

Clinical deployment requires additional independent validation, regulatory assessment, security review, and appropriate authorization.

## PART 1 — PRESERVE EVERYTHING ALREADY BUILT

Before changing code:

1. Inspect the entire repository.
2. Inspect the current branch and Draft PR #1.
3. Read AGENTS.md and all project documentation.
4. Read RAPHAEL_MASTER_SPEC.md.
5. Read REQUIREMENTS_MATRIX.md.
6. Read FEATURE_INVENTORY.md.
7. Read ROADMAP.md and RELEASE_READINESS.md.
8. Inspect existing frontend, backend, desktop, extension, RAG, imaging, authentication, and clinical modules.
9. Identify every implemented feature and its tests.
10. Preserve all previously documented requirement IDs.

The project previously reconciled approximately 278 requirements.

Do not delete, downgrade, overwrite, or silently omit any requirement.

Where historical documents conflict, preserve the conflict for resolution instead of guessing.

Do not repeat completed engineering work.

Continue from the latest tested implementation.

## PART 2 — COMPLETE PRODUCT ARCHITECTURE

Build Raphael as a modular clinical AI platform containing:

- API backend
- Web frontend
- Patient portal
- Physician portal
- Radiology workspace
- Clinical administration
- Medical evidence engine
- Clinical reasoning engine
- Voice and activation engine
- Ambient documentation engine
- Medical imaging engine
- EHR interoperability engine
- Secure messaging engine
- Clinical safety engine
- Evaluation engine
- Enterprise administration
- Billing infrastructure
- Desktop integration
- PWA/mobile experience
- Monitoring and deployment infrastructure

Avoid unnecessary rewrites and unnecessary microservices.

Use clear internal boundaries so each subsystem can be tested and upgraded independently.

## PART 3 — CHATGPT-STYLE RAPHAEL INTERFACE

Build a sophisticated conversational medical AI interface.

Required functionality:

- Streaming AI responses
- Persistent conversation history
- Patient-specific conversation threads
- General medical research conversations
- Conversation titles and search
- Context-aware follow-up questions
- Clinical attachments
- PDF uploads
- Medical document uploads
- Image attachments
- DICOM study references
- Structured medical responses
- Clickable medical citations
- Evidence panels
- Stop-generation controls
- Response regeneration
- Error recovery
- Conversation export
- Responsive design
- Accessible controls
- Professional clinical interface

Implement a provider-independent LLM abstraction compatible with OpenAI-style APIs.

Support configurable model selection, model version tracking, request tracing, timeout handling, rate limits, and provider failures.

Use actual streaming APIs where available.

Never simulate streaming and present it as a live model response.

## PART 4 — RAPHAEL STANDARD AND HANDS-FREE MODES

Preserve the original dual-mode experience.

STANDARD MODE:
- Text conversations
- Optional push-to-talk
- Clinical file attachments
- Patient-context conversations
- Clinical documentation assistance

HANDS-FREE MODE:
- User-authorized microphone activation
- Configurable wake phrase such as "Hey Raphael"
- Voice activity detection
- Realtime speech recognition
- Natural spoken responses
- Interruptible speech
- Conversational turn-taking
- Listening and processing indicators
- Mute and pause controls
- Explicit stop controls
- Session timeout handling
- Device compatibility checks
- Safe microphone permission handling

Implement supported realtime voice APIs and appropriate audio transport.

Preserve privacy and user consent.

Do not enable undisclosed background recording or transmission.

## PART 5 — ACTIVE AND AMBIENT CLINICAL DOCUMENTATION

Build a complete clinical documentation assistant.

Required encounter workflow:

1. Clinician opens a patient encounter.
2. Raphael verifies the appropriate authorization and consent state.
3. Clinician starts an encounter recording.
4. Raphael transcribes the authorized conversation.
5. Raphael performs speaker identification where supported.
6. Raphael extracts clinical information.
7. Raphael maintains an evolving draft note.
8. Raphael updates the note as new information arrives.
9. Raphael identifies ambiguities and missing information.
10. Raphael correlates the note with authorized patient records.
11. Raphael retrieves relevant medical evidence when appropriate.
12. Raphael checks the draft for unsupported statements.
13. Clinician reviews and edits the draft.
14. Clinician approves and signs the final document.
15. Raphael records approval and provenance.
16. Approved documentation can be exported or transmitted to authorized sandbox systems.

Support:
- SOAP notes
- History and physical
- Progress notes
- Consultation notes
- Discharge summaries
- Radiology reports
- Problem lists
- Medication reconciliation drafts
- Encounter summaries
- Follow-up instructions
- Structured clinical templates

Preserve physician edits and document version history.

Never automatically sign or finalize a medical record.

## PART 6 — PATIENT PORTAL

Implement a complete patient-facing experience.

Required features:

- Patient registration architecture
- Secure login
- Identity verification architecture
- Patient dashboard
- Medical profile
- Medical history
- Laboratory results
- Laboratory trends
- Medication lists
- Allergies
- Imaging results
- Clinical notes
- Visit summaries
- Medical documents
- Care plans
- Appointments
- Follow-up reminders
- Secure physician messaging
- Message attachments
- Notifications
- Patient-friendly explanations
- Consent management
- Authorized caregiver access
- Account preferences
- Mobile-responsive experience

Use synthetic accounts and synthetic data for development.

Patient-facing clinical advice must remain within the authorized clinical review workflow.

## PART 7 — PHYSICIAN AND CLINICAL PORTAL

Implement:

- Physician dashboard
- Patient worklists
- Patient search
- Patient chart review
- Longitudinal patient timelines
- Clinical encounter management
- Clinical AI conversations
- Differential diagnosis support
- Medical evidence review
- Laboratory interpretation
- Medication review
- Imaging review
- Clinical documentation
- Note editing
- Clinician sign-off
- Clinical coding assistance
- Draft orders
- Clinical pathway assistance
- Patient messaging
- Care-team messaging
- Follow-up management
- Clinical alerts
- Audit history

The interface must support efficient clinical workflows rather than forcing clinicians to repeatedly enter information already available in authorized records.

## PART 8 — SECURE PATIENT AND CLINICIAN MESSAGING

Implement:

- Patient-to-physician messages
- Physician-to-patient messages
- Clinician-to-clinician messages
- Conversation threads
- Inbox and sent folders
- Attachments
- Message status
- Read status
- Recipient authorization
- Message routing
- Notification architecture
- Follow-up tracking
- Urgent-message policies
- AI-assisted message drafting
- Explicit clinician approval
- Communication audit logs

Separate internal portal messaging from external delivery.

No external clinical message may be sent without appropriate authorization and delivery confirmation.

## PART 9 — ADVANCED MEDICAL KNOWLEDGE ENGINE

Build a comprehensive evidence-grounded retrieval system.

Support legally accessible and appropriately licensed sources, including:

- PubMed
- PubMed Central
- ClinicalTrials.gov
- NIH
- FDA
- CDC
- WHO
- NICE
- USPSTF
- Specialty-society guidelines
- Peer-reviewed research
- Systematic reviews
- Meta-analyses
- Authorized institutional protocols
- Licensed medical references
- Drug labeling and safety communications
- Medical terminology resources

Implement:

- Source connectors
- Automated ingestion
- Source licensing checks
- Metadata extraction
- Evidence classification
- Document versioning
- Publication dates
- Update dates
- Deduplication
- Chunking
- Embeddings
- Qdrant indexing
- Hybrid search
- Reranking
- Citation verification
- Source provenance
- Guideline freshness
- Contradiction detection
- Evidence coverage reports
- Retrieval benchmarks
- Controlled evidence updates

Do not claim all medical knowledge has been indexed.

Never fabricate medical references.

## PART 10 — CLINICAL REASONING ENGINE

Implement:

- Clinical case analysis
- Working diagnoses
- Ranked differential diagnoses
- Supporting evidence
- Contradictory evidence
- Dangerous alternatives
- Missing clinical information
- Recommended diagnostic evaluation
- Guideline-based treatment considerations
- Medication safety
- Laboratory interpretation
- Vital-sign interpretation
- Longitudinal trend analysis
- Cross-specialty clinical reasoning
- Clinical uncertainty
- Appropriate abstention
- Physician review and confirmation

Medical outputs must distinguish verified evidence from model-generated inference.

Do not present uncalibrated confidence scores as validated probabilities.

## PART 11 — SAFECORE CLINICAL SAFETY ARCHITECTURE

Implement Raphael's layered safety system.

Components:

1. Input and content validation
2. Clinical risk detection
3. Evidence retrieval and verification
4. Factuality assessment
5. Contradiction checks
6. Medication and allergy safety checks
7. Uncertainty calibration
8. Appropriate abstention
9. Human clinician approval
10. Audit and traceability
11. Model failure monitoring
12. Regression evaluation

Support high-risk clinical scenario testing.

Require authorization before consequential actions.

Never allow the language model alone to authorize clinical orders, prescriptions, or external patient communications.

## PART 12 — RADIOLOGY AND MEDICAL IMAGING

Radiology is a major Raphael specialization.

Preserve existing DICOM upload and viewing.

Implement:

- DICOM validation
- DICOM metadata processing
- DICOM image viewing
- Imaging study organization
- X-ray workflows
- CT workflows
- MRI workflows
- Ultrasound workflows
- Prior-study comparison
- Clinical history correlation
- Laboratory and imaging correlation
- Structured radiology reporting
- Pre-read assistance
- Post-read discrepancy review
- Missing-information detection
- Critical finding communication
- PACS adapters
- Imaging audit records
- Imaging evaluation datasets
- Validated medical image-model integrations

Do not claim diagnostic imaging accuracy without actual model validation.

## PART 13 — SMART-ON-FHIR AND EHR INTEROPERABILITY

Implement:

- SMART-on-FHIR OAuth
- Patient-context launch
- Standalone launch
- Authorization and token handling
- FHIR Patient
- Encounter
- Observation
- Condition
- AllergyIntolerance
- MedicationRequest
- MedicationStatement
- DiagnosticReport
- ImagingStudy
- DocumentReference
- Provenance
- Clinical document exchange
- Authorized draft-write workflows
- Sandbox integration testing
- Auditability

Support standards-based EHR connectivity without hardcoding a specific hospital vendor.

Do not enable production EHR write operations without authorization.

## PART 14 — CLINICAL CODING, ORDERS, AND TERMINOLOGY

Implement:

- ICD-10-CM coding assistance
- SNOMED CT terminology integration where licensed
- CPT assistance where licensed
- Clinical order sets
- Draft laboratory orders
- Draft imaging orders
- Draft medication orders
- Clinical pathways
- Medication reconciliation
- Contraindication checks
- Allergy checks
- Physician approval
- Action auditing

Do not automatically submit clinical orders.

## PART 15 — DESKTOP, MOBILE, AND EXTENSION EXPERIENCE

Inspect and preserve the existing Electron and browser extension implementations.

Support:

- Installable PWA
- Mobile-responsive web interface
- Electron desktop application
- Desktop notifications
- Microphone access controls
- Hands-free Raphael access
- Authorized patient-context workflows
- Browser extension architecture where useful
- Secure session handling across interfaces

Do not expose patient data through insecure browser extension permissions.

## PART 16 — ENTERPRISE AND COMMERCIAL PLATFORM

Implement:

- Organization accounts
- User administration
- Role assignment
- Subscription plans
- Usage tracking
- Billing management
- Stripe test-mode integration
- Organization settings
- Feature access controls
- Administrative reporting
- Audit history
- Support workflows
- Enterprise deployment configuration

Keep live billing disabled until authorized.

## PART 17 — SECURITY AND PRIVACY

Remediate all known dependency security advisories.

Implement and test:

- Secure authentication
- Password hashing
- Session security
- Role-based authorization
- Tenant isolation
- Patient-level permissions
- Consent management
- Secure messaging
- Encrypted communications
- Appropriate encryption at rest
- Secrets management
- Audit logging
- File upload validation
- DICOM security checks
- Rate limiting
- Database migrations
- Backup and recovery
- Incident response procedures
- Secure deployment configuration

Document HIPAA and BAA requirements.

Do not claim compliance merely because safeguards are implemented.

## PART 18 — CLINICAL EVALUATION AND RESEARCH

Raphael is also intended to support serious medical AI research.

Implement reproducible evaluation workflows for:

- Diagnostic accuracy
- Differential diagnosis quality
- Clinical safety
- Citation accuracy
- Guideline adherence
- Medication safety
- Radiology evaluation
- Speech recognition accuracy
- Clinical documentation quality
- Physician usability
- Bias and subgroup performance
- Model uncertainty
- Latency and throughput
- Ablation studies
- Model comparisons
- Clinician feedback

Support authorized research datasets and synthetic evaluation cases.

Prepare publication-ready technical documentation, figures, evaluation reports, and reproducible configurations.

Never invent research results or claim clinical validation without evidence.

Preserve modularity for future specialized applications, including astronaut-health research.

## PART 19 — PRODUCT DESIGN AND USER EXPERIENCE

Replace the basic development interface with a cohesive professional application.

Implement:

- Branded Raphael interface
- Modern navigation
- Clinical dashboard layouts
- ChatGPT-style conversational workspace
- Voice interface
- Active documentation workspace
- Patient portal
- Physician portal
- Radiology workspace
- Messaging interface
- Evidence viewer
- Administrative interface
- Responsive mobile design
- Accessibility
- Loading states
- Error states
- Empty states
- Confirmation dialogs
- Clear clinical warnings
- Reliable form validation

Do not leave decorative or nonfunctional controls.

## PART 20 — TESTING AND QUALITY ASSURANCE

Run:

- Python unit tests
- Frontend tests
- API integration tests
- Browser end-to-end tests
- Authentication tests
- Authorization tests
- Tenant-isolation tests
- Patient-data access tests
- Messaging tests
- Clinical note tests
- Evidence citation tests
- RAG evaluation
- DICOM tests
- FHIR sandbox tests
- Voice integration tests where supported
- Dependency security scans
- Database persistence tests
- Application restart tests
- Deployment smoke tests

Create tests for every corrected defect.

Preserve previous passing tests.

Do not claim success for unexecuted tests.

## PART 21 — COMPLETE SYNTHETIC CLINICAL WORKFLOW

Demonstrate an integrated scenario:

1. Start Raphael.
2. Sign in as a synthetic patient.
3. Review the medical chart.
4. Send a secure internal message.
5. Sign out.
6. Sign in as a physician.
7. Review the patient's chart and message.
8. Open a clinical encounter.
9. Activate Raphael's voice interface with permission.
10. Transcribe a synthetic encounter.
11. Generate and incrementally update a clinical note.
12. Review medical evidence.
13. Generate a structured differential diagnosis.
14. Identify uncertainty and missing information.
15. Review laboratory results and medications.
16. View a synthetic DICOM study.
17. Edit and approve the clinical note.
18. Draft and approve a response to the patient.
19. Verify the patient receives the internal reply.
20. Verify the audit trail.
21. Verify unauthorized users cannot access the encounter.
22. Restart the application and verify persistence.

Use real development integrations where securely configured.

Clearly label mocked and unverified steps.

## PART 22 — DEPLOYMENT AND DELIVERY

Deliver a reproducible, deployable development application.

Include:

- Complete source code
- Dependency lock or reproducible dependency configuration
- Database migrations
- Development configuration
- Synthetic account provisioning
- Startup scripts
- Render-compatible staging configuration
- Environment-variable validation
- Health checks
- Readiness checks
- CI/CD checks
- Test reports
- Rollback instructions
- Security findings
- Release notes

The owner should not need to manually rewrite source files.

Provide exact copy-and-paste deployment instructions only where necessary.

Do not deploy to production, merge protected branches, enable real patient data, or activate live billing without approval.

## PART 23 — EXECUTION STRATEGY

This is a large project. Work autonomously through dependency-aware implementation steps within the available execution context.

Do not repeatedly stop to request approval for ordinary code changes or test execution.

Prioritize complete end-to-end features rather than isolated scaffolding.

Execution priority:

1. Preserve and stabilize the existing codebase.
2. Remediate security and dependency issues.
3. Complete the clinical conversational interface.
4. Integrate live development AI and evidence retrieval.
5. Complete patient and physician workflows.
6. Complete realtime voice and activation.
7. Complete ambient documentation.
8. Integrate clinical safety.
9. Expand imaging and FHIR.
10. Complete administration and commercial infrastructure.
11. Run full end-to-end tests.
12. Prepare the deployable development release.

Where possible, implement independent workstreams in parallel without conflicting changes.

If a service, credential, dataset, license, or authorization is missing, complete the safe implementable portions and continue with other work.

Never invent successful integration results.

## PART 24 — PERMANENT REQUIREMENTS TRACKING

Update the existing requirements matrix without losing any historical entries.

Every requirement must have:

- Unique ID
- Description
- Priority
- Implementation status
- Source files
- Test coverage
- Acceptance criteria
- Dependencies
- Remaining work
- Release milestone

Use these statuses:

- Implemented and verified
- Implemented but unverified
- Partially implemented
- Mocked for development
- Blocked by external dependency
- Not implemented
- Awaiting clinical validation

Do not call a feature complete unless it meets its acceptance criteria.

## PART 25 — FINAL DELIVERABLE

Implement as much of the complete Raphael product as technically possible.

Before stopping:

1. Run relevant tests.
2. Fix reproducible defects.
3. Preserve all working features.
4. Save code changes to the development branch.
5. Update Draft PR #1 where appropriate.
6. Update the requirements matrix.
7. Provide an accessible browser preview where supported.
8. Document the exact development startup procedure.
9. Report live integration status.
10. Identify remaining blockers.
11. Record the exact next executable engineering task.

Do not merge or deploy to production without explicit approval.

The final response must clearly distinguish working, tested functionality from incomplete or simulated features.

Do not provide another roadmap instead of implementation.

DO NOT STOP AFTER WRITING DOCUMENTATION.

BUILD, INTEGRATE, TEST, DEBUG, AND DELIVER THE MOST COMPLETE FUNCTIONING RAPHAEL DEVELOPMENT APPLICATION POSSIBLE.

BEGIN NOW.

## Core intelligence implementation directive

# RAPHAEL AI — BUILD THE CORE INTELLIGENCE EXPERIENCE

Continue from the latest tested Raphael-AI development branch and Draft PR #1.

Preserve every existing feature, security improvement, and documented requirement.

This is an implementation assignment, not a planning or security-audit-only task.

The previous execution completed dependency remediation and JWT hardening. Do not repeat that work unless a new regression requires it.

## PRIMARY OBJECTIVE

Deliver a working, integrated Raphael clinical AI experience with:

1. ChatGPT-style streaming conversations.
2. Realtime voice interaction and configurable "Hey Raphael" activation.
3. Consent-controlled ambient clinical documentation.
4. Incrementally updated clinical notes.
5. Integration with the existing physician and patient portals.
6. Evidence-grounded clinical responses.
7. Actual end-to-end browser demonstrations.

Implement working code and tests rather than generating additional roadmaps.

## WORKSTREAM A — CONVERSATIONAL CLINICAL AI

Build a polished ChatGPT-style clinical conversation workspace.

Implement:

- Streaming responses
- Persistent conversation history
- Patient-specific clinical threads
- General medical research threads
- Context-aware follow-up conversations
- Conversation titles
- Markdown and structured clinical rendering
- Verified medical citations
- Clinical attachments
- Stop-generation controls
- Error recovery
- Model configuration
- Appropriate clinical uncertainty

Use the existing OpenAI-compatible API architecture.

When development credentials are available, validate actual provider responses.

Otherwise, implement deterministic integration tests and clearly identify the live-provider blocker.

## WORKSTREAM B — RAPHAEL VOICE AND ACTIVATION

Implement:

- User-controlled microphone permissions
- Push-to-talk
- Realtime speech recognition where configured
- Speech synthesis
- Interruptible spoken responses
- Voice activity detection
- Configurable wake phrase such as "Hey Raphael"
- Standard and Hands-Free modes
- Visible listening status
- Mute, pause, and stop controls
- Appropriate audio privacy controls

Use real browser APIs and supported provider integrations.

Do not represent mock voice output as live speech recognition.

## WORKSTREAM C — ACTIVE CLINICAL DOCUMENTATION

Implement a working synthetic clinical encounter workflow.

The clinician must be able to:

1. Open a synthetic patient encounter.
2. Confirm recording consent.
3. Start an authorized transcription session.
4. View an evolving transcript.
5. Generate a structured draft note.
6. Update the note as additional clinical information arrives.
7. Review uncertain or unsupported statements.
8. Edit the note.
9. Approve and sign the note.
10. Review the audit history.

Support SOAP and progress-note formats initially, using extensible templates for other note types.

Never automatically sign clinical documentation.

## WORKSTREAM D — INTEGRATION

Connect these experiences to the existing Raphael physician portal.

Ensure:

- Patient authorization is enforced.
- Encounter data persists.
- Notes are linked to the correct synthetic patient.
- Clinical conversations are appropriately scoped.
- Medical evidence references are traceable.
- Existing messaging continues working.
- Existing DICOM functionality remains working.
- Existing JWT security improvements remain intact.

## WORKSTREAM E — PROFESSIONAL INTERFACE

Create a cohesive clinical workspace with:

- ChatGPT-style conversation panel
- Voice controls
- Clinical encounter sidebar
- Live transcript panel
- Active documentation editor
- Evidence viewer
- Clinician approval controls
- Responsive layout
- Accessible interface

Do not leave nonfunctional placeholder buttons.

## WORKSTREAM F — TESTING

Run:

- Existing Python tests
- Existing frontend tests
- Existing browser tests
- Streaming conversation tests
- Voice permission tests
- Consent-gating tests
- Active note update tests
- Physician approval tests
- Authorization regression tests
- Persistence tests

Perform a browser demonstration of a synthetic physician opening a patient chart, starting an authorized encounter, interacting with Raphael, updating a clinical note, and approving the documentation.

Clearly distinguish real integrations from simulated development fixtures.

## DELIVERY REQUIREMENTS

- Implement working code.
- Fix reproducible defects.
- Preserve all existing functionality.
- Push changes to the current development branch.
- Update Draft PR #1.
- Provide a working browser preview where supported.
- Report tests actually executed.
- Report exactly which features are functional.
- Report which live services require credentials.
- Provide the precise steps to test Raphael.

Do not merge or deploy automatically.

Do not use real patient data.

Do not stop after documentation.

PRIORITIZE A WORKING INTEGRATED CLINICAL AI EXPERIENCE OVER ADDITIONAL PLANNING.

## Connected-care master product directive

# RAPHAEL AI — MASTER PRODUCT IMPLEMENTATION DIRECTIVE

## Mission

Continue developing the EXISTING Raphael AI application in the GitHub repository `AlGeniusAlien/Raphael-AI`. Do not create a replacement repository, disposable prototype, or unrelated scaffold.

Raphael's mission is to become a unified, multimodal, evidence-grounded healthcare intelligence and coordination platform serving patients, families, physicians, specialists, hospitals, urgent care centers, pediatric practices, radiologists, and other authorized healthcare professionals.

The intended experience is a connected continuum of care: prevention, routine visits, primary care, specialty consultations, pediatric care, urgent care, emergency care, hospitalization, discharge, rehabilitation, and long-term follow-up.

Raphael should help the right information reach the right authorized professional at the right time, reducing repeated paperwork, missing medical histories, unnecessary delays, and fragmented care.

Build for worldwide use, including iPhone and eventually Android, while recognizing that interoperability, privacy law, clinical validation, and provider participation vary by jurisdiction.

The goal is a functioning, commercially viable, exceptionally designed product—not a collection of mockups.

## Mandatory engineering rules

1. Inspect the existing repository, draft PR #1, AGENTS.md, architecture documents, requirements matrix, existing tests, and implementation status before changing code.
2. Preserve all working features, tests, security improvements, and requirement IDs.
3. Continue work in the existing draft PR. Do not merge or deploy to production without explicit approval.
4. Use synthetic patient data only. No real patient records or production clinical communications.
5. Never print, commit, expose, or log OpenAI or Qdrant secrets.
6. Verify implementations through executable tests, real integration checks where credentials exist, and actual browser scenarios.
7. Distinguish implemented, tested, simulated, blocked, and clinically validated capabilities. Never claim medical-grade readiness based solely on software tests.
8. Prioritize complete end-to-end workflows over shallow implementations across hundreds of features.
9. Do not stop after updating documentation or fixing a single dependency. Execute as much of the implementation as safely possible in this run.
10. Record remaining work precisely and maintain a prioritized continuation plan.

## Phase 1 — Connect the actual AI and evidence infrastructure

The Codex environment now contains:

- `LLM_API_KEY`
- `LLM_BASE_URL=https://api.openai.com/v1`
- `LLM_MODEL=gpt-4o-mini`
- `QDRANT_URL`
- `QDRANT_API_KEY`

Verify that the application actually reads these values.

Perform authenticated connectivity checks to OpenAI and Qdrant. Report success or failure without disclosing credentials.

Inspect Raphael's existing embeddings implementation, vector dimensions, collection naming, ingestion pipeline, and retrieval code.

Create or reuse an appropriately configured Qdrant development collection. Never recreate or destroy an existing collection unnecessarily.

Implement and test:

- Reliable ingestion and indexing of approved medical evidence
- Source identifiers, titles, publication dates, revisions, licensing, and specialty metadata
- Chunking and embedding consistency
- Semantic and hybrid retrieval where supported
- Metadata filtering and relevance ranking
- Retrieval provenance and citation verification
- Clear handling of missing, outdated, conflicting, or insufficient evidence
- Collection health checks and reproducible ingestion
- Evidence versioning and update mechanisms

Start with a small, curated, legally reusable corpus from authoritative sources, such as appropriate NIH, CDC, FDA, PubMed, and other approved references. Respect source licenses and access restrictions.

Do not claim that an empty Qdrant database contains medical knowledge.

Demonstrate actual retrieval results and verify that Raphael can incorporate those results into its clinical conversation workflow.

## Phase 2 — Complete Raphael's conversational medical intelligence

Inspect and finish the existing streaming conversation implementation.

Deliver a professional, responsive conversational interface supporting:

- Patient and physician conversations
- Persistent history and context
- Real model streaming
- Cancellation, regeneration, and error recovery
- File and document attachments
- Evidence retrieval and verifiable citations
- Patient-specific chart context only when authorized
- Structured clinical summaries
- Uncertainty and evidence-quality indicators
- Clear emergency escalation
- Conversation export
- Model/provider abstraction and usage monitoring

Raphael must offer clinically useful reasoning rather than generic disclaimers.

For authorized clinician workflows, support ranked differential diagnoses, supporting and contradicting findings, dangerous alternatives, missing information, and suggested confirmatory or exclusion criteria.

Do not represent AI suggestions as definitive diagnoses or replace licensed clinical judgment.

## Phase 3 — Build the unified longitudinal patient record

A patient's authorized medical information should follow them throughout their healthcare journey.

Implement a structured longitudinal health record containing:

- Demographics and verified identity
- Emergency contacts
- Medical and surgical history
- Active and historical conditions
- Medication lists
- Allergies and adverse reactions
- Immunizations
- Laboratory results and trends
- Imaging and radiology reports
- Encounters and hospitalizations
- Procedures
- Clinical notes
- Care plans
- Appointments
- Relevant family history
- Patient-reported symptoms and outcomes
- Documents and attachments
- Provenance, corrections, and update history

Support pediatric patients and guardian/proxy permissions, including age-dependent consent and access rules.

Build explicit patient-controlled sharing with revocation, expiration, purpose limitation, audit trails, and emergency-access workflows subject to applicable law.

A patient should not have to repeatedly enter the same history at every participating facility.

Use FHIR-compatible data models and interoperable interfaces where practical.

## Phase 4 — Design the entire healthcare journey

Implement complete, testable workflows for:

### Primary care and pediatrics

Appointment preparation, intake, history review, preventive care, vaccinations, developmental information, encounter notes, clinician-approved recommendations, follow-up, and secure family communication.

### Urgent care

Pre-arrival symptom intake, current medication/allergy review, risk escalation, clinician worklist, visit documentation, discharge instructions, and primary-care follow-up.

### Emergency departments

A patient or authorized caregiver should be able to prepare an emergency information packet containing essential medical history, allergies, medications, emergency contacts, relevant conditions, and current symptoms.

Support structured pre-arrival intake and secure transmission to an explicitly participating, authenticated facility.

Build an emergency encounter dashboard that helps receiving professionals review the information before arrival.

Never imply that a message was received, accepted, or monitored by emergency personnel without a verified integration and acknowledgment.

Never delay emergency services, emergency transport, or lifesaving care to complete Raphael intake.

### Hospitalization

Admission history, medication reconciliation, care-team coordination, progress notes, handoffs, imaging/lab review, discharge planning, and follow-up.

### Specialists

Referral packets, prior findings, diagnostic questions, consultation notes, care-team communication, and treatment-plan coordination.

### Post-discharge and long-term care

Medication reconciliation, follow-up appointments, recovery tracking, patient education, rehabilitation, preventive monitoring, and escalation of concerning changes.

For each journey, create a synthetic end-to-end test demonstrating what the patient sees, what the clinician sees, and what information is exchanged.

## Phase 5 — International healthcare interoperability

Design Raphael for patients who travel, live abroad, or need medical care outside their home country.

Implement:

- Patient-authorized international record sharing
- Multilingual interfaces and medical summaries
- Translation with preservation of original clinical terminology
- Clearly marked translation uncertainty
- Time-limited access for authenticated foreign clinicians
- Identity and professional credential verification
- Secure communication with the patient's existing care team
- FHIR-based exchange where participating systems support it
- Exportable structured emergency and clinical summaries
- Consent and jurisdiction-aware privacy controls
- Access logging and revocation
- Country-specific configuration and legal review checkpoints

Do not imply universal access to every hospital or national health system. Support real integrations where available and secure patient-mediated sharing elsewhere.

## Phase 6 — iPhone and mobile-first Raphael

Make mobile access a first-class product requirement.

Evaluate the existing web and desktop architecture and select a maintainable approach for an iOS application, such as a shared web frontend with a properly supported native shell or a cross-platform mobile application.

Deliver an implementation plan and begin building a real mobile client with:

- Secure sign-in and biometric authentication where supported
- Patient dashboard and medical timeline
- Emergency information access
- Care-team messaging
- Appointments and intake
- AI conversations
- Voice interaction
- Consent-managed document and image uploads
- Push notifications
- Accessible, polished mobile navigation
- Secure offline access to a minimal emergency summary where appropriate
- Privacy-conscious session and device management
- Pediatric caregiver access

Prepare the architecture for eventual Apple App Store distribution, including privacy disclosures, permission flows, accessibility, account deletion, data practices, and medical-app review requirements.

Do not claim the app is App Store approved or published without completing the actual Apple review process.

## Phase 7 — Voice, hands-free operation, and ambient clinical documentation

Finish and verify the features already implemented:

- "Hey Raphael" activation where technically supported
- Explicit microphone consent
- Real speech recognition
- Interruptible voice conversations
- Speech synthesis
- Clear recording/listening indicators
- Pause, mute, and stop controls
- Consent-based clinical encounter transcription
- Incremental note drafting
- SOAP, H&P, progress, consultation, and discharge notes
- Version history and preservation of clinician edits
- Explicit clinician review and signature
- Auditability and provenance

Implement real-device microphone testing and document unsupported browsers or environments.

Never silently record patients or publish AI-generated notes without authorized clinician approval.

## Phase 8 — Physician intelligence and multimodal medicine

Advance the clinician portal across specialties, including radiology.

Support:

- Patient worklists
- Chart review
- Evidence-grounded differential diagnosis
- Laboratory interpretation and longitudinal trends
- Medication/allergy and interaction checking using validated data
- Draft orders requiring clinician approval
- Radiology image and report workflows
- DICOM upload and viewing
- Prior-study comparison where data is available
- Structured clinical documentation
- Guideline lookup
- Clinical coding support with appropriate licensed terminology
- Secure clinician collaboration
- Clinical decision audit trails

Use modular model adapters for imaging and other modalities. Do not claim diagnostic imaging performance without representative validation.

## Phase 9 — Holistic, preventive, and lifestyle medicine

Raphael should recognize that excellent healthcare includes prevention and evidence-informed nonpharmacologic interventions, not only medications and procedures.

Build a dedicated preventive and integrative-health knowledge layer covering:

- Exercise and physical activity
- Nutrition
- Sleep
- Stress reduction
- Mindfulness and relaxation
- Evidence-based breathing practices
- Physical therapy and rehabilitation
- Smoking cessation
- Preventive screenings
- Appropriate complementary therapies

Include an evidence review of the Wim Hof Method and related breathing/cold-exposure practices.

For every intervention, distinguish established benefits, preliminary evidence, unsupported claims, contraindications, and known risks.

Explicitly prohibit breath-holding exercises in water, while driving, or in other dangerous circumstances. Provide screening and caution around cold exposure and relevant medical conditions.

Do not recommend replacing proven necessary medical treatment with unvalidated alternatives. Support informed shared decision-making and clinician oversight.

## Phase 10 — Secure communication and care coordination

Complete patient-clinician and clinician-clinician messaging.

Support authenticated care teams, routing, read status, attachments, notifications, clinician-reviewed AI drafts, and documented handoffs.

Implement an emergency pre-arrival communication workflow with clear distinctions between:

- Drafted information
- Successfully transmitted information
- Delivery acknowledgment
- Clinical receipt or acceptance
- Facility response

Never fabricate receipt or promise an immediate clinical response.

Support organizational boundaries, permissions, and escalation.

## Phase 11 — EHR and hospital integrations

Build and test standards-based interoperability using SMART on FHIR and appropriate FHIR resources.

Include sandbox integrations for patient records, observations, conditions, medications, allergies, encounters, reports, documents, and provenance.

Design integration adapters for participating EHR, PACS, laboratory, pharmacy, and scheduling systems.

Use least-privilege authorization, explicit patient context, clinician-approved writes, and comprehensive audit logs.

Do not claim universal interoperability until each real integration is independently verified.

## Phase 12 — Security, privacy, reliability, and commercial readiness

Implement and verify:

- Tenant and organization isolation
- Patient-level authorization
- Secure authentication and sessions
- Encryption in transit and at rest
- Secret management
- Audit trails
- Rate limiting
- Dependency scanning
- Secure file handling
- Backup and recovery planning
- Database migrations
- Health and readiness endpoints
- CI/CD testing
- Accessibility
- Observability and error monitoring
- Staging deployment configuration
- Billing and subscription architecture in test mode
- Enterprise administration

Prepare for HIPAA, applicable state privacy laws, international privacy obligations, medical-device regulatory assessment, and healthcare security review.

Do not assert compliance, certification, regulatory clearance, or readiness to handle live protected health information without independent verification and necessary agreements.

## Phase 13 — Clinical evaluation and scientific credibility

Develop reproducible evaluation suites for:

- Evidence retrieval accuracy
- Citation correctness
- Clinical factuality
- Differential diagnosis quality
- Dangerous-condition detection
- Medication safety
- Note completeness and fidelity
- Voice transcription accuracy
- Imaging workflows
- Fairness and subgroup performance
- Abstention and uncertainty calibration
- Data leakage and unauthorized access
- Emergency escalation behavior

Use synthetic and properly licensed public benchmark data initially.

Create clinician-review protocols and research-ready evaluation documentation.

## Phase 14 — Product experience and design quality

Raphael must look and feel like a serious, modern, world-class healthcare platform.

Deliver cohesive, accessible interfaces for:

1. Patients and families
2. Physicians and specialists
3. Hospital and facility care teams
4. Organizational administrators
5. Mobile users

Avoid raw developer interfaces, confusing navigation, placeholder-only workflows, and inconsistent visual design.

Make the most important patient information and clinical actions immediately understandable.

## Phase 15 — Execution requirements for this run

Do not merely restate the vision.

FIRST:
1. Inspect the current repository and draft PR.
2. Verify OpenAI and Qdrant connectivity.
3. Repair any integration failures.
4. Create and populate a small verified development evidence collection.
5. Demonstrate a real evidence-grounded Raphael conversation.
6. Run the existing test suite and browser tests.

THEN:
7. Implement the highest-value missing end-to-end patient/clinician workflow, prioritizing emergency pre-arrival intake, authorized record sharing, and receiving-clinician review.
8. Extend the existing patient and clinician interfaces rather than creating disconnected screens.
9. Establish the mobile app architecture and implement an initial working mobile patient experience if feasible.
10. Add security and workflow regression tests.
11. Update the requirements matrix and roadmap.
12. Push all verified changes to existing draft PR #1.

If time or tool limits prevent completion, stop at a coherent, tested checkpoint. Do not pretend unfinished features are complete.

## Final required report

At the end of the run, provide:

- Exact changes made
- Files modified
- OpenAI connection result
- Qdrant connection result
- Qdrant collection name, dimensions, and indexed document counts
- Real evidence retrieval and citation test results
- End-to-end workflows completed
- Tests passed and failed
- Browser/mobile testing performed
- Features still simulated or incomplete
- Security and regulatory blockers
- Draft PR link
- Exact steps to run and preview Raphael
- The next highest-priority implementation task

Raphael must evolve into a trustworthy, connected, clinically useful healthcare platform. Optimize for real functionality, patient safety, physician utility, interoperability, privacy, reliability, and an exceptional user experience.

## Priority-ordered principal engineer directive

# RAPHAEL AI — PRINCIPAL ENGINEER EXECUTION DIRECTIVE
## Priority: Working AI + Medical Evidence + Connected Patient-to-Clinician Journey

You are the principal software architect, lead implementation engineer, security engineer, and quality assurance lead for Raphael AI.

Your objective is not to write another proposal or produce superficial features. Your objective is to make the EXISTING Raphael application materially more complete, functional, integrated, reliable, and testable during this execution.

We have limited Codex execution capacity. Maximize verified working functionality per run. Do not waste execution on repeated descriptions, unnecessary rewrites, cosmetic changes, or speculative features.

## 1. Existing project — mandatory context

Repository: `AlGeniusAlien/Raphael-AI`

Continue the existing development branch and draft PR #1. Inspect the actual repository and PR state before making changes.

The application already includes or has recently reported work on:

- FastAPI backend and browser frontend
- Patient and clinician portals
- Authentication and scoped patient access
- Structured patient records
- Secure internal messaging
- Medical evidence retrieval architecture
- DICOM handling
- Clinical conversations and streaming
- Browser voice controls
- Encounter transcription and versioned clinical notes
- Clinician approval workflows
- Development runtime and browser tests
- Security and dependency remediation

These capabilities have different levels of completion and verification. Determine their actual status from the code and tests; do not assume the prior reports prove production readiness.

Read `AGENTS.md`, the master specification, requirements matrix, roadmap, architecture documentation, and existing tests. Preserve the existing requirements and their identifiers.

Do not rebuild functionality that already works.

## 2. Permanent product mission — preserve this vision

Raphael is a unified, patient-centered, multimodal healthcare intelligence and coordination platform.

It must ultimately support the entire continuum of care:

- Preventive and primary care
- Pediatrics and caregiver-managed care
- Urgent care
- Emergency departments and pre-arrival preparation
- Hospitals and inpatient care
- Specialists and referrals
- Imaging and radiology
- Rehabilitation and recovery
- Long-term and chronic-condition management
- International travel and cross-border healthcare
- Clinical research and evaluation

Patients should have a longitudinal health record that can be securely shared with authorized healthcare professionals.

Participating clinicians should be able to access relevant information quickly, communicate with care teams, reduce redundant intake, and make better-informed clinical decisions.

Raphael must be mobile-first, with an eventual iOS App Store application and Android support.

Its AI should provide evidence-grounded assistance, voice interaction, consented ambient documentation, and multimodal clinical support.

Raphael should include preventive and integrative healthcare, including evidence-informed lifestyle interventions and a balanced assessment of Wim Hof–style breathing and cold exposure, with contraindications and safety precautions.

Do not automatically favor alternative therapies over necessary evidence-based treatment.

The ultimate ambition is a global healthcare platform. Preserve that mission in the repository's permanent product documentation, but do not confuse the long-term vision with features that can be safely delivered in this execution.

## 3. Execution strategy

Perform a brief repository assessment, then immediately implement.

Use the following priority order:

P0 — Verify real OpenAI and Qdrant connectivity.
P1 — Deliver a complete evidence-grounded clinical conversation.
P2 — Deliver an integrated synthetic patient-to-clinician care journey.
P3 — Establish a mobile-ready implementation and emergency information workflow.
P4 — Address directly related reliability, security, and usability issues.

Do not start P2 until the essential P0/P1 integration works or is conclusively blocked by an external dependency.

Do not spend the run attempting every item in the master roadmap.

When a task is blocked, document the precise blocker, isolate it, and proceed to the next independent task.

## 4. P0 — Real infrastructure integration

The Codex development environment has been configured with:

`LLM_API_KEY`
`LLM_BASE_URL`
`LLM_MODEL`
`QDRANT_URL`
`QDRANT_API_KEY`

Never display or commit secret values.

Verify whether these credentials are actually available to the application runtime. Environment configuration alone does not prove successful integration.

### OpenAI

Perform a minimal authenticated inference test using the configured model.

Verify:

- Authentication
- Model availability
- Response handling
- Timeouts and useful error messages
- Streaming compatibility
- Reasonable usage controls

Do not repeatedly retry failed paid requests without a bounded retry policy.

### Qdrant

Connect to the existing development cluster:

`raphael-medical-evidence-dev`

Do not create a second cluster.

Inspect the repository's current embedding model, dimensions, vector naming, collection configuration, and retrieval pipeline.

Use a compatible embedding strategy, preferring the existing implementation unless a concrete technical problem requires a change.

Ensure embedding dimensions match the collection schema. Do not silently mix incompatible embedding models.

Create the development collection only if it does not already exist.

Implement idempotent ingestion and avoid duplicate documents.

Select a small, authoritative, legally reusable medical evidence corpus with verifiable source URLs, document titles, dates, source identifiers, and licensing metadata.

If internet access or source licensing prevents verified ingestion, do not fabricate source content. Use clearly labeled fixtures for local tests and report the external blocker.

Run actual authenticated retrieval tests against Qdrant.

Validate retrieved metadata, ranking, and source references.

Provide the real collection name, embedding model, vector dimensions, document counts, and retrieval test results.

## 5. P1 — Complete evidence-grounded AI conversations

Integrate the live model and Qdrant retrieval into Raphael's existing patient and clinician conversation interface.

The finished workflow must support:

- Authenticated conversation access
- Correct patient-context authorization
- Persistent conversation history
- Live model streaming
- Cancellation and regeneration
- Evidence retrieval
- Source attribution
- Verified clickable citations
- Explicit uncertainty
- Missing-evidence handling
- Clinical red-flag escalation
- Safe error handling
- Browser-level regression testing

A citation must refer to a genuine retrieved source. Never invent references, publication dates, guidelines, or supporting quotations.

Ensure the system distinguishes:

1. Facts documented in the authorized patient chart
2. Findings retrieved from external medical evidence
3. Model-generated clinical interpretation
4. Missing or uncertain information

For clinician-facing responses, structure differential diagnoses around supporting evidence, contradicting evidence, important alternatives, and unanswered questions.

For patient-facing responses, use understandable language and appropriate safety escalation.

Do not allow patient-facing AI to impersonate a licensed clinician or issue unauthorized clinical orders.

### Required P1 demonstration

Using synthetic data, show that:

A clinician signs in, opens an authorized patient chart, asks Raphael a clinical question, receives a real model-generated response, and can inspect the supporting retrieved evidence.

Demonstrate that another unauthorized clinician cannot access the same patient context.

The workflow must be exercised through actual application endpoints and, where feasible, a real browser test.

## 6. P2 — Connected patient-to-clinician workflow

Build upon the existing patient and physician portals.

Implement one complete synthetic care journey:

1. Patient signs in.
2. Patient reviews or updates permitted history and symptoms.
3. Patient initiates a pre-visit or urgent-care intake.
4. Existing medications, allergies, conditions, and relevant history are incorporated with provenance.
5. Patient explicitly authorizes sharing with a participating synthetic facility or clinician.
6. The receiving clinician signs in and sees the intake and authorized record.
7. Clinician reviews the information and can use Raphael's evidence-grounded assistant.
8. Clinician prepares a note or care summary.
9. Clinician reviews and approves the note.
10. Patient sees the appropriate approved summary and follow-up information.
11. All significant access, sharing, and approval events are auditable.

Use existing database models and application patterns wherever possible.

Do not bypass tenant isolation or patient-level authorization to make the demonstration work.

Implement a consent expiration/revocation test.

Do not simulate external hospital connectivity as though it were a real integration.

## 7. P3 — Mobile and emergency-care foundation

Raphael must eventually be accessible through the Apple App Store and on Android.

Inspect the existing frontend and recommend the simplest maintainable path to a secure mobile application.

Avoid building an entirely separate mobile backend.

Implement the highest-value mobile-ready improvement feasible within this run, prioritizing responsive patient access and an emergency information view.

The emergency information view should present:

- Patient identity information
- Important medical conditions
- Medication list
- Allergies
- Emergency contacts
- Relevant clinical history
- Last-updated timestamps
- Source/provenance indicators

Protect sensitive data with appropriate authorization and device security.

For a future emergency pre-arrival workflow, design explicit states:

DRAFT → AUTHORIZED → SENT → DELIVERY CONFIRMED → CLINICALLY ACKNOWLEDGED

Do not mark a message delivered or acknowledged without real confirmation.

Do not imply that an emergency facility is monitoring Raphael without a verified partnership or integration.

Emergency features must never delay calling emergency services.

Prepare an iOS implementation decision record covering the native app approach, authentication, biometrics, push notifications, privacy disclosures, Apple developer requirements, and App Store review dependencies.

Do not claim an App Store submission or approval unless one actually occurred.

## 8. Permanent roadmap requirements

Update the existing product specification and requirements matrix to preserve and organize these future capabilities:

- International record portability and multilingual access
- Verified foreign clinician identity and authorization
- Cross-border consent and privacy requirements
- SMART on FHIR and EHR interoperability
- Hospital, urgent care, primary care, and pediatric integrations
- Emergency pre-arrival coordination
- Radiology and multimodal imaging intelligence
- Medication safety and validated clinical data
- Secure physician-to-physician communication
- Consent-based voice and ambient documentation
- Native iOS and Android distribution
- Preventive, lifestyle, and integrative health
- Evidence-based Wim Hof Method evaluation and safety
- Research benchmarking and clinical evaluation
- Enterprise administration, billing, and deployment
- Clinical and regulatory readiness

Do not mark roadmap features implemented unless verified in code.

## 9. Engineering and security standards

Use the existing architecture and coding conventions.

Maintain modularity, typed interfaces where practical, testable boundaries, and readable code.

Avoid unnecessary dependencies and broad rewrites.

Preserve:

- Tenant isolation
- Patient-level authorization
- Secure authentication
- Explicit consent
- Audit trails
- Data provenance
- Clinician approval requirements
- Secret protection
- Safe file handling
- Existing tests and migrations

Use synthetic data only.

No production patient data.
No live hospital messaging.
No autonomous clinical decision execution.
No live billing charges.
No production deployment.
No merge without explicit approval.

Do not claim HIPAA compliance, medical-device clearance, clinical validation, or universal healthcare interoperability merely because software functionality exists.

## 10. Validation and acceptance criteria

Before completing the run:

1. Run the existing Python and frontend test suites.
2. Add focused tests for the integration changes.
3. Execute real OpenAI and Qdrant smoke tests if permitted by the environment.
4. Exercise the evidence-grounded conversation through the actual backend.
5. Run real Chromium browser tests for the main patient/clinician workflow where supported.
6. Verify unauthorized access is denied.
7. Verify consent and approval boundaries.
8. Check that no secrets or sensitive information appear in code, commits, logs, or screenshots.
9. Review the final diff for regressions and incomplete implementation.
10. Update the requirements matrix with truthful completion states.

A mocked test does not count as a verified live provider integration.

A passing unit test does not establish clinical accuracy.

A working browser screen does not establish production readiness.

## 11. Efficient execution rules

Minimize time spent explaining intentions.

Inspect, plan briefly, implement, test, fix, and report.

Use a short internal execution plan with checkpoints. Update it as implementation proceeds.

Prioritize fixing blockers that prevent a complete vertical workflow.

Do not introduce a new framework, database, or architecture without a demonstrated need.

Do not spend the run polishing unrelated UI details or rewriting documentation already present.

If the remaining execution budget becomes constrained, finish the highest-priority feature already underway, run its tests, commit the verified checkpoint, and record exact continuation instructions.

Never declare a feature complete simply because its code compiles.

## 12. Git and handoff

Continue using the existing development branch and draft PR #1 if access permits.

Do not force-push or overwrite unrelated work.

Commit coherent changes and push them when authorized and technically possible.

If GitHub permissions prevent a push, preserve the local changes and report the exact blocker.

At the end, provide a concise engineering report with:

- Branch and PR status
- Implemented features
- Files changed
- OpenAI live connection result
- Qdrant live connection result
- Collection name and embedding dimensions
- Indexed evidence count
- Actual retrieval/citation verification
- Completed patient-to-clinician workflow
- Browser tests performed
- Python/frontend test results
- Security checks
- Unfinished or simulated functionality
- Exact preview/run instructions
- Next three implementation priorities

The most important outcome of this run is a functioning, evidence-grounded Raphael clinical intelligence experience connected to real infrastructure and integrated with a coherent patient-to-clinician workflow.

Execute now. Do not stop at a plan or ask for routine implementation decisions that can be safely resolved from the existing codebase.