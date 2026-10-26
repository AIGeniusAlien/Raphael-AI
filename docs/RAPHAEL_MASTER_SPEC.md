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
