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