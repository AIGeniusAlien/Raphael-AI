# Raphael — complete-system synthetic operational acceptance protocol

**Release posture:** NOT READY FOR FULL OPERATIONAL CLINICAL USE. This
document defines the repeatable *synthetic test-first milestone* for the full
Raphael vision: comprehensive clinical reasoning, EHR/charting, multimodal
imaging, voice, nursing/clinical coordination, operations, patient experiences
and research. The software may contain partial implementations of these
workflows. A passing test suite is not evidence of clinical competence.

## Strict test environments

- **A. Isolated engineering tests:** dedicated Python 3.12 environment, Node,
  Playwright/Chromium, synthetic SQLite, locally protected keys, no PHI,
  controlled mock/sandbox services.
- **B. Authorized staging and native devices:** named accountable operator,
  isolated infrastructure, verified backups/restores, configured synthetic-only
  provider interfaces, dedicated test identities, current iPhone/hardware;
  audit and incident simulation; no public network exposure by default.
- **C. Qualified clinician and hospital evaluations:** independently
  adjudicated, rights-cleared cases with explicit intended use, specialist
  expertise, population distribution and reference standard. Separate approval
  for real PHI/hospital network connectivity. This tier is NOT provided by A
  or B.

The built-in `consumer_candidate` mode is a **patient-only preview**; it
is not a full clinician operational demo. For full module exercises use the
restricted synthetic development mode, never a publicly accessible instance.

## Execution sequence

1. Identify the exact candidate Git commit and all repository/dependency
   versions. Require a dedicated operator, synthetic fixtures and written
   signoff for environment A.
2. Run `python -m dev_verify --all --require-browser --output <private-dir>`
   using Python 3.12 and complete dependencies. Read each per-check log,
   `results.json`, safety evaluation and screenshots. Do not accept
   skipped checks in place of passed ones.
3. Confirm all four GitHub Actions jobs at that **same exact commit**:
   `regression`, `ios-simulator`, `dependency-audit`,
   `engineering-watch`. Do not treat older successful runs as new evidence.
4. Produce a reviewed CI summary and run
   `python -m scripts.full_system_acceptance --local <private-dir>/results.json --ci <reviewed-ci.json> --source-commit <40-hex-SHA> --output <private-dir>/system-acceptance.json`.
   See `CONTROLLED_SYNTHETIC_PREVIEW_HANDOFF.md` for the CI summary schema,
   review and trust limits. The CLI only summarizes software test gates; its
   CI input is an untrusted reviewer assertion, not cryptographic proof.
5. For every system below, record operator, environment, test case, actual
   outcome, artifacts, failure severity, reproduced defect and signoff.
   Every production-significant defect must be fixed and rerun *at its final
   source commit*. A feature without an exercised fixture and specialist
   acceptance remains explicitly blocked.
6. Perform separate staging, real hardware, performance, resilience and
   external-authorization tests before any broader evaluation. Require an
   independent decision on legal/privacy/regulatory intended use.

## Full-system test suites (required, not implied completed)

| System | Synthetic operational demonstration | Additional go/no-go evidence |
|---|---|---|
| Clinician conversation and medical evidence | Authenticated multi-turn case; streaming/cancel; citations resolve; missing evidence abstains; patient switch removes private context | Source licensing and versions, blinded specialist scoring, calibration and unsafe advice rate |
| Radiology / imaging | Valid synthetic DICOM upload, frame navigation/prior comparison, manual draft, explicit approval, withdrawal, communication vs acknowledged states | Independent image-model validation across modality/site/device, real PACS, missed critical findings and clinician signoff |
| Charting and longitudinal records | Correct patient/encounter, histories, allergies, lab values, medication reconciliation, reviewed notes, closed/reopened encounters and edit provenance | Clinician signature and order authority, certified integration, chart correctness and safety oversight |
| Voice/ambient listening | Explicit consent, start/stop, encounter close, remote revoke, background tab, timeout, vendor loss, no duplication or cross-patient dictation | Measured medical ASR/diarization performance, privacy/consent law, real clinical environment validation |
| Emergency / critical results | Simulated alert/risk cases fail safely; manual radiology contact and separate recipient acknowledgment; no fabricated delivery | Proven escalation, uptime, callback/receipt verification, red-team misses, qualified emergency clinician review |
| Nursing, pharmacy and procedures | Authorized synthetic care tasks, handoff and medication checks, procedure checklists, approvals and revocations | Licensed pathways, real dispensing/order controls, medication-error analysis and professional responsibility |
| Hospital administration / scheduling | Synthetic intake, encounters, scheduling conflicts, organization roles, referrals/workflows and audit | Connected billing/RCM, real insurance/payer rules, workforce and workflow authorization |
| Patient experience | Separate patient permissions, approved-only results, education, downloads, feedback, session expiry, account controls | Patient usability, language accessibility, support, consent management, account legal obligations |
| Interoperability | FHIR sandbox read-only preview, DICOMweb QIDO synthetic study discovery, patient mapping changes, network failure | SMART-on-FHIR/Epic site authorization, actual EHR and PACS partner conformance and contracts |
| Security / operations | Tenant isolation, session revocation, encryption controls, tamper detection, backup/restore and response withholding | Independent penetration testing, threat model, secrets custody, retention/erasure, incident drills, HA/RTO/RPO |
| Native/mobile | Actual iOS simulator login/MFA/session tests and full screen accessibility smoke | Physical iPhone, signing, distribution, device accessibility and secure staging |
| Clinical evaluation and research | Synthetic differential metrics, evidence coverage, fail-closed claim screening and regression cases | Prospectively defined intended use, reference standard, independent adjudication, outcome, bias and monitoring |

## Exit criteria

- **Synthetic engineering acceptance:** all supported automated checks and
  end-to-end synthetic workflows pass at one exact source commit; no
  unexplained failures; reviewer confirms isolation and signed results.
- **Complete system operational acceptance:** all rows above have real
  environment and domain-expert acceptance as applicable, with unresolved
  issues triaged and no critical safety/security blocker. This is not
  established by existing tests.
- **Clinical deployment:** requires separate site governance, privacy/security,
  professional, regulatory, legal and monitored pilot approvals. Do not
  activate medical autonomy or bypass clinicians' responsibility.

The authoritative output is evidence and identified blockers, not a claim of
perfection. No system can be certified "perfect" by static code or a synthetic
unit test.
