# Raphael — controlled synthetic preview handoff (2026-11-12)

## Release decision

**No real-patient clinical launch or unrestricted public distribution is authorized.**
The project remains a developing comprehensive clinical intelligence platform.
Do not advertise that it performs every job in medicine or has independently
validated diagnostic performance. Existing patient and clinician workflows are
synthetic development functionality; the restricted **consumer_candidate**
boundary exposes only a subset of the patient records/account experience.
It does not expose Raphael's physician conversation, imaging, diagnosis, ambient
listening, ordering, emergency care, or administrative functions.

This document provides a repeatable checkpoint toward an **operator-controlled,
provisioned-account synthetic demonstration**; it is not an App Store, TestFlight,
HIPAA, FDA, medical device, or hospital production launch certification.

## Before a controlled technical demonstration

1. Select a dedicated test host/environment, scoped private synthetic database,
   separate accounts, and documented operator. Confirm its isolation and no PHI
   with human review. No untrusted third-party access.
2. Inspect actual GitHub Actions jobs **for the same full commit SHA**. Required
   jobs: regression, ios-simulator, dependency-audit, engineering-watch. Keep
   actual run links, logs and any failed attempts as evidence. Never infer pass
   from a previous source SHA or queued CI.
3. Run source-level verification on an environment with Python dependencies,
   Node and Chromium installed:
   `python -m dev_verify --all --require-browser --output <reviewed-output-directory>`
   Read its `results.json`. Do not count its skipped native/CI checks as passes.
4. Configure `RAPHAEL_DEV_PORTAL=1`,
   `RAPHAEL_DEV_SYNTHETIC_ONLY=1`,
   `RAPHAEL_RELEASE_SCOPE=consumer_candidate`,
   and an operator-owned private directory + `RAPHAEL_RELEASE_PAUSE_FILE`
   (see `release_policy.py`). Verify real file ownership, permissions,
   start/stop controls, fail-closed pause and account continuity **on the
   deployment host**. A configuration string alone does not prove safety.
5. After independently inspecting GitHub job metadata for the exact SHA, create
   a *locally reviewed* CI evidence object in a private non-PHI review directory.
   It must have this exact structure with the **actual** head SHA and results:

   ```json
   {
     "source": "github_actions_api",
     "head_sha": "<full-reviewed-40-character-commit>",
     "jobs": {
       "regression": "success",
       "ios-simulator": "success",
       "dependency-audit": "success",
       "engineering-watch": "success"
     }
   }
   ```

   This is an **unsigned reviewer transcription**, not automatically trusted
   evidence. It can be falsified; independently consult the GitHub API/run logs
   before accepting it. Do not create it with guessed statuses.
6. Run the fail-closed technical preflight. Replace angle-bracket placeholders
   with the verified paths/SHAs; **the source commit and verified commit must
   match exactly**:

   ```bash
   python -m scripts.release_preflight \
     --verification-report <reviewed-output-directory>/results.json \
     --ci-evidence <reviewed-ci-evidence.json> \
     --commit <full-40-character-SHA> \
     --verified-commit <same-SHA> \
     --output <private-review-directory>/launch-preflight.json
   ```

   Exit code **2** means blocked. A passed *technical* preflight is NOT
   distribution authorization. The tool does not read or emit secrets, cannot
   verify whether data are truly synthetic, and cannot authenticate a provided
   CI JSON file. Independent human signoffs and host/security checks remain.
7. Perform a real staging smoke test with provisioned synthetic patient
   accounts: login/MFA, own-record visibility, cross-tenant denial, approved
   notes vs drafts, protected export, account controls, withdrawal/pause,
   restart/persistence, sign-out/session invalidation, screenshots/accessibility,
   and backup/restore drills. Test on actual intended OS/browser/iPhone hardware.
   Retain a signed operator decision and incident rollback instructions.
8. Do not enable paid integrations, internet-facing deployment, real-PHI import,
   automated medical orders, external critical messages, distribution, or
   clinician diagnosis based solely on this technical gate.

## Explicit blocking work for an actual Raphael clinical launch

- **Medical intelligence**: versioned, rights-reviewed evidence across
  specialties/procedures, adjudicated patient cohorts, clinician review and
  robust uncertainty/abstention. Current evidence is limited and cannot
  demonstrate medicine-wide coverage.
- **Radiology**: validated DICOM interpretation, PACS connectivity, calibrated
  sensitivity/miss-rate and human review. Existing DICOM view and DICOMweb
  preview are not validated automated diagnosis.
- **Voice and emergency**: real-world transcription/diarization evaluation,
  consent/privacy, urgent escalation, human acknowledgment and fail-safe
  operations. Browser speech and manual result-contact attestations are not
  hospital critical-result notification delivery.
- **Clinical and operational workflow**: clinician-authenticated EHR/FHIR/
  SMART integration, order signatures, pharmacy, nursing, surgical,
  scheduling, billing/claims and administrative systems with least privilege;
  independent acceptance tests and verified external partner agreements.
- **Production**: full PHI/data lifecycle including backups, deletion/retention
  governance and key custody, threat modeling, security testing, identity,
  SSO/MFA, monitoring/incident response, auditability, HA/DR, regulatory
  intended-use assessment, legal/privacy review, accessibility and hospital
  pilot approvals. No clinical deployment until these are addressed.

## Safety policy

Keep development PRs draft and unmerged until exact-source CI is passing;
even full CI does not authorize medicine. Preserve test history and do not
replace missing evidence with assertions. The end product remains Raphael's
comprehensive multimodal medical intelligence platform, *not* an Epic clone.
