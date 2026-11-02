# Current foundation implementation evidence

**The entire definitive directive is not complete.** Two integrated development slices were implemented on the existing branch/draft PR #1: opt-in synthetic SQLCipher database protection and reviewed source-bound local terminology. No merge, deployment, Apple submission, real patient data, clinical communications or paid provider request occurred. Existing live evidence/corpus and native features are preserved.

## Actual implemented behavior

1. **Protected storage:** See [migration, rotation, backup and operational commands](DATABASE_STORAGE_CHECKPOINT.md). Actual SQLCipher **4.12.0 community** protects database pages, including stored original-document bytes and clinical text, only when explicitly selected. Complete-schema copies are integrity/fingerprint verified and exclusively published; source and existing destination data are preserved. Wrong/missing keys, tampering and missing driver fail closed. Actual authenticated UI reports mode and exclusions. Default plaintext remains compatible;this is not an automatic production migration.
2. **Reviewed terminology:** Existing knowledge workspace now accepts immutable local canonical/alias mappings bound to an exact same-tenant approved source-version hash. Canonical text must occur in that source. New mappings are pending until explicit clinician review;content corrections create a distinct mapping. Revoked/superseded source versions immediately become ineligible. Search expansion is explicitly opt-in and exact-phrase based. Ambiguous aliases abstain from expansion;query/coverage limits prevent unbounded work. Normalization provenance and resolved original citations are returned. Specialty facets count actual active local terms/source versions, with clinical validation explicitly absent. This is **not** a licensed standard ontology, universal specialty registry, semantic/vector retrieval improvement or clinical competence claim. Live Qdrant and clinical generation defaults are unchanged.
3. **Traceability:** All1,161previous IDs/descriptions/acceptance criteria are unchanged;1,164rows now include three additive narrowly scoped requirements. Canonical ledger includes the requested verification vocabulary and explicit source/risk/API/UI/data/release fields. Legacy individual upload attribution,risk classification and contract inventories are marked unadjudicated rather than invented. The36-section latest directive is preserved verbatim;section inventory does not establish complete requirement-by-requirement reconciliation or implementation.

## Verified local result

[Consolidated results](FOUNDATION_LOCAL_RESULTS.json): **326 Python tests,29 frontend tests,8 Chromium workflows,5 mobile-source tests;all9locally runnable checks passed;0failed**. Three explicit separate/skipped checks: Xcode unavailable on Linux,live paid-service checks not selected,and advisory audit delegated to CI. [Actual encrypted restart smoke](FOUNDATION_RUNTIME_RESULTS.json) passed two real FastAPI starts,health/readiness,patient/document/MFA persistence,password-only denial and recovery login.

Seven new terminology API tests cover pending/review gating,idempotence,persistence,source revocation/supersession,conflicting aliases,no-evidence cases,exact citation provenance,wrong version,immutable correction,tenant/role/logout denial,input and query budgets. Actual Chromium covers create→review→opt-in retrieval→stored citation→source revocation,390px/320px enlarged/1280px layouts. Mobile screenshot was visually inspected. First browser attempt failed because selector population was inserted into the wrong JavaScript branch;corrected before passing rerun and full regression. The original failure log remains private workspace evidence,not success.

Raw logs/screenshots:`/workspace/.raphael/foundation-layout-final`;separate initial failures:`/workspace/.raphael/terms-browser-first.log`,`sqlcipher-tests-first.log` where present. These workspace paths are not durable CI artifacts. Historical provider results and27-case simulator acceptance retain their original source scope. Current-head CI must be inspected independently;this pre-push report does not infer its future outcome. Pinned Linux CPython3.12 wheel checksum previously verified:`c8a7076a18b4f6fc9b580b0f24aa526d6a47605b1e3e6491ee3e9f2977940d8e`;neither that checksum nor passing tests imply FIPS/HIPAA certification or independent supply-chain review.

## Release decisions

| Gate | Result | Outstanding work / owner |
|---|---|---|
| Isolated synthetic local web demonstration | PASS | Use private synthetic DB/keys and clinician review;no real PHI |
| Complete synthetic TestFlight beta | BLOCKED | Engineering completes remaining scoped acceptance;owner supplies authorized HTTPS staging,Apple team/bundle/signing,approved assets/privacy/support URLs;actual device,VoiceOver,OS share/retention and owner acceptance required |
| Controlled clinical pilot | FAIL | Qualified clinical intended-use/population/evaluation,licensed evidence,complete PHI lifecycle and accountable operational/legal review incomplete |
| Clinical production | FAIL | Pilot gates plus enterprise identity,external integrations,monitoring/load/recovery/change management and independent security/privacy/regulatory approvals incomplete |

Engineering still owns independently achievable missing features;these are not reclassified as external blockers. Comprehensive procedure pathways,source-reviewed terminology/coverage standards,real ambient medical ASR,validated differential/emergency support,full clinical/admin workflows,roles/proxies,external EHR/PACS/payer/notification acceptance,Android/desktop and comprehensive evaluation remain unfinished. Database pages do not protect plaintext source copies,exports,upload spooling,OS caches or separately restored copies. Operator-controlled KMS/escrow/retention/rotation/secure erasure and production DR remain required.

## Reproduce and continue

```bash
python -m dev_verify --all --require-browser --output /private/NEW-results
python -m unittest discover -s tests -p test_medical_terms.py -v
python -m unittest discover -s e2e -p test_terms_browser.py -v
python -m scripts.build_engineering_ledger --check
python -m scripts.build_platform_acceptance --check
```

Provision a dedicated synthetic clinician using the documented private-key launcher. Visit `/static/knowledge.html`,intake legally usable test-owned synthetic text,review the source,create a canonical phrase/alias mapping,read and explicitly review it,then opt into alias search and resolve citations. Revoking the source removes mapping eligibility. No medical assertions need be invented to reproduce this software workflow.

Continue with qualified terminology/licensing/coverage standards and clinician-reviewed procedure pathways in existing modules,including patient/clinician UI,persistence,versioned provenance,consent/role/tenant/failure tests. Do not substitute metadata for clinical validation. Preserve all historical requirements,work and draft PR #1. No merge/deploy/Apple/real patient use is authorized.

## CI-discovered layout defect and preserved failure

First current-source CI [41388665250](https://github.com/AIGeniusAlien/Raphael-AI/actions/runs/41388665250) on `13a1628bca3caec2754ff7036ab1cd74fc5729dd` failed the enlarged account-screen width assertion while dependency/source scanning passed. This overrides any inference that the initial local pass established current CI acceptance. Native was still running at this evidence snapshot;read final job outcome separately. [Failure metadata](FOUNDATION_FIRST_CI_RESULTS.json) is preserved.

Actual local reproduction:DejaVu Sans20px at320px measured document339px;the comma-joined public storage-status token measured325.303px. Scoped wrapping and readable spaces fix the content;the browser test now explicitly exercises this font fallback without relaxing its width assertion or authentication checks. The reproduction now stays320px across five fonts. Full corrected local run again passes326Python/29frontend/8Chromium/5mobile-source and all9checks. [Source digest manifest](FOUNDATION_VERIFIED_SOURCE_MANIFEST.json) identifies the exact exercised application/browser bytes before their commit. Initial local results remain preserved separately. Follow-up current-head CI must pass before claiming remote acceptance;the draft PR records its actual outcome without a self-referential future commit/CI claim here.
