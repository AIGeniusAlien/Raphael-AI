# Synthetic database encryption implementation checkpoint

The complete platform directive is **not complete**. This checkpoint implements opt-in database-page protection for isolated synthetic Linux development, not clinical-production approval. Starting source: `4cd2a24b484d14df59e620a2938a5d3c64a57eb2`. Inspect current Git HEAD/PR #1 for the resulting commit; local results below preceded that commit and do not imply subsequent CI success.

## Implemented and actually exercised

- Pinned Linux SQLCipher driver (`sqlcipher3-binary==0.6.0`); actual runtime reports **4.12.0 community**. All application/startup/account provisioning/recovery/backup engine paths share a fail-closed factory. Existing plaintext development mode stays explicit and compatible. A configured database key without encrypted mode is refused.
- New file encryption requires explicit synthetic scope, owner-only files and a distinct 256-bit component key. Missing/wrong keys, tampering, missing driver, invalid mode and unsafe paths fail without a plaintext fallback. Key values are absent from URLs, diagnostics and status responses.
- Bounded copy migration (64 MiB/200,000 table rows) preserves complete schema, tables, indexes, triggers, FTS, original-document BLOBs, IDs, autoincrement state and application/user version. SQLCipher export is reopened before integrity, foreign-key and all-table fingerprint checks. A private staged file is published exclusively only after verification. Existing destination and source are never overwritten. The source remains accessible with its original key/format. Rotation is a second verified copy, not in-place rekey or secure erasure.
- Runtime snapshots and outer encrypted backups retain SQLCipher inner protection; restore verifies keyed integrity/schema before publishing a new file. Plaintext restore into encrypted mode is refused. Authenticated account UI reports actual database mode and exclusions.
- Actual Chromium account/MFA/document/lifecycle workflow now runs on encrypted storage. Mobile-width storage status screenshot was visually inspected; no overflow or key values observed.

## Actual verification

Consolidated `python -m dev_verify --all --require-browser --output /private/NEW` passed **9 locally runnable checks**: **319 Python, 29 frontend, 7 Chromium, 5 mobile-source tests**, traceability, acceptance freshness, safety evaluation and protected runtime. Zero local check failures. Xcode, fresh paid live services and advisory audit were explicitly skipped/separate; they are not included in passing claims. Encrypted protected-runtime smoke performed two real application starts, health/readiness, synthetic patient persistence, attachment decryption, MFA persistence, password-only denial and recovery login after restart.

Eight added SQLCipher regressions use real encryption, not a substituted driver. Original failure attempts are preserved in workspace logs: virtual-table schema required reopen after export; SQLCipher page-size pragma returned a string. Both were fixed before the complete passing run. Existing backup and key-file tests also passed independently. Machine evidence: `DATABASE_STORAGE_LOCAL_RESULTS.json`, `DATABASE_STORAGE_RUNTIME_RESULTS.json`. Raw logs/screenshots remain in `/workspace/.raphael/cipher-full-initial`; these local artifacts are not durable CI artifacts.

## Reproduce safely

Install the pinned dependencies using the existing setup script or `python -m dev_start --setup`; Linux includes the new driver. Create three **distinct, NEW** owner-only keys outside Git using `python -m dev_keys create --output /private/NEW-component.key` for database, documents and MFA. Never print key values or regenerate keys for an existing database.

```bash
# New isolated synthetic database, private account provisioning:
python -m dev_start --provision --database /private/NEW-synthetic.db --database-key-file /private/db.key --document-key-file /private/document.key --mfa-key-file /private/mfa.key
# Stop all source writers before a copy; source must be a ready synthetic DB:
RAPHAEL_DEV_SYNTHETIC_ONLY=1 python -m protected_database --source /private/old-synthetic.db --output /private/NEW-protected.db --target-key-file /private/db.key
# Rotation: retain old source/key until separately authorized retention/cutover:
RAPHAEL_DEV_SYNTHETIC_ONLY=1 python -m protected_database --source /private/old-protected.db --source-key-file /private/old-db.key --output /private/NEW-rotated.db --target-key-file /private/new-db.key
python -m scripts.verify_protected_runtime --encrypted-database --output /private/NEW-report.json
```

Restart via `--database` with the exact selected new path and same component keys. Migration does **not** automatically switch the application; operator verifies the copy, stops writers and chooses cutover. Snapshot migration is not live replication. Default startup makes no provider requests. Do not use these synthetic tools against production data.

## Remaining limits and owner actions

This addresses database pages (including stored original bytes), not complete PHI protection. Plaintext legacy/source copies, exports, upload spooling, caches, separate restored files and externally held copies remain outside scope. No automatic deletion or production data migration occurred. Production KMS/escrow, rotation/retention policy, secure erasure, platform key stores, licensed corpus expansion, independent clinical/security/privacy evaluation, qualified external integrations, staging/signing/device/VoiceOver acceptance and operational recovery are still required. SQLCipher Mac/Windows acceptance is not established. A pinned wheel/TLS/checksum is not independent cryptographic/FIPS certification.

| Release gate | Result | Reason |
|---|---|---|
| Isolated synthetic local web workflow | PASS | Actual software/runtime/browser regressions above |
| Controlled synthetic TestFlight | BLOCKED | Existing staging, signing, physical-device, accessibility and owner acceptance gates remain |
| Clinical pilot / clinical production | FAIL | Clinical validation, full PHI lifecycle and operational/legal/integration gates incomplete |

Historical 27-case native simulator acceptance and historical live OpenAI/Qdrant evidence retain their original commit scope; neither was rerun locally here. No merge, deployment, Apple submission, real patient use or paid provider request occurred in this checkpoint.
