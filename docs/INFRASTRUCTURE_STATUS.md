> Release audit,2026-10-28: [Current source-scoped readiness audit](RC_RELEASE_AUDIT.md). Sections below are historical checkpoints;older "current" or "final" wording is not a release approval.

## Current verified recovery result — 2026-10-26

See [recovery implementation report](RECOVERY_IMPLEMENTATION_REPORT.md) for exact owner commands and next actions,and IMPLEMENTATION_REPORT.json/RECOVERY_LIVE_RESULTS.json for current results. Python181/181,frontend26/26,Chromium6/6 and mobileLinux5/5passed. Actual physician AND patient Chromium workflows completed with realOpenAI/Qdrant,384-dimensional text-embedding-3-small,6CDCdocuments/31vectors,repeat0writes,resolving citations,fresh original-source checks and automated support screens passed. Earlier failed screens are retained in RECOVERY_LIVE_ATTEMPTS.json. This does not establish clinical accuracy. Chart grants/revocation,patient AI and account controls are integrated. Account deletion is review-only;iOS source exists but Xcode/device/signing unverified;Android native absent. No staging deployment,AppStore readiness,merge or production-data claim. All1,112prior requirement IDs/descriptions preserved;machine ledger now1,117rows.

Earlier checkpoint text below is historical and may describe resolved infrastructure or feature gaps;use the current report first.

# Current infrastructure result

The live medical intelligence vertical workflow now passes using real OpenAI inference/streaming,real OpenAI text-embedding-3-small384-dimensional vectors and authenticatedQdrantHTTPS443access. raphael_evidence_v1 contains3CDCdocuments and20active/total vectors. Repeat ingestion wrote0. HuggingFace metadata andCDCsource/reuse-policy fetches now pass;localMiniLMload still failedOSError and an explicit hosted fallback was verified instead.

Read LIVE_MEDICAL_INTELLIGENCE.md,CDC_EVIDENCE_MANIFEST.json,LIVE_MEDICAL_WORKFLOW_RESULT.json and TEST_RESULTS.md for actual provenance,source IDs,counts,source checks,failed attempts and limitations. The actual physician Chromium run uses synthetic context and no provider fixtures. Its answer is unapproved clinician decision support,not medically validated output. Current setup requires RAPHAEL_EMBEDDING_BACKEND=openai and RAPHAEL_STREAM_EVIDENCE_BACKEND=qdrant;do not mix this encoder withMiniLMmerely because both have384dimensions. Existing encoder/model payload guards preserve data.

Both secrets are delivered through secure runtime configuration;no values were printed or stored in source. Earlier401/403andnetwork-blocked checkpoints are historical. No additional credentials or broad network access are requested. Saved startup guidance is configuration metadata,not runtime application/publication. No application merge/deployment or external clinical message occurred.
