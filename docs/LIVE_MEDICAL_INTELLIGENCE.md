## Current verified recovery result — 2026-10-26

See [recovery implementation report](RECOVERY_IMPLEMENTATION_REPORT.md) for exact owner commands and next actions,and IMPLEMENTATION_REPORT.json/RECOVERY_LIVE_RESULTS.json for current results. Python181/181,frontend26/26,Chromium6/6 and mobileLinux5/5passed. Actual physician AND patient Chromium workflows completed with realOpenAI/Qdrant,384-dimensional text-embedding-3-small,6CDCdocuments/31vectors,repeat0writes,resolving citations,fresh original-source checks and automated support screens passed. Earlier failed screens are retained in RECOVERY_LIVE_ATTEMPTS.json. This does not establish clinical accuracy. Chart grants/revocation,patient AI and account controls are integrated. Account deletion is review-only;iOS source exists but Xcode/device/signing unverified;Android native absent. No staging deployment,AppStore readiness,merge or production-data claim. All1,112prior requirement IDs/descriptions preserved;machine ledger now1,117rows.

Earlier checkpoint text below is historical and may describe resolved infrastructure or feature gaps;use the current report first.

# First live medical intelligence development workflow

This is an actual end-to-end synthetic development result,not clinical validation or an approved patient recommendation. See LIVE_MEDICAL_WORKFLOW_RESULT.json for the exact provisional response,retrieved excerpt IDs,scores,source metadata and verification flags. No fixture provider was used for that result. The case was a generated synthetic adult with cough,sore throat and rhinorrhea,no exam/testing/allergy verification and no red-flag assessment.

## Encoder and collection

Hugging Face metadata andCDCpages now returnHTTP200. The pinned local sentence-transformers/all-MiniLM-L6-v2 load attempt failed withOSError;no local-model success is claimed. The authorized fallback,OpenAI text-embedding-3-small,returned actual384-dimensional finite vectors. It is explicitly selected;384equal dimensions do not make its embedding space interchangeable withMiniLM.

raphael_evidence_v1 was absent before this run. After actual hosted-vector verification,the development collection was created with unnamed384-dimensional cosine vectors. Every point records its encoder and dimensions. Ingestion inventories existing points and refuses a different or untagged encoder without deletion,recreation or index changes. The legacy main path also refuses known differently tagged encoders. Legacy local behavior remains available with its original configuration/separate compatible collection;legacy untagged ingestion into the hosted collection fails closed with409.

The managed cluster requires payload indexes for filtered reads/writes. The first ingestion stopped with400after4vectors because those indexes were absent;read-only diagnosis confirmed an index requirement. Adding keyword/boolean indexes for the actual filters allowed resumption:4vectors were deduplicated and16added. Final actual inventory:3documents,20active vectors,20total vectors. A repeat ingestion wrote0and deduplicated20. Source version changes retain prior vectors and mark them noncurrent;there is no deletion path or automatic background ingestion.

## Legally reusable source scope

Source:CDC. Original material is freely available on the CDC website. Raphael is not endorsed byCDC,HHS or the U.S.Government. Only contiguous text excerpts were used;no logos,images or third-party works were copied. CDC's reuse policy,including exceptions,attribution,no-endorsement,unaltered-substantive-text and free-original-source requirements was inspected at https://www.cdc.gov/other/agencymaterials.html. This records a bounded engineering reuse review,not independent legal certification or a universal license for every CDC asset.

| Source | Publisher date | Role |
| --- | --- | --- |
| [Healthy Habits:Antibiotic Do's and Don'ts](https://www.cdc.gov/antibiotic-use/about/index.html) |2025-09-23|Patient education |
| [Outpatient Clinical Care for Adults](https://www.cdc.gov/antibiotic-use/hcp/clinical-care/adult-outpatient.html) |2024-04-16|Official clinical guidance summary |
| [Outpatient Clinical Care for Pediatric Populations](https://www.cdc.gov/antibiotic-use/hcp/clinical-care/pediatric-outpatient.html) |2024-04-22|Official clinical guidance summary |

CDC_EVIDENCE_MANIFEST.json preserves title,URL,date,text hash,retrieval time,license declaration,policy URL/hash. Remote payloads also preserve HTML hash,version,chunk ordinal,encoder and attribution notice. Retrieval scores are cosine similarities,not medical confidence. Tier labels are application heuristics;official guidance/education is not independentlyGRADE-rated. The2024pages are displayed with an age warning,which does not itself establish withdrawal or supersession.

## Actual physician portal demonstration

scripts.verify_live_medical created private random-password accounts and a temporary SQLite database,started actual main:app,logged in through Chromium's real physician portal,opened a patient-authorized intake with separate AI permission,and streamed an actual OpenAI response using actual hosted embeddings and remoteQdrantsearch. Six excerpts were returned;cosine scores were approximately0.62–0.68. Citation endpoints resolved and the evidence reloaded from stored generation snapshots. FreshCDCfetches matched all returned content hashes,dates,titles and exact excerpt substrings. A real missing-evidence search returned0.

The model initially produced malformed prefixed citations and,in another attempt,an invented citation. Strict checks rejected those drafts;no failed draft was approved. The corrected protocol gives the model short registered retrieval labels,validates every returned reference against retrieved labels/IDs,and resolves only valid labels into immutable source hashes before completion/persistence. Unknown labels fail. This avoids requiring the model to copy long hashes without admitting invented references. The final recorded browser run passed with no page errors. Earlier failures are preserved in TEST_RESULTS.md rather than reported as successful answers.

The response discusses antibiotics/testing and explicitly states missing case information. It remains unapproved decision-support text. Source-ID membership and source-text/hash verification do not prove every generated clinical assertion,adult/pediatric applicability,source currency or treatment recommendation. One displayed supporting excerpt is pediatric;population applicability and all medical wording require clinician review. No record was signed,no patient instructions sent,and no clinical validation,calibrated confidence or autonomous decision is claimed.

## Reproduce deliberately

From the existing checkout with secure development bindings already injected:

```bash
export PATH="/workspace/.raphael/venv/bin:$PATH"
export RAPHAEL_EMBEDDING_BACKEND=openai
export RAPHAEL_STREAM_EVIDENCE_BACKEND=qdrant
export EMBED_DIM=384
python -m curated_evidence --ingest
python -m scripts.verify_live_medical --live
```

The first command fetches only the fixed three CDC URLs and reuse policy,validates the actual hosted encoder and compatible schema,then ingests and repeats to verify idempotence. It makes bounded paid embedding requests and development-cluster writes. It refuses another embedding space;never recreate a populated incompatible collection. The second uses actual providers,no runtime mocks,and an isolated synthetic DB;its JSON and screenshot are written outside the checkout by default. Private generated test passwords are never printed. It stops only its own server and deletes the temporary database. These are opt-in live tools,not CI tests or a hosted deployment.

For interactive operator testing,follow CONNECTED_RELEASE_CANDIDATE.md's private-password bootstrap and dedicated synthetic DB,retain the two backend exports above,and run run_local.sh. Do not unsetQdrantfor this configured hosted workflow. The provider model remains controlled byLLM_MODEL. Use a patient-authorized intake if only snapshot context should be shared. The onboardingUI does not support hosted localhost previews;no public preview is claimed.

## Safety and remaining scope

Regression coverage uses explicitly labeled local fixtures for missing/conflicting evidence,provider failure,unauthorized role/tenant/patient/thread access,expiry/revocation,citation mismatch and source-version/model-space preservation. A conflicting-evidence fixture checks both passages reach the model with conflict/uncertainty instructions;there is no validated contradiction detector or medical adjudication engine. Actual provider failure was not deliberately induced with invalid credentials.

The base clinician chart policy remains broader than snapshot consent. Guardian/cross-organization policy,comprehensive literature coverage,independent source/clinical evaluation,source update monitoring,native mobile/device voice and real facility delivery remain incomplete. Existing local SQLite evidence mode,voice,documentation,messaging,DICOM andFHIR workflows are preserved. No merge,deployment,real patient data,real clinical communication or billing was used.
