# Development infrastructure verification

Actual attached-machine check found LLM_API_KEY,LLM_BASE_URL,LLM_MODEL,QDRANT_URL and QDRANT_API_KEY all unset. Standard project/workspace .env files were absent. Saved draft metadata declares LLM_API_KEY and pending variable requirements,but it is not evidence of runtime injection. No raw values or credential files were printed. OpenAI/Qdrant smoke checks are conclusively blocked before authentication; no paid request,collection creation,recreation or ingestion was attempted. No new cluster was created.

Expected existing configuration from code:collection raphael_evidence_v1 unless QDRANT_COLLECTION overrides it;sentence-transformers/all-MiniLM-L6-v2;384-dimensional cosine vectors unless corresponding explicit settings change. These are configuration defaults,not observed remote facts. The user-identified development cluster is raphael-medical-evidence-dev; its actual endpoint/schema/counts could not be read without a runtime URL/binding. Actual collection existence,dimensions,document/point count and authenticated retrieval remain unknown. Never report this as an empty or populated medical corpus.

Apply the saved development environment variables/secrets to the attached runtime in environment settings,including secure QDRANT_API_KEY scoped to the selected endpoint. Do not paste keys into chat or source. Saving a requirement alone does not apply its value; no duplicate credential is requested based on missing token variables alone. Then run:

```bash
python -m dev_services
python -m dev_services --live
```

The first command inspects presence only. --live makes at most one tiny nonclinical inference and one tiny stream request with retries disabled,then read-only Qdrant collection/schema/count checks. It never recreates a collection. Errors report class/status only. LIVE_INTEGRATION_RESULT.json records this execution. A failed or missing binding is not an authentication-success/failure claim.

Next:verify compatible model/dimension/collection metadata;cache the approved embedding model;ingest a small genuinely licensed,reviewed authoritative corpus idempotently with provenance;verify actual retrieval and source-supported citations;then exercise the application endpoint and Chromium with the real provider. Existing nonclinical evidence fixtures remain labeled fixtures. No live citation/medical factuality result is claimed here.
