# Current authenticated service verification — 2026-10-26

All five runtime variables are present and both saved secret bindings are delivered. Both configured HTTPS destinations match their secret scopes and Qdrant has a saved egress rule. No secret values were printed,placed in .env,or copied to source/logs.

## Actual OpenAI result

A bounded real inference request authenticated successfully and returned nonempty content from gpt-4o-mini-2024-07-18. A separate bounded actual SSE request returned nonempty streamed text. Retries were disabled and per-request timeouts25seconds. This establishes authentication,configured model access and streaming,not medical factuality or a grounded clinical interaction. Earlier401results are historical and do not describe this runtime.

## Actual Qdrant result and corrected routing

The first real collection-list request returned403UnexpectedResponse. Offline SDK inspection established that pinned qdrant-client1.9.1 maps a standard HTTPS URL's default port to6333. The configured URL uses standard HTTPS semantics. One controlled read-only request on443then authenticated successfully. This is a route correction,not a credential workaround;the same scoped credential and hostname were retained. The target raphael_evidence_v1 is absent. No other collection's points or detailed schema were read or modified. Point/vector configuration for an absent target is not applicable;other cluster document/point counts were not enumerated.

qdrant_config.connection_options now preserves standard HTTPS443 across main,diagnostics,PubMed,evidence and RAG clients,while retaining explicit custom ports and existing local HTTP behavior. A real SDK construction regression tests the route without making requests. The main collection initializer also checks existing unnamed cosine schema and an actual finite encoder-output dimension probe before creation. Wrong dimensions,named vectors or incompatible distance fail without replacing existing collections. These regressions use labeled synthetic vectors,not the real model.

## Remaining ingestion blocker

Expected encoder configuration is sentence-transformers/all-MiniLM-L6-v2 and384dimensions. No cached model directory exists;an actual offline Hugging Face snapshot lookup also returnedLocalEntryNotFoundError. Requests to Hugging Face model metadata and CDC evidence both failed withProxyError through the current network path. Actual encoder output/dimension and compatibility are therefore not verified. No model download,remote collection creation,ingestion,medical search or evidence-grounded conversation was claimed. Ingestion this execution:0;collection mutations:0;no live medical citation examples exist.

Saved configuration adds huggingface.co and www.cdc.gov to the existing custom network allowance and updates start_skill. The save reports requires_publish=true. Review/save and publish this environment configuration through settings,then reconnect the runtime to apply it. This is environment configuration publication,not application deployment. The agent did not publish or apply it. If model artifacts redirect to additional hosts,inspect those actual destinations and allow only those required;do not bypass proxy/TLS restrictions. No further credential replacement is indicated by the successful443read.

After network access changes:fetch authoritative model metadata and weights with normal TLS/integrity verification,cache a pinned revision,encode a nonclinical probe and assert actual384-dimensional finite vectors. Recheck the target collection's state and schema before any mutation;create only the compatible development collection if still absent. Retrieve and inspect legally reusable official medical text and its reuse policy/provenance,ingest a bounded idempotent corpus,then test actual Qdrant search and an authenticated Raphael conversation. Do not substitute fixture text or a fresh unverified embedding model for these steps. Normal application startup can initialize a missing target only after the model probe passes;diagnostic service checks remain read-only.

Latest regression evidence is TEST_RESULTS.md. No merge,deployment,real patient data or external clinical messages occurred. Draft PR#1 preserves the existing connected synthetic workflow.
