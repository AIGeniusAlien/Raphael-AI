# Connected synthetic development candidate

This is a tested local synthetic workflow candidate,not full-product or live-service release acceptance. Real OpenAI inference/stream and QdrantHTTPS443collection access now pass. Model/source downloads remainProxyErrorblocked,so an actual evidence-grounded medical interaction remains incomplete. See INFRASTRUCTURE_STATUS.md and TEST_RESULTS.md for current evidence;never present a local provider fixture as authenticated external inference.

## Integrated encounter workflow

Use the existing checkout/branch and installed virtual environment. For an operator-run local service:

```bash
cd /workspace/Raphael-AI
export PATH="/workspace/.raphael/venv/bin:$PATH"
export RAPHAEL_DEV_PORTAL=1 RAPHAEL_DEV_SYNTHETIC_ONLY=1
export XDG_STATE_HOME=/workspace/.raphael/state
# Use a dedicated synthetic path;never point this at an existing patient database.
mkdir -p "$XDG_STATE_HOME/raphael"
export DATABASE_URL="sqlite:////workspace/.raphael/state/raphael/connected-synthetic.db"
# During infrastructure diagnosis,keep remote services out of application startup:
unset QDRANT_URL QDRANT_API_KEY
python -m auth.dev_bootstrap --database-url "$DATABASE_URL" --tenant synthetic --username clinician --role clinician
python -m auth.dev_bootstrap --database-url "$DATABASE_URL" --tenant synthetic --username receiver --role clinician
python -m auth.dev_bootstrap --database-url "$DATABASE_URL" --tenant synthetic --username patient --role patient
bash run_local.sh
```

Use that path only for synthetic Raphael records;choose a new dedicated filename if its provenance is unknown. Bootstrap prompts privately;no default/shared password is installed. Provision only once;existing account creation rejects safely. Reuse existing authorized accounts on later starts. `run_local.sh` uses loopback and main:app;GET /healthz must return status ok. The onboarding UI does not support forwarded localhost preview links,so no accessible hosted preview is claimed. Real Chromium validates the actual local service internally. Do not expose the unauthenticated legacy mode or enable public/mobile network binding without staging/security review.

Clinician creator:link a new synthetic chart to the patient username and add entered medications/allergies/history. Patient:open Connected intake and emergency information,append a reported update/contact,create a snapshot,select the receiving clinician,purpose,expiration and separate AI-processing permission. Receiver:open only the explicitly shared intake and use Open consented snapshot in Raphael.

For a grounded real response,apply the saved model/source network allowance,verify/cache the actual encoder,and add reviewed legally reusable evidence first. Provider authentication andHTTPS443collection listing have now passed. In automated E2E only,a clearly labeled loopback HTTP SSE provider and test-owned nonclinical excerpt supply the response. No runtime fixture provider or fake medical corpus is installed in the application.

After an evidence-linked completed response,reload to inspect its stored evidence snapshot;Return to intake and review summary takes the clinician back to the same authorized intake. Use selected AI draft in summary explicitly stages editable text. Clinician must enter follow-up,save a new draft version and explicitly approve the current saved version. Patient cannot see draft versions;approved summary/follow-up and its AI provenance appear in care and the approved note appears in the portal. Neither staging nor saving auto-approves or sends a message. Original response citation membership is not verification of clinician-edited text or clinical factuality.

Source response ID,thread,generation,model and retrieved evidence persist with the summary and carry forward across manual edits. Historical evidence describes retrieval time;it is not a claim the source remains current,approved or unretracted. Expiration/revocation denies future source-thread/handoff access. Previously approved patient notes are durable records;revocation does not delete approved documentation or recall already transmitted context.

## Readiness checklist

- [x] Dedicated synthetic SQLite,local password bootstrap and authenticated role workflows.
- [x] Consent snapshot,expiry/revocation,AI permission,source-thread ownership and cross-intake denial.
- [x] Persistent AI evidence snapshots and explicit clinician draft handoff with provenance.
- [x] Versioned editing,separate current-version approval,approved patient summary and follow-up.
- [x] Real390px Chromium journey plus restart persistence regression;current counts in TEST_RESULTS.md.
- [x] Successful authenticated OpenAI inference/stream and Qdrant collection listing overHTTPS443.
- [ ] Actual Qdrant medical search and evidence-grounded clinical interaction.
- [ ] Actual embedding-model/schema verification and curated legally reusable medical corpus.
- [ ] Whole-chart patient-level clinician assignment/guardian/cross-organization policy;current snapshot consent does not replace the broader tenant chart policy.
- [ ] Participating facility directory/transport,real delivery acknowledgments and international controls.
- [ ] Patient-facing AI,physical iPhone/Android/microphone verification,native build/account deletion/App Store requirements.
- [ ] Externally accessible synthetic staging with reviewed transport/session/privacy controls.

Mobile work this sprint is continuation of the responsive web journey;no native shell,device test or App Store submission was added. MOBILE_ADR.md remains the packaging decision record. Real facility delivery,translation/clinical equivalence,guardian identity and licensed references require separate implementation and authorized service/legal relationships.

No merge or deployment is authorized. Draft PR#1 is the review surface. Required owner action now is review/save/publication of the model/source network allowance and runtime reconnect;external staging publication/native distribution requires explicit authorization after the remaining gates are addressed.
