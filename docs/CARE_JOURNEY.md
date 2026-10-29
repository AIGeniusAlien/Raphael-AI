# Connected synthetic patient-to-clinician journey

Start the existing authenticated synthetic portal using DEPLOYMENT.md. No account/password defaults or new cluster are created. A patient account must already be linked to a synthetic patient record. Provision a separate receiving clinician with the existing private-password CLI,for example:

```bash
python -m auth.dev_bootstrap --database-url "$DATABASE_URL" --tenant synthetic --username receiving-clinician --role clinician
```

Open **Connected intake and emergency information** from the portal. Patient:save a reported history/symptom/contact update; inspect the current emergency information; create a pre-visit or urgent-care snapshot; choose an active synthetic clinician in the same organization; authorize an explicit purpose and expiration. Optional AI processing needs a separate unchecked-by-default consent because it can send the snapshot to a configured provider. The snapshot includes provenance-marked non-draft chart entries and approved notes and never overwrites clinician records.

Receiving clinician:sign in separately,open the same care link,refresh the authorized intake list,inspect the source timestamps/medications/allergies/history and reported symptoms. Draft/edit a summary and follow-up; save the current version before explicit approval. Only the approved summary becomes patient-visible and enters the existing patient notes view. The patient can inspect the access/approval audit,expire/revoke future receiver access and retain their own copy. An approved summary is immutable in this flow.

If separately permitted,the clinician can open **Open consented snapshot in Raphael**. Its AI thread is bound to the authorized immutable snapshot,not a live chart. Other clinicians cannot read that intake or thread. Expiration/revocation denies further access and cancels later streamed tokens. It cannot recall data already viewed or sent to an external provider. Live inference is still blocked until development bindings are applied.

Important scope:these permissions protect the intake snapshot and its linked AI context. They do not replace the existing tenant-clinician chart policy or private-document sharing model. This is not full longitudinal-record consent,guardian access,international identity verification or a real facility integration. No user is represented as a verified licensed professional. No hospital transmission,delivery,monitoring,emergency triage or clinical acceptance is performed. Never delay emergency services to complete intake.

Emergency information is authenticated patient-mediated display/export only. Stored DRAFT/AUTHORIZED/REVOKED consent states and effective EXPIRED access must not be confused with transport states. Future external adapters must distinguish DRAFT→AUTHORIZED→SENT→DELIVERY_CONFIRMED→CLINICALLY_ACKNOWLEDGED using genuine provider/recipient confirmations. No current endpoint manufactures those transport states.

See MOBILE_ADR.md for the shared web/iOS decision and unsupported native features.
