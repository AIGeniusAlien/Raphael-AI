# Privacy notice draft — not approved or published

Owner/operator identity, jurisdiction, lawful bases, approved public policy URL, support contact and effective date require owner/legal review before distribution. This draft describes the current synthetic development architecture; it does not establish HIPAA or international compliance.

The app is restricted to synthetic development records. Do not enter real patient data until the operator's independent privacy, clinical, organizational and security reviews authorize a defined deployment. The native client connects only to the operator-configured HTTPS Raphael service. All AI/database operations use server-side credentials; those keys are never distributed in the app.

## Data inventory for App Privacy review

| Data | Purpose and processing | Linkage and retention questions |
| --- | --- | --- |
| Account username, organization and opaque session cookie | Authentication and authorization at the Raphael server; cookie held by an ephemeral WebKit store | Linked to the test account. Server session expiry/revocation follows backend policy; process restart discards native web session. |
| Entered demographics, history, symptoms, medications, allergies, documents and clinical records | Existing authorized backend patient/clinician workflows | Health information linked to the account. Backend retention, backups and clinical-record obligations require a published approved policy. |
| Manually entered emergency contacts/names/phone numbers | Patient-managed emergency information | Linked record content. No contacts-address-book permission is requested. |
| AI questions and authorized record context | General education/provisional clinician support through configured OpenAI services; query embeddings support Qdrant retrieval | User content/health data linked to the workflow. Explicit consent/authorization controls are server-owned. Provider contracts, international transfers, retention and training settings must be reviewed; this draft does not invent provider guarantees. |
| Public medical source excerpts and citations | Evidence provenance and source inspection | CDC source data is public; patient context is not sent to CDC. Opening a source in system Safari causes normal source-site/network metadata processing. |
| Audit/access records | Security, consent and provenance | Backend audit retention and access limitations require operator approval. No chart data or secrets belong in general application logs. |

The native privacy manifest conservatively lists Health, Name, Phone Number, User ID and Other User Content for app functionality, linked to the user and not for tracking. The actual release manifest/App Store labels must be reconciled against the shipped backend, all SDKs and provider agreements. The app adds no analytics SDK, advertising/tracking, push token registration or automatic contact-book access. Microphone capture is disabled in native code. No microphone/provider accuracy or audio privacy validation is claimed.

The app has no offline medical database, credential vault or record cache. An ephemeral WebKit store reduces persistent web storage; it is not a guarantee against all OS memory, temporary storage or user screenshots. The background privacy cover, transport, WebKit lifecycle and sign-out behavior require physical-device verification. Native downloads are blocked; do not describe backend export as an on-device completed export.

## Controls and deletion

Account/profile, sharing consent, device sessions and data export use the existing backend. Sign-out attempts remote session revocation and clears local session state; failure to confirm server revocation is disclosed. Revocation cannot recall information already viewed or transmitted. Approved clinical/audit records may have retention obligations.

The account page exposes a **request for deletion review**, not a promise of immediate erasure. Before App Store submission, implement and validate the final account-deletion workflow, identity verification, retention exceptions, backup lifecycle, notifications and appeal/support process. Do not label a recorded review request as completed deletion.

Publish an owner/legal-approved policy at a stable publicly accessible HTTPS URL before App Review. No approved public policy URL, operating-entity declaration or cross-border legal arrangement is supplied by this draft.
