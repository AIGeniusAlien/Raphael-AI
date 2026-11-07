"""Deterministic preparation from authorized snapshots; no clinical inference."""
import json


SECTIONS={
    'SOAP':('O — Not independently observed.','A — Not assessed.','P — Pending clinician review.'),
    'progress':('Interval changes — Not independently verified.','Assessment and plan: Pending clinician review.'),
    'history_physical':('History of present illness — Requires clinician synthesis.','Physical examination — Not documented;do not infer normal findings.','Assessment — Not assessed.','Plan — Pending clinician decision.'),
    'consultation':('Consultation question — Clinician input required.','Findings — Requires clinician review.','Opinion and recommendations — Not assessed.'),
    'discharge':('Admission/discharge dates — Not documented.','Hospital course/procedures — Not documented.','Discharge diagnoses — Not established.','Medication reconciliation — Not completed.','Disposition/follow-up — Clinician decision required.'),
    'referral':('Recipient and referral reason — Clinician input required.','Requested evaluation — Not ordered.','Urgency and transfer details — Not assessed.'),
    'after_visit':('What was discussed — Clinician synthesis required.','Medication changes — No changes authorized by this draft.','Instructions and return precautions — Clinician input required.','Follow-up — Not scheduled by this draft.'),
}


def render_document(format,snapshot):
    if format not in SECTIONS:raise ValueError('Unsupported documentation format.')
    transcript='\n'.join(segment['content'] for segment in snapshot['transcript_segments'])
    heading='S — Reported transcript:' if format=='SOAP' else 'Reported transcript — not independently verified:'
    chart=json.dumps(snapshot['chart_entries'],ensure_ascii=False,indent=2)
    return ('SYNTHETIC TRANSCRIPT TEMPLATE — NOT AI INFERENCE. Clinician review required.\n'
            'Document type: '+format+'\n'+heading+'\n'+transcript+'\n\n'
            'Authorized chart snapshot — reported record data,not proof of examination or services:\n'+chart+'\n\n'
            +'\n'.join(SECTIONS[format])+'\n\n'
            'Missing fields above require explicit clinician completion. No orders,claims,prescriptions or external correspondence have been submitted.')


# These are clinician-authored preparations, never payer submissions or AI-coded claims.
STRUCTURED_FORMATS = {
    'care_plan': ('goals', 'actions', 'follow_up', 'return_precautions'),
    'medical_necessity': ('requested_service', 'clinical_rationale', 'alternatives', 'recipient'),
    'prior_authorization': ('requested_service', 'insurance_reference', 'clinical_rationale', 'prior_treatments'),
    'insurance_appeal': ('decision_reference', 'requested_service', 'clinical_rationale', 'requested_review'),
    'structured_form': ('purpose', 'reported_findings', 'requested_action'),
    'coding_review': ('documented_service', 'candidate_codes', 'supporting_documentation', 'verification_needed'),
}


def render_structured_document(format, snapshot, fields):
    if format not in STRUCTURED_FORMATS: raise ValueError('Unsupported structured format.')
    allowed = STRUCTURED_FORMATS[format]
    if set(fields) - set(allowed): raise ValueError('Field does not belong to this document format.')
    chart_ids = {r['id'] for r in snapshot['chart_entries']}
    segment_ids = {r['id'] for r in snapshot['transcript_segments']}
    output = ['SYNTHETIC CLINICIAN-AUTHORED PREPARATION — review required.', 'Document type: '+format]
    missing = []
    for name in allowed:
        field = fields.get(name)
        if not field or not field['text'].strip():
            missing.append(name); output.append(name.replace('_',' ').title()+': NOT DOCUMENTED — clinician completion required.'); continue
        if set(field['chart_entry_ids']) - chart_ids or set(field['transcript_segment_ids']) - segment_ids:
            raise ValueError('Source reference is outside the authorized documentation snapshot.')
        output.append(name.replace('_',' ').title()+': '+field['text'])
        output.append('Provenance: explicit clinician input; chart entry IDs '+json.dumps(field['chart_entry_ids'])+'; transcript segment IDs '+json.dumps(field['transcript_segment_ids'])+'. References identify provenance, not independent factual validation.')
    output.append('Missing fields: '+(', '.join(missing) if missing else 'None structurally; clinical and administrative accuracy still require review.'))
    output.append('No diagnosis, order, claim, payer decision, medication change or external transmission is authorized by this preparation.')
    return '\n\n'.join(output), missing
