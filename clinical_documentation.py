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
