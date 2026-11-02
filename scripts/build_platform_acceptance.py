"""Derive the owner's status vocabulary without rewriting canonical requirements."""
import argparse
from collections import Counter
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
STATUS={'VERIFIED':'VERIFIED IN SYNTHETIC TESTING','PARTIAL':'IN PROGRESS','IMPLEMENTED_UNVERIFIED':'IMPLEMENTED BUT UNVERIFIED','NOT_IMPLEMENTED':'NOT STARTED','BLOCKED_EXTERNAL':'EXTERNALLY BLOCKED','REQUIRES_CLINICAL_VALIDATION':'IMPLEMENTED BUT UNVERIFIED'}
CAUSE={'VERIFIED':'NONE_WITHIN_RECORDED_SCOPE','PARTIAL':'ENGINEERING_INCOMPLETE','IMPLEMENTED_UNVERIFIED':'VERIFICATION_MISSING','NOT_IMPLEMENTED':'IMPLEMENTATION_MISSING','BLOCKED_EXTERNAL':'EXTERNAL_DEPENDENCY','REQUIRES_CLINICAL_VALIDATION':'INDEPENDENT_VALIDATION_MISSING'}

def build():
    ledger=json.loads((ROOT/'docs/ENGINEERING_LEDGER.json').read_text())
    lines={line.split('|')[1].strip():number for number,line in enumerate((ROOT/'docs/REQUIREMENTS_MATRIX.md').read_text().splitlines(),1) if line.startswith('|')}
    rows=[]
    for row in ledger['requirements']:
        identifier=row['requirement_id']
        rows.append({**row,'canonical_state':row['state'],'status':('FAILED' if identifier=='PLATFORM-006' and row['state']!='VERIFIED' else row['verification_status']),
            'original_source':{**row['original_source'],'preserved_canonical_matrix':'docs/REQUIREMENTS_MATRIX.md','line':lines[identifier],'requirement_id':identifier,'historical_directive_attribution':'See preserved master directives;individual original-upload attribution has not been independently reconciled.'},
            'functional_area':row['functional_area'],
            'exact_required_behavior':row['description'],
            'safety_classification':'UNREVIEWED:qualified intended-use risk assessment required;software verification does not establish clinical safety.',
            'root_cause_type':('ACCEPTANCE_TEST_FAILURE' if identifier=='PLATFORM-006' and row['state']!='VERIFIED' else CAUSE[row['state']]),
            'owner_controlled_blockers':{'assessment':'Not individually adjudicated;dependencies must not be assumed to be external-only blockers.','recorded_dependencies':row['external_dependency'],'release_gate_record':'docs/RAPHAEL_FINAL_ACCEPTANCE.json'},
            'verification_evidence':row['test_coverage'],
            'status_scope':'Verification is bounded to the preserved row acceptance/evidence;IN PROGRESS includes historical partial implementation and does not assert an active assigned worker. No scope deferral is inferred.'})
    return {'schema_version':1,'source':'docs/ENGINEERING_LEDGER.json','requirement_count':len(rows),'status_counts':dict(Counter(row['status'] for row in rows)),'clinical_validation':False,'individual_original_source_and_risk_review_complete':False,'requirements':rows}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true');args=parser.parse_args()
    path=ROOT/'docs/PLATFORM_REQUIREMENT_ACCEPTANCE.json';encoded=json.dumps(build(),indent=2,ensure_ascii=False)+'\n'
    if args.check:
        if not path.exists() or path.read_text()!=encoded:raise SystemExit('Platform acceptance view is stale.')
    else:path.write_text(encoded)
    print(json.dumps({'requirements':build()['requirement_count'],'acceptance_view':'current' if args.check else 'written'}))
if __name__=='__main__':main()
