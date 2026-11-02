"""Derive the permanent machine ledger without rewriting requirement descriptions."""
import argparse
from collections import Counter
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
STATES={'Implemented and verified':'VERIFIED','Implemented but unverified':'IMPLEMENTED_UNVERIFIED','Partially implemented':'PARTIAL','Not implemented':'NOT_IMPLEMENTED','awaiting_validation':'REQUIRES_CLINICAL_VALIDATION','verified':'VERIFIED','partial':'PARTIAL','missing':'NOT_IMPLEMENTED','blocked':'BLOCKED_EXTERNAL','untested':'IMPLEMENTED_UNVERIFIED'}

VERIFICATION={'VERIFIED':'VERIFIED IN SYNTHETIC TESTING','IMPLEMENTED_UNVERIFIED':'IMPLEMENTED BUT UNVERIFIED','PARTIAL':'IN PROGRESS','NOT_IMPLEMENTED':'NOT STARTED','REQUIRES_CLINICAL_VALIDATION':'IMPLEMENTED BUT UNVERIFIED','BLOCKED_EXTERNAL':'EXTERNALLY BLOCKED'}

def build():
    rows=[];seen=set()
    for line in (ROOT/'docs/REQUIREMENTS_MATRIX.md').read_text().splitlines():
        if not line.startswith('|'):continue
        cells=[cell.strip() for cell in line.strip().strip('|').split('|')]
        if len(cells)!=10 or cells[3] not in STATES:continue
        identifier,description,priority,state,code,dependency,tests,criteria,remaining,milestone=cells
        if identifier in seen:raise ValueError('Duplicate requirement identifier: '+identifier)
        seen.add(identifier)
        rows.append(dict(requirement_id=identifier,description=description,state=STATES[state],code_location=code,test_coverage=tests,external_dependency=dependency,remaining_work=remaining,acceptance_criteria=criteria,next_implementation_action=remaining,priority=priority,milestone=milestone,verification_scope='Only the bounded behavior and evidence in this row;historical verification is not a current full-product or clinical-validation claim.'))
    if not rows:raise ValueError('No requirements found.')
    for row in rows:
        row.update(verification_status=VERIFICATION[row['state']],original_source={'canonical_matrix':'docs/REQUIREMENTS_MATRIX.md','historical_upload_attribution':'Not individually adjudicated;preserved master directives remain authoritative.'},functional_area={'requirement_family':row['requirement_id'].split('-')[0],'individual_domain_review':'UNREVIEWED'},risk_classification='UNREVIEWED:qualified intended-use assessment required.',implementation_files=row['code_location'],api_contracts='Inspect implementation references;individual contract inventory not yet adjudicated.',ui_routes='Inspect implementation references;individual route inventory not yet adjudicated.',data_contracts='Inspect implementation references;individual data inventory not yet adjudicated.',verification_artifacts=row['test_coverage'],clinical_validation_requirements='Independent intended-use,population,safety and clinician review required;not established by software tests.',release_gate='docs/RAPHAEL_FINAL_ACCEPTANCE.json;docs/DATABASE_STORAGE_CHECKPOINT.md',external_blockers=row['external_dependency'])
    return {'schema_version':2,'source':'docs/REQUIREMENTS_MATRIX.md','state_counts':dict(Counter(row['state'] for row in rows)),'requirement_count':len(rows),'verification_status_counts':dict(Counter(row['verification_status'] for row in rows)),'clinical_validation':False,'individual_source_contract_and_risk_adjudication_complete':False,'requirements':rows}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
    output=ROOT/'docs/ENGINEERING_LEDGER.json';encoded=json.dumps(build(),indent=2,ensure_ascii=False)+'\n'
    if args.check:
        if not output.exists() or output.read_text()!=encoded:raise SystemExit('Ledger is stale;run python -m scripts.build_engineering_ledger')
    else:output.write_text(encoded)
    print(json.dumps({'requirements':build()['requirement_count'],'ledger':'current' if args.check else 'written'}))
if __name__=='__main__':main()
