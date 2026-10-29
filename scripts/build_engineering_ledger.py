"""Derive the permanent machine ledger without rewriting requirement descriptions."""
import argparse
from collections import Counter
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
STATES={'Implemented and verified':'VERIFIED','Implemented but unverified':'IMPLEMENTED_UNVERIFIED','Partially implemented':'PARTIAL','Not implemented':'NOT_IMPLEMENTED','awaiting_validation':'REQUIRES_CLINICAL_VALIDATION','verified':'VERIFIED','partial':'PARTIAL','missing':'NOT_IMPLEMENTED','blocked':'BLOCKED_EXTERNAL','untested':'IMPLEMENTED_UNVERIFIED'}

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
    return {'schema_version':1,'source':'docs/REQUIREMENTS_MATRIX.md','state_counts':dict(Counter(row['state'] for row in rows)),'requirement_count':len(rows),'clinical_validation':False,'requirements':rows}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
    output=ROOT/'docs/ENGINEERING_LEDGER.json';encoded=json.dumps(build(),indent=2,ensure_ascii=False)+'\n'
    if args.check:
        if not output.exists() or output.read_text()!=encoded:raise SystemExit('Ledger is stale;run python -m scripts.build_engineering_ledger')
    else:output.write_text(encoded)
    print(json.dumps({'requirements':build()['requirement_count'],'ledger':'current' if args.check else 'written'}))
if __name__=='__main__':main()
