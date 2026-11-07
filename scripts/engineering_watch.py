"""Bounded read-only engineering audit;never implements,merges,deploys or contacts patients.

A one-shot result does not establish a registered persistent scheduler. Reports
contain repository/test metadata only;no clinical database or provider is opened.
"""
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import re
import subprocess

REPOSITORY='AIGeniusAlien/Raphael-AI'
STATES={'VERIFIED','PARTIAL','IMPLEMENTED_UNVERIFIED','NOT_IMPLEMENTED','REQUIRES_CLINICAL_VALIDATION','BLOCKED_EXTERNAL'}
CONCLUSIONS={'success','failure','cancelled','timed_out','action_required','neutral','skipped','stale','startup_failure',None}


def read_json(path,limit=10*1024*1024):
    if not path.is_file() or path.stat().st_size>limit:raise ValueError('Missing or oversized engineering metadata.')
    return json.loads(path.read_text())


def summarize_runs(payload,head):
    runs=payload.get('workflow_runs')
    if not isinstance(runs,list) or len(runs)>20:raise ValueError('Invalid bounded GitHub run inventory.')
    selected=[]
    for row in runs:
        if row.get('head_sha')!=head:continue
        if type(row.get('id')) is not int or row['id']<=0 or row.get('status') not in {'queued','in_progress','completed','waiting','requested','pending'} or row.get('conclusion') not in CONCLUSIONS:raise ValueError('Invalid GitHub run metadata.')
        selected.append({'run_id':row['id'],'status':row['status'],'conclusion':row.get('conclusion'),'url':f"https://github.com/{REPOSITORY}/actions/runs/{row['id']}"})
    return selected


def github_runs(head):
    if not re.fullmatch('[0-9a-f]{40}',head):raise ValueError('Exact source commit required.')
    try:
        result=subprocess.run(['gh','api','--hostname','github.com',f'repos/{REPOSITORY}/actions/runs?head_sha={head}&per_page=20'],capture_output=True,text=True,timeout=25,check=True)
        if len(result.stdout)>2*1024*1024:raise ValueError()
        return summarize_runs(json.loads(result.stdout),head)
    except (OSError,ValueError,subprocess.SubprocessError):raise ValueError('GitHub metadata unavailable;check scoped actions-read access and network. No retry or credential value disclosed.') from None


def audit(root,*,head,clean,runs=None):
    if not re.fullmatch('[0-9a-f]{40}',head):raise ValueError('Exact source commit required.')
    ledger=read_json(root/'docs/ENGINEERING_LEDGER.json');rows=ledger.get('requirements');seen=set()
    if not isinstance(rows,list) or not rows or len(rows)>10000:raise ValueError('Bounded canonical ledger required.')
    for row in rows:
        identifier=row.get('requirement_id')
        if not isinstance(identifier,str) or not re.fullmatch('[A-Z0-9]+-[0-9]+',identifier) or identifier in seen or row.get('state') not in STATES:raise ValueError('Canonical requirement identity/state invalid.')
        seen.add(identifier)
    if ledger.get('requirement_count')!=len(rows):raise ValueError('Canonical requirement count mismatch.')
    acceptance=read_json(root/'docs/RAPHAEL_FINAL_ACCEPTANCE.json')
    gates=acceptance.get('release_decisions',{})
    if not isinstance(gates,dict) or any(not isinstance(k,str) or v not in {'PASS','FAIL','BLOCKED'} for k,v in gates.items()):raise ValueError('Invalid recorded release gates.')
    unfinished=sorted((r for r in rows if r['state']!='VERIFIED'),key=lambda r:(r.get('priority','UNREVIEWED'),r['requirement_id']))
    return {'schema_version':1,'generated_at':datetime.now(timezone.utc).isoformat(),'repository':REPOSITORY,'head_sha':head,'working_tree_clean':bool(clean),'requirement_count':len(rows),'state_counts':dict(Counter(r['state'] for r in rows)),'proposed_work':[{'requirement_id':r['requirement_id'],'state':r['state'],'priority':r.get('priority','UNREVIEWED')} for r in unfinished[:20]],'recorded_release_gates':gates,'gate_scope':'Recorded requirements evidence;not independently revalidated by this metadata audit.','github_checks':runs if runs is not None else [],'github_metadata_checked':runs is not None,'persistent_execution':'not_proven_by_one_shot','automatic_code_changes':False,'clinical_database_accessed':False,'provider_requests':False,'merge_or_deployment':False}


def write_report(path,report):
    # Never replace an existing report or follow its symlink. Artifact metadata
    # remains private until the caller intentionally publishes the CI artifact.
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0),0o600)
    with os.fdopen(fd,'w') as output:json.dump(report,output,indent=2);output.write('\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--github',action='store_true');parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    try:
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True,timeout=10).strip()
        clean=not subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True,timeout=10)
        report=audit(root,head=head,clean=clean,runs=github_runs(head) if args.github else None)
        write_report(args.output,report)
        print(json.dumps({'status':'metadata_audited','head_sha':head,'requirements':report['requirement_count'],'github_metadata_checked':report['github_metadata_checked'],'persistent_execution':report['persistent_execution']}));return 0
    except (OSError,ValueError,subprocess.SubprocessError):
        print('Engineering audit refused;verify bounded canonical metadata,GitHub read access and a NEW output path. No credential values disclosed.');return 2
if __name__=='__main__':raise SystemExit(main())
