"""Owner-directed product feedback;no training,clinical correction or deployment."""
import hashlib
import json
from pathlib import Path,PurePosixPath
import re
from typing import Literal
from uuid import UUID
from fastapi import APIRouter,Depends,HTTPException,Query,Request
from pydantic import BaseModel,ConfigDict,Field,StrictBool,StrictInt
from sqlalchemy import text
from feedback.store import ensure_portal_feedback_tables as init_feedback_schema
from dev_account import session_active
from dev_intelligence import now

class Input(BaseModel):model_config=ConfigDict(extra='forbid',str_strip_whitespace=True)
class Report(Input):
    request_id:UUID
    category:Literal['bug','missing_feature','usability','documentation','clinical_information','missing_reference','accessibility','performance','other']
    impact:Literal['routine','blocking','possible_safety_concern']
    title:str=Field(min_length=5,max_length=200)
    detail:str=Field(min_length=10,max_length=3000)
    synthetic_only_ack:StrictBool
    no_sensitive_data_ack:StrictBool
class Action(Input):
    request_id:UUID
    expected_revision:StrictInt=Field(gt=0)
    action:Literal['triage','reject','create_task','start_task','assign_task','resolve','note']
    note:str=Field(min_length=5,max_length=2000)
    priority:Literal['P0','P1','P2']='P1'
    requirement_id:str|None=Field(default=None,max_length=60)
    assignee_id:StrictInt|None=Field(default=None,gt=0)
    evidence_reference:str|None=Field(default=None,max_length=500)

def fingerprint(body):return hashlib.sha256(json.dumps(body.model_dump(mode='json'),sort_keys=True).encode()).hexdigest()
def catalog():
    try:
        rows=json.loads((Path(__file__).resolve().parent/'docs/ENGINEERING_LEDGER.json').read_text())['requirements']
        return {row['requirement_id']:{'id':row['requirement_id'],'description':row['description'],'state':row['state']} for row in rows}
    except (OSError,ValueError,KeyError,TypeError):raise HTTPException(503,'Requirements catalog unavailable;no task link was guessed.') from None

def evidence_pointer(value):
    if not value:return False
    return bool(re.fullmatch(r'[a-f0-9]{40}|https://github\.com/AIGeniusAlien/Raphael-AI/actions/runs/[0-9]+',value) or (value.startswith('docs/') and '..' not in PurePosixPath(value).parts and re.fullmatch(r'docs/[A-Za-z0-9_./-]+',value)))
def sensitive(value):return bool(re.search(r'\bsk-[A-Za-z0-9_-]{12,}|\b(?:password|api[_ -]?key|access[_ -]?token)\s*[:=]\s*\S+',value,re.I))

def install_feedback(app,ctx):
    router=APIRouter(prefix='/api/dev/feedback',dependencies=[Depends(ctx.same_origin_json)])
    def admin(user=Depends(ctx.user)):
        if user['role']!='org_admin':raise HTTPException(403,'Organization product administrator required.')
        return user
    def active(c,request,user):
        if not session_active(c,request,user):raise HTTPException(401,'Session revoked;feedback was not changed.')
    def report_for(c,identifier,user):
        row=c.execute(text('SELECT * FROM dev_product_feedback WHERE id=:id AND tenant=:t'),{'id':identifier,'t':user['tenant']}).mappings().first()
        if not row or (user['role']!='org_admin' and row['owner_id']!=user['id']):raise HTTPException(404,'Feedback not found.')
        return dict(row)
    def result(c,row,user,history_limit=20):
        task=c.execute(text('SELECT * FROM dev_feedback_tasks WHERE feedback_id=:id'),{'id':row['id']}).mappings().first()
        actions=c.execute(text('SELECT action,note,result_revision,created_at FROM dev_feedback_actions WHERE feedback_id=:id ORDER BY id LIMIT :limit'),{'id':row['id'],'limit':history_limit+1}).mappings().all()
        if user['role']!='org_admin':
            actions=[{k:v for k,v in item.items() if k!='note'} for item in actions]
            if task:task={k:v for k,v in task.items() if k not in ('assigned_to','created_at','updated_at')}
        return {**{k:v for k,v in row.items() if k not in ('request_id','payload_hash')},'task':dict(task) if task else None,'actions':[dict(a) for a in actions[:history_limit]],'actions_truncated':len(actions)>history_limit,'clinical_validation':False,'external_delivery':False,'evidence_pointer_verified':False}
    @router.post('/reports')
    def submit(body:Report,request:Request,user=Depends(ctx.user)):
        if not body.synthetic_only_ack or not body.no_sensitive_data_ack:raise HTTPException(422,'Confirm synthetic,de-identified product feedback without patient records or credentials.')
        if sensitive(body.title+' '+body.detail):raise HTTPException(422,'Remove credential-like text. Do not submit patient records or secrets.')
        digest=fingerprint(body)
        with ctx.engine.begin() as c:
            c.exec_driver_sql('BEGIN IMMEDIATE');active(c,request,user)
            old=c.execute(text('SELECT * FROM dev_product_feedback WHERE tenant=:t AND owner_id=:u AND request_id=:r'),{'t':user['tenant'],'u':user['id'],'r':str(body.request_id)}).mappings().first()
            if old:
                if old['payload_hash']!=digest:raise HTTPException(409,'Submission identifier was reused with different content.')
                return result(c,dict(old),user)
            if c.execute(text('SELECT count(*) FROM dev_product_feedback WHERE tenant=:t AND owner_id=:u'),{'t':user['tenant'],'u':user['id']}).scalar()>=500:raise HTTPException(413,'Product feedback limit reached;no partial submission was retained.')
            identifier=c.execute(text("INSERT INTO dev_product_feedback(tenant,owner_id,request_id,payload_hash,category,impact,title,detail,state,revision,created_at,updated_at) VALUES(:t,:u,:r,:h,:category,:impact,:title,:detail,'submitted',1,:at,:at)"),{'t':user['tenant'],'u':user['id'],'r':str(body.request_id),'h':digest,'category':body.category,'impact':body.impact,'title':body.title,'detail':body.detail,'at':now()}).lastrowid
            ctx.audit(c,user,'submit_product_feedback','product_feedback',identifier)
            return result(c,report_for(c,identifier,user),user)
    @router.get('/reports')
    def reports(after:int=Query(0,ge=0),user=Depends(ctx.user)):
        with ctx.engine.begin() as c:
            rows=c.execute(text("SELECT * FROM dev_product_feedback WHERE tenant=:t AND id>:after AND (:admin=1 OR owner_id=:u) ORDER BY id LIMIT 101"),{'t':user['tenant'],'after':after,'admin':int(user['role']=='org_admin'),'u':user['id']}).mappings().all()
            return {'items':[result(c,dict(row),user) for row in rows[:100]],'next_after':rows[99]['id'] if len(rows)>100 else None,'scope':'Internal product feedback only;not urgent care,clinical messaging,automatic training or deployment. Administrator evidence pointers are recorded,not automatically verified.'}
    @router.get('/reports/{identifier}')
    def report(identifier:int,user=Depends(ctx.user)):
        with ctx.engine.begin() as c:return result(c,report_for(c,identifier,user),user,history_limit=200)
    @router.get('/requirements')
    def requirements(q:str=Query('',max_length=200),user=Depends(admin)):
        matches=[row for row in catalog().values() if q.casefold() in (row['id']+' '+row['description']).casefold()]
        return {'items':matches[:50],'truncated':len(matches)>50,'scope':'Catalog lookup does not change requirement acceptance.'}
    @router.get('/assignees')
    def assignees(user=Depends(admin)):
        with ctx.engine.connect() as c:return [dict(row) for row in c.execute(text("SELECT id,username FROM dev_users WHERE tenant=:t AND role='org_admin' AND active=1 ORDER BY id LIMIT 200"),{'t':user['tenant']}).mappings()]
    @router.post('/reports/{identifier}/actions')
    def act(identifier:int,body:Action,request:Request,user=Depends(admin)):
        if sensitive(body.note):raise HTTPException(422,'Remove credential-like text;no secrets belong in feedback notes.')
        digest=fingerprint(body)
        with ctx.engine.begin() as c:
            c.exec_driver_sql('BEGIN IMMEDIATE');active(c,request,user);row=report_for(c,identifier,user)
            replay=c.execute(text('SELECT actor_id,payload_hash,result_revision FROM dev_feedback_actions WHERE feedback_id=:id AND request_id=:r'),{'id':identifier,'r':str(body.request_id)}).mappings().first()
            if replay:
                if replay['actor_id']!=user['id'] or replay['payload_hash']!=digest:raise HTTPException(409,'Action identifier conflicts with an earlier operation.')
                return {**result(c,row,user),'operation_replayed':True,'original_revision':replay['result_revision']}
            if row['revision']!=body.expected_revision:raise HTTPException(409,'Feedback changed;refresh and review the current version.')
            count=c.execute(text('SELECT count(*) FROM dev_feedback_actions WHERE feedback_id=:id'),{'id':identifier}).scalar()
            if count>=200:raise HTTPException(413,'Feedback action history exceeds development limits;no partial change was applied.')
            task=c.execute(text('SELECT * FROM dev_feedback_tasks WHERE feedback_id=:id'),{'id':identifier}).mappings().first()
            state=row['state'];target=state
            if body.action=='triage':
                if state not in ('submitted','triaged'):raise HTTPException(409,'Only submitted or triaged feedback can be triaged.')
                target='triaged'
            elif body.action=='reject':
                if state not in ('submitted','triaged'):raise HTTPException(409,'Started tasks cannot be silently rejected.')
                target='rejected'
            elif body.action=='create_task':
                if state!='triaged' or task:raise HTTPException(409,'Triage is required before creating one task.')
                if body.requirement_id not in catalog():raise HTTPException(422,'Select an existing requirement identifier;no requirement was invented.')
                assignee=body.assignee_id or user['id']
                if not c.execute(text("SELECT 1 FROM dev_users WHERE id=:id AND tenant=:t AND role='org_admin' AND active=1"),{'id':assignee,'t':user['tenant']}).first():raise HTTPException(404,'Active product administrator assignee not found.')
                c.execute(text("INSERT INTO dev_feedback_tasks(feedback_id,requirement_id,assigned_to,state,created_at,updated_at) VALUES(:id,:r,:u,'planned',:at,:at)"),{'id':identifier,'r':body.requirement_id,'u':assignee,'at':now()});target='task_created'
            elif body.action=='start_task':
                if state!='task_created' or not task:raise HTTPException(409,'A planned task is required.')
                c.execute(text("UPDATE dev_feedback_tasks SET state='in_progress',updated_at=:at WHERE feedback_id=:id"),{'at':now(),'id':identifier});target='in_progress'
            elif body.action=='assign_task':
                if state not in ('task_created','in_progress') or not task or not body.assignee_id:raise HTTPException(409,'An open task and explicit assignee are required.')
                if not c.execute(text("SELECT 1 FROM dev_users WHERE id=:id AND tenant=:t AND role='org_admin' AND active=1"),{'id':body.assignee_id,'t':user['tenant']}).first():raise HTTPException(404,'Active product administrator assignee not found.')
                c.execute(text('UPDATE dev_feedback_tasks SET assigned_to=:u,updated_at=:at WHERE feedback_id=:id'),{'u':body.assignee_id,'at':now(),'id':identifier})
            elif body.action=='resolve':
                if state!='in_progress' or not task:raise HTTPException(409,'An in-progress task is required before technical resolution.')
                if row['category']=='clinical_information' or row['impact']=='possible_safety_concern':raise HTTPException(409,'Clinical safety feedback requires a qualified-review workflow,which is not enabled. No medical correction was approved.')
                if not evidence_pointer(body.evidence_reference):raise HTTPException(422,'Record a commit SHA,repository CI run URL or docs reference. This pointer is not automatic proof of a passed test.')
                c.execute(text("UPDATE dev_feedback_tasks SET state='done',evidence_reference=:e,updated_at=:at WHERE feedback_id=:id"),{'e':body.evidence_reference,'at':now(),'id':identifier});target='resolved'
            elif state in ('resolved','rejected'):raise HTTPException(409,'Closed feedback is immutable;submit a new linked report for further work.')
            revision=row['revision']+1
            c.execute(text('UPDATE dev_product_feedback SET state=:s,priority=:p,revision=:v,updated_at=:at WHERE id=:id'),{'s':target,'p':body.priority if body.action=='triage' else row['priority'],'v':revision,'at':now(),'id':identifier})
            c.execute(text('INSERT INTO dev_feedback_actions(feedback_id,actor_id,request_id,payload_hash,action,note,result_revision,created_at) VALUES(:id,:u,:r,:h,:a,:n,:v,:at)'),{'id':identifier,'u':user['id'],'r':str(body.request_id),'h':digest,'a':body.action,'n':body.note,'v':revision,'at':now()})
            ctx.audit(c,user,'product_feedback_'+body.action,'product_feedback',identifier)
            return result(c,report_for(c,identifier,user),user)
    app.include_router(router)
