"""Patient-authorized, expiring internal synthetic intake snapshots.

This grants access to an immutable intake snapshot only, not the whole chart.
There is no external facility transport or emergency-services integration.
"""
import json
import re
from datetime import datetime, timedelta, timezone
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StrictBool
from sqlalchemy import text

from database import execute_sqlite_script
from dev_portal import _require_synthetic

SCHEMA = """
CREATE TABLE IF NOT EXISTS dev_patient_reports (
 id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id INTEGER NOT NULL,
 author_id INTEGER NOT NULL, history TEXT NOT NULL, symptoms TEXT NOT NULL,
 contacts TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS dev_care_intakes (
 id INTEGER PRIMARY KEY AUTOINCREMENT, tenant TEXT NOT NULL, patient_id INTEGER NOT NULL,
 owner_id INTEGER NOT NULL, kind TEXT NOT NULL, symptoms TEXT NOT NULL, snapshot TEXT NOT NULL,
 state TEXT NOT NULL DEFAULT 'DRAFT', recipient_id INTEGER, purpose TEXT,
 expires_at TEXT, allow_ai INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL,
 authorized_at TEXT, revoked_at TEXT
);
CREATE TABLE IF NOT EXISTS dev_care_summary_provenance (
 summary_id INTEGER PRIMARY KEY, source_message_id INTEGER NOT NULL,
 source_thread_id INTEGER NOT NULL, generation_id TEXT NOT NULL,
 model TEXT, evidence TEXT NOT NULL, clinician_edited INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS dev_care_summaries (
 id INTEGER PRIMARY KEY AUTOINCREMENT, intake_id INTEGER NOT NULL, version INTEGER NOT NULL,
 content TEXT NOT NULL, followup TEXT NOT NULL, status TEXT NOT NULL,
 approved_by INTEGER, approved_at TEXT, published_note_id INTEGER,
 UNIQUE(intake_id,version)
);
"""


def now():return datetime.now(timezone.utc).isoformat()


def init_care_schema(engine):
    _require_synthetic(engine)
    with engine.begin() as c:execute_sqlite_script(c,SCHEMA)


def authorized_intake(c, identifier, user, require_ai=False):
    row=c.execute(text('SELECT * FROM dev_care_intakes WHERE id=:id AND tenant=:t'),{'id':identifier,'t':user['tenant']}).mappings().first()
    if not row:raise HTTPException(404,'Intake not found.')
    patient_owner=user['role']=='patient' and row['owner_id']==user['id']
    receiver=(user['role']=='clinician' and row['recipient_id']==user['id'] and row['state']=='AUTHORIZED' and row['expires_at'] and row['expires_at']>now())
    if not (patient_owner or receiver) or (require_ai and (not receiver or not row['allow_ai'])):
        raise HTTPException(404,'Intake is unavailable or consent is no longer active.')
    result=dict(row);result['snapshot']=json.loads(result['snapshot'])
    result['effective_access']='EXPIRED' if row['state']=='AUTHORIZED' and row['expires_at']<=now() else row['state']
    return result


class Input(BaseModel):model_config=ConfigDict(extra='forbid',str_strip_whitespace=True)
class Contact(Input):
    name:str=Field(min_length=1,max_length=200)
    relationship:str=Field(min_length=1,max_length=100)
    phone:str=Field(min_length=3,max_length=64)
class Report(Input):
    history:str=Field(default='',max_length=10000)
    symptoms:str=Field(default='',max_length=10000)
    contacts:list[Contact]=Field(default_factory=list,max_length=5)
class IntakeIn(Input):
    kind:Literal['pre_visit','urgent_care','emergency_information']
    symptoms:str=Field(min_length=1,max_length=10000)
class Share(Input):
    recipient_id:int=Field(gt=0)
    purpose:Literal['pre_visit_review','urgent_care_review','emergency_information_review']
    expires_in_minutes:int=Field(ge=1,le=10080)
    allow_ai:StrictBool=False
class Summary(Input):
    source_message_id:int | None=Field(default=None,gt=0)
    expected_version:int=Field(ge=0)
    content:str=Field(min_length=1,max_length=20000)
    followup:str=Field(min_length=1,max_length=10000)
class Approval(Input):expected_version:int=Field(gt=0)


def install_care(app,ctx):
    engine=ctx.engine
    router=APIRouter(prefix='/api/dev/care',dependencies=[Depends(ctx.same_origin_json)])
    def patient_user(user=Depends(ctx.user)):
        if user['role']!='patient':raise HTTPException(403,'Patient account required.')
        return user
    def packet(c,patient_id,user):
        patient=ctx.patient_for(c,patient_id,user)
        entries=[dict(r) for r in c.execute(text("SELECT id,kind,label,data,recorded_at,status,revision FROM dev_chart_entries WHERE patient_id=:id AND status!='draft' ORDER BY recorded_at DESC,id DESC LIMIT 100"),{'id':patient_id}).mappings()]
        for entry in entries:entry['data']=json.loads(entry['data']);entry['provenance']={'type':'stored_chart_entry','id':entry['id'],'recorded_at':entry['recorded_at'],'revision':entry['revision']}
        report=c.execute(text('SELECT * FROM dev_patient_reports WHERE patient_id=:id ORDER BY id DESC LIMIT 1'),{'id':patient_id}).mappings().first()
        notes=[dict(r) for r in c.execute(text("SELECT id,content,approved_at FROM dev_notes WHERE patient_id=:id AND status='approved' ORDER BY id DESC LIMIT 10"),{'id':patient_id}).mappings()]
        return {'patient':{'id':patient['id'],'name':patient['name'],'dob':patient['dob'],'mrn':patient['mrn'],'identity_status':'synthetic_linked_account_not_verified_identity'},'chart_entries':entries,'approved_notes':notes,
                'patient_report':dict(report,contacts=json.loads(report['contacts']),provenance='patient_reported_not_clinically_verified') if report else None,
                'generated_at':now(),'scope':'immutable_internal_intake_snapshot','limitations':'Latest 100 non-draft chart entries and 10 approved notes. Not a complete medical record or emergency triage. No offline copy or external facility delivery.'}
    @router.get('/clinicians')
    def clinicians(user=Depends(patient_user)):
        with engine.connect() as c:return [dict(r) for r in c.execute(text("SELECT id,username FROM dev_users WHERE tenant=:t AND role='clinician' AND active=1 ORDER BY username"),{'t':user['tenant']}).mappings()]
    @router.get('/patients/{patient_id}/emergency')
    def emergency(patient_id:int,user=Depends(patient_user)):
        with engine.begin() as c:
            result=packet(c,patient_id,user);ctx.audit(c,user,'read_emergency_information','patient',patient_id);return result
    @router.post('/patients/{patient_id}/reports')
    def report(patient_id:int,body:Report,user=Depends(patient_user)):
        with engine.begin() as c:
            ctx.patient_for(c,patient_id,user)
            identifier=c.execute(text('INSERT INTO dev_patient_reports(patient_id,author_id,history,symptoms,contacts,created_at) VALUES (:p,:u,:h,:s,:v,:at)'),{'p':patient_id,'u':user['id'],'h':body.history,'s':body.symptoms,'v':json.dumps([v.model_dump() for v in body.contacts]),'at':now()}).lastrowid
            ctx.audit(c,user,'append_patient_report','patient_report',identifier);return {'id':identifier,'provenance':'patient_reported_not_clinically_verified'}
    @router.post('/patients/{patient_id}/intakes')
    def create(patient_id:int,body:IntakeIn,user=Depends(patient_user)):
        with engine.begin() as c:
            snapshot=packet(c,patient_id,user)
            identifier=c.execute(text("INSERT INTO dev_care_intakes(tenant,patient_id,owner_id,kind,symptoms,snapshot,created_at) VALUES (:t,:p,:u,:k,:s,:v,:at)"),{'t':user['tenant'],'p':patient_id,'u':user['id'],'k':body.kind,'s':body.symptoms,'v':json.dumps(snapshot),'at':now()}).lastrowid
            ctx.audit(c,user,'draft_care_intake','care_intake',identifier);return authorized_intake(c,identifier,user)
    @router.get('/intakes')
    def listing(user=Depends(ctx.user)):
        if user['role'] not in ('patient','clinician'):raise HTTPException(403,'Patient or clinician role required.')
        with engine.begin() as c:
            where="owner_id=:u" if user['role']=='patient' else "recipient_id=:u AND state='AUTHORIZED' AND expires_at>:at"
            result=[dict(r) for r in c.execute(text('SELECT id,kind,state,patient_id,created_at,expires_at FROM dev_care_intakes WHERE tenant=:t AND '+where+' ORDER BY id DESC'),{'t':user['tenant'],'u':user['id'],'at':now()}).mappings()]
            ctx.audit(c,user,'list_care_intakes','care_intake');return result
    @router.get('/intakes/{identifier}')
    def read(identifier:int,user=Depends(ctx.user)):
        with engine.begin() as c:
            record=authorized_intake(c,identifier,user)
            clause=" AND status='approved'" if user['role']=='patient' else ''
            record['summaries']=[dict(r) for r in c.execute(text('SELECT * FROM dev_care_summaries WHERE intake_id=:id'+clause+' ORDER BY version'),{'id':identifier}).mappings()]
            for summary in record['summaries']:
                provenance=c.execute(text('SELECT * FROM dev_care_summary_provenance WHERE summary_id=:id'),{'id':summary['id']}).mappings().first()
                summary['ai_provenance']=dict(provenance,evidence=json.loads(provenance['evidence']),citation_check='original_response_retrieved_id_membership_only_not_factuality') if provenance else None
            record['transport_status']='not_transmitted_externally';ctx.audit(c,user,'read_care_intake','care_intake',identifier);return record
    @router.post('/intakes/{identifier}/authorize')
    def authorize(identifier:int,body:Share,user=Depends(patient_user)):
        with engine.begin() as c:
            c.exec_driver_sql('BEGIN IMMEDIATE');record=authorized_intake(c,identifier,user)
            if record['state']!='DRAFT':raise HTTPException(409,'Create a new intake snapshot to grant new sharing consent.')
            recipient=c.execute(text("SELECT id FROM dev_users WHERE id=:id AND tenant=:t AND role='clinician' AND active=1"),{'id':body.recipient_id,'t':user['tenant']}).scalar()
            if not recipient:raise HTTPException(400,'Choose an active participating synthetic clinician in this organization.')
            expected={'pre_visit':'pre_visit_review','urgent_care':'urgent_care_review','emergency_information':'emergency_information_review'}
            if expected[record['kind']]!=body.purpose:raise HTTPException(422,'Sharing purpose must match the intake workflow.')
            expiry=(datetime.now(timezone.utc)+timedelta(minutes=body.expires_in_minutes)).isoformat()
            c.execute(text("UPDATE dev_care_intakes SET state='AUTHORIZED',recipient_id=:r,purpose=:p,expires_at=:e,allow_ai=:a,authorized_at=:at WHERE id=:id"),{'r':recipient,'p':body.purpose,'e':expiry,'a':int(body.allow_ai),'at':now(),'id':identifier})
            ctx.audit(c,user,'authorize_care_snapshot','care_intake',identifier);return {'state':'AUTHORIZED','expires_at':expiry,'transport_status':'not_transmitted_externally','scope':'intake_snapshot_only'}
    @router.post('/intakes/{identifier}/revoke')
    def revoke(identifier:int,user=Depends(patient_user)):
        with engine.begin() as c:
            authorized_intake(c,identifier,user);c.execute(text("UPDATE dev_care_intakes SET state='REVOKED',revoked_at=:at WHERE id=:id"),{'at':now(),'id':identifier});ctx.audit(c,user,'revoke_care_snapshot','care_intake',identifier)
        return {'state':'REVOKED','future_receiver_access':'denied'}
    @router.post('/intakes/{identifier}/summary')
    def summary(identifier:int,body:Summary,user=Depends(ctx.clinician)):
        with engine.begin() as c:
            c.exec_driver_sql('BEGIN IMMEDIATE');authorized_intake(c,identifier,user)
            current=c.execute(text('SELECT version,status,id FROM dev_care_summaries WHERE intake_id=:id ORDER BY version DESC LIMIT 1'),{'id':identifier}).first()
            version=current[0] if current else 0
            if body.expected_version!=version or (current and current[1]=='approved'):raise HTTPException(409,'Summary changed or is already approved.')
            provenance=None
            if body.source_message_id:
                authorized_intake(c,identifier,user,require_ai=True)
                source=c.execute(text("SELECT m.*,t.intake_id FROM dev_ai_messages m JOIN dev_ai_threads t ON t.id=m.thread_id WHERE m.id=:id AND t.tenant=:t AND t.owner_id=:u AND t.intake_id=:i AND m.role='assistant' AND m.status='complete'"),{'id':body.source_message_id,'t':user['tenant'],'u':user['id'],'i':identifier}).mappings().first()
                if not source:raise HTTPException(404,'Choose a completed response from your authorized intake conversation.')
                stored=c.execute(text('SELECT evidence FROM dev_ai_generation_evidence WHERE generation_id=:g'),{'g':source['generation_id']}).scalar()
                evidence=json.loads(stored) if stored else []
                allowed={item['chunk_id'] for item in evidence}
                citations=re.findall(r'\[([^\[\]]+)\]',source['content'])
                if not citations or any(item not in allowed for item in citations):raise HTTPException(409,'Stored source evidence is unavailable or citation membership is invalid.')
                provenance={'source_message_id':source['id'],'source_thread_id':source['thread_id'],'generation_id':source['generation_id'],'model':source['model'],'evidence':stored,'clinician_edited':int(body.content!=source['content'])}
            elif current:
                previous=c.execute(text('SELECT * FROM dev_care_summary_provenance WHERE summary_id=:id'),{'id':current[2]}).mappings().first()
                if previous:provenance={**dict(previous),'clinician_edited':1}
            summary_id=c.execute(text("INSERT INTO dev_care_summaries(intake_id,version,content,followup,status) VALUES (:id,:v,:c,:f,'draft')"),{'id':identifier,'v':version+1,'c':body.content,'f':body.followup}).lastrowid
            if provenance:
                c.execute(text('INSERT INTO dev_care_summary_provenance(summary_id,source_message_id,source_thread_id,generation_id,model,evidence,clinician_edited) VALUES (:id,:source_message_id,:source_thread_id,:generation_id,:model,:evidence,:clinician_edited)'),{**provenance,'id':summary_id})
                ctx.audit(c,user,'link_ai_draft_to_care_summary','care_intake',identifier)
            ctx.audit(c,user,'draft_care_summary','care_intake',identifier);return {'version':version+1,'status':'draft'}
    @router.post('/intakes/{identifier}/approve')
    def approve(identifier:int,body:Approval,user=Depends(ctx.clinician)):
        with engine.begin() as c:
            c.exec_driver_sql('BEGIN IMMEDIATE');record=authorized_intake(c,identifier,user)
            current=c.execute(text('SELECT * FROM dev_care_summaries WHERE intake_id=:id ORDER BY version DESC LIMIT 1'),{'id':identifier}).mappings().first()
            if not current or current['version']!=body.expected_version or current['status']!='draft':raise HTTPException(409,'Only the current draft may be approved.')
            at=now();note=c.execute(text("INSERT INTO dev_notes(patient_id,author_id,content,status,created_at,approved_by,approved_at) VALUES (:p,:u,:v,'approved',:at,:u,:at)"),{'p':record['patient_id'],'u':user['id'],'v':current['content']+'\nFollow-up: '+current['followup'],'at':at}).lastrowid
            c.execute(text("UPDATE dev_care_summaries SET status='approved',approved_by=:u,approved_at=:at,published_note_id=:p WHERE id=:id"),{'u':user['id'],'at':at,'p':note,'id':current['id']});ctx.audit(c,user,'approve_care_summary','care_intake',identifier)
            return {'status':'approved','version':current['version'],'patient_note_id':note}
    @router.get('/intakes/{identifier}/audit')
    def audit(identifier:int,user=Depends(ctx.user)):
        with engine.begin() as c:
            authorized_intake(c,identifier,user)
            return [dict(r) for r in c.execute(text("SELECT event,created_at,actor_id FROM dev_audit WHERE tenant=:t AND resource_type='care_intake' AND resource_id=:id ORDER BY id"),{'t':user['tenant'],'id':identifier}).mappings()]
    app.include_router(router)
