"""Authenticated development account controls; deletion requests are not erasure."""
import hashlib
import json
import uuid
from datetime import datetime,timezone
from zoneinfo import ZoneInfo,ZoneInfoNotFoundError
from fastapi import APIRouter,Depends,HTTPException,Request,Response
from pydantic import BaseModel,ConfigDict,Field,StrictBool,field_validator
from sqlalchemy import text
from database import execute_sqlite_script
from dev_portal import _require_synthetic,COOKIE_NAME

SCHEMA='''
CREATE TABLE IF NOT EXISTS dev_account_profiles (
 user_id INTEGER PRIMARY KEY REFERENCES dev_users(id), display_name TEXT NOT NULL,
 language TEXT NOT NULL,time_zone TEXT NOT NULL,internal_notifications INTEGER NOT NULL,
 updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS dev_account_deletion_requests (
 id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL REFERENCES dev_users(id),
 tenant TEXT NOT NULL,state TEXT NOT NULL CHECK(state IN ('REQUESTED','ACKNOWLEDGED','CANCELLED')),
 requested_at TEXT NOT NULL,updated_at TEXT NOT NULL,reviewed_by INTEGER,review_note TEXT
);
'''


def now():return datetime.now(timezone.utc).isoformat()
def session_id(value):return str(uuid.uuid5(uuid.NAMESPACE_URL,'raphael-session-display:'+value))
def session_active(c,request,user):
    token=request.cookies.get(COOKIE_NAME,'')
    return bool(token and c.execute(text('SELECT 1 FROM dev_sessions s JOIN dev_users u ON u.id=s.user_id WHERE s.token_hash=:h AND s.user_id=:u AND s.expires_at>:at AND u.active=1'),{'h':hashlib.sha256(token.encode()).hexdigest(),'u':user['id'],'at':now()}).scalar())
def init_account_schema(engine):
    _require_synthetic(engine)
    with engine.begin() as c:execute_sqlite_script(c,SCHEMA)


class Profile(BaseModel):
    model_config=ConfigDict(extra='forbid',str_strip_whitespace=True)
    display_name:str=Field(min_length=1,max_length=100)
    language:str=Field(pattern=r'^[a-z]{2}(?:-[A-Z]{2})?$')
    time_zone:str=Field(max_length=80)
    internal_notifications:StrictBool=True
    @field_validator('time_zone')
    @classmethod
    def valid_zone(cls,value):
        try:ZoneInfo(value)
        except (ZoneInfoNotFoundError,ValueError):raise ValueError('Use a supported IANA time zone.')
        return value
class DeleteRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    confirm_review_request:StrictBool
class Review(BaseModel):
    model_config=ConfigDict(extra='forbid',str_strip_whitespace=True)
    note:str=Field(min_length=1,max_length=1000)


def install_account(app,ctx,patient_ai_exporter=None):
    engine=ctx.engine
    router=APIRouter(prefix='/api/dev/account',dependencies=[Depends(ctx.same_origin_json)])
    @router.get('')
    def account(user=Depends(ctx.user)):
        with engine.begin() as c:
            row=c.execute(text('SELECT display_name,language,time_zone,internal_notifications,updated_at FROM dev_account_profiles WHERE user_id=:u'),{'u':user['id']}).mappings().first()
            profile=dict(row) if row else {'display_name':user['username'],'language':'en','time_zone':'UTC','internal_notifications':True,'updated_at':None}
            return {'identity':user,'profile':profile,'identity_verification':'local_synthetic_account_only','translation':'preference_only_no_clinical_translation','external_notifications':'not_enabled','deletion':'review_request_only_no_erasure_policy_approved'}
    @router.post('/profile')
    def profile(body:Profile,user=Depends(ctx.user)):
        values=body.model_dump()
        with engine.begin() as c:
            c.execute(text('INSERT INTO dev_account_profiles(user_id,display_name,language,time_zone,internal_notifications,updated_at) VALUES (:u,:display_name,:language,:time_zone,:internal_notifications,:at) ON CONFLICT(user_id) DO UPDATE SET display_name=excluded.display_name,language=excluded.language,time_zone=excluded.time_zone,internal_notifications=excluded.internal_notifications,updated_at=excluded.updated_at'),dict(values,u=user['id'],at=now()))
            ctx.audit(c,user,'update_account_preferences','user',user['id'])
        return {'saved':True,'patient_chart_changed':False,'translation_or_external_notifications_enabled':False}
    @router.get('/sessions')
    def sessions(request:Request,user=Depends(ctx.user)):
        current=hashlib.sha256(request.cookies.get(COOKIE_NAME,'').encode()).hexdigest()
        with engine.connect() as c:return [{'id':session_id(row['token_hash']),'expires_at':row['expires_at'],'current':row['token_hash']==current,'device_identity':'not_verified'} for row in c.execute(text('SELECT token_hash,expires_at FROM dev_sessions WHERE user_id=:u AND expires_at>:at'),{'u':user['id'],'at':now()}).mappings()]
    @router.post('/sessions/{identifier}/revoke')
    def revoke_session(identifier:str,response:Response,request:Request,user=Depends(ctx.user)):
        with engine.begin() as c:
            token=next((row[0] for row in c.execute(text('SELECT token_hash FROM dev_sessions WHERE user_id=:u'),{'u':user['id']}) if session_id(row[0])==identifier),None)
            if token is None:raise HTTPException(404,'Session not found.')
            c.execute(text('DELETE FROM dev_sessions WHERE token_hash=:h AND user_id=:u'),{'h':token,'u':user['id']});ctx.audit(c,user,'revoke_account_session','user',user['id'])
        current=hashlib.sha256(request.cookies.get(COOKIE_NAME,'').encode()).hexdigest()==token
        if current:response.delete_cookie(COOKIE_NAME,path='/api/dev',httponly=True,samesite='strict')
        return {'revoked':True,'current_session':current}
    @router.get('/export')
    def export(user=Depends(ctx.user)):
        if user['role']!='patient':raise HTTPException(403,'This export is for your linked patient records only.')
        with engine.begin() as c:
            params={'u':user['id'],'t':user['tenant']}
            def bounded(query):
                rows=[dict(r) for r in c.execute(text(query+' LIMIT 501'),params).mappings()]
                if len(rows)>500:raise HTTPException(413,'Export exceeds development limits;no partial file was produced. Request an audited operator export.')
                return rows
            result={'format':'raphael-synthetic-patient-export-v1','generated_at':now(),'account':user,'scope':'Own linked records;unsigned clinician drafts and file bytes excluded. This is a bounded development export,not a complete legal portability determination.'}
            result['patients']=bounded('SELECT id,name,dob,mrn,created_at FROM dev_patients WHERE patient_user_id=:u AND tenant=:t ORDER BY id')
            result['entries']=bounded("SELECT e.id,e.patient_id,e.kind,e.label,e.data,e.recorded_at,e.status,e.author_id,e.revision,e.approved_at FROM dev_chart_entries e JOIN dev_patients p ON p.id=e.patient_id WHERE p.patient_user_id=:u AND p.tenant=:t AND e.status!='draft' ORDER BY e.id")
            result['approved_notes']=bounded("SELECT n.id,n.patient_id,n.content,n.author_id,n.approved_by,n.approved_at FROM dev_notes n JOIN dev_patients p ON p.id=n.patient_id WHERE p.patient_user_id=:u AND p.tenant=:t AND n.status='approved' ORDER BY n.id")
            result['appointments']=bounded('SELECT a.* FROM dev_chart_appointments a JOIN dev_patients p ON p.id=a.patient_id WHERE p.patient_user_id=:u AND p.tenant=:t ORDER BY a.id')
            result['documents']=bounded('SELECT d.id,d.patient_id,d.title,d.file_type,d.visibility,d.size_bytes,d.sha256,d.created_at FROM dev_chart_documents d JOIN dev_patients p ON p.id=d.patient_id WHERE p.patient_user_id=:u AND p.tenant=:t ORDER BY d.id')
            result['shared_messages']=bounded('SELECT m.id,m.conversation_id,m.sender_id,m.content,m.created_at FROM dev_messages m JOIN dev_conversations v ON v.id=m.conversation_id JOIN dev_patients p ON p.id=v.patient_id WHERE p.patient_user_id=:u AND p.tenant=:t ORDER BY m.id')
            result['patient_reports']=bounded('SELECT r.* FROM dev_patient_reports r JOIN dev_patients p ON p.id=r.patient_id WHERE p.patient_user_id=:u AND p.tenant=:t ORDER BY r.id')
            result['intakes']=bounded('SELECT id,patient_id,kind,symptoms,snapshot,state,recipient_id,purpose,expires_at,allow_ai,created_at,authorized_at,revoked_at FROM dev_care_intakes WHERE owner_id=:u AND tenant=:t ORDER BY id')
            result['approved_summaries']=bounded("SELECT s.* FROM dev_care_summaries s JOIN dev_care_intakes i ON i.id=s.intake_id WHERE i.owner_id=:u AND i.tenant=:t AND s.status='approved' ORDER BY s.id")
            if patient_ai_exporter:result['patient_ai']=patient_ai_exporter(c,user)
            result['file_bytes']='Download permitted files individually through their authenticated document routes.'
            encoded=json.dumps(result).encode()
            if len(encoded)>5*1024*1024:raise HTTPException(413,'Export exceeds5MiB;no partial file was produced.')
            ctx.audit(c,user,'export_own_patient_records','user',user['id'])
            return Response(encoded,media_type='application/json',headers={'Content-Disposition':'attachment; filename="raphael-synthetic-patient-export.json"','Cache-Control':'no-store'})
    @router.get('/deletion-requests')
    def deletion_requests(user=Depends(ctx.user)):
        with engine.connect() as c:
            return [dict(row) for row in c.execute(text('SELECT id,state,requested_at,updated_at,review_note FROM dev_account_deletion_requests WHERE user_id=:u AND tenant=:t ORDER BY id DESC'),{'u':user['id'],'t':user['tenant']}).mappings()]
    @router.post('/deletion-requests')
    def request_deletion(body:DeleteRequest,user=Depends(ctx.user)):
        if not body.confirm_review_request:raise HTTPException(422,'Explicit confirmation required.')
        with engine.begin() as c:
            c.exec_driver_sql('BEGIN IMMEDIATE')
            existing=c.execute(text("SELECT id,state FROM dev_account_deletion_requests WHERE user_id=:u AND tenant=:t AND state IN ('REQUESTED','ACKNOWLEDGED')"),{'u':user['id'],'t':user['tenant']}).first()
            if existing is None:existing=(c.execute(text("INSERT INTO dev_account_deletion_requests(user_id,tenant,state,requested_at,updated_at) VALUES (:u,:t,'REQUESTED',:at,:at)"),{'u':user['id'],'t':user['tenant'],'at':now()}).lastrowid,'REQUESTED')
            ctx.audit(c,user,'request_account_deletion_review','deletion_request',existing[0])
        return {'id':existing[0],'state':existing[1],'erasure_performed':False,'notice':'Review requested. Account and clinical records remain retained pending an approved retention/deletion policy. This is not completed deletion.'}
    @router.post('/deletion-requests/{identifier}/cancel')
    def cancel_deletion(identifier:int,user=Depends(ctx.user)):
        with engine.begin() as c:
            updated=c.execute(text("UPDATE dev_account_deletion_requests SET state='CANCELLED',updated_at=:at WHERE id=:id AND user_id=:u AND tenant=:t AND state IN ('REQUESTED','ACKNOWLEDGED')"),{'id':identifier,'u':user['id'],'t':user['tenant'],'at':now()})
            if updated.rowcount!=1:raise HTTPException(404,'Open request not found.')
            ctx.audit(c,user,'cancel_account_deletion_review','deletion_request',identifier)
        return {'state':'CANCELLED','erasure_performed':False}
    @router.get('/deletion-review')
    def review_queue(user=Depends(ctx.user)):
        if user['role']!='org_admin':raise HTTPException(403,'Organization administrator required.')
        with engine.connect() as c:return [dict(row) for row in c.execute(text("SELECT r.id,r.user_id,u.username,r.state,r.requested_at,r.updated_at FROM dev_account_deletion_requests r JOIN dev_users u ON u.id=r.user_id WHERE r.tenant=:t AND r.state IN ('REQUESTED','ACKNOWLEDGED') ORDER BY r.id"),{'t':user['tenant']}).mappings()]
    @router.post('/deletion-review/{identifier}')
    def acknowledge(identifier:int,body:Review,user=Depends(ctx.user)):
        if user['role']!='org_admin':raise HTTPException(403,'Organization administrator required.')
        with engine.begin() as c:
            result=c.execute(text("UPDATE dev_account_deletion_requests SET state='ACKNOWLEDGED',updated_at=:at,reviewed_by=:u,review_note=:v WHERE id=:id AND tenant=:t AND state IN ('REQUESTED','ACKNOWLEDGED')"),{'id':identifier,'t':user['tenant'],'u':user['id'],'v':body.note,'at':now()})
            if result.rowcount!=1:raise HTTPException(404,'Open request not found.')
            ctx.audit(c,user,'acknowledge_deletion_review','deletion_request',identifier)
        return {'state':'ACKNOWLEDGED','erasure_performed':False}
    app.include_router(router)
