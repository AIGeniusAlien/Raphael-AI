"""Source-bound synthetic procedure checklists, never autonomous procedural care.

No medical content is seeded. Clinician content/rights attestations are not
independent clinical validation. Checklist acknowledgements do not establish
medical consent, procedure performance, or legal/regulatory authorization.
"""
import hashlib
import json
from datetime import datetime, timezone
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator
from sqlalchemy import text
from database import execute_sqlite_script
from dev_portal import _require_synthetic

SECTIONS = ('indications','contraindications','patient_selection','preparation','tests_and_imaging',
            'consent_requirements','safety_checks','equipment_and_resources','procedure_stages',
            'complications','monitoring','follow_up','escalation')
NOTICE = ('Synthetic checklist coordination only. Supplied content and permission are clinician attestations, '
          'not independent clinical validation. Acknowledging a step is not medical consent or evidence '
          'that a procedure was performed. Authorized chart-granted clinicians may collaborate on published patient coordination; '
          'private intelligence-thread ownership is unchanged. No orders, scheduling, billing, or external communications occur.')
SCHEMA = '''
CREATE TABLE IF NOT EXISTS dev_procedure_templates (
 id INTEGER PRIMARY KEY AUTOINCREMENT, tenant TEXT NOT NULL, slug TEXT NOT NULL,
 version INTEGER NOT NULL, title TEXT NOT NULL, specialty TEXT NOT NULL,
 source_id INTEGER NOT NULL REFERENCES dev_evidence_sources(id), source_version_sha256 TEXT NOT NULL,
 content TEXT NOT NULL, content_sha256 TEXT NOT NULL, status TEXT NOT NULL,
 created_by INTEGER NOT NULL REFERENCES dev_users(id), created_at TEXT NOT NULL,
 reviewed_by INTEGER REFERENCES dev_users(id), reviewed_at TEXT,
 UNIQUE(tenant,slug,version), UNIQUE(tenant,slug,content_sha256),
 CHECK(status IN ('pending','approved','superseded','revoked'))
);
CREATE TABLE IF NOT EXISTS dev_procedure_runs (
 id INTEGER PRIMARY KEY AUTOINCREMENT, tenant TEXT NOT NULL,
 patient_id INTEGER NOT NULL REFERENCES dev_patients(id),
 template_id INTEGER NOT NULL REFERENCES dev_procedure_templates(id),
 created_by INTEGER NOT NULL REFERENCES dev_users(id), created_at TEXT NOT NULL,
 idempotency_key TEXT NOT NULL, snapshot TEXT NOT NULL, steps TEXT NOT NULL,
 state TEXT NOT NULL DEFAULT 'draft', consent_state TEXT NOT NULL DEFAULT 'awaiting',
 revision INTEGER NOT NULL DEFAULT 1, published_at TEXT, completed_by INTEGER, completed_at TEXT,
 UNIQUE(tenant,created_by,idempotency_key),
 CHECK(state IN ('draft','published','completed')),
 CHECK(consent_state IN ('awaiting','granted','revoked'))
);
CREATE TABLE IF NOT EXISTS dev_procedure_events (
 id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INTEGER NOT NULL REFERENCES dev_procedure_runs(id),
 revision INTEGER NOT NULL, actor_id INTEGER NOT NULL REFERENCES dev_users(id),
 kind TEXT NOT NULL, detail TEXT NOT NULL, created_at TEXT NOT NULL,
 UNIQUE(run_id,revision)
);
CREATE TRIGGER IF NOT EXISTS dev_procedure_template_content_immutable
BEFORE UPDATE OF tenant,slug,version,title,specialty,source_id,source_version_sha256,content,content_sha256,created_by,created_at ON dev_procedure_templates
BEGIN SELECT RAISE(ABORT,'Procedure template content is immutable;create a new version.'); END;
CREATE TRIGGER IF NOT EXISTS dev_procedure_run_snapshot_immutable
BEFORE UPDATE OF tenant,patient_id,template_id,created_by,created_at,idempotency_key,snapshot ON dev_procedure_runs
BEGIN SELECT RAISE(ABORT,'Procedure snapshot binding is immutable.'); END;
CREATE TRIGGER IF NOT EXISTS dev_procedure_completed_steps_immutable
BEFORE UPDATE OF steps,state,completed_by,completed_at ON dev_procedure_runs WHEN OLD.state='completed'
BEGIN SELECT RAISE(ABORT,'Completed checklist content is immutable.'); END;
CREATE TRIGGER IF NOT EXISTS dev_procedure_events_no_update BEFORE UPDATE ON dev_procedure_events
BEGIN SELECT RAISE(ABORT,'Procedure event history is append-only.'); END;
CREATE TRIGGER IF NOT EXISTS dev_procedure_events_no_delete BEFORE DELETE ON dev_procedure_events
BEGIN SELECT RAISE(ABORT,'Procedure event history is append-only.'); END;
CREATE INDEX IF NOT EXISTS dev_procedure_template_tenant ON dev_procedure_templates(tenant,status);
CREATE INDEX IF NOT EXISTS dev_procedure_runs_patient ON dev_procedure_runs(tenant,patient_id);
'''


def now(): return datetime.now(timezone.utc).isoformat()
def encoded(value): return json.dumps(value,sort_keys=True,separators=(',',':'))
def init_procedure_schema(engine):
    _require_synthetic(engine)
    with engine.begin() as c: execute_sqlite_script(c,SCHEMA)

class Input(BaseModel): model_config=ConfigDict(extra='forbid',str_strip_whitespace=True)
class Step(Input):
    key: str = Field(pattern=r'^[a-z][a-z0-9_]{0,39}$')
    phase: Literal['preparation','procedure','monitoring','follow_up','safety']
    assignee: Literal['patient','clinician']
    instruction: str = Field(min_length=3,max_length=1500)
class TemplateIn(Input):
    slug: str = Field(pattern=r'^[a-z][a-z0-9_]{1,59}$')
    title: str = Field(min_length=3,max_length=200)
    specialty: str = Field(min_length=2,max_length=100)
    source_id: int = Field(gt=0)
    source_version_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    sections: dict[str,str]
    steps: list[Step] = Field(min_length=1,max_length=30)
    @field_validator('sections')
    @classmethod
    def sections_complete(cls,value):
        if set(value)!=set(SECTIONS) or any(not 3<=len(v.strip())<=2000 for v in value.values()):
            raise ValueError('Supply every named reference section with3to2000characters;explicitly record unknown content rather than inventing it.')
        return {k:v.strip() for k,v in value.items()}
    @field_validator('steps')
    @classmethod
    def unique_steps(cls,value):
        if len({v.key for v in value})!=len(value):raise ValueError('Step keys must be unique.')
        return value
class TemplateReview(Input):
    content_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    source_version_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    content_reviewed: StrictBool
    permission_reviewed: StrictBool
class RunIn(Input):
    template_id: int = Field(gt=0)
    idempotency_key: UUID
class Revision(Input): expected_revision:int=Field(gt=0)
class Publish(Revision): content_reviewed:StrictBool
class Authorization(Revision):
    authorize:StrictBool
    understands_coordination_only:StrictBool
class StepUpdate(Revision):
    status:Literal['acknowledged','not_applicable','pending']
    note:str=Field(default='',max_length=2000)
class Completion(Revision): content_reviewed:StrictBool


def install_procedures(app,ctx):
    engine=ctx.engine
    router=APIRouter(prefix='/api/dev/procedures',tags=['synthetic reviewed procedure pathways'],dependencies=[Depends(ctx.same_origin_json)])
    def clinical(user=Depends(ctx.user)):
        if user['role'] not in ('patient','clinician'):raise HTTPException(403,'Patient or clinician role required.')
        return user
    def require_session(c,request,user):
        from dev_account import session_active
        if not session_active(c,request,user):raise HTTPException(401,'Session revoked;procedure coordination was not changed.')
    def template(c,identifier,user):
        row=c.execute(text('SELECT t.*,s.status AS source_status,s.version_sha256 AS current_source_hash,s.url AS source_url,s.title AS source_title,s.license_declaration FROM dev_procedure_templates t JOIN dev_evidence_sources s ON s.id=t.source_id AND s.tenant=t.tenant WHERE t.id=:id AND t.tenant=:tenant'),{'id':identifier,'tenant':user['tenant']}).mappings().first()
        if not row:raise HTTPException(404,'Template not found.')
        result=dict(row);result['content']=json.loads(result['content'])
        result['active_for_use']=row['status']=='approved' and row['source_status']=='approved' and row['source_version_sha256']==row['current_source_hash']
        result['independently_clinically_validated']=False;result['notice']=NOTICE
        return result
    def active_template(c,identifier,user):
        value=template(c,identifier,user)
        if not value['active_for_use']:raise HTTPException(409,'Current approved template and source version required;no automatic substitution occurs.')
        return value
    def get_run(c,identifier,user,allow_inactive_patient=False):
        row=c.execute(text('SELECT * FROM dev_procedure_runs WHERE id=:id AND tenant=:tenant'),{'id':identifier,'tenant':user['tenant']}).mappings().first()
        if not row:raise HTTPException(404,'Pathway not found.')
        ctx.patient_for(c,row['patient_id'],user)
        if user['role']=='patient' and row['state']=='draft':raise HTTPException(404,'Pathway not found.')
        if user['role']=='clinician' and (row['consent_state']=='revoked' or (row['state']=='draft' and row['created_by']!=user['id'])):raise HTTPException(404,'Pathway is private or access was revoked.')
        result=dict(row);result['snapshot']=json.loads(row['snapshot']);result['steps']=json.loads(row['steps'])
        result['active_source_and_template']=template(c,row['template_id'],user)['active_for_use']
        if user['role']=='patient' and not result['active_source_and_template'] and not allow_inactive_patient:
            raise HTTPException(409,'Published instructions are withheld because their source or template is no longer current;contact your clinician.')
        result['actionable']=result['active_source_and_template'] and row['state']=='published' and row['consent_state']=='granted'
        result['notice']=NOTICE
        return result
    def check_revision(run,body):
        if run['revision']!=body.expected_revision:raise HTTPException(409,'Pathway changed;refresh before acting.')
    def record(c,run,user,kind,detail,**changes):
        revision=run['revision']+1
        allowed={'state','consent_state','steps','published_at','completed_by','completed_at'}
        if set(changes)-allowed:raise ValueError('Invalid pathway update.')
        assignments=','.join(k+'=:'+k for k in changes)
        sql='UPDATE dev_procedure_runs SET '+(assignments+',' if assignments else '')+'revision=:new_revision WHERE id=:id AND revision=:previous_revision'
        values=dict(changes,new_revision=revision,id=run['id'],previous_revision=run['revision'])
        if c.execute(text(sql),values).rowcount!=1:raise HTTPException(409,'Pathway changed;refresh before acting.')
        c.execute(text('INSERT INTO dev_procedure_events(run_id,revision,actor_id,kind,detail,created_at) VALUES (:id,:revision,:actor,:kind,:detail,:at)'),{'id':run['id'],'revision':revision,'actor':user['id'],'kind':kind,'detail':encoded(detail),'at':now()})
        ctx.audit(c,user,kind,'procedure_run',run['id'])
    @router.get('/templates')
    def templates(user=Depends(ctx.clinician)):
        with engine.begin() as c:
            ids=c.execute(text('SELECT id FROM dev_procedure_templates WHERE tenant=:t ORDER BY id DESC LIMIT 200'),{'t':user['tenant']}).scalars().all()
            ctx.audit(c,user,'list_procedure_templates','procedure_template')
            return {'templates':[template(c,i,user) for i in ids],'limit':200,'notice':NOTICE}
    @router.get('/coverage')
    def coverage(user=Depends(ctx.clinician)):
        with engine.begin() as c:
            rows=c.execute(text("SELECT t.specialty,count(*) AS templates FROM dev_procedure_templates t JOIN dev_evidence_sources s ON s.id=t.source_id AND s.tenant=t.tenant WHERE t.tenant=:t AND t.status='approved' AND s.status='approved' AND s.version_sha256=t.source_version_sha256 GROUP BY t.specialty ORDER BY t.specialty"),{'t':user['tenant']}).mappings().all()
            return {'facets':[dict(r,independently_clinically_validated=False) for r in rows],'comprehensive_procedure_coverage':False,'notice':NOTICE}
    @router.get('/templates/{identifier}')
    def read_template(identifier:int,user=Depends(ctx.clinician)):
        with engine.begin() as c:
            value=template(c,identifier,user);ctx.audit(c,user,'read_procedure_template','procedure_template',identifier);return value
    @router.post('/templates')
    def create_template(body:TemplateIn,request:Request,user=Depends(ctx.clinician)):
        from dev_knowledge import _source
        values=body.model_dump(mode='json');digest=hashlib.sha256(encoded(values).encode()).hexdigest()
        with engine.begin() as c:
            c.exec_driver_sql('BEGIN IMMEDIATE');require_session(c,request,user);source=_source(c,body.source_id,user['tenant'])
            if source['status']!='approved' or source['version_sha256']!=body.source_version_sha256:raise HTTPException(409,'Current approved source version required.')
            existing=c.execute(text('SELECT id FROM dev_procedure_templates WHERE tenant=:t AND slug=:slug AND content_sha256=:h'),{'t':user['tenant'],'slug':body.slug,'h':digest}).scalar()
            if existing:return dict(template(c,existing,user),deduplicated=True)
            version=c.execute(text('SELECT COALESCE(MAX(version),0)+1 FROM dev_procedure_templates WHERE tenant=:t AND slug=:slug'),{'t':user['tenant'],'slug':body.slug}).scalar()
            identifier=c.execute(text("INSERT INTO dev_procedure_templates(tenant,slug,version,title,specialty,source_id,source_version_sha256,content,content_sha256,status,created_by,created_at) VALUES (:t,:slug,:version,:title,:specialty,:source_id,:source_version_sha256,:content,:h,'pending',:uid,:at)"),dict(values,t=user['tenant'],version=version,content=encoded(values),h=digest,uid=user['id'],at=now())).lastrowid
            ctx.audit(c,user,'create_procedure_template','procedure_template',identifier);return dict(template(c,identifier,user),deduplicated=False)
    def change_template(identifier,body,user,status,request):
        with engine.begin() as c:
            c.exec_driver_sql('BEGIN IMMEDIATE');require_session(c,request,user);current=template(c,identifier,user)
            if not body.content_reviewed or not body.permission_reviewed or body.content_sha256!=current['content_sha256'] or body.source_version_sha256!=current['source_version_sha256']:
                raise HTTPException(409,'Explicit content and permission review of this exact immutable version is required.')
            if status=='approved':
                if current['status']!='pending' or current['source_status']!='approved' or current['current_source_hash']!=body.source_version_sha256:raise HTTPException(409,'Only pending templates on current approved sources can be approved.')
                c.execute(text("UPDATE dev_procedure_templates SET status='superseded' WHERE tenant=:t AND slug=:slug AND status='approved'"),{'t':user['tenant'],'slug':current['slug']})
            elif current['status']=='revoked':return current
            c.execute(text('UPDATE dev_procedure_templates SET status=:status,reviewed_by=:uid,reviewed_at=:at WHERE id=:id'),{'status':status,'uid':user['id'],'at':now(),'id':identifier})
            ctx.audit(c,user,'approve_procedure_template' if status=='approved' else 'revoke_procedure_template','procedure_template',identifier);return template(c,identifier,user)
    @router.post('/templates/{identifier}/approve')
    def approve_template(identifier:int,body:TemplateReview,request:Request,user=Depends(ctx.clinician)):return change_template(identifier,body,user,'approved',request)
    @router.post('/templates/{identifier}/revoke')
    def revoke_template(identifier:int,body:TemplateReview,request:Request,user=Depends(ctx.clinician)):return change_template(identifier,body,user,'revoked',request)
    @router.get('/patients/{patient_id}/runs')
    def runs(patient_id:int,user=Depends(clinical)):
        with engine.begin() as c:
            ctx.patient_for(c,patient_id,user)
            rows=c.execute(text('SELECT id,template_id,created_by,state,consent_state,revision,created_at FROM dev_procedure_runs WHERE tenant=:t AND patient_id=:p ORDER BY id DESC LIMIT 100'),{'t':user['tenant'],'p':patient_id}).mappings().all()
            result=[]
            for row in rows:
                if (user['role']=='patient' and row['state']=='draft') or (user['role']=='clinician' and (row['consent_state']=='revoked' or (row['state']=='draft' and row['created_by']!=user['id']))):continue
                result.append(dict(row,active_source_and_template=template(c,row['template_id'],user)['active_for_use']))
            ctx.audit(c,user,'list_procedure_runs','patient',patient_id);return {'runs':result,'limit':100,'notice':NOTICE}
    @router.post('/patients/{patient_id}/runs')
    def create_run(patient_id:int,body:RunIn,request:Request,user=Depends(ctx.clinician)):
        with engine.begin() as c:
            c.exec_driver_sql('BEGIN IMMEDIATE');require_session(c,request,user);patient=ctx.patient_for(c,patient_id,user)
            if not patient['patient_user_id']:raise HTTPException(409,'A linked synthetic patient account is required for coordination authorization.')
            current=active_template(c,body.template_id,user)
            existing=c.execute(text('SELECT id,patient_id,template_id FROM dev_procedure_runs WHERE tenant=:t AND created_by=:u AND idempotency_key=:key'),{'t':user['tenant'],'u':user['id'],'key':str(body.idempotency_key)}).mappings().first()
            if existing:
                if existing['patient_id']!=patient_id or existing['template_id']!=body.template_id:raise HTTPException(409,'Idempotency key already belongs to a different request.')
                return dict(get_run(c,existing['id'],user),deduplicated=True)
            snapshot={k:current[k] for k in ('id','slug','version','title','specialty','source_id','source_version_sha256','source_url','source_title','license_declaration','content','content_sha256','reviewed_by','reviewed_at')}
            steps={v['key']:{'status':'pending','note':'','actor_id':None,'updated_at':None} for v in current['content']['steps']}
            identifier=c.execute(text('INSERT INTO dev_procedure_runs(tenant,patient_id,template_id,created_by,created_at,idempotency_key,snapshot,steps) VALUES (:t,:p,:template,:u,:at,:key,:snapshot,:steps)'),{'t':user['tenant'],'p':patient_id,'template':body.template_id,'u':user['id'],'at':now(),'key':str(body.idempotency_key),'snapshot':encoded(snapshot),'steps':encoded(steps)}).lastrowid
            c.execute(text("INSERT INTO dev_procedure_events(run_id,revision,actor_id,kind,detail,created_at) VALUES (:id,1,:u,'create_procedure_run','{}',:at)"),{'id':identifier,'u':user['id'],'at':now()})
            ctx.audit(c,user,'create_procedure_run','procedure_run',identifier);return dict(get_run(c,identifier,user),deduplicated=False)
    @router.get('/runs/{identifier}')
    def read_run(identifier:int,user=Depends(clinical)):
        with engine.begin() as c:
            result=get_run(c,identifier,user);ctx.audit(c,user,'read_procedure_run','procedure_run',identifier);return result
    @router.get('/runs/{identifier}/events')
    def events(identifier:int,user=Depends(clinical)):
        with engine.begin() as c:
            get_run(c,identifier,user);ctx.audit(c,user,'read_procedure_events','procedure_run',identifier)
            rows=c.execute(text('SELECT revision,actor_id,kind,detail,created_at FROM dev_procedure_events WHERE run_id=:id ORDER BY revision DESC LIMIT 500'),{'id':identifier}).mappings().all()
            return [dict(r,detail=json.loads(r['detail'])) for r in reversed(rows)]
    @router.post('/runs/{identifier}/publish')
    def publish(identifier:int,body:Publish,request:Request,user=Depends(ctx.clinician)):
        with engine.begin() as c:
            c.exec_driver_sql('BEGIN IMMEDIATE');require_session(c,request,user);run=get_run(c,identifier,user);check_revision(run,body);active_template(c,run['template_id'],user)
            if run['state']!='draft' or not body.content_reviewed:raise HTTPException(409,'Only explicitly reviewed draft plans can be published.')
            record(c,run,user,'publish_procedure_run',{},state='published',published_at=now());return get_run(c,identifier,user)
    @router.post('/runs/{identifier}/authorization')
    def authorize(identifier:int,body:Authorization,request:Request,user=Depends(clinical)):
        if user['role']!='patient':raise HTTPException(403,'The linked patient controls coordination authorization.')
        with engine.begin() as c:
            c.exec_driver_sql('BEGIN IMMEDIATE');require_session(c,request,user);run=get_run(c,identifier,user,allow_inactive_patient=True);check_revision(run,body)
            if body.authorize:
                active_template(c,run['template_id'],user)
                if not body.understands_coordination_only:raise HTTPException(422,'Acknowledge that coordination authorization is not medical procedure consent.')
            record(c,run,user,'authorize_procedure_coordination' if body.authorize else 'revoke_procedure_coordination',{},consent_state='granted' if body.authorize else 'revoked')
            return {'id':identifier,'revision':run['revision']+1,'consent_state':'granted' if body.authorize else 'revoked','notice':NOTICE}
    @router.post('/runs/{identifier}/steps/{step_key}')
    def update_step(identifier:int,step_key:str,body:StepUpdate,request:Request,user=Depends(clinical)):
        with engine.begin() as c:
            c.exec_driver_sql('BEGIN IMMEDIATE');require_session(c,request,user);run=get_run(c,identifier,user);check_revision(run,body);active_template(c,run['template_id'],user)
            if not run['actionable']:raise HTTPException(409,'Published plan and active patient coordination authorization required;completed checklists are immutable.')
            definition=next((v for v in run['snapshot']['content']['steps'] if v['key']==step_key),None)
            if not definition:raise HTTPException(404,'Checklist step not found.')
            if user['role']=='patient' and definition['assignee']!='patient':raise HTTPException(403,'This step requires clinician review.')
            if body.status=='not_applicable' and (user['role']!='clinician' or not body.note):raise HTTPException(422,'Only clinicians may mark a step not applicable,with an explicit rationale.')
            run['steps'][step_key]={'status':body.status,'note':body.note,'actor_id':user['id'],'updated_at':now()}
            record(c,run,user,'update_procedure_step',{'step_key':step_key,**run['steps'][step_key]},steps=encoded(run['steps']));return get_run(c,identifier,user)
    @router.post('/runs/{identifier}/complete')
    def complete(identifier:int,body:Completion,request:Request,user=Depends(ctx.clinician)):
        with engine.begin() as c:
            c.exec_driver_sql('BEGIN IMMEDIATE');require_session(c,request,user);run=get_run(c,identifier,user);check_revision(run,body);active_template(c,run['template_id'],user)
            if not run['actionable'] or not body.content_reviewed or any(v['status']=='pending' for v in run['steps'].values()):raise HTTPException(409,'Active authorization,all resolved checklist steps,and explicit clinician review are required.')
            record(c,run,user,'complete_procedure_checklist',{},state='completed',completed_by=user['id'],completed_at=now());return get_run(c,identifier,user)
    app.include_router(router)


def export_patient_procedures(c,user,allow_instructions=True):
    """Bounded own-patient published history;never silently truncate or leak drafts.

    Candidate release scopes pass allow_instructions=False so account export
    cannot bypass the experimental procedure route boundary. Source/template
    invalidation and coordination revocation also withhold instructions/notes.
    All domain records remain stored;this is not legal erasure certification.
    """
    if user['role']!='patient':return {'runs':[],'notice':NOTICE}
    rows=c.execute(text("SELECT r.*,t.status AS template_status,t.source_version_sha256,s.status AS source_status,s.version_sha256 AS current_source_hash FROM dev_procedure_runs r JOIN dev_patients p ON p.id=r.patient_id JOIN dev_procedure_templates t ON t.id=r.template_id AND t.tenant=r.tenant JOIN dev_evidence_sources s ON s.id=t.source_id AND s.tenant=t.tenant WHERE r.tenant=:t AND p.tenant=:t AND p.patient_user_id=:u AND r.state!='draft' ORDER BY r.id LIMIT 501"),{'t':user['tenant'],'u':user['id']}).mappings().all()
    if len(rows)>500:raise HTTPException(413,'Procedure history exceeds development export limits;no partial file produced. Request an audited operator export.')
    result=[];size=0
    for row in rows:
        instructions=bool(allow_instructions and row['consent_state']!='revoked' and row['template_status']=='approved' and row['source_status']=='approved' and row['source_version_sha256']==row['current_source_hash'])
        value={k:row[k] for k in ('id','patient_id','template_id','created_by','created_at','state','consent_state','revision','published_at','completed_by','completed_at')}
        snapshot=json.loads(row['snapshot']);steps=json.loads(row['steps'])
        value['instructions_withheld']=not instructions
        value['withholding_reason']=None if instructions else ('release_scope_restriction' if not allow_instructions else 'coordination_revoked' if row['consent_state']=='revoked' else 'source_or_template_not_current')
        value['snapshot']=snapshot if instructions else {k:snapshot.get(k) for k in ('id','slug','version','title','specialty','source_id','source_version_sha256','source_url','source_title','content_sha256','reviewed_by','reviewed_at')}
        value['steps']=steps if instructions else {k:{field:item.get(field) for field in ('status','actor_id','updated_at')} for k,item in steps.items()}
        events=c.execute(text('SELECT revision,actor_id,kind,detail,created_at FROM dev_procedure_events WHERE run_id=:id ORDER BY revision LIMIT 501'),{'id':row['id']}).mappings().all()
        if len(events)>500:raise HTTPException(413,'Procedure event history exceeds development export limits;no partial file produced. Request an audited operator export.')
        value['events']=[]
        for event in events:
            detail=json.loads(event['detail'])
            if not instructions:detail={k:detail[k] for k in ('step_key','status') if k in detail}
            value['events'].append(dict(event,detail=detail))
        size+=len(encoded(value).encode())
        if size>10*1024*1024:raise HTTPException(413,'Procedure history exceeds development export byte limits;no partial file produced. Request an audited operator export.')
        result.append(value)
    return {'runs':result,'notice':NOTICE,'scope':'Own linked patient published checklists only;drafts excluded. Invalid/revoked/restricted instructions and notes withheld. Not a complete legal portability determination.'}
