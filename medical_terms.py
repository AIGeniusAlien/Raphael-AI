"""Source-version-bound local vocabulary;not a licensed ontology or clinical validation.

Only explicit clinician-reviewed aliases are used,with opt-in retrieval. No
provider requests,patient context,automatic terminology import or code claims.
"""
import hashlib
import json
import re
from typing import Literal
from fastapi import APIRouter,Depends,HTTPException
from pydantic import BaseModel,ConfigDict,Field,StrictBool,field_validator
from sqlalchemy import text
from database import execute_sqlite_script

SCHEMA='''
CREATE TABLE IF NOT EXISTS dev_medical_terms (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 tenant TEXT NOT NULL,
 source_id INTEGER NOT NULL REFERENCES dev_evidence_sources(id),
 source_version_sha256 TEXT NOT NULL,
 canonical TEXT NOT NULL,
 aliases TEXT NOT NULL,
 specialty TEXT NOT NULL,
 kind TEXT NOT NULL CHECK(kind IN ('condition','symptom','procedure','test','medication','other')),
 content_sha256 TEXT NOT NULL,
 status TEXT NOT NULL CHECK(status IN ('pending','reviewed','revoked')),
 created_by INTEGER NOT NULL REFERENCES dev_users(id),
 created_at TEXT NOT NULL,
 reviewed_by INTEGER REFERENCES dev_users(id),
 reviewed_at TEXT,
 UNIQUE(tenant,content_sha256)
);
CREATE INDEX IF NOT EXISTS dev_medical_terms_tenant ON dev_medical_terms(tenant,status,source_id);
'''
NOTICE='Local clinician-attested terminology only;not a licensed standard ontology,independent clinical validation or specialty competence. Source supersession/revocation disables associated aliases.'

def init_terms(connection):execute_sqlite_script(connection,SCHEMA)

def normalized(value):return ' '.join(re.findall(r'\b\w+\b',value.casefold(),flags=re.UNICODE))

class TermIn(BaseModel):
    model_config=ConfigDict(extra='forbid',str_strip_whitespace=True)
    source_id:int=Field(gt=0)
    source_version_sha256:str=Field(pattern=r'^[0-9a-f]{64}$')
    canonical:str=Field(min_length=2,max_length=120)
    aliases:list[str]=Field(min_length=1,max_length=12)
    specialty:str=Field(min_length=2,max_length=80)
    kind:Literal['condition','symptom','procedure','test','medication','other']='other'
    @field_validator('canonical','specialty')
    @classmethod
    def phrase(cls,value):
        if not normalized(value) or any(ord(ch)<32 for ch in value):raise ValueError('Use a bounded readable term.')
        return value
    @field_validator('aliases')
    @classmethod
    def phrases(cls,values):
        if any(not isinstance(v,str) or not 2<=len(v.strip())<=120 or not normalized(v) or any(ord(ch)<32 for ch in v) for v in values):raise ValueError('Aliases require readable phrases of2to120characters.')
        return sorted(set(normalized(v) for v in values))

class ReviewIn(BaseModel):
    model_config=ConfigDict(extra='forbid')
    content_sha256:str=Field(pattern=r'^[0-9a-f]{64}$')
    source_version_sha256:str=Field(pattern=r'^[0-9a-f]{64}$')
    content_reviewed:StrictBool


def active_rows(connection,tenant):
    return connection.execute(text("SELECT t.* FROM dev_medical_terms t JOIN dev_evidence_sources s ON s.id=t.source_id WHERE t.tenant=:tenant AND s.tenant=:tenant AND t.status='reviewed' AND s.status='approved' AND t.source_version_sha256=s.version_sha256 ORDER BY t.id LIMIT 2001"),{'tenant':tenant}).mappings().all()

def expand_query(engine,tenant,query):
    """All-or-nothing bounded expansion;ambiguous aliases never choose a meaning."""
    with engine.connect() as connection:rows=active_rows(connection,tenant)
    if len(rows)>2000:return {'query':query,'status':'coverage_budget_exceeded','mappings':[]}
    lookup={};matching=[];phrase=' '+normalized(query)+' '
    for row in rows:
        for alias in json.loads(row['aliases']):
            if ' '+alias+' ' in phrase:
                lookup.setdefault(alias,set()).add(normalized(row['canonical']))
                matching.append(row)
    if any(len(values)>1 for values in lookup.values()):return {'query':query,'status':'ambiguous_no_expansion','mappings':[]}
    selected={row['id']:row for row in matching}
    terms=sorted({row['canonical'] for row in selected.values()})
    candidate=query+(' '+' '.join(terms) if terms else '')
    # Existing lexical contract supports at most20tokens. Never truncate an
    # expanded meaning or turn a long query into a misleading partial mapping.
    if len(set(re.findall(r'\b\w{2,64}\b',candidate.lower())))>20 or len(candidate)>1000:return {'query':query,'status':'query_budget_exceeded','mappings':[]}
    return {'query':candidate,'status':'expanded' if terms else 'no_reviewed_alias_match','mappings':[{'term_id':row['id'],'canonical':row['canonical'],'source_id':row['source_id'],'source_version_sha256':row['source_version_sha256']} for row in selected.values()]}


def install_terms(app,ctx):
    from dev_knowledge import _now,_source
    router=APIRouter(prefix='/api/dev/knowledge/terms',tags=['reviewed local terminology'],dependencies=[Depends(ctx.same_origin_json)])
    def get_term(c,identifier,user):
        row=c.execute(text('SELECT * FROM dev_medical_terms WHERE id=:id AND tenant=:tenant'),{'id':identifier,'tenant':user['tenant']}).mappings().first()
        if not row:raise HTTPException(404,'Term not found.')
        return row
    def output(row):return dict(row,aliases=json.loads(row['aliases']),independently_clinically_validated=False)
    @router.get('')
    def listing(user=Depends(ctx.clinician)):
        with ctx.engine.begin() as c:
            rows=c.execute(text('SELECT * FROM dev_medical_terms WHERE tenant=:tenant ORDER BY id DESC LIMIT 200'),{'tenant':user['tenant']}).mappings().all()
            eligible=active_rows(c,user['tenant'])
            active={r['id'] for r in eligible} if len(eligible)<=2000 else set()
            ctx.audit(c,user,'list_medical_terms','medical_term')
            return {'terms':[dict(output(r),active_for_search=r['id'] in active) for r in rows],'list_limit':200,'notice':NOTICE}
    @router.get('/coverage')
    def coverage(user=Depends(ctx.clinician)):
        with ctx.engine.begin() as c:
            rows=active_rows(c,user['tenant']);groups={}
            for row in rows[:2000]:groups.setdefault(row['specialty'],[]).append(row)
            ctx.audit(c,user,'review_terminology_coverage','medical_term')
            return {'facets':[{'specialty':name,'active_terms':len(items),'source_versions':len({r['source_version_sha256'] for r in items}),'clinical_validation':False} for name,items in sorted(groups.items())],'truncated':len(rows)>2000,'comprehensive_specialty_coverage':False,'notice':NOTICE}
    @router.post('')
    def create(body:TermIn,user=Depends(ctx.clinician)):
        values=body.model_dump();digest=hashlib.sha256(json.dumps(values,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        with ctx.engine.begin() as c:
            c.exec_driver_sql('BEGIN IMMEDIATE')
            source=_source(c,body.source_id,user['tenant'])
            if source['status']!='approved' or source['version_sha256']!=body.source_version_sha256:raise HTTPException(409,'Current approved source version required.')
            if ' '+normalized(body.canonical)+' ' not in ' '+normalized(source['title']+' '+source['body'])+' ':raise HTTPException(422,'Canonical phrase must occur in the approved source text.')
            existing=c.execute(text('SELECT * FROM dev_medical_terms WHERE tenant=:tenant AND content_sha256=:digest'),{'tenant':user['tenant'],'digest':digest}).mappings().first()
            if existing:return dict(output(existing),deduplicated=True)
            values.update(tenant=user['tenant'],aliases=json.dumps(body.aliases),digest=digest,uid=user['id'],at=_now())
            identifier=c.execute(text("INSERT INTO dev_medical_terms(tenant,source_id,source_version_sha256,canonical,aliases,specialty,kind,content_sha256,status,created_by,created_at) VALUES (:tenant,:source_id,:source_version_sha256,:canonical,:aliases,:specialty,:kind,:digest,'pending',:uid,:at)"),values).lastrowid
            ctx.audit(c,user,'create_medical_term','medical_term',identifier)
            return dict(output(get_term(c,identifier,user)),deduplicated=False)
    def transition(identifier,body,user,target):
        with ctx.engine.begin() as c:
            c.exec_driver_sql('BEGIN IMMEDIATE');row=get_term(c,identifier,user)
            if not body.content_reviewed or row['content_sha256']!=body.content_sha256 or row['source_version_sha256']!=body.source_version_sha256:raise HTTPException(409,'Explicit review of this exact term and source version required.')
            if target=='reviewed':
                source=_source(c,row['source_id'],user['tenant'])
                if row['status']!='pending' or source['status']!='approved' or source['version_sha256']!=body.source_version_sha256:raise HTTPException(409,'Only pending terms on current approved sources can be reviewed;create a new immutable mapping for corrections.')
            elif row['status']=='revoked':return output(row)
            c.execute(text('UPDATE dev_medical_terms SET status=:status,reviewed_by=:uid,reviewed_at=:at WHERE id=:id'),{'status':target,'uid':user['id'],'at':_now(),'id':identifier})
            ctx.audit(c,user,'review_medical_term' if target=='reviewed' else 'revoke_medical_term','medical_term',identifier)
            return output(get_term(c,identifier,user))
    @router.post('/{identifier}/review')
    def review(identifier:int,body:ReviewIn,user=Depends(ctx.clinician)):return transition(identifier,body,user,'reviewed')
    @router.post('/{identifier}/revoke')
    def revoke(identifier:int,body:ReviewIn,user=Depends(ctx.clinician)):return transition(identifier,body,user,'revoked')
    app.include_router(router)
