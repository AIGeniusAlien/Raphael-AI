"""Encrypted, explicitly reviewed synthetic text attachments for clinician threads.

Not a malware scanner, diagnostic parser or production PHI storage system.
Original chart files and conversation output retain their existing storage policy.
"""
import hashlib
import json
import os
import re
from datetime import datetime,timezone
from pathlib import PurePath
from cryptography.fernet import Fernet,InvalidToken
from fastapi import APIRouter,Depends,File,Form,HTTPException,Request,UploadFile
from pydantic import BaseModel,ConfigDict,Field,StrictBool,StrictInt
from sqlalchemy import text
from starlette.responses import JSONResponse

MAX_BYTES=65536
MAX_TEXT=6000
SCHEMA='''
CREATE TABLE IF NOT EXISTS dev_conversation_attachments (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 thread_id INTEGER NOT NULL REFERENCES dev_ai_threads(id),
 patient_id INTEGER NOT NULL REFERENCES dev_patients(id),
 source_document_id INTEGER REFERENCES dev_chart_documents(id),
 title TEXT NOT NULL,sha256 TEXT NOT NULL,text_sha256 TEXT NOT NULL,
 encrypted_content BLOB,size_bytes INTEGER NOT NULL,
 state TEXT NOT NULL CHECK(state IN ('quarantined','reviewed','deleted')),
 created_at TEXT NOT NULL,reviewed_by INTEGER,reviewed_at TEXT,deleted_at TEXT
);
CREATE TABLE IF NOT EXISTS dev_generation_attachments (
 generation_id TEXT NOT NULL REFERENCES dev_ai_messages(generation_id),
 attachment_id INTEGER NOT NULL REFERENCES dev_conversation_attachments(id),
 sha256 TEXT NOT NULL,text_sha256 TEXT NOT NULL,
 PRIMARY KEY(generation_id,attachment_id)
);
'''

def init_attachment_schema(engine):
    from database import execute_sqlite_script
    from dev_portal import _require_synthetic
    _require_synthetic(engine)
    with engine.begin() as c:execute_sqlite_script(c,SCHEMA)

def now():return datetime.now(timezone.utc).isoformat()
def cipher():
    try:return Fernet(os.environ['RAPHAEL_DOCUMENT_ENCRYPTION_KEY'].encode('ascii'))
    except (KeyError,ValueError,UnicodeError):raise HTTPException(503,'Protected document encryption key unavailable;operator configuration is required.') from None

def parse(content,patient_id):
    if not content or len(content)>MAX_BYTES:raise HTTPException(413,'Attachment must be nonempty and at most 64 KiB.')
    try:value=content.decode('utf-8')
    except UnicodeError:raise HTTPException(415,'Only UTF-8 plain text is supported;no executable or binary format.') from None
    if any(ord(ch)<32 and ch not in '\r\n\t' for ch in value):raise HTTPException(415,'Control or binary bytes are forbidden.')
    header,separator,body=value.partition('\n')
    if not separator or header.rstrip('\r')!=f'RAPHAEL_SYNTHETIC_PATIENT_ID: {patient_id}':raise HTTPException(422,'Attachment patient header must match the selected synthetic patient exactly.')
    if not body.strip() or len(body)>MAX_TEXT:raise HTTPException(413,'Attachment body must contain 1–6,000 characters;no silent truncation is performed.')
    blocked=bool(re.search(r'<\s*(?:script|iframe|object)|\b(?:ignore|override)\b.{0,50}\b(?:system|previous|instructions)\b|\b(?:system|developer)\s*:|EICAR-STANDARD-ANTIVIRUS-TEST-FILE',body,re.I|re.S))
    return body,not blocked

class AttachmentLimit:
    """Read a bounded body before multipart parsing,including chunked requests."""
    def __init__(self,app):self.app=app
    async def __call__(self,scope,receive,send):
        if scope['type']!='http' or scope.get('method')!='POST' or not re.fullmatch(r'/api/dev/intelligence/threads/\d+/attachments',scope.get('path','')):return await self.app(scope,receive,send)
        cap=MAX_BYTES+16384
        length=dict(scope.get('headers',[])).get(b'content-length')
        try:
            if length is not None and (int(length)<0 or int(length)>cap):return await JSONResponse({'detail':'Attachment request exceeds bounded size.'},413)(scope,receive,send)
        except ValueError:return await JSONResponse({'detail':'Invalid request length.'},400)(scope,receive,send)
        chunks=[];size=0
        while True:
            message=await receive()
            if message['type']=='http.disconnect':return
            chunk=message.get('body',b'');size+=len(chunk)
            if size>cap:return await JSONResponse({'detail':'Attachment request exceeds bounded size.'},413)(scope,receive,send)
            chunks.append(chunk)
            if not message.get('more_body'):break
        delivered=False
        async def replay():
            nonlocal delivered
            if not delivered:delivered=True;return {'type':'http.request','body':b''.join(chunks),'more_body':False}
            return await receive()
        await self.app(scope,replay,send)

class Review(BaseModel):
    model_config=ConfigDict(extra='forbid')
    sha256:str=Field(pattern=r'^[a-f0-9]{64}$')
    authorize_provider_use:StrictBool
class ImportDocument(BaseModel):
    model_config=ConfigDict(extra='forbid')
    document_id:StrictInt=Field(gt=0)
class Delete(BaseModel):
    model_config=ConfigDict(extra='forbid')
    confirm:StrictBool

class AttachmentStore:
    def __init__(self,ctx,thread_for):self.ctx=ctx;self.thread_for=thread_for
    def thread(self,c,identifier,user):
        thread=self.thread_for(c,identifier,user)
        if not thread['patient_id'] or thread['intake_id']:raise HTTPException(409,'Attachments require a full-chart-authorized patient conversation;intake-only or general threads cannot acquire chart context.')
        return thread
    def row(self,c,identifier,thread,user):
        row=c.execute(text('SELECT * FROM dev_conversation_attachments WHERE id=:id AND thread_id=:t'),{'id':identifier,'t':thread['id']}).mappings().first()
        if not row or row['state']=='deleted':raise HTTPException(404,'Attachment unavailable.')
        self.ctx.patient_for(c,thread['patient_id'],user)
        if row['patient_id']!=thread['patient_id']:raise HTTPException(409,'Attachment patient changed.')
        if row['source_document_id']:
            from dev_chart import chart_document_allowed
            source=c.execute(text('SELECT * FROM dev_chart_documents WHERE id=:id'),{'id':row['source_document_id']}).mappings().first()
            if not source or source['patient_id']!=row['patient_id'] or not chart_document_allowed(c,source,user,self.ctx):raise HTTPException(404,'Attachment source unavailable.')
            if source['sha256']!=row['sha256'] or hashlib.sha256(bytes(source['content'])).hexdigest()!=row['sha256']:raise HTTPException(409,'Attachment source changed;new review is required.')
        return row
    def content(self,row):
        try:content=cipher().decrypt(bytes(row['encrypted_content']))
        except (InvalidToken,TypeError):raise HTTPException(503,'Attachment cannot be decrypted or verified;operator key restoration is required.') from None
        if hashlib.sha256(content).hexdigest()!=row['sha256']:raise HTTPException(409,'Attachment integrity check failed.')
        body,reviewable=parse(content,row['patient_id'])
        if hashlib.sha256(body.encode()).hexdigest()!=row['text_sha256']:raise HTTPException(409,'Attachment text integrity check failed.')
        return body,reviewable
    @staticmethod
    def metadata(row):return {key:row[key] for key in ('id','thread_id','patient_id','source_document_id','title','sha256','text_sha256','size_bytes','state','created_at','reviewed_by','reviewed_at')}
    def listing(self,c,thread,user):
        result=[]
        for row in c.execute(text("SELECT id FROM dev_conversation_attachments WHERE thread_id=:t AND state!='deleted' ORDER BY id LIMIT 21"),{'t':thread['id']}):
            try:result.append(self.metadata(self.row(c,row[0],thread,user)))
            except HTTPException:result.append({'id':row[0],'state':'unavailable','title':'Attachment access unavailable'})
        return result
    def add(self,c,thread,user,title,content,source=None):
        body,_=parse(content,thread['patient_id']);encryption=cipher();digest=hashlib.sha256(content).hexdigest()
        existing=c.execute(text("SELECT id FROM dev_conversation_attachments WHERE thread_id=:t AND sha256=:h AND state!='deleted' AND source_document_id IS :s ORDER BY id LIMIT 1"),{'t':thread['id'],'h':digest,'s':source}).scalar()
        if existing:return self.metadata(self.row(c,existing,thread,user))
        quota=c.execute(text("SELECT count(*),coalesce(sum(CASE WHEN state!='deleted' THEN size_bytes ELSE 0 END),0),sum(CASE WHEN state!='deleted' THEN 1 ELSE 0 END) FROM dev_conversation_attachments WHERE thread_id=:t"),{'t':thread['id']}).one()
        if quota[0]>=200 or (quota[2] or 0)>=20 or quota[1]+len(content)>1024*1024:raise HTTPException(413,'Conversation attachment quota exceeded;no partial upload was retained.')
        identifier=c.execute(text("INSERT INTO dev_conversation_attachments(thread_id,patient_id,source_document_id,title,sha256,text_sha256,encrypted_content,size_bytes,state,created_at) VALUES(:t,:p,:s,:title,:h,:x,:b,:n,'quarantined',:at)"),{'t':thread['id'],'p':thread['patient_id'],'s':source,'title':title.strip(),'h':digest,'x':hashlib.sha256(body.encode()).hexdigest(),'b':encryption.encrypt(content),'n':len(content),'at':now()}).lastrowid
        self.ctx.audit(c,user,'quarantine_conversation_attachment','attachment',identifier)
        return self.metadata(self.row(c,identifier,thread,user))
    def generation_access(self,c,generation,thread,user):
        refs=c.execute(text('SELECT * FROM dev_generation_attachments WHERE generation_id=:g'),{'g':generation}).mappings().all()
        for ref in refs:
            row=self.row(c,ref['attachment_id'],thread,user)
            if row['state']!='reviewed' or row['sha256']!=ref['sha256'] or row['text_sha256']!=ref['text_sha256']:raise HTTPException(409,'Attachment authorization or reviewed version changed.')
            _,reviewable=self.content(row)
            if not reviewable:raise HTTPException(409,'Attachment remains quarantined.')
    def history_access(self,c,thread,user):
        refs=c.execute(text('SELECT DISTINCT a.attachment_id,a.sha256,a.text_sha256 FROM dev_generation_attachments a JOIN dev_ai_messages m ON m.generation_id=a.generation_id WHERE m.thread_id=:t LIMIT 201'),{'t':thread['id']}).mappings().all()
        if len(refs)>200:raise HTTPException(413,'Create a new conversation;attachment history exceeds development limits.')
        for ref in refs:
            row=self.row(c,ref['attachment_id'],thread,user)
            if row['state']!='reviewed' or row['sha256']!=ref['sha256'] or row['text_sha256']!=ref['text_sha256']:raise HTTPException(409,'Historical attachment context is no longer authorized.')
            _,reviewable=self.content(row)
            if not reviewable:raise HTTPException(409,'Historical attachment remains quarantined.')
        return bool(refs)
    def inherit(self,c,thread,generation):
        c.execute(text('INSERT OR IGNORE INTO dev_generation_attachments(generation_id,attachment_id,sha256,text_sha256) SELECT :g,a.attachment_id,a.sha256,a.text_sha256 FROM dev_generation_attachments a JOIN dev_ai_messages m ON m.generation_id=a.generation_id WHERE m.thread_id=:t AND a.generation_id!=:g'),{'g':generation,'t':thread['id']})
        return [dict(row) for row in c.execute(text('SELECT attachment_id,sha256,text_sha256 FROM dev_generation_attachments WHERE generation_id=:g ORDER BY attachment_id'),{'g':generation}).mappings()]
    def prepare(self,c,thread,user,ids,generation):
        if any(identifier<=0 for identifier in ids) or len(set(ids))!=len(ids):raise HTTPException(422,'Select distinct attachments.')
        documents=[]
        for identifier in ids:
            row=self.row(c,identifier,thread,user)
            if row['state']!='reviewed':raise HTTPException(409,'Review and explicitly authorize the attachment before model use.')
            body,reviewable=self.content(row)
            if not reviewable:raise HTTPException(409,'Suspicious instructions remain quarantined;no model transmission is allowed.')
            documents.append({**self.metadata(row),'text':body,'classification':'Untrusted reported document content;not public clinical evidence or verified findings.'})
        if len(json.dumps(documents))>18000:raise HTTPException(413,'Selected attachment context exceeds the provider budget;select a smaller source. No truncation occurred.')
        for row in documents:c.execute(text('INSERT INTO dev_generation_attachments(generation_id,attachment_id,sha256,text_sha256) VALUES(:g,:id,:h,:x)'),{'g':generation,'id':row['id'],'h':row['sha256'],'x':row['text_sha256']})
        return documents

def install_attachments(app,ctx,thread_for):
    store=AttachmentStore(ctx,thread_for)
    router=APIRouter(prefix='/api/dev/intelligence',dependencies=[Depends(ctx.same_origin_json)])
    uploads=APIRouter(prefix='/api/dev/intelligence')
    if not any(item.cls is AttachmentLimit for item in app.user_middleware):app.add_middleware(AttachmentLimit)
    @router.get('/threads/{identifier}/attachments')
    def listing(identifier:int,user=Depends(ctx.clinician)):
        with ctx.engine.begin() as c:
            thread=store.thread(c,identifier,user);items=store.listing(c,thread,user)
            try:cipher();available=True
            except HTTPException:available=False
            ctx.audit(c,user,'list_conversation_attachments','ai_thread',identifier)
            return {'items':items,'encryption_available':available,'max_bytes':MAX_BYTES,'max_text_characters':MAX_TEXT,'scope':'Protected storage ready. Preview and review each upload before use. Review is not a malware scan or clinical verification.'}
    @uploads.post('/threads/{identifier}/attachments')
    async def upload(identifier:int,request:Request,title:str=Form(...,min_length=1,max_length=200),file:UploadFile=File(...),user=Depends(ctx.clinician)):
        # ctx.clinician executes the shared request guard (Origin,Fetch Metadata,
        # session/rate budgets);multipart is allowed only on this bounded route.
        try:
            if request.headers.get('content-type','').split(';',1)[0].lower()!='multipart/form-data':raise HTTPException(415,'Multipart upload required.')
            if not title.strip():raise HTTPException(422,'Attachment title required.')
            if PurePath(file.filename or '').suffix.lower()!='.txt' or file.content_type not in ('text/plain','application/octet-stream'):raise HTTPException(415,'Only .txt plain-text attachments are supported;PDF/images/executables remain unsupported.')
            content=await file.read(MAX_BYTES+1)
            with ctx.engine.begin() as c:
                c.exec_driver_sql('BEGIN IMMEDIATE');thread=store.thread(c,identifier,user)
                return store.add(c,thread,user,title,content)
        finally:await file.close()
    @router.post('/threads/{identifier}/attachments/from-document')
    def import_document(identifier:int,body:ImportDocument,user=Depends(ctx.clinician)):
        from dev_chart import chart_document_allowed
        with ctx.engine.begin() as c:
            c.exec_driver_sql('BEGIN IMMEDIATE');thread=store.thread(c,identifier,user)
            document=c.execute(text('SELECT * FROM dev_chart_documents WHERE id=:id'),{'id':body.document_id}).mappings().first()
            if not document or document['patient_id']!=thread['patient_id'] or not chart_document_allowed(c,document,user,ctx):raise HTTPException(404,'Authorized matching document not found.')
            if document['file_type']!='txt':raise HTTPException(415,'Only the explicit synthetic text contract is supported.')
            content=bytes(document['content'])
            if hashlib.sha256(content).hexdigest()!=document['sha256']:raise HTTPException(409,'Original document integrity changed.')
            return store.add(c,thread,user,document['title'],content,document['id'])
    @router.get('/threads/{identifier}/attachments/{attachment}/preview')
    def preview(identifier:int,attachment:int,user=Depends(ctx.clinician)):
        with ctx.engine.begin() as c:
            thread=store.thread(c,identifier,user);row=store.row(c,attachment,thread,user);body,reviewable=store.content(row)
            ctx.audit(c,user,'preview_conversation_attachment','attachment',attachment)
            return {**store.metadata(row),'text':body,'reviewable':reviewable,'notice':'Untrusted reported text;not verified clinical findings. Suspicious instructions cannot be authorized. No malware-clean claim.'}
    @router.post('/threads/{identifier}/attachments/{attachment}/review')
    def review(identifier:int,attachment:int,body:Review,user=Depends(ctx.clinician)):
        if not body.authorize_provider_use:raise HTTPException(422,'Explicit authorization for model context is required.')
        with ctx.engine.begin() as c:
            c.exec_driver_sql('BEGIN IMMEDIATE');thread=store.thread(c,identifier,user);row=store.row(c,attachment,thread,user)
            if row['sha256']!=body.sha256:raise HTTPException(409,'Review hash changed.')
            _,reviewable=store.content(row)
            if not reviewable:raise HTTPException(409,'Suspicious instructions are quarantined;operator review is required. No provider use is allowed.')
            if row['state']!='reviewed':
                c.execute(text("UPDATE dev_conversation_attachments SET state='reviewed',reviewed_by=:u,reviewed_at=:at WHERE id=:id"),{'u':user['id'],'at':now(),'id':attachment});ctx.audit(c,user,'review_conversation_attachment','attachment',attachment)
            return store.metadata(store.row(c,attachment,thread,user))
    @router.post('/threads/{identifier}/attachments/{attachment}/delete')
    def delete(identifier:int,attachment:int,body:Delete,user=Depends(ctx.clinician)):
        if not body.confirm:raise HTTPException(422,'Explicit deletion confirmation required.')
        with ctx.engine.begin() as c:
            c.exec_driver_sql('BEGIN IMMEDIATE');thread=store.thread(c,identifier,user)
            row=c.execute(text('SELECT * FROM dev_conversation_attachments WHERE id=:id AND thread_id=:t'),{'id':attachment,'t':identifier}).mappings().first()
            if not row:raise HTTPException(404,'Attachment unavailable.')
            if row['state']!='deleted':
                c.execute(text("UPDATE dev_conversation_attachments SET state='deleted',encrypted_content=NULL,deleted_at=:at WHERE id=:id"),{'at':now(),'id':attachment});ctx.audit(c,user,'delete_conversation_attachment','attachment',attachment)
            return {'deleted':True,'source_document_retained':bool(row['source_document_id']),'scope':'Encrypted attachment payload removed logically;hash/audit and clinical conversations retained. Not forensic erasure,original-chart deletion or account deletion. Prior disclosed copies cannot be recalled.'}
    app.include_router(uploads);app.include_router(router)
    return store
