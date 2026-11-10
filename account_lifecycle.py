"""Credential-verified synthetic account lifecycle with conservative retention.

Limited credential/profile erasure is NOT clinical-record or legal account erasure.
Policies are operator attestations,not independent legal approval. No automatic purge.
"""
import hashlib
import secrets
from datetime import datetime,timedelta,timezone
from fastapi import APIRouter,Depends,HTTPException,Request,Response
from pydantic import BaseModel,ConfigDict,Field,SecretStr,StrictBool,StrictInt,field_validator
from sqlalchemy import text

SCHEMA='''
CREATE TABLE IF NOT EXISTS dev_account_lifecycle (
 user_id INTEGER PRIMARY KEY REFERENCES dev_users(id),
 state TEXT NOT NULL CHECK(state IN ('deactivated','reactivated','credentials_erased')),
 revision INTEGER NOT NULL,deactivated_at TEXT,erasure_requested_at TEXT,
 reactivated_at TEXT,completed_at TEXT,policy_reference TEXT
);
CREATE TABLE IF NOT EXISTS dev_retention_holds (
 user_id INTEGER PRIMARY KEY REFERENCES dev_users(id),active INTEGER NOT NULL,
 reference TEXT NOT NULL,revision INTEGER NOT NULL,updated_at TEXT NOT NULL,updated_by INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS dev_lifecycle_policies (
 tenant TEXT PRIMARY KEY,reference TEXT NOT NULL,minimum_inactive_days INTEGER NOT NULL,
 enable_credential_erasure INTEGER NOT NULL,revision INTEGER NOT NULL,updated_at TEXT NOT NULL,updated_by INTEGER NOT NULL
);
'''
def init_lifecycle_schema(engine):
    from database import execute_sqlite_script
    from dev_portal import _require_synthetic
    _require_synthetic(engine)
    with engine.begin() as c:execute_sqlite_script(c,SCHEMA)

def now():return datetime.now(timezone.utc).isoformat()
def update_account_state(c,user_id,active):
    """Hook existing admin activation;never resurrect erased credentials."""
    if not c.exec_driver_sql("SELECT 1 FROM sqlite_master WHERE type='table' AND name='dev_account_lifecycle'").first():return
    current=c.execute(text('SELECT state FROM dev_account_lifecycle WHERE user_id=:u'),{'u':user_id}).scalar()
    if current=='credentials_erased':
        if active:raise HTTPException(409,'Credential-erased accounts cannot be reactivated or reassigned to another identity.')
        return
    if active:
        c.execute(text("UPDATE dev_account_lifecycle SET state='reactivated',revision=revision+1,reactivated_at=:at,erasure_requested_at=NULL WHERE user_id=:u"),{'u':user_id,'at':now()})
    else:
        c.execute(text("INSERT INTO dev_account_lifecycle(user_id,state,revision,deactivated_at) VALUES(:u,'deactivated',1,:at) ON CONFLICT(user_id) DO UPDATE SET revision=revision+1,deactivated_at=CASE WHEN state='deactivated' THEN deactivated_at ELSE excluded.deactivated_at END,erasure_requested_at=CASE WHEN state='deactivated' THEN erasure_requested_at ELSE NULL END,state='deactivated'"),{'u':user_id,'at':now()})

class Credential(BaseModel):
    model_config=ConfigDict(extra='forbid')
    password:SecretStr=Field(min_length=1,max_length=128)
    code:SecretStr=Field(default=SecretStr(''),max_length=100)
class Deactivate(Credential):
    confirm:StrictBool
    request_credential_erasure:StrictBool=False
class ChangePassword(Credential):
    new_password:SecretStr=Field(min_length=12,max_length=128)
    confirm:StrictBool

class Policy(Credential):
    reference:str=Field(min_length=5,max_length=200)
    minimum_inactive_days:StrictInt=Field(ge=0,le=36500)
    enable_credential_erasure:StrictBool
    expected_revision:StrictInt=Field(ge=0)
    @field_validator('reference')
    @classmethod
    def reference_required(cls,value):
        if len(value.strip())<5:raise ValueError('Meaningful operator policy reference required.')
        return value.strip()
class Hold(Credential):
    active:StrictBool
    reference:str=Field(min_length=5,max_length=200)
    expected_revision:StrictInt=Field(ge=0)
    @field_validator('reference')
    @classmethod
    def reference_required(cls,value):
        if len(value.strip())<5:raise ValueError('Meaningful hold/release reference required.')
        return value.strip()
class Erase(Credential):
    confirm:StrictBool
    target_username:str=Field(min_length=1,max_length=100)
    expected_revision:StrictInt=Field(gt=0)
    policy_revision:StrictInt=Field(gt=0)

def install_lifecycle(app,ctx):
    from dev_portal import COOKIE_NAME,_password_matches,_password_hash
    from dev_mfa import verify
    router=APIRouter(prefix='/api/dev/account/lifecycle',dependencies=[Depends(ctx.same_origin_json)])
    def budget(user):
        key=hashlib.sha256(('lifecycle:'+str(user['id'])).encode()).hexdigest();at=datetime.now(timezone.utc)
        with ctx.engine.begin() as c:
            c.exec_driver_sql('BEGIN IMMEDIATE')
            c.execute(text('DELETE FROM dev_login_attempts WHERE reset_at<=:at'),{'at':at.isoformat()})
            count=c.execute(text('SELECT failures FROM dev_login_attempts WHERE key_hash=:h'),{'h':key}).scalar()
            if count is not None and count>=5:raise HTTPException(429,'Too many lifecycle management attempts. Wait five minutes.',headers={'Retry-After':'300'})
            c.execute(text('INSERT INTO dev_login_attempts(key_hash,failures,reset_at) VALUES(:h,1,:at) ON CONFLICT(key_hash) DO UPDATE SET failures=failures+1'),{'h':key,'at':(at+timedelta(minutes=5)).isoformat()})
        return user
    def management_user(user=Depends(ctx.user)):return budget(user)
    def admin(user=Depends(ctx.user)):
        if user['role']!='org_admin':raise HTTPException(403,'Organization administrator required.')
        return user
    def admin_management(user=Depends(admin)):return budget(user)
    def credentials(c,user,body):
        stored=c.execute(text('SELECT password_hash FROM dev_users WHERE id=:u AND tenant=:t AND active=1'),{'u':user['id'],'t':user['tenant']}).scalar()
        if not stored or not _password_matches(body.password.get_secret_value(),stored):raise HTTPException(403,'Credential verification failed.')
        if not verify(c,user['id'],body.code.get_secret_value()):raise HTTPException(403,'An unused authenticator or recovery code is required.')
    def target(c,identifier,user):
        row=c.execute(text('SELECT id,username,role,active FROM dev_users WHERE id=:u AND tenant=:t'),{'u':identifier,'t':user['tenant']}).mappings().first()
        if not row:raise HTTPException(404,'Account not found.')
        return row
    @router.get('')
    def own_status(user=Depends(ctx.user)):
        with ctx.engine.begin() as c:
            record=c.execute(text('SELECT state,revision,deactivated_at,erasure_requested_at FROM dev_account_lifecycle WHERE user_id=:u'),{'u':user['id']}).mappings().first()
            hold=bool(c.execute(text('SELECT active FROM dev_retention_holds WHERE user_id=:u'),{'u':user['id']}).scalar())
            return {'active':True,'lifecycle':dict(record) if record else None,'retention_hold':hold,'can_self_deactivate':user['role']!='org_admin','scope':'Deactivation blocks sign-in and revokes sessions. Clinical records,audit history,original documents and backups remain retained. Limited credential/profile erasure requires a separate owner request,operator policy,no hold and explicit administrator approval;it is not full medical-record erasure.'}
    @router.post('/deactivate')
    def deactivate(body:Deactivate,response:Response,user=Depends(management_user)):
        if not body.confirm:raise HTTPException(422,'Explicit deactivation confirmation required.')
        if user['role']=='org_admin':raise HTTPException(409,'Administrators must arrange an authorized handover;self-deactivation is not permitted.')
        with ctx.engine.begin() as c:
            c.exec_driver_sql('BEGIN IMMEDIATE');credentials(c,user,body)
            update_account_state(c,user['id'],False)
            c.execute(text('UPDATE dev_account_lifecycle SET erasure_requested_at=:at WHERE user_id=:u'),{'at':now() if body.request_credential_erasure else None,'u':user['id']})
            c.execute(text('UPDATE dev_users SET active=0 WHERE id=:u'),{'u':user['id']});c.execute(text('DELETE FROM dev_sessions WHERE user_id=:u'),{'u':user['id']})
            ctx.audit(c,user,'self_deactivate_account','user',user['id'])
            if body.request_credential_erasure:ctx.audit(c,user,'request_limited_credential_erasure','user',user['id'])
        response.delete_cookie(COOKIE_NAME,path='/api/dev',httponly=True,samesite='strict')
        return {'deactivated':True,'sessions_revoked':True,'clinical_records_deleted':False,'credential_erasure_requested':body.request_credential_erasure,'notice':'Account deactivated. All sessions revoked. Records retained. An organization administrator can reactivate access unless a separately authorized credential-erasure action is completed.'}
    @router.post('/password')
    def change_password(body:ChangePassword,request:Request,response:Response,user=Depends(management_user)):
        """Change a synthetic password and revoke every session, including this one."""
        if not body.confirm:raise HTTPException(422,'Explicit password change confirmation required.')
        new_value=body.new_password.get_secret_value()
        if not new_value.strip() or new_value!=new_value.strip():
            raise HTTPException(422,'Password must not be blank or padded with whitespace.')
        token=request.cookies.get(COOKIE_NAME,'')
        with ctx.engine.begin() as c:
            c.exec_driver_sql('BEGIN IMMEDIATE')
            session_hash=hashlib.sha256(token.encode()).hexdigest()
            if not token or not c.execute(text('SELECT 1 FROM dev_sessions s JOIN dev_users u ON u.id=s.user_id WHERE s.token_hash=:h AND s.user_id=:uid AND s.expires_at>:at AND u.active=1 AND u.tenant=:tenant AND u.role=:role'),{'h':session_hash,'uid':user['id'],'at':now(),'tenant':user['tenant'],'role':user['role']}).scalar():
                raise HTTPException(401,'Session revoked; password was not changed.')
            credentials(c,user,body)
            if _password_matches(new_value,c.execute(text('SELECT password_hash FROM dev_users WHERE id=:u'),{'u':user['id']}).scalar()):
                raise HTTPException(422,'Choose a different password.')
            c.execute(text('UPDATE dev_users SET password_hash=:h WHERE id=:u AND tenant=:t AND active=1'),{'h':_password_hash(new_value),'u':user['id'],'t':user['tenant']})
            c.execute(text('DELETE FROM dev_sessions WHERE user_id=:u'),{'u':user['id']})
            ctx.audit(c,user,'change_account_password','user',user['id'])
        response.delete_cookie(COOKIE_NAME,path='/api/dev',httponly=True,samesite='strict')
        response.headers['Cache-Control']='no-store'
        return {'password_changed':True,'sessions_revoked':True,'notice':'Password changed. Sign in again on every device.'}

    @router.get('/administration')
    def administration(user=Depends(admin)):
        with ctx.engine.begin() as c:
            policy=c.execute(text('SELECT reference,minimum_inactive_days,enable_credential_erasure,revision FROM dev_lifecycle_policies WHERE tenant=:t'),{'t':user['tenant']}).mappings().first()
            rows=c.execute(text('SELECT u.id,u.username,u.role,u.active,l.state,l.revision,l.deactivated_at,l.erasure_requested_at,h.active AS hold_active,h.reference AS hold_reference,h.revision AS hold_revision FROM dev_users u LEFT JOIN dev_account_lifecycle l ON l.user_id=u.id LEFT JOIN dev_retention_holds h ON h.user_id=u.id WHERE u.tenant=:t ORDER BY u.id DESC LIMIT 201'),{'t':user['tenant']}).mappings().all()
            ctx.audit(c,user,'review_account_lifecycle','organization')
            return {'policy':dict(policy) if policy else None,'accounts':[dict(row) for row in rows[:200]],'truncated':len(rows)>200,'scope':'Technical operator attestations only. This is not legal approval,clinical retention policy or full erasure. Clinical/audit records and original documents are always retained by this mechanism.'}
    @router.post('/policy')
    def policy(body:Policy,user=Depends(admin_management)):
        with ctx.engine.begin() as c:
            c.exec_driver_sql('BEGIN IMMEDIATE');credentials(c,user,body)
            version=c.execute(text('SELECT revision FROM dev_lifecycle_policies WHERE tenant=:t'),{'t':user['tenant']}).scalar() or 0
            if version!=body.expected_revision:raise HTTPException(409,'Policy changed;refresh before applying a new version.')
            c.execute(text('INSERT INTO dev_lifecycle_policies(tenant,reference,minimum_inactive_days,enable_credential_erasure,revision,updated_at,updated_by) VALUES(:t,:r,:days,:enabled,:v,:at,:u) ON CONFLICT(tenant) DO UPDATE SET reference=excluded.reference,minimum_inactive_days=excluded.minimum_inactive_days,enable_credential_erasure=excluded.enable_credential_erasure,revision=excluded.revision,updated_at=excluded.updated_at,updated_by=excluded.updated_by'),{'t':user['tenant'],'r':body.reference.strip(),'days':body.minimum_inactive_days,'enabled':int(body.enable_credential_erasure),'v':version+1,'at':now(),'u':user['id']})
            ctx.audit(c,user,'record_credential_retention_policy','organization')
        return {'revision':version+1,'clinical_retention_policy_created':False}
    @router.post('/accounts/{identifier}/hold')
    def hold(identifier:int,body:Hold,user=Depends(admin_management)):
        with ctx.engine.begin() as c:
            c.exec_driver_sql('BEGIN IMMEDIATE');credentials(c,user,body);target(c,identifier,user)
            version=c.execute(text('SELECT revision FROM dev_retention_holds WHERE user_id=:u'),{'u':identifier}).scalar() or 0
            if version!=body.expected_revision:raise HTTPException(409,'Hold changed;refresh before continuing.')
            c.execute(text('INSERT INTO dev_retention_holds(user_id,active,reference,revision,updated_at,updated_by) VALUES(:u,:a,:r,:v,:at,:actor) ON CONFLICT(user_id) DO UPDATE SET active=excluded.active,reference=excluded.reference,revision=excluded.revision,updated_at=excluded.updated_at,updated_by=excluded.updated_by'),{'u':identifier,'a':int(body.active),'r':body.reference.strip(),'v':version+1,'at':now(),'actor':user['id']})
            ctx.audit(c,user,'set_retention_hold' if body.active else 'release_retention_hold','user',identifier)
        return {'revision':version+1,'active':body.active}
    @router.post('/accounts/{identifier}/erase-credentials')
    def erase(identifier:int,body:Erase,user=Depends(admin_management)):
        if not body.confirm:raise HTTPException(422,'Explicit limited-erasure confirmation required.')
        with ctx.engine.begin() as c:
            c.exec_driver_sql('BEGIN IMMEDIATE');credentials(c,user,body);record=target(c,identifier,user)
            lifecycle=c.execute(text('SELECT * FROM dev_account_lifecycle WHERE user_id=:u'),{'u':identifier}).mappings().first()
            policy=c.execute(text('SELECT * FROM dev_lifecycle_policies WHERE tenant=:t'),{'t':user['tenant']}).mappings().first()
            if identifier==user['id'] or record['active'] or not lifecycle or lifecycle['state']!='deactivated' or not lifecycle['erasure_requested_at']:raise HTTPException(409,'A deactivated account with an explicit owner credential-erasure request is required.')
            if lifecycle['revision']!=body.expected_revision or record['username']!=body.target_username:raise HTTPException(409,'Account version or confirmation changed;refresh and re-review the target.')
            if not policy or not policy['enable_credential_erasure'] or policy['revision']!=body.policy_revision:raise HTTPException(409,'A matching explicitly enabled operator policy is required. No default erasure rule exists.')
            if c.execute(text('SELECT active FROM dev_retention_holds WHERE user_id=:u'),{'u':identifier}).scalar():raise HTTPException(409,'An active retention hold blocks credential erasure.')
            try:eligible=datetime.fromisoformat(lifecycle['deactivated_at'])+timedelta(days=policy['minimum_inactive_days'])<=datetime.now(timezone.utc)
            except (ValueError,TypeError):raise HTTPException(409,'Lifecycle dates require operator review.') from None
            if not eligible:raise HTTPException(409,'The configured minimum inactivity period has not elapsed.')
            c.execute(text('UPDATE dev_users SET username=:name,password_hash=:h,active=0 WHERE id=:u'),{'u':identifier,'name':'erased-'+secrets.token_hex(16),'h':_password_hash(secrets.token_urlsafe(48))})
            for table in ('dev_sessions','dev_mfa_recovery','dev_mfa_factors','dev_account_profiles'):
                c.execute(text('DELETE FROM '+table+' WHERE user_id=:u'),{'u':identifier})
            c.execute(text("UPDATE dev_account_lifecycle SET state='credentials_erased',revision=revision+1,completed_at=:at,policy_reference=:r WHERE user_id=:u"),{'u':identifier,'at':now(),'r':policy['reference']})
            ctx.audit(c,user,'erase_account_credentials_retain_clinical_records','user',identifier)
        return {'credentials_erased':True,'clinical_records_deleted':False,'original_documents_deleted':False,'audit_records_deleted':False,'backups_erased':False,'notice':'Login credentials,stored factor material and account preferences were removed logically. Clinical records,original documents,conversations,audit records and backups remain retained. This is not full account or medical-record erasure and does not promise forensic wiping.'}
    app.include_router(router)
