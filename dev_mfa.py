"""Opt-in encrypted TOTP factors for isolated synthetic accounts, not identity proofing."""
import base64
import hashlib
import hmac
import os
import secrets
import struct
import time
from datetime import datetime,timedelta,timezone
from urllib.parse import quote
from cryptography.fernet import Fernet, InvalidToken
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field, SecretStr
from sqlalchemy import text

SCHEMA='''
CREATE TABLE IF NOT EXISTS dev_mfa_factors (
 user_id INTEGER PRIMARY KEY REFERENCES dev_users(id), encrypted_secret TEXT NOT NULL,
 enabled INTEGER NOT NULL DEFAULT 0 CHECK(enabled IN(0,1)), last_step INTEGER NOT NULL DEFAULT -1,
 pending_expires INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS dev_mfa_recovery (
 user_id INTEGER NOT NULL REFERENCES dev_users(id), code_hash TEXT NOT NULL,
 PRIMARY KEY(user_id,code_hash)
);
'''

def cipher():
    try:
        key=os.environ.get('RAPHAEL_MFA_ENCRYPTION_KEY','')
        if not key:raise ValueError()
        return Fernet(key.encode('ascii'))
    except (ValueError,UnicodeError):
        raise HTTPException(503,'MFA key unavailable. Operator must restore the protected MFA encryption key.') from None

def totp(secret,step):
    digest=hmac.new(base64.b32decode(secret),struct.pack('>Q',step),hashlib.sha1).digest()
    offset=digest[-1]&15
    return str((struct.unpack('>I',digest[offset:offset+4])[0]&0x7fffffff)%1000000).zfill(6)

def verify(c,user_id,code,*,pending=False,at=None):
    row=c.execute(text('SELECT * FROM dev_mfa_factors WHERE user_id=:u'),{'u':user_id}).mappings().first()
    if not row or (not pending and not row['enabled']):return not pending
    if not code:return False
    now=int(time.time() if at is None else at)
    if pending and (row['enabled'] or row['pending_expires']<now):return False
    try:secret=cipher().decrypt(row['encrypted_secret'].encode()).decode('ascii')
    except (InvalidToken,UnicodeError):raise HTTPException(503,'MFA key unavailable. Operator must restore the protected MFA encryption key.') from None
    if len(code)==6 and code.isascii() and code.isdigit():
        for step in (now//30-1,now//30,now//30+1):
            if step>row['last_step'] and hmac.compare_digest(totp(secret,step),code):
                return c.execute(text('UPDATE dev_mfa_factors SET last_step=:s WHERE user_id=:u AND last_step<:s'),{'u':user_id,'s':step}).rowcount==1
    if not pending:
        return c.execute(text('DELETE FROM dev_mfa_recovery WHERE user_id=:u AND code_hash=:h'),{'u':user_id,'h':hashlib.sha256(code.encode()).hexdigest()}).rowcount==1
    return False

class Credential(BaseModel):
    model_config=ConfigDict(extra='forbid')
    password:SecretStr=Field(min_length=1,max_length=128)
class Verification(Credential):
    code:SecretStr=Field(min_length=1,max_length=100)

def install_mfa(app,ctx):
    from dev_portal import _password_matches,COOKIE_NAME
    router=APIRouter(prefix='/api/dev/account/mfa',dependencies=[Depends(ctx.same_origin_json)])
    def password(c,user,value):
        stored=c.execute(text('SELECT password_hash FROM dev_users WHERE id=:u AND active=1'),{'u':user['id']}).scalar()
        if not stored or not _password_matches(value,stored):raise HTTPException(403,'Credential verification failed.')
    def revoke_others(c,user,request):
        token=hashlib.sha256(request.cookies.get(COOKIE_NAME,'').encode()).hexdigest()
        c.execute(text('DELETE FROM dev_sessions WHERE user_id=:u AND token_hash!=:h'),{'u':user['id'],'h':token})
    def management_user(user=Depends(ctx.user)):
        # Persist a per-account budget even when the subsequent credential check fails.
        key=hashlib.sha256(('mfa-management:'+str(user['id'])).encode()).hexdigest()
        now=datetime.now(timezone.utc)
        with ctx.engine.begin() as c:
            c.exec_driver_sql('BEGIN IMMEDIATE')
            c.execute(text('DELETE FROM dev_login_attempts WHERE reset_at<=:at'),{'at':now.isoformat()})
            row=c.execute(text('SELECT failures FROM dev_login_attempts WHERE key_hash=:h'),{'h':key}).first()
            if row and row[0]>=5:raise HTTPException(429,'Too many MFA management attempts. Wait five minutes.',headers={'Retry-After':'300'})
            c.execute(text('INSERT INTO dev_login_attempts(key_hash,failures,reset_at) VALUES(:h,1,:e) ON CONFLICT(key_hash) DO UPDATE SET failures=failures+1'),{'h':key,'e':(now+timedelta(minutes=5)).isoformat()})
        return user
    @router.get('')
    def status(user=Depends(ctx.user)):
        with ctx.engine.connect() as c:
            enabled=bool(c.execute(text('SELECT enabled FROM dev_mfa_factors WHERE user_id=:u'),{'u':user['id']}).scalar())
            count=c.execute(text('SELECT count(*) FROM dev_mfa_recovery WHERE user_id=:u'),{'u':user['id']}).scalar()
        try:cipher();available=True
        except HTTPException:available=False
        return {'enabled':enabled,'enrollment_available':available,'recovery_codes_remaining':count,'identity_proofing':False}
    @router.post('/enroll')
    def enroll(body:Credential,response:Response,user=Depends(management_user)):
        encryption=cipher()
        with ctx.engine.begin() as c:
            c.exec_driver_sql('BEGIN IMMEDIATE');password(c,user,body.password.get_secret_value())
            if c.execute(text('SELECT enabled FROM dev_mfa_factors WHERE user_id=:u'),{'u':user['id']}).scalar():raise HTTPException(409,'MFA is already enabled.')
            secret=base64.b32encode(secrets.token_bytes(20)).decode('ascii')
            c.execute(text('INSERT INTO dev_mfa_factors(user_id,encrypted_secret,pending_expires) VALUES(:u,:s,:e) ON CONFLICT(user_id) DO UPDATE SET encrypted_secret=excluded.encrypted_secret,pending_expires=excluded.pending_expires,last_step=-1'),{'u':user['id'],'s':encryption.encrypt(secret.encode()).decode(),'e':int(time.time())+600})
            ctx.audit(c,user,'mfa_enrollment_started','user',user['id'])
        response.headers['Cache-Control']='no-store'
        label=quote('Raphael:'+user['tenant']+':'+user['username'],safe='')
        return {'secret':secret,'otpauth_uri':'otpauth://totp/'+label+'?secret='+secret+'&issuer=Raphael&algorithm=SHA1&digits=6&period=30','expires_in_seconds':600,'enabled':False}
    @router.post('/confirm')
    def confirm(body:Verification,request:Request,response:Response,user=Depends(management_user)):
        with ctx.engine.begin() as c:
            c.exec_driver_sql('BEGIN IMMEDIATE');password(c,user,body.password.get_secret_value())
            if not verify(c,user['id'],body.code.get_secret_value(),pending=True):raise HTTPException(403,'Factor verification failed or enrollment expired.')
            c.execute(text('UPDATE dev_mfa_factors SET enabled=1 WHERE user_id=:u'),{'u':user['id']})
            codes=[secrets.token_hex(10) for _ in range(10)]
            c.execute(text('DELETE FROM dev_mfa_recovery WHERE user_id=:u'),{'u':user['id']})
            for code in codes:c.execute(text('INSERT INTO dev_mfa_recovery(user_id,code_hash) VALUES(:u,:h)'),{'u':user['id'],'h':hashlib.sha256(code.encode()).hexdigest()})
            revoke_others(c,user,request);ctx.audit(c,user,'mfa_enabled','user',user['id'])
        response.headers['Cache-Control']='no-store'
        return {'enabled':True,'recovery_codes':codes,'notice':'Save these codes privately; each works once with your password. Other sessions were revoked. Wait for the next authenticator code before signing in again.'}
    @router.post('/disable')
    def disable(body:Verification,request:Request,user=Depends(management_user)):
        with ctx.engine.begin() as c:
            c.exec_driver_sql('BEGIN IMMEDIATE');password(c,user,body.password.get_secret_value())
            row=c.execute(text('SELECT enabled FROM dev_mfa_factors WHERE user_id=:u'),{'u':user['id']}).scalar()
            if not row or not verify(c,user['id'],body.code.get_secret_value()):raise HTTPException(403,'Factor verification failed.')
            c.execute(text('DELETE FROM dev_mfa_recovery WHERE user_id=:u'),{'u':user['id']});c.execute(text('DELETE FROM dev_mfa_factors WHERE user_id=:u'),{'u':user['id']})
            revoke_others(c,user,request);ctx.audit(c,user,'mfa_disabled','user',user['id'])
        return {'enabled':False,'other_sessions_revoked':True}
    app.include_router(router)
