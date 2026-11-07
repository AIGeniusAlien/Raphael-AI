"""Actual launcher/restart verification;temporary synthetic data and keys only."""
import argparse
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch
import httpx
from sqlalchemy import create_engine
from dev_keys import create_key_file,load_key_file
from dev_runtime import migrate
from dev_portal import create_dev_user
from dev_mfa import totp


def run(output,encrypted_database=False):
    stage='setup';result={'status':'failed','real_launcher':True,'real_provider_requests':False,'production_data_used':False}
    process=None
    def stop():
        nonlocal process
        if process is None:return
        try:os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
        process=None
        # The launcher parent can exit before uvicorn finishes graceful shutdown.
        # Wait for the old listener to close before considering restart readiness.
        deadline=time.monotonic()+10
        with httpx.Client(base_url=base,timeout=.3,trust_env=False) as check:
            while time.monotonic()<deadline:
                try:check.get('/readyz')
                except httpx.ConnectError:return
                except httpx.TransportError:pass
                time.sleep(.05)
        raise RuntimeError('Owned listener did not stop within its bound.')
    try:
        with tempfile.TemporaryDirectory(prefix='raphael-protected-runtime-') as directory,tempfile.TemporaryFile(mode='w+') as log:
            private=Path(directory);database=private/'synthetic.db';url='sqlite:///'+str(database)
            document=private/'document.key';mfa=private/'mfa.key'
            database_key=private/'database.key'
            create_key_file(document);create_key_file(mfa)
            if encrypted_database:create_key_file(database_key)
            password=secrets.token_urlsafe(30)
            from protected_database import application_engine
            with patch.dict(os.environ,{'RAPHAEL_DEV_SYNTHETIC_ONLY':'1'}):engine=application_engine(url,encrypted=encrypted_database,encryption_key=load_key_file(database_key) if encrypted_database else None)
            try:
                with patch.dict(os.environ,{'RAPHAEL_DEV_SYNTHETIC_ONLY':'1'}):
                    migrate(engine)
                    for name,role in (('doctor','clinician'),('patient','patient')):create_dev_user(engine,'synthetic',name,password,role)
            finally:engine.dispose()
            environment={k:v for k,v in os.environ.items() if k not in ('RAPHAEL_DOCUMENT_ENCRYPTION_KEY','RAPHAEL_MFA_ENCRYPTION_KEY','RAPHAEL_DATABASE_ENCRYPTION_KEY','RAPHAEL_DATABASE_ENCRYPTED')}
            with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
            base='http://127.0.0.1:'+str(port)
            command=[sys.executable,'-m','dev_start','--database',str(database),'--document-key-file',str(document),'--mfa-key-file',str(mfa),'--port',str(port)]
            if encrypted_database:command+=['--database-key-file',str(database_key)]
            def start():
                nonlocal process
                process=subprocess.Popen(command,env=environment,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
                deadline=time.monotonic()+30
                with httpx.Client(base_url=base,timeout=2,trust_env=False) as check:
                    while time.monotonic()<deadline:
                        if process.poll() is not None:raise RuntimeError('Launcher exited.')
                        try:
                            if check.get('/readyz').status_code==200:return
                        except httpx.TransportError:pass
                        time.sleep(.1)
                raise RuntimeError('Launcher readiness deadline exceeded.')
            try:
                stage='first_start';start()
                with httpx.Client(base_url=base,timeout=10,trust_env=False) as web:
                    assert web.get('/healthz').status_code==200
                    login={'tenant':'synthetic','username':'doctor','password':password}
                    web.post('/api/dev/login',json=login).raise_for_status()
                    record=web.post('/api/dev/patients',json={'name':'Synthetic protected runtime','patient_username':'patient'});record.raise_for_status();patient=record.json()['id']
                    thread=web.post('/api/dev/intelligence/threads',json={'title':'Synthetic protected restart','patient_id':patient});thread.raise_for_status();path='/api/dev/intelligence/threads/'+str(thread.json()['id'])
                    content=f'RAPHAEL_SYNTHETIC_PATIENT_ID: {patient}\nSynthetic protected restart text.'.encode()
                    attachment=web.post(path+'/attachments',data={'title':'Synthetic restart source'},files={'file':('synthetic.txt',content,'text/plain')});attachment.raise_for_status();item=attachment.json()
                    web.post(path+'/attachments/'+str(item['id'])+'/review',json={'sha256':item['sha256'],'authorize_provider_use':True}).raise_for_status()
                    enrollment=web.post('/api/dev/account/mfa/enroll',json={'password':password});enrollment.raise_for_status()
                    factor=totp(enrollment.json()['secret'],int(time.time())//30)
                    confirmation=web.post('/api/dev/account/mfa/confirm',json={'password':password,'code':factor});confirmation.raise_for_status();recovery=confirmation.json()['recovery_codes'][0]
                    web.post('/api/dev/logout',json={}).raise_for_status()
                stop();stage='same_key_restart';start()
                with httpx.Client(base_url=base,timeout=10,trust_env=False) as web:
                    assert web.post('/api/dev/login',json=login).status_code==401
                    web.post('/api/dev/login',json={**login,'otp':recovery}).raise_for_status()
                    assert web.get('/api/dev/account/mfa').json()['enabled'] is True
                    assert web.get('/api/dev/account/storage').json()['sqlite_pages_encrypted']==encrypted_database
                    stored=web.get(path+'/attachments');stored.raise_for_status();assert stored.json()['items'][0]['id']==item['id']
                    preview=web.get(path+'/attachments/'+str(item['id'])+'/preview');preview.raise_for_status()
                    assert 'Synthetic protected restart text.' in preview.json()['text']
                    web.post('/api/dev/logout',json={}).raise_for_status()
                result.update(status='passed',healthz=True,readyz=True,starts=2,synthetic_patient_persisted=True,encrypted_attachment_decrypts_after_restart=True,mfa_key_persists=True,password_only_rejected_after_restart=True,recovery_authentication_after_restart=True,provider_mode='disabled',sqlite_pages_encrypted=encrypted_database)
            finally:stop()
    except Exception as error:result.update(failed_stage=stage,error_type=type(error).__name__)
    if output:output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result));return 0 if result['status']=='passed' else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path);parser.add_argument('--encrypted-database',action='store_true');args=parser.parse_args();return run(args.output,args.encrypted_database)
if __name__=='__main__':raise SystemExit(main())
