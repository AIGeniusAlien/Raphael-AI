"""Prepare actual main:app on ephemeral, simulator-trusted HTTPS; CI only.

No remote service credentials or patient data. A random, temporary synthetic
password is compiled into the UI TEST bundle only, never the Raphael app.
"""
import json
import os
from pathlib import Path
import platform
import secrets
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def prepare():
    if platform.system()!='Darwin' or os.getenv('GITHUB_ACTIONS')!='true':
        raise SystemExit('This helper is restricted to ephemeral macOS GitHub Actions runners.')
    destination=os.environ['RAPHAEL_SIMULATOR_DESTINATION']
    identifier=destination.split('id=',1)[1]
    private=Path(tempfile.mkdtemp(prefix='raphael-native-ci-'))
    password=secrets.token_urlsafe(32)
    # No inherited provider credentials or database may reach this local backend.
    environment={key:value for key,value in os.environ.items() if key in ('PATH','HOME','TMPDIR','LANG')}
    database='sqlite:///'+str(private/'synthetic.db')
    os.environ['RAPHAEL_DEV_SYNTHETIC_ONLY']='1'
    from sqlalchemy import create_engine
    from dev_runtime import migrate
    from dev_portal import create_dev_user
    engine=create_engine(database)
    try:
        migrate(engine)
        for name,role in (('native-patient','patient'),('native-clinician','clinician')):
            create_dev_user(engine,'native-ci',name,password,role)
    finally:engine.dispose()
    certificate=private/'certificate.pem';key=private/'key.pem';config=private/'openssl.cnf'
    config.write_text('[req]\ndistinguished_name=dn\nx509_extensions=extensions\nprompt=no\n[dn]\nCN=raphael-ci.test\n[extensions]\nsubjectAltName=DNS:raphael-ci.test\nbasicConstraints=critical,CA:TRUE\nkeyUsage=critical,digitalSignature,keyEncipherment,keyCertSign\nextendedKeyUsage=serverAuth\n')
    subprocess.run(['openssl','req','-x509','-newkey','rsa:2048','-nodes','-sha256','-days','2','-config',str(config),'-keyout',str(key),'-out',str(certificate)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    key.chmod(0o600)
    subprocess.run(['sudo','sh','-c','echo "127.0.0.1 raphael-ci.test" >> /etc/hosts'],check=True)
    subprocess.run(['xcrun','simctl','boot',identifier],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    subprocess.run(['xcrun','simctl','bootstatus',identifier,'-b'],check=True,timeout=240)
    subprocess.run(['xcrun','simctl','keychain',identifier,'add-root-cert',str(certificate)],check=True)
    server_environment={**environment,'DATABASE_URL':database,'RAPHAEL_DEV_PORTAL':'1','RAPHAEL_DEV_SYNTHETIC_ONLY':'1','PYTHONDONTWRITEBYTECODE':'1'}
    command=['sudo','env',*[key+'='+value for key,value in server_environment.items()],sys.executable,'-m','uvicorn','main:app','--host','127.0.0.1','--port','443','--ssl-keyfile',str(key),'--ssl-certfile',str(certificate),'--no-access-log']
    with (private/'backend.log').open('w') as log:
        process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=log,start_new_session=True)
    (private/'process.json').write_text(json.dumps({'process_group':process.pid}))
    import ssl
    import httpx
    tls=ssl.create_default_context(cafile=str(certificate))
    with httpx.Client(base_url='https://raphael-ci.test',verify=tls,trust_env=False,timeout=10) as client:
        deadline=time.monotonic()+60
        while True:
            if process.poll() is not None or time.monotonic()>deadline:raise RuntimeError('Synthetic HTTPS backend failed readiness;inspect private runner log.')
            try:
                if client.get('/readyz').status_code==200:break
            except httpx.TransportError:pass
            time.sleep(.2)
        client.post('/api/dev/login',json={'tenant':'native-ci','username':'native-clinician','password':password}).raise_for_status()
        record=client.post('/api/dev/patients',json={'name':'Synthetic native patient','patient_username':'native-patient','dob':'1994-01-01'});record.raise_for_status()
        note=client.post(f"/api/dev/patients/{record.json()['id']}/notes",json={'content':'Synthetic native approved note. No real clinical findings.'});note.raise_for_status()
        client.post(f"/api/dev/notes/{note.json()['id']}/approve",json={}).raise_for_status()
        client.post('/api/dev/logout',json={}).raise_for_status()
    test=ROOT/'mobile/ios/RaphaelUITests/RaphaelUITests.swift'
    source=test.read_text()
    if source.count('__CI_SYNTHETIC_PASSWORD__')!=1:raise RuntimeError('Expected exactly one private UI-test placeholder.')
    test.write_text(source.replace('__CI_SYNTHETIC_PASSWORD__',password))
    with open(os.environ['GITHUB_ENV'],'a') as output:
        output.write('RAPHAEL_AUTHENTICATED_CI=1\nRAPHAEL_NATIVE_CI_PRIVATE='+str(private)+'\n')
    print('Actual synthetic main:app HTTPS ready;simulator trust installed;no provider requests.')


def cleanup():
    private=Path(os.environ['RAPHAEL_NATIVE_CI_PRIVATE'])
    group=json.loads((private/'process.json').read_text())['process_group']
    subprocess.run(['sudo','kill','-TERM','--','-'+str(group)],check=False)
    remaining=subprocess.check_output(['sudo',sys.executable,'-c',"import sqlite3,sys; c=sqlite3.connect('file:'+sys.argv[1]+'?mode=ro',uri=True); print(c.execute('SELECT count(*) FROM dev_sessions').fetchone()[0]); c.close()",str(private/'synthetic.db')],text=True,timeout=15).strip()
    if remaining!='0':raise RuntimeError('Native workflow left an active synthetic server session;logout was not verified.')
    print('::notice title=Raphael native backend verification::'+json.dumps({'synthetic_sessions_remaining':0,'provider_requests':0,'backend':'actual_main_app_https'}))
    print('Ephemeral synthetic backend stopped. Runner destroys its private database and test certificate.')


if __name__=='__main__':
    cleanup() if sys.argv[1:]==['cleanup'] else prepare()
