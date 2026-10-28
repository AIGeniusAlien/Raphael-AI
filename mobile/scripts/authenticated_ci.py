"""Prepare actual main:app on ephemeral, simulator-trusted HTTPS; CI only.

No remote service credentials or patient data. A random, temporary synthetic
password is compiled into the UI TEST bundle only, never the Raphael app.
"""
import json
import os
from pathlib import Path
import platform
from collections import Counter
import secrets
import shutil
import signal
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
    (private/'password').write_text(password)
    (private/'password').chmod(0o600)
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
    server_environment={**environment,'DATABASE_URL':database,'RAPHAEL_DEV_PORTAL':'1','RAPHAEL_DEV_SYNTHETIC_ONLY':'1','PYTHONDONTWRITEBYTECODE':'1','GITHUB_ACTIONS':'true'}
    forwarder=shutil.which('socat')
    if not forwarder:raise RuntimeError('Install socat for the isolated loopback443 TCP forwarder.')
    # Keep the actual application unprivileged. The forwarder carries encrypted
    # bytes only;TLS still terminates in uvicorn with the verified test certificate.
    command=[sys.executable,str(ROOT/'mobile/scripts/ci_backend.py'),'--key',str(key),'--cert',str(certificate)]
    with (private/'backend.log').open('w') as log:
        process=subprocess.Popen(command,cwd=ROOT,env=server_environment,stdout=log,stderr=log,start_new_session=True)
    with (private/'forwarder.log').open('w') as log:
        bridge=subprocess.Popen(['sudo','-n',forwarder,'TCP-LISTEN:443,bind=127.0.0.1,reuseaddr,fork','TCP:127.0.0.1:8443'],stdout=log,stderr=log,start_new_session=True)
    (private/'process.json').write_text(json.dumps({'server_group':process.pid,'forwarder_group':bridge.pid}))
    with open(os.environ['GITHUB_ENV'],'a') as output:output.write('RAPHAEL_NATIVE_CI_PRIVATE='+str(private)+'\n')
    import ssl
    import httpx
    tls=ssl.create_default_context(cafile=str(certificate))
    with httpx.Client(base_url='https://raphael-ci.test',verify=tls,trust_env=False,timeout=10) as client:
        deadline=time.monotonic()+60
        last_status,last_transport_error=None,None
        while True:
            if process.poll() is not None or bridge.poll() is not None or time.monotonic()>deadline:
                # This CI-only backend has no provider credentials or real PHI.
                # Expose startup exceptions,not the private account password.
                startup=(private/'backend.log').read_text(errors='replace')[-10000:]+'\nForwarder tail:\n'+(private/'forwarder.log').read_text(errors='replace')[-1000:]
                Path('native-backend-startup.log').write_text(startup.replace(password,'[SYNTHETIC PASSWORD REDACTED]'))
                print('Backend startup error: '+json.dumps({'process_exit_code':process.poll(),'forwarder_exit_code':bridge.poll(),'last_http_status':last_status,'last_transport_error_type':last_transport_error}),flush=True)
                raise RuntimeError('Synthetic HTTPS backend failed readiness;bounded startup log published separately.')
            try:
                last_status=client.get('/readyz').status_code
                if last_status==200:
                    process.send_signal(signal.SIGUSR1)
                    break
            except httpx.TransportError as error:last_transport_error=type(error).__name__
            time.sleep(.2)
        client.post('/api/dev/login',json={'tenant':'native-ci','username':'native-clinician','password':password}).raise_for_status()
        record=client.post('/api/dev/patients',json={'name':'Synthetic native patient','patient_username':'native-patient','dob':'1994-01-01'});record.raise_for_status()
        note=client.post(f"/api/dev/patients/{record.json()['id']}/notes",json={'content':'Synthetic native approved note. No real clinical findings.'});note.raise_for_status()
        client.post(f"/api/dev/notes/{note.json()['id']}/approve",json={}).raise_for_status()
        client.post('/api/dev/patients',json={'name':'Synthetic native unlinked record'}).raise_for_status()
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
    groups=json.loads((private/'process.json').read_text())
    subprocess.run(['sudo','-n','kill','-TERM','--','-'+str(groups['forwarder_group'])],check=False)
    try:os.killpg(groups['server_group'],signal.SIGTERM)
    except ProcessLookupError:pass
    import sqlite3
    with sqlite3.connect('file:'+str(private/'synthetic.db')+'?mode=ro',uri=True) as connection:
        remaining=connection.execute('SELECT count(*) FROM dev_sessions').fetchone()[0]
        events={event:count for event,count in connection.execute("SELECT event,count(*) FROM dev_audit WHERE event IN ('login','logout','read_patient','read_notes','list_patients') GROUP BY event")}
    routes=Counter()
    for line in (private/'backend.log').read_text(errors='replace').splitlines():
        if line.startswith('RAPHAEL_CI_HTTP '):
            event=json.loads(line.split(' ',1)[1])
            routes[event['method']+' '+event['path']+' '+str(event['status'])]+=1
    print('::notice title=Raphael native HTTP trace::'+json.dumps(dict(routes)),flush=True)
    print('::notice title=Raphael native backend verification::'+json.dumps({'synthetic_sessions_remaining':remaining,'provider_requests':0,'backend':'actual_main_app_https','audit_event_counts_including_provisioning':events}),flush=True)
    if remaining!=0:raise RuntimeError('Native workflow left an active synthetic server session;logout was not verified.')
    print('Ephemeral synthetic backend stopped. Runner destroys its private database and test certificate.')


if __name__=='__main__':
    cleanup() if sys.argv[1:]==['cleanup'] else prepare()
