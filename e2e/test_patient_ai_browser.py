"""Actual patient Chromium and SDK SSE against an explicitly nonclinical fixture."""
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from unittest.mock import patch
import httpx
from playwright.sync_api import sync_playwright,expect
from sqlalchemy import create_engine
from dev_portal import init_portal_schema,create_dev_user


class PatientAIBrowserTests(unittest.TestCase):
    def test_patient_education_consent_stream_sources_reload_cancel_and_revoke_mobile(self):
        requests=[]
        class Provider(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                body=json.loads(self.rfile.read(int(self.headers['Content-Length'])));requests.append(body)
                evidence=json.loads(body['messages'][0]['content'].split('EVIDENCE: ',1)[1])
                self.send_response(200);self.send_header('Content-Type','text/event-stream');self.end_headers()
                for value in ('Synthetic nonclinical educational fixture. ','['+evidence[0]['chunk_id']+']'):
                    try:self.wfile.write(('data: '+json.dumps({'id':'patient-fixture','choices':[{'index':0,'delta':{'content':value}}]})+'\n\n').encode());self.wfile.flush()
                    except (BrokenPipeError,ConnectionResetError):return
                    time.sleep(1 if 'cancel' in body['messages'][-1]['content'] else .05)
                try:self.wfile.write(b'data: [DONE]\n\n');self.wfile.flush()
                except (BrokenPipeError,ConnectionResetError):pass
        provider=ThreadingHTTPServer(('127.0.0.1',0),Provider);threading.Thread(target=provider.serve_forever,daemon=True).start();self.addCleanup(provider.server_close);self.addCleanup(provider.shutdown)
        with tempfile.TemporaryDirectory(prefix='raphael-patient-browser-') as directory:
            database='sqlite:///'+directory+'/synthetic.db';engine=create_engine(database);password=secrets.token_urlsafe(28)
            with patch.dict(os.environ,{'RAPHAEL_DEV_SYNTHETIC_ONLY':'1'}):
                init_portal_schema(engine)
                for name,role in (('doctor','clinician'),('patient','patient'),('outsider','patient')):create_dev_user(engine,'synthetic',name,password,role)
            engine.dispose()
            with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
            base=f'http://127.0.0.1:{port}'
            environment=dict(os.environ,DATABASE_URL=database,RAPHAEL_DEV_PORTAL='1',RAPHAEL_DEV_SYNTHETIC_ONLY='1',LLM_BASE_URL=f'http://127.0.0.1:{provider.server_port}/v1',LLM_API_KEY='synthetic-test-fixture-only',LLM_MODEL='synthetic-fixture',RAPHAEL_ALLOW_LOCAL_LLM='1',RAPHAEL_STREAM_EVIDENCE_BACKEND='sqlite',RAPHAEL_VERIFY_CLAIMS='0',QDRANT_URL='',QDRANT_API_KEY='',STATIC_PATH='static',PYTHONDONTWRITEBYTECODE='1')
            with tempfile.TemporaryFile(mode='w+') as log:
                process=subprocess.Popen([sys.executable,'-m','uvicorn','main:app','--host','127.0.0.1','--port',str(port)],env=environment,cwd=Path(__file__).resolve().parents[1],stdout=log,stderr=log,start_new_session=True)
                try:
                    with httpx.Client(base_url=base,trust_env=False) as web:
                        deadline=time.monotonic()+45
                        while True:
                            if process.poll() is not None or time.monotonic()>deadline:raise AssertionError('Synthetic app did not become ready.')
                            try:
                                if web.get('/healthz').status_code==200:break
                            except httpx.TransportError:pass
                            time.sleep(.1)
                    with sync_playwright() as p:
                        browser=p.chromium.launch(executable_path=os.getenv('CHROMIUM_EXECUTABLE','/usr/bin/chromium'),headless=True,args=['--no-sandbox','--disable-dev-shm-usage'])
                        try:
                            doctor=browser.new_context(base_url=base);doctor.request.post('/api/dev/login',data={'tenant':'synthetic','username':'doctor','password':password})
                            patient=doctor.request.post('/api/dev/patients',data={'name':'Synthetic patient educational record','patient_username':'patient','dob':'2000-01-01'}).json()
                            self.assertIn('id',patient)
                            source=doctor.request.post('/api/dev/knowledge/sources',data={'title':'Nonclinical patient fixture','url':'https://example.org/patient-fixture','content':'Violet calibrator baseline. Test-owned synthetic nonclinical education fixture.','license_declaration':'Test-owned synthetic fixture content.','permission_attested':True,'tier':'D','org':'Synthetic fixture','source_date':'2024-01-01'}).json()
                            self.assertEqual(doctor.request.post(f"/api/dev/knowledge/sources/{source['id']}/approve",data={}).status,200)
                            context=browser.new_context(base_url=base,viewport={'width':390,'height':844});page=context.new_page();errors=[];page.on('pageerror',lambda error:errors.append(str(error)))
                            page.goto('/');page.locator('#tenant').fill('synthetic');page.locator('#username').fill('patient');page.locator('#password').fill(password);page.locator('#login-form button').click();expect(page.locator('#workspace')).to_be_visible()
                            page.goto('/static/patient-ai.html');expect(page.locator('#patient-ai')).to_be_visible();page.locator('#title').fill('Synthetic patient education');page.locator('#consent').check();page.locator('#context').check();page.locator('#new-thread button').click();expect(page.locator('#thread-title')).to_have_text('Synthetic patient education')
                            page.locator('#prompt').fill('violet calibrator');page.locator('#prompt-form button').click();expect(page.locator('#status')).to_contain_text('Educational draft complete');expect(page.locator('#messages')).to_contain_text('Synthetic nonclinical educational fixture.');expect(page.locator('#evidence')).to_contain_text('Nonclinical patient fixture')
                            thread=context.request.get('/api/dev/patient-ai/threads').json()[0]['id']
                            citation=page.locator('#evidence a').first.get_attribute('href');self.assertIn('/api/dev/patient-ai/threads/',citation);self.assertEqual(context.request.get(citation).status,200)
                            outsider=browser.new_context(base_url=base);outsider.request.post('/api/dev/login',data={'tenant':'synthetic','username':'outsider','password':password});self.assertEqual(outsider.request.get(f'/api/dev/patient-ai/threads/{thread}').status,404)
                            page.reload();expect(page.locator('#patient-ai')).to_be_visible();page.locator('#threads button').first.click();expect(page.locator('#messages')).to_contain_text('Synthetic nonclinical educational fixture.');expect(page.locator('#evidence')).to_contain_text('Nonclinical patient fixture')
                            page.locator('#prompt').fill('violet calibrator cancel');page.locator('#prompt-form button').click();expect(page.locator('#messages .chat').last).to_contain_text('Synthetic nonclinical educational fixture.');page.locator('#stop').click();expect(page.locator('#status')).to_contain_text('Generation stopped')
                            page.once('dialog',lambda dialog:dialog.accept());page.locator('#revoke').click();expect(page.locator('#scope')).to_contain_text('AI consent revoked');expect(page.locator('#send')).to_be_disabled()
                            self.assertEqual(context.request.post(f'/api/dev/patient-ai/threads/{thread}/stream',data={'content':'No request after revoked consent'}).status,409)
                            page.goto('/static/account.html');expect(page.locator('#account')).to_be_visible()
                            page.locator('#display-name').fill('Synthetic account preference')
                            page.locator('#profile-form button').click();expect(page.locator('#status')).to_contain_text('preferences saved')
                            page.reload();expect(page.locator('#display-name')).to_have_value('Synthetic account preference')
                            page.locator('#confirm-deletion').check();page.locator('#request-deletion').click()
                            expect(page.locator('#deletion-requests')).to_contain_text('REQUESTED')
                            page.locator('#deletion-requests button').click();expect(page.locator('#deletion-requests')).to_contain_text('CANCELLED')
                            page.once('dialog',lambda dialog:dialog.accept())
                            with page.expect_download() as download_info:page.locator('#export').click()
                            downloaded=download_info.value
                            self.assertEqual(downloaded.suggested_filename,'raphael-synthetic-patient-export.json')
                            exported=json.loads(Path(downloaded.path()).read_text());self.assertIn('patient_ai',exported)
                            self.assertNotIn('password_hash',json.dumps(exported))
                            self.assertTrue(page.evaluate('() => document.documentElement.scrollWidth <= innerWidth'));self.assertFalse(errors,errors);self.assertGreaterEqual(len(requests),1);self.assertIn('patient_selected_approved_notes_and_self_report',requests[0]['messages'][0]['content'])
                        finally:browser.close()
                finally:
                    os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
