"""Actual Chromium UI + actual SDK HTTP stream against an explicit synthetic fixture.

The provider below is a deterministic test server, not a live medical model.
No microphone or external clinical evidence is used or claimed validated.
"""
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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

import httpx
from playwright.sync_api import sync_playwright, expect
from sqlalchemy import create_engine
from dev_portal import init_portal_schema, create_dev_user


class IntelligenceBrowserTests(unittest.TestCase):
    def test_streamed_fixture_encounter_consent_incremental_notes_edit_and_signoff(self):
        requests=[]
        class ProviderFixture(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                data=json.loads(self.rfile.read(int(self.headers['Content-Length'])));requests.append(data)
                evidence=json.loads(data['messages'][0]['content'].split('EVIDENCE: ',1)[1])
                self.send_response(200);self.send_header('Content-Type','text/event-stream');self.end_headers()
                for value in ['Synthetic nonclinical fixture response. ', '['+evidence[0]['chunk_id']+']']:
                    try:
                        self.wfile.write(('data: '+json.dumps({'id':'test-fixture','choices':[{'index':0,'delta':{'content':value}}]})+'\n\n').encode());self.wfile.flush()
                    except (BrokenPipeError,ConnectionResetError):
                        return
                    time.sleep(1 if 'cancel' in data['messages'][-1]['content'] else .1)
                self.wfile.write(b'data: [DONE]\n\n');self.wfile.flush()
        server=ThreadingHTTPServer(('127.0.0.1',0),ProviderFixture);server_thread=threading.Thread(target=server.serve_forever,daemon=True);server_thread.start()
        self.addCleanup(server.server_close);self.addCleanup(server.shutdown)
        with tempfile.TemporaryDirectory(prefix='raphael-intelligence-browser-') as directory:
            url='sqlite:///'+directory+'/synthetic.db';engine=create_engine(url);password=secrets.token_urlsafe(24)
            with patch.dict(os.environ,{'RAPHAEL_DEV_SYNTHETIC_ONLY':'1'}):
                init_portal_schema(engine);create_dev_user(engine,'synthetic','physician',password,'clinician')
            engine.dispose()
            with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
            base=f'http://127.0.0.1:{port}'
            environment=dict(os.environ,DATABASE_URL=url,RAPHAEL_DEV_PORTAL='1',RAPHAEL_DEV_SYNTHETIC_ONLY='1',LLM_BASE_URL=f'http://127.0.0.1:{server.server_port}/v1',LLM_API_KEY='synthetic-provider-fixture',RAPHAEL_ALLOW_LOCAL_LLM='1',QDRANT_URL='',STATIC_PATH='static',PYTHONDONTWRITEBYTECODE='1')
            with tempfile.TemporaryFile(mode='w+') as log:
                process=subprocess.Popen([sys.executable,'-m','uvicorn','main:app','--host','127.0.0.1','--port',str(port)],cwd=Path(__file__).resolve().parents[1],env=environment,stdout=log,stderr=log,start_new_session=True)
                try:
                    with httpx.Client(base_url=base,trust_env=False) as client:
                        deadline=time.monotonic()+45
                        while True:
                            if process.poll() is not None or time.monotonic()>deadline:log.seek(0);self.fail(log.read())
                            try:
                                if client.get('/healthz').status_code==200:break
                            except httpx.TransportError:pass
                            time.sleep(.1)
                    with sync_playwright() as p:
                        browser=p.chromium.launch(executable_path=os.getenv('CHROMIUM_EXECUTABLE','/usr/bin/chromium'),headless=True,args=['--no-sandbox','--disable-dev-shm-usage'])
                        try:
                            context=browser.new_context(base_url=base);page=context.new_page();errors=[];page.on('pageerror',lambda error:errors.append(str(error)))
                            context.request.post('/api/dev/login',data={'tenant':'synthetic','username':'physician','password':password})
                            patient=context.request.post('/api/dev/patients',data={'name':'Synthetic intelligence patient'}).json()
                            source=context.request.post('/api/dev/knowledge/sources',data={'title':'Nonclinical synthetic fixture','url':'https://example.org/synthetic-fixture','content':'Violet calibrator baseline measurement. Synthetic nonclinical fixture text.','license_declaration':'Test-owned nonclinical fixture text.','permission_attested':True,'tier':'D','org':'Synthetic test','source_date':'2024-01-01'}).json()
                            self.assertIn('id',source);self.assertEqual(context.request.post(f"/api/dev/knowledge/sources/{source['id']}/approve",data={}).status,200)
                            page.goto('/static/intelligence.html?patient='+str(patient['id']));expect(page.locator('#intelligence')).to_be_visible()
                            page.locator('#title').fill('Synthetic clinical thread');page.locator('#thread-form button').click();expect(page.locator('#thread-title')).to_have_text('Synthetic clinical thread')
                            page.locator('#prompt').fill('violet calibrator');page.locator('#prompt-form button').click();expect(page.locator('#messages')).to_contain_text('Synthetic nonclinical fixture response.');expect(page.locator('#status')).to_contain_text('Draft complete')
                            expect(page.locator('#evidence')).to_contain_text('Nonclinical synthetic fixture');expect(page.locator('#evidence a[href^="/api/dev/knowledge/citations/"]')).to_have_attribute('href',__import__('re').compile(r'/api/dev/knowledge/citations/'))
                            # Explicit browser-only SSE fixture tests the blocked terminal UI,
                            # never used as proof of provider or clinical validation.
                            page.route('**/api/dev/intelligence/threads/*/stream',lambda route:route.fulfill(status=200,content_type='text/event-stream',body='event: blocked\ndata: '+json.dumps({'detail':'Synthetic unsupported-claim fixture. Not approved.'})+'\n\n'),times=1)
                            page.locator('#prompt').fill('Synthetic blocked UI case');page.locator('#prompt-form button').click()
                            expect(page.locator('#status')).to_contain_text('Draft blocked: Synthetic unsupported-claim fixture.')
                            expect(page.locator('#send')).to_be_enabled()
                            page.locator('#prompt').fill('violet calibrator cancel');page.locator('#prompt-form button').click()
                            expect(page.locator('#messages .chat-entry').last.locator('.chat-text')).to_contain_text('Synthetic nonclinical fixture response.')
                            expect(page.locator('#send')).to_be_disabled();page.locator('#cancel').click();expect(page.locator('#status')).to_contain_text('Generation stopped')
                            page.locator('#new-encounter').click();expect(page.locator('#encounter option')).to_have_count(1)
                            page.locator('#start-session').click();expect(page.locator('#status')).to_contain_text('Confirm recording consent')
                            page.locator('#consent').check();page.locator('#start-session').click();expect(page.locator('#session-id')).not_to_have_value('')
                            page.locator('#manual').fill('Synthetic reported symptom.');page.locator('#append').click();expect(page.locator('#note')).to_have_value(__import__('re').compile('Synthetic reported symptom'))
                            page.locator('#manual').fill('Additional synthetic history.');page.locator('#append').click();expect(page.locator('#note')).to_have_value(__import__('re').compile('Additional synthetic history'))
                            page.locator('#note').fill('Physician edited synthetic SOAP note.');page.locator('#save-note').click();expect(page.locator('#note-status')).to_contain_text('Version 3')
                            page.once('dialog',lambda dialog:dialog.accept());page.locator('#approve').click();expect(page.locator('#note-status')).to_contain_text('approved');expect(page.locator('#audit')).to_contain_text('approve_active_note')
                            session_id=page.locator('#session-id').input_value();page.locator('#stop-session').click();expect(page.locator('#voice-status')).to_contain_text('consent revoked')
                            page.reload();expect(page.locator('#intelligence')).to_be_visible();page.locator('#session-id').fill(session_id);page.locator('#resume').click();expect(page.locator('#note')).to_have_value('Physician edited synthetic SOAP note.');expect(page.locator('#note-status')).to_contain_text('approved')
                            pending=[]
                            page.route('**/api/dev/intelligence/sessions/'+session_id,lambda route:route.continue_() if pending else pending.append(route))
                            page.locator('#resume').click()
                            page.wait_for_function('() => document.querySelector("#session-id").value !== ""')
                            self.assertTrue(pending)
                            page.locator('#new-draft').click()
                            with page.expect_response(lambda response:response.url.endswith('/intelligence/sessions/'+session_id)):
                                pending[0].continue_()
                            page.evaluate('() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))')
                            expect(page.locator('#note')).to_have_value('');expect(page.locator('#thread-title')).to_have_text('New conversation')
                            self.assertTrue(requests[0]['stream']);self.assertFalse(errors,errors)
                        finally:browser.close()
                finally:
                    os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
