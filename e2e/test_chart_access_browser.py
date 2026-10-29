"""Actual390px Chromium grants/revokes full chart scope;no external services."""
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import httpx
from playwright.sync_api import sync_playwright,expect
from sqlalchemy import create_engine
from dev_portal import init_portal_schema,create_dev_user


class ChartAccessBrowserTests(unittest.TestCase):
    def test_patient_explicit_grant_and_revoke_block_other_and_originating_clinicians(self):
        with tempfile.TemporaryDirectory(prefix='raphael-chart-access-browser-') as directory:
            url='sqlite:///'+directory+'/synthetic.db';engine=create_engine(url);password=secrets.token_urlsafe(24)
            with patch.dict(os.environ,{'RAPHAEL_DEV_SYNTHETIC_ONLY':'1'}):
                init_portal_schema(engine)
                for name,role in (('creator','clinician'),('consultant','clinician'),('patient','patient')):create_dev_user(engine,'synthetic',name,password,role)
            engine.dispose()
            with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
            base=f'http://127.0.0.1:{port}'
            env=dict(os.environ,DATABASE_URL=url,RAPHAEL_DEV_PORTAL='1',RAPHAEL_DEV_SYNTHETIC_ONLY='1',QDRANT_URL='',QDRANT_API_KEY='',LLM_BASE_URL='',LLM_API_KEY='',STATIC_PATH='static',PYTHONDONTWRITEBYTECODE='1')
            with tempfile.TemporaryFile(mode='w+') as log:
                process=subprocess.Popen([sys.executable,'-m','uvicorn','main:app','--host','127.0.0.1','--port',str(port)],env=env,stdout=log,stderr=log,start_new_session=True)
                try:
                    with httpx.Client(base_url=base,trust_env=False) as web:
                        deadline=time.monotonic()+45
                        while True:
                            if process.poll() is not None or time.monotonic()>deadline:log.seek(0);self.fail(log.read())
                            try:
                                if web.get('/healthz').status_code==200:break
                            except httpx.TransportError:pass
                            time.sleep(.1)
                    with sync_playwright() as playwright:
                        browser=playwright.chromium.launch(executable_path=os.getenv('CHROMIUM_EXECUTABLE','/usr/bin/chromium'),headless=True,args=['--no-sandbox','--disable-dev-shm-usage'])
                        try:
                            contexts={name:browser.new_context(base_url=base,viewport={'width':390,'height':844}) for name in ('creator','consultant','patient')}
                            for name,context in contexts.items():self.assertEqual(context.request.post('/api/dev/login',data={'tenant':'synthetic','username':name,'password':password}).status,200)
                            creator=contexts['creator'];consultant=contexts['consultant'];patient=contexts['patient']
                            record=creator.request.post('/api/dev/patients',data={'name':'Synthetic chart access patient','patient_username':'patient'}).json();path='/api/dev/patients/'+str(record['id'])
                            self.assertEqual(consultant.request.get(path).status,404)
                            page=patient.new_page();errors=[];page.on('pageerror',lambda error:errors.append(str(error)))
                            page.goto('/static/chart-access.html');expect(page.locator('#chart-access')).to_be_visible();expect(page.locator('#permissions')).to_contain_text('originating_clinician')
                            page.locator('#clinician').select_option('consultant');page.locator('#purpose').fill('Synthetic specialist chart review');page.locator('#authorize').check();page.locator('#grant-form button').click();expect(page.locator('#status')).to_contain_text('Full-chart access granted')
                            self.assertEqual(consultant.request.get(path).status,200)
                            conversation=consultant.request.post(path+'/conversations',data={'title':'Synthetic authorized thread'}).json()['id']
                            page.once('dialog',lambda dialog:dialog.accept());page.get_by_role('button',name='Revoke consultant access',exact=True).click();expect(page.locator('#status')).to_contain_text('Future full-chart access revoked')
                            self.assertEqual(consultant.request.get(path).status,404);self.assertEqual(consultant.request.get('/api/dev/conversations/'+str(conversation)+'/messages').status,404)
                            self.assertEqual(consultant.request.get('/api/dev/inbox').json()['threads'],[])
                            page.once('dialog',lambda dialog:dialog.accept());page.get_by_role('button',name='Revoke creator access',exact=True).click();expect(page.locator('#history')).to_contain_text('Patient revoked')
                            self.assertEqual(creator.request.get(path).status,404)
                            page.reload();expect(page.locator('#history')).to_contain_text('Patient revoked')
                            self.assertTrue(page.evaluate('() => document.documentElement.scrollWidth <= innerWidth'));self.assertFalse(errors,errors)
                        finally:browser.close()
                finally:
                    os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
