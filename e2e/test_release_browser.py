"""Real browser candidate boundary, own records, preferences, feedback and pause."""
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
from fastapi import FastAPI
from fastapi.testclient import TestClient
from playwright.sync_api import expect,sync_playwright
from sqlalchemy import create_engine
from dev_chart import install_chart
from dev_portal import create_dev_user,install_dev_portal
from dev_runtime import migrate
from dev_verify import isolated_environment


class ReleaseBrowserTests(unittest.TestCase):
    def test_candidate_patient_workflow_direct_bypass_denial_and_operator_recovery(self):
        with tempfile.TemporaryDirectory(prefix='raphael-release-browser-') as directory:
            private=Path(directory);url='sqlite:///'+str(private/'synthetic.db');engine=create_engine(url);password=secrets.token_urlsafe(24)
            with patch.dict(os.environ,{'RAPHAEL_DEV_SYNTHETIC_ONLY':'1'}):
                migrate(engine)
                for name,role in [('doctor','clinician'),('patient','patient'),('other','patient')]:create_dev_user(engine,'synthetic',name,password,role)
                bootstrap=FastAPI();ctx=install_dev_portal(bootstrap,engine);install_chart(bootstrap,ctx)
                with TestClient(bootstrap) as setup:
                    setup.post('/api/dev/login',json={'tenant':'synthetic','username':'doctor','password':password}).raise_for_status()
                    patient=setup.post('/api/dev/patients',json={'name':'Synthetic preview own record','patient_username':'patient'}).json()
                    other=setup.post('/api/dev/patients',json={'name':'OTHER-PRIVATE-RECORD','patient_username':'other'}).json()
                    setup.post(f"/api/dev/patients/{patient['id']}/entries",json={'kind':'history','label':'Synthetic fixture history','text':'Nonclinical synthetic record;no medical advice.'}).raise_for_status()
                    note=setup.post(f"/api/dev/patients/{patient['id']}/notes",json={'content':'Synthetic approved shared note.'}).json()
                    setup.post(f"/api/dev/notes/{note['id']}/approve",json={}).raise_for_status()
                    setup.post(f"/api/dev/patients/{patient['id']}/notes",json={'content':'UNSIGNED-DOCTOR-DRAFT'}).raise_for_status()
                    document=setup.post(f"/api/dev/patients/{patient['id']}/documents",data={'title':'Synthetic shared document','visibility':'care_team'},files={'file':('synthetic-preview.txt',b'Synthetic nonclinical release preview document.','text/plain')})
                    document.raise_for_status()
            engine.dispose()
            control=private/'control';control.mkdir(mode=0o700);pause=control/'paused'
            with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
            base=f'http://127.0.0.1:{port}';env=isolated_environment();env.update(DATABASE_URL=url,RAPHAEL_DEV_PORTAL='1',RAPHAEL_DEV_SYNTHETIC_ONLY='1',RAPHAEL_RELEASE_SCOPE='consumer_candidate',RAPHAEL_RELEASE_PAUSE_FILE=str(pause),STATIC_PATH='static')
            with tempfile.TemporaryFile(mode='w+') as log:
                process=subprocess.Popen([sys.executable,'-m','uvicorn','main:app','--host','127.0.0.1','--port',str(port)],cwd=Path(__file__).resolve().parents[1],env=env,stdout=log,stderr=log,start_new_session=True)
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
                        browser=p.chromium.launch(executable_path=os.getenv('CHROMIUM_EXECUTABLE','/usr/bin/chromium'),headless=True,args=['--no-sandbox','--disable-dev-shm-usage']);context=browser.new_context(base_url=base,viewport={'width':390,'height':844});page=context.new_page();errors=[];page.on('pageerror',lambda error:errors.append(str(error)))
                        try:
                            page.goto('/');expect(page).to_have_url(base+'/static/release-preview.html');expect(page.locator('#login-panel')).to_be_visible()
                            page.locator('#tenant').fill('synthetic');page.locator('#username').fill('patient');page.locator('#password').fill(password);page.locator('#login-form button').click()
                            expect(page.locator('#workspace')).to_be_visible();expect(page.locator('#entries')).to_contain_text('Synthetic fixture history');expect(page.locator('#entries')).to_contain_text('Recorded information: Nonclinical synthetic record;no medical advice.');expect(page.locator('#entries')).not_to_contain_text('encounter_id');expect(page.locator('#entries')).not_to_contain_text('null');expect(page.locator('#notes')).to_contain_text('Synthetic approved shared note.');expect(page.locator('#notes')).not_to_contain_text('UNSIGNED-DOCTOR-DRAFT');expect(page.locator('#documents')).to_contain_text('Synthetic shared document');self.assertEqual(page.locator('#patient option').count(),1)
                            download=context.request.get(page.locator('#documents a').get_attribute('href'));self.assertEqual(download.status,200);self.assertEqual(download.body(),b'Synthetic nonclinical release preview document.')
                            self.assertEqual(context.request.get(f"/api/dev/patients/{other['id']}/notes").status,404)
                            for target in ['/api/dev/procedures','/api/dev/jobs','/api/dev/intelligence/threads','/static/intelligence.html','/static/procedures.html']:
                                self.assertEqual(context.request.get(target).status,403,target)
                            self.assertEqual(context.request.post('/api/dev/intelligence/threads',data={}).status,403)
                            page.locator('#display-name').fill('Synthetic saved preference');page.locator('#profile-form button').click();expect(page.locator('#status')).to_contain_text('Preferences saved')
                            page.locator('#feedback-title-input').fill('Synthetic preview feedback');page.locator('#feedback-detail').fill('Synthetic reproducible software-only feedback without medical records.');page.locator('#feedback-ack').check();page.locator('#feedback-form button').click();expect(page.locator('#reports')).to_contain_text('Synthetic preview feedback');expect(page.locator('#status')).to_contain_text('received internally')
                            page.reload();expect(page.locator('#display-name')).to_have_value('Synthetic saved preference');expect(page.locator('#reports')).to_contain_text('Synthetic preview feedback')
                            fd=os.open(pause,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.close(fd);page.locator('#refresh').click();expect(page.locator('#status')).to_contain_text('temporarily paused');self.assertEqual(context.request.get('/api/dev/account/export').status,503);expect(page.locator('#entries')).to_be_empty();expect(page.locator('#notes')).to_be_empty();expect(page.locator('#workspace')).to_be_visible();expect(page.locator('#logout')).to_be_visible()
                            page.locator('#display-name').fill('Preference saved while paused');page.locator('#profile-form button').click();expect(page.locator('#status')).to_contain_text('Preferences saved')
                            pause.unlink();page.locator('#refresh').click();expect(page.locator('#entries')).to_contain_text('Synthetic fixture history');expect(page.locator('#status')).to_contain_text('Records refreshed');expect(page.locator('#policy')).not_to_contain_text('paused')
                            output=Path(os.getenv('RAPHAEL_VISUAL_OUTPUT',directory));output.mkdir(parents=True,exist_ok=True)
                            for name,width,size in [('release-mobile',390,16),('release-enlarged',320,20),('release-desktop',1280,16)]:
                                page.set_viewport_size({'width':width,'height':844});page.evaluate('(size)=>{document.documentElement.style.fontSize=size+"px";document.documentElement.style.fontFamily="DejaVu Sans,sans-serif"}',size);self.assertTrue(page.evaluate('document.documentElement.scrollWidth <= innerWidth'),name);page.screenshot(path=str(output/(name+'.png')),full_page=True)
                            page.locator('#logout').click();expect(page.locator('#login-panel')).to_be_visible();expect(page.locator('#workspace')).to_be_hidden();self.assertEqual(context.request.get('/api/dev/patients').status,401)
                            self.assertEqual(context.request.post('/api/dev/login',data={'tenant':'synthetic','username':'doctor','password':password}).status,200);self.assertEqual(context.request.get('/api/dev/patients').status,403);self.assertEqual(context.request.post('/api/dev/logout',data={}).status,200)
                            self.assertFalse(errors,errors)
                        finally:browser.close()
                finally:
                    os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)
