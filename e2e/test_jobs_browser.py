"""Actual browser→durable queue→separate worker→retained outcome;synthetic only."""
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
import unittest
from unittest.mock import patch
import httpx
from playwright.sync_api import expect,sync_playwright
from sqlalchemy import create_engine
from dev_portal import init_portal_schema,create_dev_user
from dev_verify import isolated_environment

class JobsBrowserTests(unittest.TestCase):
    def test_background_inventory_browser_closure_worker_and_cancellation(self):
        with tempfile.TemporaryDirectory(prefix='raphael-jobs-browser-') as directory:
            url='sqlite:///'+directory+'/synthetic.db';engine=create_engine(url);password=secrets.token_urlsafe(24)
            with patch.dict(os.environ,{'RAPHAEL_DEV_SYNTHETIC_ONLY':'1'}):
                init_portal_schema(engine)
                for tenant,name,role in [('synthetic','doctor','clinician'),('synthetic','patient','patient'),('other','outsider','clinician')]:create_dev_user(engine,tenant,name,password,role)
            engine.dispose()
            with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
            base=f'http://127.0.0.1:{port}';env=isolated_environment();env.update(DATABASE_URL=url,RAPHAEL_DEV_PORTAL='1',RAPHAEL_DEV_SYNTHETIC_ONLY='1',STATIC_PATH='static')
            with tempfile.TemporaryFile(mode='w+') as log:
                app=subprocess.Popen([sys.executable,'-m','uvicorn','main:app','--host','127.0.0.1','--port',str(port)],cwd=Path(__file__).resolve().parents[1],env=env,stdout=log,stderr=log,start_new_session=True)
                try:
                    with httpx.Client(base_url=base,trust_env=False) as client:
                        deadline=time.monotonic()+45
                        while True:
                            if app.poll() is not None or time.monotonic()>deadline:log.seek(0);self.fail(log.read())
                            try:
                                if client.get('/healthz').status_code==200:break
                            except httpx.TransportError:pass
                            time.sleep(.1)
                    with sync_playwright() as p:
                        browser=p.chromium.launch(executable_path=os.getenv('CHROMIUM_EXECUTABLE','/usr/bin/chromium'),headless=True,args=['--no-sandbox','--disable-dev-shm-usage'])
                        try:
                            context=browser.new_context(base_url=base,viewport={'width':390,'height':844});page=context.new_page();errors=[];page.on('pageerror',lambda error:errors.append(str(error)))
                            self.assertEqual(context.request.post('/api/dev/login',data={'tenant':'synthetic','username':'doctor','password':password}).status,200)
                            source=context.request.post('/api/dev/knowledge/sources',data={'title':'Synthetic inventory source','url':'https://example.org/job-inventory','content':'Violet calibrator inventory fixture. This is synthetic nonclinical text.','permission_attested':True,'license_declaration':'Test-owned synthetic fixture text only.','org':'Synthetic fixture'});self.assertEqual(source.status,200)
                            self.assertEqual(context.request.post(f"/api/dev/knowledge/sources/{source.json()['id']}/approve",data={}).status,200)
                            page.goto('/static/jobs.html');expect(page.locator('#workspace')).to_be_visible();expect(page.locator('#kind option')).to_have_count(1);expect(page.locator('#worker-status')).to_contain_text('No recent active worker')
                            page.locator('#enqueue').click();expect(page.locator('#jobs')).to_contain_text('queued');expect(page.locator('#status')).to_contain_text('Execution requires a separate worker')
                            identifier=context.request.get('/api/dev/jobs').json()['items'][0]['id'];page.close()
                            # Browser closure does not own or execute the durable task.
                            page=context.new_page();page.on('pageerror',lambda error:errors.append(str(error)));page.goto('/static/jobs.html');expect(page.locator('#jobs')).to_contain_text('queued')
                            completed=subprocess.run([sys.executable,'-m','dev_jobs','--database-url',url,'--tenant','synthetic'],env=env,cwd=Path(__file__).resolve().parents[1],text=True,capture_output=True,timeout=15)
                            self.assertEqual(completed.returncode,0,completed.stdout+completed.stderr);self.assertEqual(json.loads(completed.stdout)['jobs_attempted'],1)
                            page.locator('#refresh').click();expect(page.locator('#jobs')).to_contain_text('succeeded');expect(page.locator('#jobs')).to_contain_text('"approved": 1');expect(page.locator('#jobs')).to_contain_text('"approved_chunks": 1')
                            page.reload();expect(page.locator(f'article[data-job-id="{identifier}"]')).to_contain_text('succeeded')
                            page.locator('#delay').fill('60');page.locator('#enqueue').click();expect(page.locator('#status')).to_contain_text('queued');items=context.request.get('/api/dev/jobs').json()['items'];delayed=items[0]['id'];self.assertNotEqual(delayed,identifier)
                            page.locator(f'article[data-job-id="{delayed}"] button').click();expect(page.locator('#status')).to_contain_text('Job cancelled');expect(page.locator(f'article[data-job-id="{delayed}"]')).to_contain_text('cancelled')
                            output=Path(os.getenv('RAPHAEL_VISUAL_OUTPUT',directory));output.mkdir(parents=True,exist_ok=True)
                            for name,width,size in [('jobs-mobile',390,16),('jobs-enlarged',320,20),('jobs-desktop',1280,16)]:
                                page.set_viewport_size({'width':width,'height':844});page.evaluate('(size)=>document.documentElement.style.fontSize=size+"px"',size)
                                self.assertTrue(page.evaluate('document.documentElement.scrollWidth <= innerWidth'),name);page.screenshot(path=str(output/(name+'.png')),full_page=True)
                            denied=browser.new_context(base_url=base)
                            self.assertEqual(denied.request.post('/api/dev/login',data={'tenant':'other','username':'outsider','password':password}).status,200)
                            self.assertEqual(denied.request.get(f'/api/dev/jobs/{identifier}').status,404);self.assertEqual(denied.request.get('/api/dev/jobs').json()['items'],[])
                            self.assertEqual(denied.request.post('/api/dev/logout',data={}).status,200)
                            self.assertEqual(denied.request.post('/api/dev/login',data={'tenant':'synthetic','username':'patient','password':password}).status,200)
                            self.assertEqual(denied.request.get('/api/dev/jobs').status,403);patient=denied.new_page();patient.goto('/static/jobs.html');expect(patient.locator('#workspace')).to_be_hidden();expect(patient.locator('#status')).to_contain_text('permitted workflow operator')
                            self.assertFalse(errors,errors)
                        finally:browser.close()
                finally:
                    os.killpg(app.pid,signal.SIGTERM)
                    try:app.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(app.pid,signal.SIGKILL);app.wait(timeout=10)
