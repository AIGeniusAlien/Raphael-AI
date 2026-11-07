"""Real Chromium local term review/retrieval/revocation;no medical/model claims."""
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

class TermsBrowserTests(unittest.TestCase):
    def test_reviewed_alias_search_coverage_and_source_revocation(self):
        with tempfile.TemporaryDirectory(prefix='raphael-terms-browser-') as directory:
            url='sqlite:///'+directory+'/synthetic.db';engine=create_engine(url);password=secrets.token_urlsafe(24)
            with patch.dict(os.environ,{'RAPHAEL_DEV_SYNTHETIC_ONLY':'1'}):
                init_portal_schema(engine);create_dev_user(engine,'synthetic','doctor',password,'clinician')
            engine.dispose()
            with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
            base=f'http://127.0.0.1:{port}';env=isolated_environment();env.update(DATABASE_URL=url,RAPHAEL_DEV_PORTAL='1',RAPHAEL_DEV_SYNTHETIC_ONLY='1',STATIC_PATH='static')
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
                        browser=p.chromium.launch(executable_path=os.getenv('CHROMIUM_EXECUTABLE','/usr/bin/chromium'),headless=True,args=['--no-sandbox']);context=browser.new_context(base_url=base,viewport={'width':390,'height':844});page=context.new_page();errors=[];page.on('pageerror',lambda error:errors.append(str(error)))
                        try:
                            self.assertEqual(context.request.post('/api/dev/login',data={'tenant':'synthetic','username':'doctor','password':password}).status,200)
                            response=context.request.post('/api/dev/knowledge/sources',data={'title':'Synthetic terminology source','url':'https://example.org/browser-terms','content':'Violet calibrator baseline fixture. This is nonclinical synthetic text, not medical evidence.','license_declaration':'Test-owned synthetic fixture only.','permission_attested':True,'org':'Synthetic fixture'})
                            self.assertEqual(response.status,200);source=response.json();self.assertEqual(context.request.post(f"/api/dev/knowledge/sources/{source['id']}/approve",data={}).status,200)
                            page.goto('/static/knowledge.html');expect(page.locator('#workspace')).to_be_visible();expect(page.locator('#term-source option')).to_have_count(2)
                            page.locator('#term-source').select_option(str(source['id']));page.locator('#term-canonical').fill('violet calibrator');page.locator('#term-aliases').fill('violetwand');page.locator('#term-specialty').fill('Synthetic calibration');page.locator('#term-form button').click();expect(page.locator('#terms')).to_contain_text('pending');expect(page.locator('#term-coverage')).to_contain_text('No active')
                            page.locator('#query').fill('violetwand');page.locator('#search-form button').click();expect(page.locator('#results')).to_contain_text('No approved excerpt matched')
                            page.locator('#terms input[type=checkbox]').check();page.get_by_role('button',name='Review term mapping',exact=True).click();expect(page.locator('#terms')).to_contain_text('active for opt-in search');expect(page.locator('#term-coverage')).to_contain_text('1 active terms')
                            page.locator('#search-form button').click();expect(page.locator('#status')).to_contain_text('disabled');expect(page.locator('#results')).to_contain_text('No approved excerpt matched')
                            page.locator('#use-terms').check();page.locator('#search-form button').click();expect(page.locator('#status')).to_contain_text('expanded');expect(page.locator('#results')).to_contain_text('Synthetic terminology source')
                            page.get_by_role('button',name='Resolve stored citation and checksum').click();expect(page.locator('#citation')).to_contain_text('Violet calibrator')
                            output=Path(os.getenv('RAPHAEL_VISUAL_OUTPUT',directory));output.mkdir(parents=True,exist_ok=True)
                            for name,width,size in [('terms-mobile',390,16),('terms-enlarged',320,20),('terms-desktop',1280,16)]:
                                page.set_viewport_size({'width':width,'height':844});page.evaluate('(size)=>document.documentElement.style.fontSize=size+"px"',size)
                                self.assertTrue(page.evaluate('document.documentElement.scrollWidth <= innerWidth'),name);page.screenshot(path=str(output/(name+'.png')),full_page=True)
                            self.assertEqual(context.request.post(f"/api/dev/knowledge/sources/{source['id']}/revoke",data={}).status,200)
                            page.locator('#refresh').click();expect(page.locator('#terms')).to_contain_text('inactive for search');expect(page.locator('#term-coverage')).to_contain_text('No active')
                            page.locator('#search-form button').click();expect(page.locator('#results')).to_contain_text('No approved excerpt matched');expect(page.locator('#status')).to_contain_text('no_reviewed_alias_match')
                            self.assertFalse(errors,errors)
                        finally:browser.close()
                finally:
                    os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)
