"""Actual clinician/patient Chromium checklist journey;nonclinical fixtures only."""
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
from dev_procedures import SECTIONS
from dev_verify import isolated_environment

class ProceduresBrowserTests(unittest.TestCase):
    def test_review_publish_patient_authorize_acknowledge_signoff_and_revoke(self):
        with tempfile.TemporaryDirectory(prefix='raphael-procedures-browser-') as directory:
            url='sqlite:///'+directory+'/synthetic.db';engine=create_engine(url);password=secrets.token_urlsafe(24)
            with patch.dict(os.environ,{'RAPHAEL_DEV_SYNTHETIC_ONLY':'1'}):
                init_portal_schema(engine)
                for name,role in [('doctor','clinician'),('patient','patient'),('other','patient')]:create_dev_user(engine,'synthetic',name,password,role)
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
                        browser=p.chromium.launch(executable_path=os.getenv('CHROMIUM_EXECUTABLE','/usr/bin/chromium'),headless=True,args=['--no-sandbox','--disable-dev-shm-usage']);contexts={name:browser.new_context(base_url=base,viewport={'width':390,'height':844}) for name in ['doctor','patient','other']};errors=[]
                        try:
                            for name,context in contexts.items():self.assertEqual(context.request.post('/api/dev/login',data={'tenant':'synthetic','username':name,'password':password}).status,200)
                            doctor=contexts['doctor'].new_page();patient=contexts['patient'].new_page()
                            for page in [doctor,patient]:page.on('pageerror',lambda error:errors.append(str(error)))
                            response=contexts['doctor'].request.post('/api/dev/patients',data={'name':'Synthetic procedure patient','patient_username':'patient'});self.assertEqual(response.status,200);patient_id=response.json()['id']
                            response=contexts['doctor'].request.post('/api/dev/knowledge/sources',data={'title':'Nonclinical procedure reference fixture','url':'https://example.org/browser-procedures','content':'Violet calibrator workflow. Synthetic clinician and patient fixture markers can be acknowledged. No medical facts or instructions.','license_declaration':'Test-owned nonclinical synthetic material only.','permission_attested':True,'org':'Synthetic test'});self.assertEqual(response.status,200);source=response.json();self.assertEqual(contexts['doctor'].request.post(f"/api/dev/knowledge/sources/{source['id']}/approve",data={}).status,200)
                            doctor.goto('/static/procedures.html');expect(doctor.locator('#workspace')).to_be_visible();expect(doctor.locator('#source option')).to_have_count(2)
                            doctor.locator('#slug').fill('synthetic_fixture');doctor.locator('#title').fill('Synthetic nonclinical checklist');doctor.locator('#specialty').fill('Synthetic calibration');doctor.locator('#source').select_option(str(source['id']))
                            for key in SECTIONS:doctor.locator('#reference-'+key).fill('Not established: nonclinical synthetic fixture only.')
                            first=doctor.locator('.step-editor').nth(0);first.locator('textarea').fill('Clinician fixture marker review.');first.locator('select').nth(1).select_option('safety');doctor.locator('#add-step').click()
                            second=doctor.locator('.step-editor').nth(1);second.locator('textarea').fill('Patient fixture marker review.');second.locator('select').nth(0).select_option('patient')
                            doctor.get_by_role('button',name='Save pending template version',exact=True).click();expect(doctor.locator('#templates')).to_contain_text('pending');expect(doctor.locator('#coverage')).to_contain_text('No current approved')
                            card=doctor.locator('#templates article').first;card.locator('summary').click();expect(card).to_contain_text('Not established: nonclinical synthetic fixture only.');card.locator('input[type=checkbox]').nth(0).check();card.locator('input[type=checkbox]').nth(1).check();card.get_by_role('button',name='Approve template version',exact=True).click();expect(doctor.locator('#coverage')).to_contain_text('1 current templates')
                            template=contexts['doctor'].request.get('/api/dev/procedures/templates').json()['templates'][0];doctor.locator('#run-template').select_option(str(template['id']));doctor.get_by_role('button',name='Create private draft checklist',exact=True).click();expect(doctor.locator('#run-detail')).to_contain_text('draft')
                            run_id=contexts['doctor'].request.get(f'/api/dev/procedures/patients/{patient_id}/runs').json()['runs'][0]['id'];self.assertEqual(contexts['patient'].request.get(f'/api/dev/procedures/runs/{run_id}').status,404)
                            self.assertEqual(contexts['other'].request.get(f'/api/dev/procedures/runs/{run_id}').status,404)
                            patient.goto('/static/procedures.html');expect(patient.locator('#runs')).to_contain_text('No visible checklists');expect(patient.locator('#clinician-workspace')).to_be_hidden()
                            doctor.locator('#run-detail input[type=checkbox]').check();doctor.get_by_role('button',name='Publish reviewed checklist',exact=True).click();expect(doctor.locator('#run-detail')).to_contain_text('published')
                            visible=contexts['patient'].request.get(f'/api/dev/procedures/patients/{patient_id}/runs')
                            self.assertEqual(visible.status,200,visible.text())
                            self.assertEqual([row['id'] for row in visible.json()['runs']],[run_id],
                                             'Published workflow must be visible to the authorized patient in the API before UI review.')
                            patient.locator('#refresh').click()
                            try:
                                expect(patient.locator('#runs button')).to_have_count(1,timeout=15000)
                            except AssertionError:
                                self.fail('Patient refresh did not render the authorized published checklist: '
                                          +repr({'status':patient.locator('#status').inner_text(),
                                                 'runs':patient.locator('#runs').inner_text(),
                                                 'selected_patient':patient.locator('#patient').input_value(),
                                                 'page_errors':errors}))
                            patient.locator('#runs button').click();expect(patient.locator('#run-detail')).to_contain_text('coordination awaiting');expect(patient.locator('[data-step=step_2] button').first).to_be_disabled()
                            patient.locator('#run-detail input[type=checkbox]').check();patient.get_by_role('button',name='Authorize coordination',exact=True).click();expect(patient.locator('#run-detail')).to_contain_text('coordination granted')
                            patient.locator('[data-step=step_2]').get_by_role('button',name='Acknowledge step',exact=True).click();expect(patient.locator('[data-step=step_2]')).to_contain_text('acknowledged')
                            doctor.locator('#refresh').click();doctor.locator('#runs button').click();doctor.locator('[data-step=step_1]').get_by_role('button',name='Acknowledge step',exact=True).click();expect(doctor.locator('[data-step=step_1]')).to_contain_text('acknowledged')
                            doctor.locator('#run-detail input[type=checkbox]').check();doctor.get_by_role('button',name='Sign off completed checklist',exact=True).click();expect(doctor.locator('#run-detail')).to_contain_text('completed')
                            patient.locator('#refresh').click();patient.locator('#runs button').click();expect(patient.locator('#run-detail')).to_contain_text('completed');expect(patient.locator('[data-step=step_2] button').first).to_be_disabled()
                            output=Path(os.getenv('RAPHAEL_VISUAL_OUTPUT',directory));output.mkdir(parents=True,exist_ok=True)
                            for label,page in [('clinician',doctor),('patient',patient)]:
                                for name,width,size in [('mobile',390,16),('enlarged',320,20),('desktop',1280,16)]:
                                    page.set_viewport_size({'width':width,'height':844});page.evaluate('(size)=>document.documentElement.style.fontSize=size+"px"',size)
                                    self.assertTrue(page.evaluate('document.documentElement.scrollWidth<=innerWidth'),label+' '+name);page.screenshot(path=str(output/f'procedures-{label}-{name}.png'),full_page=True)
                            patient.get_by_role('button',name='Revoke coordination',exact=True).click();expect(patient.locator('#run-detail')).to_contain_text('coordination revoked');self.assertEqual(contexts['doctor'].request.get(f'/api/dev/procedures/runs/{run_id}').status,404)
                            patient.locator('#run-detail input[type=checkbox]').check();patient.get_by_role('button',name='Authorize coordination',exact=True).click();expect(patient.locator('#run-detail')).to_contain_text('coordination granted')
                            self.assertEqual(contexts['doctor'].request.post(f"/api/dev/knowledge/sources/{source['id']}/revoke",data={}).status,200)
                            patient.locator('#refresh').click();expect(patient.locator('#runs')).to_contain_text('reference no longer current');patient.locator('#runs button').click();expect(patient.locator('#status')).to_contain_text('instructions are withheld');expect(patient.locator('#run-detail')).to_be_empty()
                            self.assertEqual(contexts['patient'].request.get(f'/api/dev/procedures/runs/{run_id}').status,409);historical=contexts['doctor'].request.get(f'/api/dev/procedures/runs/{run_id}').json();self.assertFalse(historical['active_source_and_template']);self.assertEqual(historical['snapshot']['source_version_sha256'],source['version_sha256'])
                            self.assertFalse(errors,errors)
                        finally:browser.close()
                finally:
                    os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)
