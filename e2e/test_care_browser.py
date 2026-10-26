"""Real mobile Chromium care-sharing workflow; no external hospital/model service."""
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

class CareBrowserTests(unittest.TestCase):
 def test_patient_authorization_receiver_review_approval_revocation_and_emergency_mobile(self):
  with tempfile.TemporaryDirectory(prefix='raphael-care-browser-') as directory:
   url='sqlite:///'+directory+'/synthetic.db';engine=create_engine(url);password=secrets.token_urlsafe(24)
   with patch.dict(os.environ,{'RAPHAEL_DEV_SYNTHETIC_ONLY':'1'}):
    init_portal_schema(engine)
    for name,role in [('creator','clinician'),('receiver','clinician'),('outsider','clinician'),('patient','patient')]:create_dev_user(engine,'synthetic',name,password,role)
   engine.dispose()
   with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
   base=f'http://127.0.0.1:{port}';env=dict(os.environ,DATABASE_URL=url,RAPHAEL_DEV_PORTAL='1',RAPHAEL_DEV_SYNTHETIC_ONLY='1',LLM_BASE_URL='',LLM_API_KEY='',QDRANT_URL='',STATIC_PATH='static',PYTHONDONTWRITEBYTECODE='1')
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
      browser=p.chromium.launch(executable_path=os.getenv('CHROMIUM_EXECUTABLE','/usr/bin/chromium'),headless=True,args=['--no-sandbox','--disable-dev-shm-usage'])
      try:
       contexts={name:browser.new_context(base_url=base,viewport={'width':390,'height':844}) for name in ['creator','receiver','outsider','patient']}
       for name,c in contexts.items():self.assertEqual(c.request.post('/api/dev/login',data={'tenant':'synthetic','username':name,'password':password}).status,200)
       creator=contexts['creator'];patient_id=creator.request.post('/api/dev/patients',data={'name':'Synthetic mobile patient','patient_username':'patient'}).json()['id']
       for kind,label in [('medication','Synthetic medication'),('allergy','Synthetic allergy')]:self.assertEqual(creator.request.post(f'/api/dev/patients/{patient_id}/entries',data={'kind':kind,'label':label,'dose':'Synthetic entered dose'} if kind=='medication' else {'kind':kind,'label':label,'reaction':'Synthetic entered reaction'}).status,200)
       patient=contexts['patient'].new_page();doctor=contexts['receiver'].new_page();errors=[]
       for page in (patient,doctor):page.on('pageerror',lambda error:errors.append(str(error)))
       patient.goto('/');expect(patient.get_by_role('link',name='Connected intake and emergency information')).to_be_visible();patient.get_by_role('link',name='Connected intake and emergency information').click();expect(patient.locator('#patient-panel')).to_be_visible()
       patient.locator('#history').fill('Synthetic patient-reported medical history');patient.locator('#symptoms').fill('Synthetic reported concern');patient.locator('#contact-name').fill('Synthetic family contact');patient.locator('#contact-relation').fill('family');patient.locator('#contact-phone').fill('TEST-0000');patient.locator('#report-form button').click();expect(patient.locator('#status')).to_contain_text('Patient report saved')
       patient.locator('#emergency').click();expect(patient.locator('#emergency-packet')).to_contain_text('Synthetic medication');expect(patient.locator('#emergency-packet')).to_contain_text('Synthetic family contact');expect(patient.locator('#status')).to_contain_text('No emergency facility was contacted')
       patient.locator('#intake-symptoms').fill('Synthetic visit request');patient.locator('#intake-form button').click();expect(patient.locator('#sharing-state')).to_contain_text('DRAFT')
       intake_id=contexts['patient'].request.get('/api/dev/care/intakes').json()[0]['id']
       doctor.goto('/static/care.html');expect(doctor.locator('#intakes')).to_contain_text('No authorized intakes')
       patient.locator('#recipient').select_option(label='receiver');patient.locator('#allow-ai').check();patient.locator('#share-form button').click();expect(patient.locator('#sharing-state')).to_contain_text('AUTHORIZED')
       doctor.locator('#refresh').click();expect(doctor.locator('#intakes button')).to_have_count(1);doctor.locator('#intakes button').click();expect(doctor.locator('#packet')).to_contain_text('Synthetic visit request');expect(doctor.locator('#packet')).to_contain_text('Synthetic allergy')
       self.assertEqual(contexts['outsider'].request.get(f'/api/dev/care/intakes/{intake_id}').status,404)
       doctor.locator('#summary').fill('Synthetic clinician reviewed summary');doctor.locator('#followup').fill('Synthetic approved followup');doctor.locator('#summary-form button').click();expect(doctor.locator('#summaries')).to_contain_text('draft')
       patient.locator('#refresh').click();patient.locator('#intakes button').click();expect(patient.locator('#summaries')).to_have_text('')
       doctor.locator('#summary').fill('Unsaved change');doctor.locator('#approve').click();expect(doctor.locator('#status')).to_contain_text('Save your summary edits')
       doctor.locator('#summary-form button').click();expect(doctor.locator('#summaries')).to_contain_text('Summary version 2');doctor.once('dialog',lambda dialog:dialog.accept());doctor.locator('#approve').click();expect(doctor.locator('#status')).to_contain_text('Summary approved')
       patient.locator('#refresh').click();patient.locator('#intakes button').click();expect(patient.locator('#summaries')).to_contain_text('Unsaved change');expect(patient.locator('#summaries')).to_contain_text('Synthetic approved followup')
       doctor.locator('#assistant').click();expect(doctor.locator('#thread-title')).to_contain_text('consented snapshot only');expect(doctor.locator('#thread-scope')).to_contain_text('No live chart is included')
       thread_id=contexts['receiver'].request.get('/api/dev/intelligence/threads').json()[0]['id']
       patient.once('dialog',lambda dialog:dialog.accept());patient.locator('#revoke').click();expect(patient.locator('#sharing-state')).to_contain_text('REVOKED')
       self.assertEqual(contexts['receiver'].request.get(f'/api/dev/care/intakes/{intake_id}').status,404);self.assertEqual(contexts['receiver'].request.get(f'/api/dev/intelligence/threads/{thread_id}').status,404)
       self.assertTrue(patient.evaluate('() => document.documentElement.scrollWidth <= innerWidth'))
       self.assertFalse(errors,errors)
      finally:browser.close()
    finally:
     os.killpg(process.pid,signal.SIGTERM)
     try:process.wait(timeout=10)
     except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
