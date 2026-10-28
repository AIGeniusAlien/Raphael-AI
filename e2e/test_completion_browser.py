"""Actual browser/backend MFA and clinician-paperwork journey; no model or remote clinical service."""
import os,re,secrets,signal,socket,subprocess,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
import httpx
from cryptography.fernet import Fernet
from playwright.sync_api import sync_playwright,expect
from sqlalchemy import create_engine
from dev_runtime import migrate
from dev_portal import create_dev_user
from dev_mfa import totp

class CompletionBrowserTests(unittest.TestCase):
 def test_authenticator_recovery_structured_approval_and_patient_visibility(self):
  with tempfile.TemporaryDirectory(prefix='raphael-completion-browser-') as directory:
   url='sqlite:///'+directory+'/synthetic.db';password=secrets.token_urlsafe(25);key=Fernet.generate_key().decode()
   engine=create_engine(url)
   with patch.dict(os.environ,{'RAPHAEL_DEV_SYNTHETIC_ONLY':'1'}):
    migrate(engine)
    for name,role in [('doctor','clinician'),('patient','patient')]:create_dev_user(engine,'synthetic',name,password,role)
   engine.dispose()
   with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
   base=f'http://127.0.0.1:{port}'
   env=dict(os.environ,DATABASE_URL=url,RAPHAEL_DEV_PORTAL='1',RAPHAEL_DEV_SYNTHETIC_ONLY='1',RAPHAEL_MFA_ENCRYPTION_KEY=key,LLM_API_KEY='',LLM_BASE_URL='',QDRANT_URL='',QDRANT_API_KEY='')
   with tempfile.TemporaryFile(mode='w+') as log:
    process=subprocess.Popen([sys.executable,'-m','uvicorn','main:app','--host','127.0.0.1','--port',str(port)],env=env,stdout=log,stderr=log,start_new_session=True)
    try:
     with httpx.Client(base_url=base,trust_env=False) as client:
      deadline=time.monotonic()+45
      while True:
       if process.poll() is not None or time.monotonic()>deadline:self.fail('Synthetic app did not start; private startup log retained only in temporary storage.')
       try:
        if client.get('/healthz').status_code==200:break
       except httpx.TransportError:pass
       time.sleep(.1)
     with sync_playwright() as p:
      browser=p.chromium.launch(executable_path=os.getenv('CHROMIUM_EXECUTABLE','/usr/bin/chromium'),headless=True,args=['--no-sandbox','--disable-dev-shm-usage'])
      try:
       context=browser.new_context(base_url=base);page=context.new_page();errors=[];page.on('pageerror',lambda error:errors.append(str(error)))
       self.assertEqual(context.request.post('/api/dev/login',data={'tenant':'synthetic','username':'doctor','password':password}).status,200)
       patient=context.request.post('/api/dev/patients',data={'name':'Synthetic completion patient','patient_username':'patient'}).json()
       page.goto('/static/account.html');expect(page.locator('#account')).to_be_visible();expect(page.locator('#mfa-enroll')).to_be_enabled()
       page.locator('#mfa-password').fill(password);page.locator('#mfa-enroll').click();expect(page.locator('#mfa-private')).to_be_visible()
       secret=re.search(r'key \(expires in 10 minutes\): ([A-Z2-7]+)',page.locator('#mfa-private').inner_text()).group(1)
       page.locator('#mfa-password').fill(password);page.locator('#mfa-code').fill(totp(secret,int(time.time())//30));page.locator('#mfa-confirm').click()
       expect(page.locator('#mfa-status')).to_contain_text('Authenticator enabled');recovery=page.locator('#mfa-private').inner_text().splitlines()[1]
       page.locator('#mfa-clear').click();expect(page.locator('#mfa-private')).to_be_hidden()
       context.request.post('/api/dev/logout',data={});page.goto('/')
       page.locator('#tenant').fill('synthetic');page.locator('#username').fill('doctor');page.locator('#password').fill(password);page.locator('#otp').fill(recovery);page.locator('#login-form button').click();expect(page.locator('#workspace')).to_be_visible()
       page.goto('/static/intelligence.html?patient='+str(patient['id']));expect(page.locator('#intelligence')).to_be_visible()
       page.locator('#new-encounter').click();expect(page.locator('#encounter option')).to_have_count(1);page.locator('#consent').check();page.locator('#start-session').click();expect(page.locator('#session-id')).not_to_have_value('')
       page.locator('#format').select_option('care_plan');page.locator('#document-field-goals').fill('Synthetic patient-stated goal for review.');page.locator('#document-field-actions').fill('Clinician-entered action for review; no treatment instruction.');page.locator('#generate-note').click()
       expect(page.locator('#note')).to_have_value(re.compile('Synthetic patient-stated goal'));expect(page.locator('#note')).to_have_value(re.compile('Follow Up: NOT DOCUMENTED'))
       if os.getenv('RAPHAEL_VISUAL_OUTPUT'):
        output=Path(os.environ['RAPHAEL_VISUAL_OUTPUT']);output.mkdir(parents=True,exist_ok=True)
        for name,width,height,size in [('desktop',1440,1000,16),('tablet',768,1024,16),('mobile',390,844,16),('enlarged',320,900,20)]:
         page.set_viewport_size({'width':width,'height':height});page.evaluate('(s)=>document.documentElement.style.fontSize=s+"px"',size);page.locator('#documentation-fields').scroll_into_view_if_needed();page.screenshot(path=str(output/('completion-'+name+'.png')),full_page=False)
         self.assertLessEqual(page.evaluate('document.documentElement.scrollWidth'),width+1,name)
       page.locator('#note').fill('Synthetic reviewed care plan. Patient-stated goal retained. No treatment authorized.');page.locator('#save-note').click();expect(page.locator('#note-status')).to_contain_text('Version 2');page.once('dialog',lambda dialog:dialog.accept());page.locator('#approve').click();expect(page.locator('#note-status')).to_contain_text('approved')
       other=browser.new_context(base_url=base);other.request.post('/api/dev/login',data={'tenant':'synthetic','username':'patient','password':password});result=other.request.get('/api/dev/patients/'+str(patient['id'])+'/notes');self.assertEqual(result.status,200);self.assertIn('Synthetic reviewed care plan',result.text());self.assertNotIn(secret,result.text());other.close()
       self.assertEqual(errors,[]);context.close()
      finally:browser.close()
    finally:
     os.killpg(process.pid,signal.SIGTERM)
     try:process.wait(timeout=10)
     except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
