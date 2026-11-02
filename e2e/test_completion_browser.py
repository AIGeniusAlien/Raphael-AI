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
   url='sqlite:///'+directory+'/synthetic.db';password=secrets.token_urlsafe(25);key=Fernet.generate_key().decode();database_key=Fernet.generate_key().decode()
   from protected_database import application_engine
   with patch.dict(os.environ,{'RAPHAEL_DEV_SYNTHETIC_ONLY':'1'}):
    engine=application_engine(url,encrypted=True,encryption_key=database_key)
    migrate(engine)
    for name,role in [('doctor','clinician'),('patient','patient'),('admin','org_admin')]:create_dev_user(engine,'synthetic',name,password,role)
   engine.dispose()
   with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
   base=f'http://127.0.0.1:{port}'
   env=dict(os.environ,DATABASE_URL=url,RAPHAEL_DEV_PORTAL='1',RAPHAEL_DEV_SYNTHETIC_ONLY='1',RAPHAEL_MFA_ENCRYPTION_KEY=key,RAPHAEL_DATABASE_ENCRYPTED='1',RAPHAEL_DATABASE_ENCRYPTION_KEY=database_key,LLM_API_KEY='',LLM_BASE_URL='',QDRANT_URL='',QDRANT_API_KEY='')
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
       page.goto('/static/account.html');expect(page.locator('#account')).to_be_visible();expect(page.locator('#storage-status')).to_contain_text('SQLCipher database-page encryption verified');expect(page.locator('#mfa-enroll')).to_be_enabled()
       page.locator('#mfa-password').fill(password);page.locator('#mfa-enroll').click();expect(page.locator('#mfa-private')).to_be_visible()
       secret=re.search(r'key \(expires in 10 minutes\): ([A-Z2-7]+)',page.locator('#mfa-private').inner_text()).group(1)
       page.locator('#mfa-password').fill(password);page.locator('#mfa-code').fill(totp(secret,int(time.time())//30));page.locator('#mfa-confirm').click()
       expect(page.locator('#mfa-status')).to_contain_text('Authenticator enabled');recovery=page.locator('#mfa-private').inner_text().splitlines()[1]
       # Verify sensitive content layout without ever capturing its plaintext.
       page.set_viewport_size({'width':320,'height':900})
       self.assertLessEqual(page.evaluate('document.documentElement.scrollWidth'),321,'private setup/recovery layout')
       page.locator('#mfa-clear').click();expect(page.locator('#mfa-private')).to_be_hidden()
       if os.getenv('RAPHAEL_VISUAL_OUTPUT'):
        output=Path(os.environ['RAPHAEL_VISUAL_OUTPUT']);output.mkdir(parents=True,exist_ok=True)
        for name,width,height in [('desktop',1440,1000),('tablet',768,1024),('mobile',390,844),('enlarged',320,900)]:
         page.set_viewport_size({'width':width,'height':height});page.evaluate('(s)=>document.documentElement.style.fontSize=s+"px"',20 if name=='enlarged' else 16)
         page.screenshot(path=str(output/('account-'+name+'.png')),full_page=True)
         self.assertLessEqual(page.evaluate('document.documentElement.scrollWidth'),width+1,name)
       page.set_viewport_size({'width':1440,'height':1000});page.evaluate('document.documentElement.style.fontSize="16px"')
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
       page.get_by_text('Your accessible workflow checks',exact=True).click();page.locator('#workflow-refresh').click();expect(page.locator('#workflow-checks')).to_contain_text('accessible generations');expect(page.locator('#workflow-checks')).to_contain_text('Observed total latency')
       page.get_by_text('Supported evidence scope',exact=True).click();page.locator('#coverage-refresh').click();expect(page.locator('#coverage-map')).to_contain_text('Live curated coverage has not been checked')
       page.goto('/static/chart.html?patient_id='+str(patient['id']));expect(page.locator('#chart-content')).to_be_visible()
       from observation_extraction import HEADERS
       source=(','.join(HEADERS)+f"\n{patient['id']},lab,Synthetic trend,2020-01-01T00:00:00+00:00,5,test-unit,,\n{patient['id']},lab,Synthetic trend,2020-01-02T00:00:00+00:00,9,test-unit,,\n").encode()
       page.locator('#document-title').fill('Synthetic tabular observations')
       page.locator('#document-file').set_input_files({'name':'synthetic-observations.txt','mimeType':'text/plain','buffer':source})
       page.locator('#document-form button').click();expect(page.locator('#status')).to_contain_text('Synthetic document stored')
       page.get_by_role('button',name='Review tabular observations',exact=True).click();expect(page.locator('.observation-preview')).to_contain_text('Source rows awaiting your review')
       expect(page.locator('.observation-preview button')).to_be_disabled()
       for checkbox in page.locator('.observation-preview input[type=checkbox]').all():checkbox.check()
       page.once('dialog',lambda dialog:dialog.accept());page.locator('.observation-preview button').click()
       expect(page.locator('#status')).to_contain_text('2 reviewed observations created')
       expect(page.locator('#longitudinal')).to_contain_text('Entered first-to-last change: 4 test-unit')
       expect(page.locator('#entries')).to_contain_text('approved')
       if os.getenv('RAPHAEL_VISUAL_OUTPUT'):
        output=Path(os.environ['RAPHAEL_VISUAL_OUTPUT']);output.mkdir(parents=True,exist_ok=True)
        for name,width,height,size in [('desktop',1440,1000,16),('tablet',768,1024,16),('mobile',390,844,16),('enlarged',320,900,20)]:
         page.set_viewport_size({'width':width,'height':height});page.evaluate('(s)=>document.documentElement.style.fontSize=s+"px"',size);page.locator('#longitudinal-heading').evaluate('(element)=>element.scrollIntoView({block:"start"})');page.screenshot(path=str(output/('longitudinal-'+name+'.png')),full_page=False)
         self.assertLessEqual(page.evaluate('document.documentElement.scrollWidth'),width+1,name)
       patient_view=browser.new_context(base_url=base,viewport={'width':390,'height':844});patient_view.request.post('/api/dev/login',data={'tenant':'synthetic','username':'patient','password':password})
       patient_page=patient_view.new_page();patient_page.goto('/static/chart.html?patient_id='+str(patient['id']));expect(patient_page.locator('#longitudinal')).to_contain_text('Entered first-to-last change: 4 test-unit')
       self.assertEqual(patient_page.get_by_role('button',name='Review tabular observations').count(),0)
       # Integrated lifecycle uses the same account/privacy interface and
       # existing administrator activation API;never a real erasure policy.
       patient_page.on('pageerror',lambda error:errors.append(str(error)))
       uid=patient_view.request.get('/api/dev/session').json()['id']
       patient_page.goto('/static/account.html');expect(patient_page.locator('#lifecycle-status')).to_contain_text('Deactivation blocks sign-in')
       if os.getenv('RAPHAEL_VISUAL_OUTPUT'):
        for name,width,height in [('desktop',1440,1000),('mobile',390,844),('enlarged',320,900)]:
         patient_page.set_viewport_size({'width':width,'height':height});patient_page.evaluate('(s)=>document.documentElement.style.fontSize=s+"px"',20 if name=='enlarged' else 16)
         patient_page.locator('#lifecycle-section').screenshot(path=str(output/('lifecycle-patient-'+name+'.png')))
         self.assertLessEqual(patient_page.evaluate('document.documentElement.scrollWidth'),width+1,name+' lifecycle')
       patient_page.locator('#lifecycle-password').fill(password);patient_page.locator('#lifecycle-confirm').check();patient_page.locator('#lifecycle-request-erasure').check();patient_page.once('dialog',lambda dialog:dialog.accept());patient_page.locator('#lifecycle-deactivate').click()
       expect(patient_page.locator('#account')).to_be_hidden();expect(patient_page.locator('#status')).to_contain_text('Account deactivated')
       self.assertEqual(patient_view.request.get('/api/dev/session').status,401)
       admin=browser.new_context(base_url=base);admin.request.post('/api/dev/login',data={'tenant':'synthetic','username':'admin','password':password});admin_page=admin.new_page();admin_page.on('pageerror',lambda error:errors.append(str(error)))
       admin_page.goto('/static/account.html');expect(admin_page.locator('#lifecycle-admin')).to_be_visible()
       row=admin_page.locator('[data-lifecycle-id="'+str(uid)+'"]');admin_page.once('dialog',lambda dialog:dialog.accept());row.get_by_role('button',name='Reactivate preserved account',exact=True).click()
       expect(admin_page.locator('#status')).to_contain_text('Account reactivated');self.assertEqual(patient_view.request.get('/api/dev/session').status,401)
       self.assertEqual(patient_view.request.post('/api/dev/login',data={'tenant':'synthetic','username':'patient','password':password}).status,200)
       # Real owner submission and organization task workflow;no provider,
       # model training,medical correction or automatic evidence verification.
       patient_page.goto('/static/feedback.html');expect(patient_page.locator('#feedback')).to_be_visible()
       patient_page.locator('#title').fill('<script>synthetic feedback title</script>')
       patient_page.locator('#detail').fill('Synthetic reproducible product issue without patient records.')
       patient_page.locator('#synthetic-ack').check();patient_page.locator('#private-ack').check();patient_page.locator('#submit-report').click()
       expect(patient_page.locator('#status')).to_contain_text('persisted')
       expect(patient_page.locator('#reports')).to_contain_text('<script>synthetic feedback title</script>')
       self.assertEqual(patient_page.locator('#reports script').count(),0)
       report_id=patient_page.locator('[data-feedback-id]').first.get_attribute('data-feedback-id')
       admin_page.goto('/static/feedback.html');expect(admin_page.locator('#administration')).to_be_visible()
       admin_page.locator('#requirement-search').fill('CORE-007');admin_page.locator('#search').click();expect(admin_page.locator('#requirements')).to_contain_text('CORE-007')
       report=admin_page.locator('[data-feedback-id="'+report_id+'"]')
       def feedback_action(kind,note):
        report.locator('select[id^="action-"]').select_option(kind);report.locator('textarea').fill(note);report.get_by_role('button',name='Apply action to report '+report_id,exact=True).click()
        expect(admin_page.locator('#status')).to_contain_text('Action recorded')
        # Avoid accepting an earlier success before the new revision arrives.
       if os.getenv('RAPHAEL_VISUAL_OUTPUT'):
        for name,width,height in [('mobile',390,844),('enlarged',320,900)]:
         admin_page.set_viewport_size({'width':width,'height':height});admin_page.evaluate('(s)=>document.documentElement.style.fontSize=s+"px"',20 if name=='enlarged' else 16)
         report.screenshot(path=str(output/('feedback-admin-controls-'+name+'.png')))
         self.assertLessEqual(admin_page.evaluate('document.documentElement.scrollWidth'),width+1,name+' active feedback controls')
        admin_page.set_viewport_size({'width':1440,'height':1000});admin_page.evaluate('document.documentElement.style.fontSize="16px"')
       feedback_action('triage','Private synthetic triage note');expect(report).to_contain_text('revision 2')
       report.locator('input[id^="requirement-"]').fill('CORE-007');feedback_action('create_task','Private synthetic linked task');expect(report).to_contain_text('task_created')
       feedback_action('start_task','Private synthetic work started');expect(report).to_contain_text('in_progress')
       report.locator('input[id^="evidence-"]').fill('docs/RELEASE_READINESS.md');feedback_action('resolve','Private synthetic technical resolution only');expect(report).to_contain_text('resolved')
       expect(report).to_contain_text('unverified')
       patient_page.locator('#refresh').click();expect(patient_page.locator('#reports')).to_contain_text('resolved');expect(patient_page.locator('#reports')).not_to_contain_text('Private synthetic')
       if os.getenv('RAPHAEL_VISUAL_OUTPUT'):
        for name,width,height in [('desktop',1440,1000),('mobile',390,844),('enlarged',320,900)]:
         for label,view in [('owner',patient_page),('admin',admin_page)]:
          view.set_viewport_size({'width':width,'height':height});view.evaluate('(s)=>document.documentElement.style.fontSize=s+"px"',20 if name=='enlarged' else 16)
          view.screenshot(path=str(output/('feedback-'+label+'-'+name+'.png')),full_page=True)
          self.assertLessEqual(view.evaluate('document.documentElement.scrollWidth'),width+1,name+' feedback '+label)
       admin_page.goto('/static/account.html');expect(admin_page.locator('#lifecycle-admin')).to_be_visible()
       patient_page.goto('/static/account.html');expect(patient_page.locator('#lifecycle-status')).to_contain_text('Deactivation blocks sign-in')
       patient_page.locator('#lifecycle-password').fill(password);patient_page.locator('#lifecycle-confirm').check();patient_page.locator('#lifecycle-request-erasure').check();patient_page.once('dialog',lambda dialog:dialog.accept());patient_page.locator('#lifecycle-deactivate').click();expect(patient_page.locator('#account')).to_be_hidden()
       admin_page.locator('#lifecycle-admin-refresh').click();expect(row).to_contain_text('deactivated')
       admin_page.locator('#lifecycle-policy-reference').fill('SYNTHETIC TEST POLICY NOT LEGAL APPROVAL');admin_page.locator('#lifecycle-policy-days').fill('0');admin_page.locator('#lifecycle-policy-enabled').check();admin_page.locator('#lifecycle-password').fill(password);admin_page.locator('#lifecycle-policy-save').click();expect(admin_page.locator('#status')).to_contain_text('Policy reference/version recorded')
       row.locator('input[id^="hold-reference-"]').fill('SYNTHETIC HOLD');admin_page.locator('#lifecycle-password').fill(password);row.get_by_role('button',name='Place retention hold',exact=True).click();expect(row).to_contain_text('Retention hold active')
       expect(row.get_by_role('button',name='Erase credentials and preferences; retain clinical records',exact=True)).to_be_disabled()
       if os.getenv('RAPHAEL_VISUAL_OUTPUT'):
        for name,width,height in [('desktop',1440,1000),('mobile',390,844),('enlarged',320,900)]:
         admin_page.set_viewport_size({'width':width,'height':height});admin_page.evaluate('(s)=>document.documentElement.style.fontSize=s+"px"',20 if name=='enlarged' else 16)
         admin_page.locator('#lifecycle-admin').screenshot(path=str(output/('lifecycle-admin-'+name+'.png')))
         self.assertLessEqual(admin_page.evaluate('document.documentElement.scrollWidth'),width+1,name+' lifecycle admin')
       row.locator('input[id^="hold-reference-"]').fill('SYNTHETIC HOLD RELEASE');admin_page.locator('#lifecycle-password').fill(password);row.get_by_role('button',name='Release retention hold',exact=True).click();expect(row).to_contain_text('No active retention hold')
       row.locator('input[id^="erase-target-"]').fill('patient');admin_page.locator('#lifecycle-password').fill(password);admin_page.once('dialog',lambda dialog:dialog.accept());row.get_by_role('button',name='Erase credentials and preferences; retain clinical records',exact=True).click();expect(admin_page.locator('#status')).to_contain_text('not full account or medical-record erasure')
       self.assertEqual(patient_view.request.post('/api/dev/login',data={'tenant':'synthetic','username':'patient','password':password}).status,401)
       self.assertEqual(admin.request.get('/api/dev/patients').status,403)
       retained=context.request.get('/api/dev/patients/'+str(patient['id'])+'/notes');self.assertEqual(retained.status,200);self.assertIn('Synthetic reviewed care plan',retained.text())
       admin.close();patient_view.close();self.assertEqual(errors,[]);context.close()
      finally:browser.close()
    finally:
     os.killpg(process.pid,signal.SIGTERM)
     try:process.wait(timeout=10)
     except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
