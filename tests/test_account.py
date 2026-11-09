"""Synthetic account controls,not legal erasure or identity validation."""
import os,tempfile,unittest
from unittest.mock import patch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine,text
from dev_runtime import migrate
from dev_portal import create_dev_user,install_dev_portal
from dev_chart import install_chart
from dev_care import install_care
from dev_account import install_account


class AccountTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.engine=create_engine('sqlite:///'+self.tmp.name+'/synthetic.db');self.addCleanup(self.engine.dispose)
  env=patch.dict(os.environ,{'RAPHAEL_DEV_SYNTHETIC_ONLY':'1'});env.start();self.addCleanup(env.stop);migrate(self.engine);self.password='Synthetic-account-password'
  for tenant,name,role in [('one','patient','patient'),('one','other','patient'),('one','doctor','clinician'),('one','admin','org_admin'),('two','foreign','org_admin')]:create_dev_user(self.engine,tenant,name,self.password,role)
  self.app=FastAPI();self.ctx=ctx=install_dev_portal(self.app,self.engine);install_chart(self.app,ctx);install_care(self.app,ctx);install_account(self.app,ctx)
  self.client=TestClient(self.app);self.addCleanup(self.client.close);self.login('doctor')
  self.pid=self.client.post('/api/dev/patients',json={'name':'Synthetic own record','patient_username':'patient'}).json()['id']
  self.client.post('/api/dev/patients',json={'name':'OTHER-PATIENT-MUST-NOT-EXPORT','patient_username':'other'}).raise_for_status();self.login()
 def login(self,name='patient',tenant='one',client=None):
  (client or self.client).post('/api/dev/login',json={'tenant':tenant,'username':name,'password':self.password}).raise_for_status()
 def test_profile_is_private_and_does_not_overwrite_chart(self):
  self.client.post('/api/dev/account/profile',json={'display_name':'Account preference','language':'es','time_zone':'America/New_York','internal_notifications':False}).raise_for_status()
  self.assertEqual(self.client.get('/api/dev/account').json()['profile']['display_name'],'Account preference')
  self.assertEqual(self.client.get('/api/dev/patients').json()[0]['name'],'Synthetic own record')
  self.login('other');self.assertEqual(self.client.get('/api/dev/account').json()['profile']['display_name'],'other')
 def test_export_only_own_approved_content_no_credentials_or_drafts(self):
  self.login('doctor');self.client.post(f'/api/dev/patients/{self.pid}/notes',json={'content':'UNSIGNED-DOCTOR-DRAFT'}).raise_for_status()
  approved=self.client.post(f'/api/dev/patients/{self.pid}/notes',json={'content':'Approved synthetic note'}).json()['id'];self.client.post(f'/api/dev/notes/{approved}/approve',json={}).raise_for_status();self.login()
  response=self.client.get('/api/dev/account/export');self.assertEqual(response.status_code,200);self.assertEqual(response.json()['approved_notes'][0]['content'],'Approved synthetic note')
  for forbidden in ('OTHER-PATIENT-MUST-NOT-EXPORT','UNSIGNED-DOCTOR-DRAFT','password_hash','token_hash',self.password):self.assertNotIn(forbidden,response.text)
  self.login('admin');self.assertEqual(self.client.get('/api/dev/account/export').status_code,403)
 def test_session_revocation_is_own_scoped_and_clears_correct_cookie_path(self):
  second=TestClient(self.app);self.addCleanup(second.close);self.login(client=second)
  sessions=self.client.get('/api/dev/account/sessions').json();self.assertEqual(len(sessions),2);target=next(s['id'] for s in sessions if not s['current'])
  self.client.post('/api/dev/account/sessions/'+target+'/revoke',json={}).raise_for_status();self.assertEqual(second.get('/api/dev/session').status_code,401)
  current=self.client.get('/api/dev/account/sessions').json()[0]['id'];self.login('other');self.assertEqual(self.client.post('/api/dev/account/sessions/'+current+'/revoke',json={}).status_code,404)
  self.login();own=next(s for s in self.client.get('/api/dev/account/sessions').json() if s['current']);response=self.client.post('/api/dev/account/sessions/'+own['id']+'/revoke',json={});self.assertIn('Path=/api/dev',response.headers['set-cookie']);self.assertEqual(self.client.get('/api/dev/session').status_code,401)
 def test_revoke_other_sessions_preserves_current_and_scopes_to_account(self):
  second=TestClient(self.app);self.addCleanup(second.close);self.login(client=second)
  other=TestClient(self.app);self.addCleanup(other.close);self.login('other',client=other)
  self.assertEqual(len(self.client.get('/api/dev/account/sessions').json()),2)
  result=self.client.post('/api/dev/account/sessions/revoke-others',json={})
  self.assertEqual(result.status_code,200,result.text)
  self.assertEqual(result.json(),{'revoked_other_sessions':1,'current_session_preserved':True})
  self.assertEqual(second.get('/api/dev/session').status_code,401)
  self.assertEqual(self.client.get('/api/dev/session').status_code,200)
  self.assertEqual(other.get('/api/dev/session').status_code,200)
  self.assertEqual(self.client.post('/api/dev/account/sessions/revoke-others',json={}).json()['revoked_other_sessions'],0)

 def test_revoke_other_sessions_fails_closed_if_session_revoked_during_request(self):
  from fastapi import Request
  original=self.ctx.user
  second=TestClient(self.app);self.addCleanup(second.close);self.login(client=second)
  def revoke_after_auth(request:Request):
   user=original(request)
   from dev_portal import COOKIE_NAME
   import hashlib
   token=request.cookies.get(COOKIE_NAME,'')
   with self.engine.begin() as c:
    c.execute(text('DELETE FROM dev_sessions WHERE token_hash=:h'),{'h':hashlib.sha256(token.encode()).hexdigest()})
   return user
  self.app.dependency_overrides[original]=revoke_after_auth
  try:
   result=self.client.post('/api/dev/account/sessions/revoke-others',json={})
   self.assertEqual(result.status_code,401,result.text)
   self.assertEqual(second.get('/api/dev/session').status_code,200)
  finally:self.app.dependency_overrides.clear()

 def test_targeted_revoke_fails_closed_if_current_session_revoked_after_auth(self):
  from fastapi import Request
  from dev_portal import COOKIE_NAME
  import hashlib
  second=TestClient(self.app);self.addCleanup(second.close);self.login(client=second)
  target=next(s['id'] for s in self.client.get('/api/dev/account/sessions').json() if not s['current'])
  original=self.ctx.user
  def revoke_after_auth(request:Request):
   user=original(request)
   token=request.cookies.get(COOKIE_NAME,'')
   with self.engine.begin() as c:
    c.execute(text('DELETE FROM dev_sessions WHERE token_hash=:h'),{'h':hashlib.sha256(token.encode()).hexdigest()})
   return user
  self.app.dependency_overrides[original]=revoke_after_auth
  try:
   result=self.client.post('/api/dev/account/sessions/'+target+'/revoke',json={})
   self.assertEqual(result.status_code,401,result.text)
   self.assertEqual(second.get('/api/dev/session').status_code,200)
  finally:self.app.dependency_overrides.clear()

 def test_deletion_review_persists_is_idempotent_and_never_erases(self):
  path='/api/dev/account/deletion-requests';body={'confirm_review_request':True}
  first=self.client.post(path,json=body).json();self.assertFalse(first['erasure_performed']);self.assertEqual(self.client.post(path,json=body).json()['id'],first['id'])
  self.login('other');self.assertEqual(self.client.post(path+'/'+str(first['id'])+'/cancel',json={}).status_code,404)
  self.login('foreign','two');self.assertEqual(self.client.get('/api/dev/account/deletion-review').json(),[])
  self.login('admin');self.client.post('/api/dev/account/deletion-review/'+str(first['id']),json={'note':'Retention policy review pending;no erasure.'}).raise_for_status()
  self.login();self.assertEqual(self.client.post(path,json=body).json()['state'],'ACKNOWLEDGED');self.assertTrue(self.client.get('/api/dev/patients').json())
  migrate(self.engine);self.assertEqual(self.client.get(path).json()[0]['state'],'ACKNOWLEDGED');self.client.post(path+'/'+str(first['id'])+'/cancel',json={}).raise_for_status();self.assertEqual(self.client.get(path).json()[0]['state'],'CANCELLED')

 def test_export_rejects_clinicians_and_revoked_owner_session(self):
  self.login('doctor');self.assertEqual(self.client.get('/api/dev/account/export').status_code,403)
  self.login();owner=TestClient(self.app);self.addCleanup(owner.close);self.login(client=owner)
  sessions=owner.get('/api/dev/account/sessions').json()
  others=[row for row in sessions if not row['current']]
  self.assertTrue(others)
  for row in others:owner.post('/api/dev/account/sessions/'+row['id']+'/revoke',json={}).raise_for_status()
  self.assertEqual(self.client.get('/api/dev/account/export').status_code,401)
  self.assertEqual(owner.get('/api/dev/account/export').status_code,200)

 def test_export_limits_fail_without_a_partial_attachment(self):
  with self.engine.begin() as c:
   doctor=c.execute(text("SELECT id FROM dev_users WHERE tenant='one' AND username='doctor'")).scalar_one()
   c.execute(text("INSERT INTO dev_notes(patient_id,author_id,content,status,created_at,approved_by,approved_at) VALUES(:p,:u,:v,'approved','2020-01-01',:u,'2020-01-01')"),{'p':self.pid,'u':doctor,'v':'SYNTHETIC-OVERSIZE-'+'x'*(5*1024*1024)})
  result=self.client.get('/api/dev/account/export')
  self.assertEqual(result.status_code,413)
  self.assertNotIn('content-disposition',result.headers)
  self.assertNotIn('SYNTHETIC-OVERSIZE',result.text)

 def test_export_revalidates_session_and_role_after_authentication(self):
  from fastapi import Request
  original=self.ctx.user
  for mutation in ('DELETE FROM dev_sessions WHERE user_id=:u',"UPDATE dev_users SET role='org_admin' WHERE id=:u"):
   def revoke_after_dependency(request:Request):
    user=original(request)
    with self.engine.begin() as c:c.execute(text(mutation),{'u':user['id']})
    return user
   self.app.dependency_overrides[original]=revoke_after_dependency
   result=self.client.get('/api/dev/account/export')
   self.assertEqual(result.status_code,401,result.text)
   self.assertNotIn('content-disposition',result.headers)
   self.assertNotIn('Synthetic own record',result.text)
   self.app.dependency_overrides.clear()
   with self.engine.begin() as c:c.execute(text("UPDATE dev_users SET role='patient' WHERE tenant='one' AND username='patient'"))
   self.login()
