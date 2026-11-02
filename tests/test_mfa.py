"""Real API/SQLite factor lifecycle; synthetic keys, no remote identity provider."""
import base64
import hashlib
import os
import unittest
from unittest.mock import patch
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient
from sqlalchemy import text
from tests import test_account as account_fixture
from dev_mfa import totp
from dev_runtime import migrate

class MFATests(unittest.TestCase):
 setUp=account_fixture.AccountTests.setUp
 login=account_fixture.AccountTests.login
 def configure(self):
  key=patch.dict(os.environ,{'RAPHAEL_MFA_ENCRYPTION_KEY':Fernet.generate_key().decode()});key.start();self.addCleanup(key.stop)
  clock=patch('dev_mfa.time.time',return_value=1700000000);clock.start();self.addCleanup(clock.stop);self.clock=clock
 def enroll(self):
  self.configure()
  response=self.client.post('/api/dev/account/mfa/enroll',json={'password':self.password});response.raise_for_status();self.secret=response.json()['secret']
  self.code=totp(self.secret,1700000000//30)
  response=self.client.post('/api/dev/account/mfa/confirm',json={'password':self.password,'code':self.code});response.raise_for_status();return response.json()['recovery_codes']
 def sign_in(self,code='',client=None):return (client or self.client).post('/api/dev/login',json={'tenant':'one','username':'patient','password':self.password,'otp':code})
 def test_rfc6238_sha1_six_digit_vector(self):
  secret=base64.b32encode(b'12345678901234567890').decode();self.assertEqual(totp(secret,1),'287082')
 def test_unconfigured_enrollment_fails_closed_without_breaking_existing_login(self):
  with patch.dict(os.environ,{'RAPHAEL_MFA_ENCRYPTION_KEY':''}):
   self.assertFalse(self.client.get('/api/dev/account/mfa').json()['enrollment_available'])
   self.assertEqual(self.client.post('/api/dev/account/mfa/enroll',json={'password':self.password}).status_code,503)
   self.sign_in().raise_for_status()
 def test_enable_encrypts_secret_hashes_recovery_and_revokes_other_sessions(self):
  other=TestClient(self.app);self.addCleanup(other.close);self.login(client=other)
  codes=self.enroll();self.assertEqual(other.get('/api/dev/session').status_code,401)
  with self.engine.connect() as c:
   stored=c.execute(text('SELECT encrypted_secret FROM dev_mfa_factors')).scalar();hashes=list(c.execute(text('SELECT code_hash FROM dev_mfa_recovery')).scalars())
  self.assertNotIn(self.secret,stored);self.assertEqual(len(hashes),10);self.assertNotIn(codes[0],hashes);self.assertIn(hashlib.sha256(codes[0].encode()).hexdigest(),hashes)
  status=self.client.get('/api/dev/account/mfa');self.assertTrue(status.json()['enabled']);self.assertNotIn(self.secret,status.text)
 def test_login_requires_factor_blocks_replay_and_accepts_next_step(self):
  self.enroll();self.assertEqual(self.sign_in().status_code,401);self.assertEqual(self.sign_in(self.code).status_code,401)
  with patch('dev_mfa.time.time',return_value=1700000030):
   next_code=totp(self.secret,1700000030//30);self.sign_in(next_code).raise_for_status();self.assertEqual(self.sign_in(next_code).status_code,401)
 def test_recovery_code_is_single_use_even_after_repeat_safe_migration(self):
  codes=self.enroll();migrate(self.engine);self.sign_in(codes[0]).raise_for_status();self.assertEqual(self.sign_in(codes[0]).status_code,401)
  self.assertEqual(self.client.get('/api/dev/account/mfa').json()['recovery_codes_remaining'],9)
 def test_missing_or_wrong_encryption_key_never_bypasses_enabled_factor(self):
  codes=self.enroll()
  for key in ['',Fernet.generate_key().decode()]:
   with patch.dict(os.environ,{'RAPHAEL_MFA_ENCRYPTION_KEY':key}):self.assertEqual(self.sign_in(codes[0]).status_code,503)
  self.sign_in(codes[0]).raise_for_status()
 def test_schema_stamp_refuses_pre_mfa_runtime_downgrade_without_erasing_factor(self):
  self.enroll()
  import dev_runtime
  old_components=tuple(row for row in dev_runtime.COMPONENTS if row[0]!='mfa')
  with patch.object(dev_runtime,'COMPONENTS',old_components):
   with self.assertRaisesRegex(RuntimeError,'Unsupported development schema version'):migrate(self.engine)
  self.assertTrue(self.client.get('/api/dev/account/mfa').json()['enabled'])
  self.assertEqual(self.sign_in().status_code,401)
 def test_management_requires_password_and_factor_and_has_persistent_budget(self):
  codes=self.enroll()
  for _ in range(3):self.assertEqual(self.client.post('/api/dev/account/mfa/disable',json={'password':'incorrect','code':codes[0]}).status_code,403)
  self.assertEqual(self.client.post('/api/dev/account/mfa/disable',json={'password':self.password,'code':codes[0]}).status_code,429)
  self.assertTrue(self.client.get('/api/dev/account/mfa').json()['enabled'])
 def test_disable_revokes_other_sessions_and_recovery_material(self):
  codes=self.enroll();other=TestClient(self.app);self.addCleanup(other.close);self.sign_in(codes[0],other).raise_for_status()
  self.client.post('/api/dev/account/mfa/disable',json={'password':self.password,'code':codes[1]}).raise_for_status();self.assertEqual(other.get('/api/dev/session').status_code,401)
  self.assertFalse(self.client.get('/api/dev/account/mfa').json()['enabled']);self.sign_in().raise_for_status()
 def test_factor_routes_reject_cross_origin_and_unauthenticated_requests(self):
  self.configure();self.assertEqual(self.client.post('/api/dev/account/mfa/enroll',json={'password':self.password},headers={'Origin':'https://attacker.example'}).status_code,403)
  anonymous=TestClient(self.app);self.addCleanup(anonymous.close);self.assertEqual(anonymous.get('/api/dev/account/mfa').status_code,401)
 def test_pending_enrollment_expires_and_does_not_enable_factor(self):
  self.configure();r=self.client.post('/api/dev/account/mfa/enroll',json={'password':self.password});r.raise_for_status();secret=r.json()['secret']
  with patch('dev_mfa.time.time',return_value=1700000601):
   self.assertEqual(self.client.post('/api/dev/account/mfa/confirm',json={'password':self.password,'code':totp(secret,1700000601//30)}).status_code,403)
  self.assertFalse(self.client.get('/api/dev/account/mfa').json()['enabled'])
