"""Synthetic credential lifecycle;these tests are not legal erasure approval."""
import os
import unittest
from unittest.mock import patch
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient
from sqlalchemy import text
from tests import test_account as fixture
from tests import test_mfa as mfa_fixture
from dev_portal import create_dev_user
from dev_runtime import migrate

class AccountLifecycleTests(unittest.TestCase):
    setUp=fixture.AccountTests.setUp
    login=fixture.AccountTests.login
    configure=mfa_fixture.MFATests.configure
    enroll=mfa_fixture.MFATests.enroll
    def uid(self,name='patient',tenant='one'):
        with self.engine.connect() as c:return c.execute(text('SELECT id FROM dev_users WHERE username=:n AND tenant=:t'),{'n':name,'t':tenant}).scalar()
    def deactivate(self,request=True,**changes):
        body={'password':self.password,'confirm':True,'request_credential_erasure':request};body.update(changes)
        return self.client.post('/api/dev/account/lifecycle/deactivate',json=body)
    def policy(self,days=0,version=0,enabled=True):
        return self.client.post('/api/dev/account/lifecycle/policy',json={'password':self.password,'reference':'SYNTHETIC-TEST-POLICY-NOT-LEGAL-APPROVAL','minimum_inactive_days':days,'enable_credential_erasure':enabled,'expected_revision':version})
    def erase(self,identifier,**changes):
        body={'password':self.password,'confirm':True,'target_username':'patient','expected_revision':1,'policy_revision':1};body.update(changes)
        return self.client.post(f'/api/dev/account/lifecycle/accounts/{identifier}/erase-credentials',json=body)

    def test_password_change_revokes_every_session_and_requires_new_credentials(self):
        other=TestClient(self.app);self.addCleanup(other.close);self.login(client=other)
        path='/api/dev/account/lifecycle/password'
        body={'password':self.password,'new_password':'Different-Synthetic-Password-2026','confirm':True}
        self.assertEqual(self.client.post(path,json={**body,'confirm':False}).status_code,422)
        self.assertEqual(self.client.post(path,json={**body,'password':'wrong'}).status_code,403)
        self.assertEqual(self.client.get('/api/dev/session').status_code,200)
        response=self.client.post(path,json=body)
        self.assertEqual(response.status_code,200,response.text)
        self.assertTrue(response.json()['sessions_revoked'])
        self.assertEqual(self.client.get('/api/dev/session').status_code,401)
        self.assertEqual(other.get('/api/dev/session').status_code,401)
        self.assertEqual(self.client.post('/api/dev/login',json={'tenant':'one','username':'patient','password':self.password}).status_code,401)
        self.client.post('/api/dev/login',json={'tenant':'one','username':'patient','password':body['new_password']}).raise_for_status()
        self.assertEqual(self.client.get('/api/dev/session').status_code,200)

    def test_password_change_requires_mfa_and_preserves_other_accounts(self):
        codes=self.enroll()
        other=TestClient(self.app);self.addCleanup(other.close);self.login('other',client=other)
        body={'password':self.password,'new_password':'Another-Synthetic-Password-2026','confirm':True}
        self.assertEqual(self.client.post('/api/dev/account/lifecycle/password',json=body).status_code,403)
        response=self.client.post('/api/dev/account/lifecycle/password',json={**body,'code':codes[0]})
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(other.get('/api/dev/session').status_code,200)

    def test_deactivate_revokes_all_sessions_preserves_records_and_admin_restores_access(self):
        identifier=self.uid();other=TestClient(self.app);self.addCleanup(other.close);self.login(client=other)
        self.assertEqual(self.deactivate(password='WRONG-SYNTHETIC-PASSWORD').status_code,403)
        self.assertEqual(self.client.get('/api/dev/session').status_code,200)
        response=self.deactivate();self.assertEqual(response.status_code,200,response.text);self.assertIn('Path=/api/dev',response.headers['set-cookie'])
        self.assertEqual(self.client.get('/api/dev/session').status_code,401);self.assertEqual(other.get('/api/dev/session').status_code,401)
        self.assertEqual(self.client.post('/api/dev/login',json={'tenant':'one','username':'patient','password':self.password}).status_code,401)
        with self.engine.connect() as c:self.assertEqual(c.execute(text('SELECT count(*) FROM dev_patients')).scalar(),2)
        self.login('admin');self.client.patch(f'/api/dev/admin/users/{identifier}',json={'active':True}).raise_for_status()
        self.assertEqual(other.get('/api/dev/session').status_code,401)
        self.login();state=self.client.get('/api/dev/account/lifecycle').json()['lifecycle'];self.assertEqual(state['state'],'reactivated');self.assertIsNone(state['erasure_requested_at'])
        self.assertEqual(self.client.get('/api/dev/patients').json()[0]['id'],self.pid)

    def test_mfa_requires_unused_factor_for_self_deactivation(self):
        codes=self.enroll();identifier=self.uid()
        self.assertEqual(self.deactivate().status_code,403)
        self.assertEqual(self.deactivate(code=self.code).status_code,403) # enrollment code is consumed
        self.assertEqual(self.deactivate(code=codes[0]).status_code,200)
        self.login('admin');self.client.patch(f'/api/dev/admin/users/{identifier}',json={'active':True}).raise_for_status()
        denied=self.client.post('/api/dev/login',json={'tenant':'one','username':'patient','password':self.password,'otp':codes[0]});self.assertEqual(denied.status_code,401)
        accepted=self.client.post('/api/dev/login',json={'tenant':'one','username':'patient','password':self.password,'otp':codes[1]});self.assertEqual(accepted.status_code,200)
        self.assertTrue(self.client.get('/api/dev/account/mfa').json()['enabled'])

    def test_limited_erasure_removes_credentials_not_clinical_identity_links(self):
        identifier=self.uid()
        self.client.post('/api/dev/account/profile',json={'display_name':'Synthetic private preference','language':'en','time_zone':'UTC','internal_notifications':False}).raise_for_status()
        original=self.client.post(f'/api/dev/patients/{self.pid}/documents',data={'title':'Synthetic retained original'},files={'file':('retained.txt',b'Synthetic retained original document','text/plain')});self.assertEqual(original.status_code,200)
        with patch.dict(os.environ,{'RAPHAEL_MFA_ENCRYPTION_KEY':Fernet.generate_key().decode()}):self.client.post('/api/dev/account/mfa/enroll',json={'password':self.password}).raise_for_status()
        self.deactivate().raise_for_status();migrate(self.engine);self.login('admin')
        self.assertEqual(self.erase(identifier).status_code,409) # no default erasure policy
        self.policy().raise_for_status();response=self.erase(identifier);self.assertEqual(response.status_code,200,response.text)
        for key in ('clinical_records_deleted','original_documents_deleted','audit_records_deleted','backups_erased'):self.assertFalse(response.json()[key])
        self.assertEqual(self.client.patch(f'/api/dev/admin/users/{identifier}',json={'active':True}).status_code,409)
        self.assertEqual(self.erase(identifier).status_code,409) # stale retry has no second effect
        self.client.patch(f'/api/dev/admin/users/{identifier}',json={'active':False}).raise_for_status()
        self.assertEqual(self.client.patch(f'/api/dev/admin/users/{identifier}',json={'active':True}).status_code,409)
        with self.engine.connect() as c:
            for table in ('dev_sessions','dev_mfa_factors','dev_mfa_recovery','dev_account_profiles'):self.assertEqual(c.execute(text('SELECT count(*) FROM '+table+' WHERE user_id=:u'),{'u':identifier}).scalar(),0)
            self.assertEqual(c.execute(text('SELECT patient_user_id FROM dev_patients WHERE id=:p'),{'p':self.pid}).scalar(),identifier)
            self.assertEqual(c.execute(text('SELECT count(*) FROM dev_chart_documents')).scalar(),1)
            self.assertEqual(c.execute(text("SELECT count(*) FROM dev_audit WHERE event='erase_account_credentials_retain_clinical_records'")).scalar(),1)
        # Reusing a freed login name must never inherit the retained clinical record.
        create_dev_user(self.engine,'one','patient',self.password,'patient');self.login();self.assertEqual(self.client.get('/api/dev/patients').json(),[])

    def test_hold_and_operator_minimum_period_block_erasure(self):
        identifier=self.uid();self.deactivate().raise_for_status();self.login('admin');self.policy(days=7).raise_for_status()
        path=f'/api/dev/account/lifecycle/accounts/{identifier}/hold'
        self.client.post(path,json={'password':self.password,'active':True,'reference':'SYNTHETIC-HOLD','expected_revision':0}).raise_for_status()
        self.assertEqual(self.erase(identifier).status_code,409)
        self.client.post(path,json={'password':self.password,'active':False,'reference':'SYNTHETIC-RELEASE','expected_revision':1}).raise_for_status()
        self.assertEqual(self.erase(identifier).status_code,409) # policy period still has not elapsed
        with self.engine.connect() as c:self.assertEqual(c.execute(text('SELECT username FROM dev_users WHERE id=:u'),{'u':identifier}).scalar(),'patient')

    def test_owner_request_is_required_even_with_enabled_policy(self):
        identifier=self.uid();self.deactivate(request=False).raise_for_status();self.login('admin');self.policy().raise_for_status()
        self.assertEqual(self.erase(identifier).status_code,409)

    def test_policy_versions_admin_authorization_and_cross_tenant_denial(self):
        identifier=self.uid();self.assertEqual(self.client.get('/api/dev/account/lifecycle/administration').status_code,403)
        self.assertEqual(self.policy().status_code,403)
        self.login('foreign','two')
        self.assertEqual(self.erase(identifier).status_code,404)
        self.assertEqual(self.client.post(f'/api/dev/account/lifecycle/accounts/{identifier}/hold',json={'password':self.password,'active':True,'reference':'SYNTHETIC-HOLD','expected_revision':0}).status_code,404)
        self.login('admin');self.policy().raise_for_status();self.assertEqual(self.policy(version=0).status_code,409)
        self.policy(version=1,enabled=False).raise_for_status();self.assertFalse(self.client.get('/api/dev/account/lifecycle/administration').json()['policy']['enable_credential_erasure'])
        self.assertEqual(self.deactivate().status_code,409) # no administrator self-deactivation bypass

    def test_failed_sensitive_credentials_are_bounded_without_changing_access(self):
        for _ in range(5):self.assertEqual(self.deactivate(password='WRONG-SYNTHETIC-PASSWORD').status_code,403)
        response=self.deactivate();self.assertEqual(response.status_code,429);self.assertEqual(response.headers['retry-after'],'300');self.assertEqual(self.client.get('/api/dev/session').status_code,200)

    def test_new_privacy_components_block_old_runtime_and_repeat_migration_keeps_state(self):
        import dev_runtime
        identifier=self.uid();self.deactivate().raise_for_status();original=dev_runtime.COMPONENTS
        with patch.object(dev_runtime,'COMPONENTS',tuple(row for row in original if row[0] not in ('conversation_attachments','account_lifecycle'))):
            with self.assertRaisesRegex(RuntimeError,'Unsupported development schema'):migrate(self.engine)
        migrate(self.engine)
        with self.engine.connect() as c:
            self.assertEqual(c.execute(text('SELECT active FROM dev_users WHERE id=:u'),{'u':identifier}).scalar(),0)
            self.assertEqual(c.execute(text('SELECT state FROM dev_account_lifecycle WHERE user_id=:u'),{'u':identifier}).scalar(),'deactivated')
