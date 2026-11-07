import os,tempfile,unittest
from unittest.mock import patch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine,text
from auth.dev_recover import recover_dev_user
from dev_portal import init_portal_schema,create_dev_user,install_dev_portal

class OperatorRecoveryTests(unittest.TestCase):
    def setUp(self):
        directory=tempfile.TemporaryDirectory();self.addCleanup(directory.cleanup)
        self.engine=create_engine('sqlite:///'+directory.name+'/synthetic.db');self.addCleanup(self.engine.dispose)
        flag=patch.dict(os.environ,{'RAPHAEL_DEV_SYNTHETIC_ONLY':'1'});flag.start();self.addCleanup(flag.stop)
        init_portal_schema(self.engine)
        for tenant in ('one','two'):create_dev_user(self.engine,tenant,'patient','Original-synthetic-password','patient')
        app=FastAPI();install_dev_portal(app,self.engine)
        self.first=TestClient(app);self.second=TestClient(app);self.addCleanup(self.first.close);self.addCleanup(self.second.close)
        for client,tenant in ((self.first,'one'),(self.second,'two')):self.assertEqual(client.post('/api/dev/login',json={'tenant':tenant,'username':'patient','password':'Original-synthetic-password'}).status_code,200)
    def test_recovery_atomically_revokes_only_target_sessions_and_audits_operator(self):
        result=recover_dev_user(self.engine,'one','patient','Replacement-synthetic-password');self.assertEqual(result['sessions_revoked'],1)
        self.assertEqual(self.first.get('/api/dev/session').status_code,401);self.assertEqual(self.second.get('/api/dev/session').status_code,200)
        self.assertEqual(self.first.post('/api/dev/login',json={'tenant':'one','username':'patient','password':'Original-synthetic-password'}).status_code,401)
        self.assertEqual(self.first.post('/api/dev/login',json={'tenant':'one','username':'patient','password':'Replacement-synthetic-password'}).status_code,200)
        with self.engine.connect() as c:
            row=c.execute(text("SELECT actor_id,tenant,resource_type FROM dev_audit WHERE event='local_operator_password_recovery'")).first();self.assertEqual(tuple(row),(None,'one','user'))
    def test_weak_missing_or_disabled_account_cannot_change_sessions_or_activate_account(self):
        for name,password in (('patient','short'),('missing','Replacement-password')):
            with self.assertRaises(ValueError):recover_dev_user(self.engine,'one',name,password)
        self.assertEqual(self.first.get('/api/dev/session').status_code,200)
        with self.engine.begin() as c:c.execute(text("UPDATE dev_users SET active=0 WHERE tenant='one'"))
        with self.assertRaises(ValueError):recover_dev_user(self.engine,'one','patient','Replacement-password')
        with self.engine.connect() as c:self.assertEqual(c.execute(text("SELECT active FROM dev_users WHERE tenant='one'")).scalar(),0)
    def test_missing_synthetic_flag_denies_recovery(self):
        with patch.dict(os.environ,{'RAPHAEL_DEV_SYNTHETIC_ONLY':'0'}):
            with self.assertRaises(RuntimeError):recover_dev_user(self.engine,'one','patient','Replacement-password')
        self.assertEqual(self.first.get('/api/dev/session').status_code,200)
