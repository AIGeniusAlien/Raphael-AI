"""CI harness restrictions;these fixture tests are not live-provider evidence."""
import importlib.util
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from mobile.scripts.fixture_control import FixtureControl


class NativeFixtureControlTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.private=Path(self.temp.name)
        self.token='SYNTHETIC-CONTROL-TOKEN-'+'x'*24
        (self.private/'control-token').write_text(self.token)
        self.env=patch.dict(os.environ,{'GITHUB_ACTIONS':'true','RAPHAEL_DEV_SYNTHETIC_ONLY':'1','DATABASE_URL':'sqlite:///'+str(self.private/'synthetic.db')});self.env.start();self.addCleanup(self.env.stop)
        with sqlite3.connect(self.private/'synthetic.db') as c:
            c.executescript("CREATE TABLE dev_users(id INTEGER,tenant TEXT,username TEXT); CREATE TABLE dev_sessions(user_id INTEGER,expires_at TEXT); INSERT INTO dev_users VALUES(1,'native-ci','native-expiry-patient'),(2,'native-ci','other'); INSERT INTO dev_sessions VALUES(1,'2099-01-01'),(2,'2099-01-01');")
        app=FastAPI()
        @app.get('/readyz')
        def ready():return {'status':'ready'}
        self.client=TestClient(FixtureControl(app,self.private));self.addCleanup(self.client.close)

    def control(self,operation,token=None):
        return self.client.post('/_ci/native-control',json={'operation':operation},headers={'X-Raphael-CI-Control':self.token if token is None else token})

    def test_production_or_other_database_configuration_is_refused(self):
        for values in ({'GITHUB_ACTIONS':'false'},{'RAPHAEL_DEV_SYNTHETIC_ONLY':'0'},{'DATABASE_URL':'sqlite:///other.sqlite'}):
            with patch.dict(os.environ,values),self.assertRaises(RuntimeError):FixtureControl(FastAPI(),self.private)

    def test_unauthenticated_control_cannot_inject_faults_or_expire_sessions(self):
        self.assertEqual(self.control('expire_fixture_session','wrong').status_code,404)
        self.assertEqual(self.client.get('/readyz').status_code,200)
        with sqlite3.connect(self.private/'synthetic.db') as c:self.assertEqual(c.execute("SELECT count(*) FROM dev_sessions WHERE expires_at='2099-01-01'").fetchone()[0],2)

    def test_failure_and_recovery_affect_only_fixed_native_read_paths(self):
        self.assertEqual(self.control('unavailable').status_code,200);self.assertEqual(self.client.get('/readyz').status_code,503)
        self.assertEqual(self.client.get('/api/dev/account/export').status_code,503)
        self.assertEqual(self.client.get('/unrelated').status_code,404)
        self.assertEqual(self.control('healthy').status_code,200);self.assertEqual(self.client.get('/readyz').status_code,200)
        self.assertEqual(self.control('arbitrary-command').status_code,422)
        self.assertEqual(self.client.post('/_ci/native-control',content=b'x'*257,headers={'X-Raphael-CI-Control':self.token}).status_code,413)

    def test_expiration_preserves_rows_and_cannot_target_other_users(self):
        self.assertEqual(self.control('expire_fixture_session').status_code,200)
        with sqlite3.connect(self.private/'synthetic.db') as c:
            self.assertEqual(c.execute('SELECT count(*) FROM dev_sessions').fetchone()[0],2)
            self.assertEqual(c.execute('SELECT expires_at FROM dev_sessions WHERE user_id=2').fetchone()[0],'2099-01-01')
            self.assertTrue(c.execute('SELECT expires_at FROM dev_sessions WHERE user_id=1').fetchone()[0].startswith('2000'))
