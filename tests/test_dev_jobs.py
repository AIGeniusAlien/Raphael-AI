"""Actual SQLite/API/process worker persistence tests;no clinical/provider calls."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from uuid import uuid4
from fastapi import FastAPI,Request
from fastapi.testclient import TestClient
from sqlalchemy import create_engine,text
from dev_jobs import init_jobs_schema,install_jobs,claim,finish,execute_job,heartbeat
from dev_knowledge import init_knowledge_schema
from dev_portal import create_dev_user,init_portal_schema,install_dev_portal
from dev_verify import isolated_environment

class JobsTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory(prefix='raphael-jobs-test-');self.addCleanup(temp.cleanup);self.directory=Path(temp.name)
        self.env=isolated_environment();self.env['RAPHAEL_DEV_SYNTHETIC_ONLY']='1'
        flag=patch.dict(os.environ,{'RAPHAEL_DEV_SYNTHETIC_ONLY':'1'});flag.start();self.addCleanup(flag.stop)
        self.url='sqlite:///'+str(self.directory/'synthetic.db');self.engine=create_engine(self.url);self.addCleanup(self.engine.dispose)
        init_portal_schema(self.engine);init_knowledge_schema(self.engine);init_jobs_schema(self.engine)
        self.password='Synthetic-only-jobs-password'
        for tenant,name,role in [('one','doctor','clinician'),('one','peer','clinician'),('one','admin','org_admin'),('one','patient','patient'),('two','foreign','clinician')]:create_dev_user(self.engine,tenant,name,self.password,role)
        self.app=FastAPI();self.ctx=install_dev_portal(self.app,self.engine);install_jobs(self.app,self.ctx)
        self.client=TestClient(self.app);self.addCleanup(self.client.close);self.login()
    def login(self,name='doctor',tenant='one'):
        r=self.client.post('/api/dev/login',json={'username':name,'tenant':tenant,'password':self.password});self.assertEqual(r.status_code,200,r.text)
    def enqueue(self,**changes):
        body={'request_id':str(uuid4()),'kind':'evidence_inventory'};body.update(changes)
        return self.client.post('/api/dev/jobs',json=body)
    def read(self,identifier):return self.client.get(f'/api/dev/jobs/{identifier}')
    def worker(self,*args,env=None):
        return subprocess.run([sys.executable,'-m','dev_jobs','--database-url',self.url,'--tenant','one',*args],cwd=Path(__file__).resolve().parents[1],env=env or self.env,text=True,capture_output=True,timeout=25)
    def test_idempotent_api_worker_process_restart_and_append_only_history(self):
        request=str(uuid4());r=self.enqueue(request_id=request);self.assertEqual(r.status_code,200,r.text);identifier=r.json()['id']
        self.assertTrue(self.enqueue(request_id=request).json()['deduplicated']);self.assertEqual(self.enqueue(request_id=request,priority=2).status_code,409)
        init_jobs_schema(self.engine)
        worker=self.worker();self.assertEqual(worker.returncode,0,worker.stdout+worker.stderr);self.assertEqual(json.loads(worker.stdout)['provider_requests'],0)
        self.client.close();self.client=TestClient(self.app);self.login();row=self.read(identifier).json()
        self.assertEqual(row['state'],'succeeded');self.assertEqual(row['attempts'],1);self.assertEqual(row['result']['approved_chunks'],0)
        self.assertEqual([e['event'] for e in row['events']],['enqueued','claimed','succeeded']);self.assertNotIn('lease_token_hash',row);self.assertNotIn('token',row)
        self.assertEqual(self.worker().returncode,0);self.assertEqual(self.read(identifier).json()['attempts'],1)
        with self.engine.connect() as c:
            with self.assertRaises(Exception):c.execute(text('DELETE FROM dev_job_events WHERE job_id=:id'),{'id':identifier})
    def test_role_tenant_owner_and_input_boundaries(self):
        row=self.enqueue().json();identifier=row['id']
        for name,tenant in [('peer','one'),('foreign','two'),('admin','one')]:
            self.login(name,tenant);self.assertEqual(self.read(identifier).status_code,404);self.assertEqual(self.client.post(f'/api/dev/jobs/{identifier}/cancel',json={}).status_code,404);self.assertEqual(self.client.get('/api/dev/jobs').json()['items'],[])
        self.assertEqual(self.enqueue().status_code,403)
        self.assertEqual(self.enqueue(kind='requirements_snapshot').status_code,200)
        self.login('patient');self.assertEqual(self.client.get('/api/dev/jobs/status').status_code,403)
        self.login();self.assertEqual(self.enqueue(kind='prescribe').status_code,422);self.assertEqual(self.enqueue(url='https://example.org').status_code,422);self.assertEqual(self.enqueue(max_attempts=4).status_code,422);self.assertEqual(self.enqueue(delay_seconds=True).status_code,422)
        self.client.post('/api/dev/logout',json={});self.assertEqual(self.read(identifier).status_code,401)
    def test_real_crashed_claim_is_recovered_by_new_worker_process(self):
        row=self.enqueue().json();identifier=row['id']
        code="from protected_database import application_engine;from dev_jobs import claim;import os;e=application_engine(os.environ['DATABASE_URL']);j=claim(e,'one',lease_seconds=1);os._exit(23 if j else 24)"
        env=dict(self.env,DATABASE_URL=self.url)
        crashed=subprocess.run([sys.executable,'-c',code],env=env,cwd=Path(__file__).resolve().parents[1],timeout=10)
        self.assertEqual(crashed.returncode,23);self.assertEqual(self.read(identifier).json()['state'],'running')
        recovered=self.worker('--max-jobs','1','--run-seconds','8');self.assertEqual(recovered.returncode,0,recovered.stdout+recovered.stderr)
        row=self.read(identifier).json();self.assertEqual(row['state'],'succeeded');self.assertEqual(row['attempts'],2);self.assertIn('lease_expired',[e['event'] for e in row['events']])
    def test_concurrent_claims_cancellation_and_stale_result_fencing(self):
        identifier=self.enqueue().json()['id']
        with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(lambda _:claim(self.engine,'one'),range(2)))
        leased=[x for x in results if x is not None];self.assertEqual(len(leased),1);job=leased[0]
        self.assertTrue(heartbeat(self.engine,job))
        self.assertEqual(self.client.post(f'/api/dev/jobs/{identifier}/cancel',json={}).status_code,200)
        self.assertFalse(finish(self.engine,job,result={'untrusted':'must not persist'}));self.assertFalse(heartbeat(self.engine,job))
        row=self.read(identifier).json();self.assertEqual(row['state'],'cancelled');self.assertIsNone(row['result']);self.assertEqual(self.client.post(f'/api/dev/jobs/{identifier}/cancel',json={}).status_code,200)
    def test_expired_token_cannot_finish_and_attempt_budget_dead_letters(self):
        identifier=self.enqueue(max_attempts=2).json()['id'];at=time.time()+1
        old=claim(self.engine,'one',at=at,lease_seconds=1)
        self.assertIsNone(claim(self.engine,'one',at=at+2,lease_seconds=1))
        new=claim(self.engine,'one',at=at+5,lease_seconds=1);self.assertIsNotNone(new)
        self.assertFalse(finish(self.engine,old,result={'stale':True},at=at+5))
        self.assertIsNone(claim(self.engine,'one',at=at+7));row=self.read(identifier).json();self.assertEqual(row['state'],'dead_letter');self.assertEqual(row['attempts'],2);self.assertEqual(row['last_error_code'],'lease_expired')
    def test_actual_handler_failure_retries_backoff_and_dead_letters(self):
        identifier=self.enqueue().json()['id']
        with self.engine.begin() as c:c.execute(text('DROP TABLE dev_evidence_sources'))
        outcome=self.worker('--max-jobs','3','--run-seconds','12');self.assertEqual(outcome.returncode,0,outcome.stdout+outcome.stderr)
        row=self.read(identifier).json();self.assertEqual(row['state'],'dead_letter');self.assertEqual(row['attempts'],3);self.assertEqual(row['last_error_code'],'worker_operation_failed');self.assertIsNone(row['result'])
        self.assertEqual([e['event'] for e in row['events']].count('retry_wait'),2)
    def test_owner_deactivation_and_session_revocation_block_changes(self):
        identifier=self.enqueue().json()['id'];job=claim(self.engine,'one')
        with self.engine.begin() as c:c.execute(text("UPDATE dev_users SET active=0 WHERE username='doctor' AND tenant='one'"))
        self.assertTrue(finish(self.engine,job,result={'discard':'private result'}))
        with self.engine.connect() as c:
            row=c.execute(text('SELECT state,result_json FROM dev_jobs WHERE id=:id'),{'id':identifier}).first();self.assertEqual(row,('cancelled',None))
        self.assertEqual(self.read(identifier).status_code,401)
        with self.engine.begin() as c:c.execute(text("UPDATE dev_users SET active=1 WHERE username='doctor' AND tenant='one'"))
        self.login()
        original=self.ctx.user
        def revoke_after_auth(request:Request):
            user=original(request)
            with self.engine.begin() as c:c.execute(text('DELETE FROM dev_sessions WHERE user_id=:u'),{'u':user['id']})
            return user
        self.app.dependency_overrides[self.ctx.user]=revoke_after_auth
        self.assertEqual(self.enqueue().status_code,401)
    def test_scheduling_priority_and_requirements_snapshot(self):
        delayed=self.enqueue(delay_seconds=100,priority=5).json();ready=self.enqueue(priority=1).json()
        job=claim(self.engine,'one');self.assertEqual(job['id'],ready['id']);finish(self.engine,job,result=execute_job(self.engine,job))
        self.assertIsNone(claim(self.engine,'one'));self.assertEqual(self.read(delayed['id']).json()['state'],'queued')
        self.login('admin');snapshot=self.enqueue(kind='requirements_snapshot').json();self.assertEqual(self.worker().returncode,0)
        row=self.read(snapshot['id']).json();self.assertEqual(row['state'],'succeeded');self.assertGreaterEqual(row['result']['requirement_count'],1164);self.assertEqual(len(row['result']['catalog_sha256']),64)
    def test_bounded_worker_heartbeat_is_not_permanent_scheduler(self):
        process=subprocess.Popen([sys.executable,'-m','dev_jobs','--database-url',self.url,'--tenant','one','--max-jobs','100','--run-seconds','2'],env=self.env,cwd=Path(__file__).resolve().parents[1],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        try:
            deadline=time.monotonic()+5
            while True:
                status=self.client.get('/api/dev/jobs/status').json()
                if status['fresh_worker_heartbeat']:break
                if time.monotonic()>deadline:self.fail('No actual worker heartbeat observed.')
                time.sleep(.05)
            self.assertFalse(status['persistent_scheduler_configured']);out,err=process.communicate(timeout=10);self.assertEqual(process.returncode,0,out+err)
            self.assertFalse(self.client.get('/api/dev/jobs/status').json()['fresh_worker_heartbeat'])
        finally:
            if process.poll() is None:process.kill();process.wait(timeout=5)
    @unittest.skipUnless(sys.platform=='linux' and importlib.util.find_spec('sqlcipher3') is not None,'Actual Linux SQLCipher runtime required;no encrypted-worker acceptance on this host.')
    def test_actual_encrypted_database_worker_process(self):
        from cryptography.fernet import Fernet
        from protected_database import application_engine
        key=Fernet.generate_key().decode();env=dict(self.env,RAPHAEL_DATABASE_ENCRYPTED='1',RAPHAEL_DATABASE_ENCRYPTION_KEY=key)
        encrypted_url='sqlite:///'+str(self.directory/'encrypted.db')
        with patch.dict(os.environ,env):
            encrypted=application_engine(encrypted_url);init_portal_schema(encrypted);init_jobs_schema(encrypted);create_dev_user(encrypted,'one','admin',self.password,'org_admin')
            app=FastAPI();ctx=install_dev_portal(app,encrypted);install_jobs(app,ctx)
            with TestClient(app) as client:
                self.assertEqual(client.post('/api/dev/login',json={'tenant':'one','username':'admin','password':self.password}).status_code,200)
                queued=client.post('/api/dev/jobs',json={'request_id':str(uuid4()),'kind':'requirements_snapshot'}).json()
                result=subprocess.run([sys.executable,'-m','dev_jobs','--database-url',encrypted_url,'--tenant','one'],cwd=Path(__file__).resolve().parents[1],env=env,text=True,capture_output=True,timeout=15)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr);self.assertNotIn(key,result.stdout+result.stderr)
                self.assertEqual(client.get(f"/api/dev/jobs/{queued['id']}").json()['state'],'succeeded')
            encrypted.dispose()
        with sqlite3.connect(self.directory/'encrypted.db') as plain:
            with self.assertRaises(sqlite3.DatabaseError):plain.execute('SELECT * FROM dev_jobs').fetchall()
