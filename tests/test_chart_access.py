"""Synthetic patient-scoped chart authorization,provenance migration and revocation."""
import os
import tempfile
import unittest
from unittest.mock import patch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine,text
from dev_portal import init_portal_schema,install_dev_portal,create_dev_user
from dev_chart import init_chart_schema,install_chart
from dev_messaging import init_messaging_schema,install_messaging


class ChartAccessTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='raphael-chart-access-');self.addCleanup(self.temp.cleanup)
        self.engine=create_engine('sqlite:///'+self.temp.name+'/synthetic.db');self.addCleanup(self.engine.dispose)
        flag=patch.dict(os.environ,{'RAPHAEL_DEV_SYNTHETIC_ONLY':'1'});flag.start();self.addCleanup(flag.stop)
        for initializer in (init_portal_schema,init_chart_schema,init_messaging_schema):initializer(self.engine)
        self.password='Synthetic-chart-access-password'
        self.users={}
        for tenant,name,role in [('one','creator','clinician'),('one','other','clinician'),('one','patient','patient'),('one','otherpatient','patient'),('one','admin','org_admin'),('two','foreign','clinician')]:self.users[name]=create_dev_user(self.engine,tenant,name,self.password,role)
        self.app=FastAPI();self.ctx=install_dev_portal(self.app,self.engine);install_chart(self.app,self.ctx);install_messaging(self.app,self.ctx)
        self.client=TestClient(self.app);self.addCleanup(self.client.close);self.login('creator')
        self.patient=self.client.post('/api/dev/patients',json={'name':'Synthetic scoped patient','patient_username':'patient'}).json()['id'];self.path=f'/api/dev/patients/{self.patient}'
        self.entry=self.client.post(self.path+'/entries',json={'kind':'history','label':'Synthetic history','text':'Synthetic only'}).json()['id']
    def login(self,name,tenant='one'):
        result=self.client.post('/api/dev/login',json={'tenant':tenant,'username':name,'password':self.password});self.assertEqual(result.status_code,200,result.text)
    def grant(self,username='other'):
        result=self.client.post(self.path+'/chart-access',json={'clinician_username':username,'expires_in_minutes':60,'purpose':'Synthetic full-chart consultation','authorize_full_chart':True});self.assertEqual(result.status_code,200,result.text)

    def test_same_tenant_other_clinician_denied_until_explicit_patient_grant(self):
        self.login('other');self.assertEqual(self.client.get('/api/dev/patients').json(),[])
        for route in (self.path,self.path+'/entries',self.path+'/notes',self.path+'/conversations'):
            self.assertEqual(self.client.get(route).status_code,404,route)
        self.assertEqual(self.client.post(self.path+'/notes',json={'content':'Must not be accepted'}).status_code,404)
        self.login('patient');self.grant();self.login('other')
        self.assertEqual(self.client.get(self.path).status_code,200);self.assertEqual(self.client.get(self.path+'/entries').json()[0]['id'],self.entry)
        self.assertEqual(len(self.client.get('/api/dev/patients').json()),1)
        thread=self.client.post(self.path+'/conversations',json={'title':'Authorized synthetic conversation'}).json()['id']
        self.assertEqual(self.client.post(f'/api/dev/conversations/{thread}/messages',json={'content':'Synthetic message'}).status_code,200)
        self.login('patient');self.client.post(self.path+'/chart-access/'+str(self.users['other']['id'])+'/revoke',json={}).raise_for_status();self.login('other')
        self.assertEqual(self.client.get(self.path).status_code,404);self.assertEqual(self.client.get(f'/api/dev/conversations/{thread}/messages').status_code,404)
        inbox=self.client.get('/api/dev/inbox').json();self.assertEqual(inbox['threads'],[]);self.assertEqual(inbox['unread_count'],0)

    def test_patient_can_revoke_originating_clinician_and_expiry_does_not_restore_baseline(self):
        self.login('patient');self.client.post(self.path+'/chart-access/'+str(self.users['creator']['id'])+'/revoke',json={}).raise_for_status();self.login('creator');self.assertEqual(self.client.get(self.path).status_code,404)
        self.login('patient');self.grant('creator');self.login('creator');self.assertEqual(self.client.get(self.path).status_code,200)
        with self.engine.begin() as c:c.execute(text("UPDATE dev_chart_access_decisions SET expires_at='2000-01-01T00:00:00+00:00' WHERE decision='grant'"))
        self.assertEqual(self.client.get(self.path).status_code,404)
        init_portal_schema(self.engine);self.assertEqual(self.client.get(self.path).status_code,404)
        self.login('patient');self.assertEqual(self.client.get(self.path).status_code,200);self.assertEqual(len(self.client.get(self.path+'/chart-access').json()['history']),2)

    def test_patient_only_controls_refuse_admin_foreign_and_accidental_authorization(self):
        for name in ('creator','admin','otherpatient'):
            self.login(name);self.assertIn(self.client.get(self.path+'/chart-access').status_code,(403,404))
            self.assertIn(self.client.post(self.path+'/chart-access',json={'clinician_username':'other','expires_in_minutes':60,'purpose':'Synthetic sharing','authorize_full_chart':True}).status_code,(403,404))
        self.login('patient')
        self.assertEqual(self.client.post(self.path+'/chart-access',json={'clinician_username':'other','expires_in_minutes':60,'purpose':'Synthetic sharing','authorize_full_chart':False}).status_code,422)
        for recipient in ('admin','foreign','patient','unknown'):
            self.assertEqual(self.client.post(self.path+'/chart-access',json={'clinician_username':recipient,'expires_in_minutes':60,'purpose':'Synthetic sharing','authorize_full_chart':True}).status_code,404)
        self.login('foreign','two');self.assertEqual(self.client.get(self.path).status_code,404)

    def test_migration_assigns_only_unambiguous_creator_audit_and_never_infers_from_old_reads(self):
        with self.engine.begin() as c:
            c.execute(text('DELETE FROM dev_schema_migrations WHERE version=3'))
            c.execute(text('UPDATE dev_patients SET created_by_id=NULL WHERE id=:id'),{'id':self.patient})
            for label in ('Unknown provenance','Ambiguous provenance'):
                c.execute(text("INSERT INTO dev_patients(tenant,name,created_at) VALUES ('one',:label,'2024-01-01')"),{'label':label})
            unknown=c.execute(text("SELECT id FROM dev_patients WHERE name='Unknown provenance'")).scalar()
            ambiguous=c.execute(text("SELECT id FROM dev_patients WHERE name='Ambiguous provenance'")).scalar()
            c.execute(text("INSERT INTO dev_audit(tenant,actor_id,event,resource_type,resource_id,created_at) VALUES ('one',:u,'read_patient','patient',:p,'2024-01-01')"),{'u':self.users['creator']['id'],'p':unknown})
            for name in ('creator','other'):
                c.execute(text("INSERT INTO dev_audit(tenant,actor_id,event,resource_type,resource_id,created_at) VALUES ('one',:u,'create_patient','patient',:p,'2024-01-01')"),{'u':self.users[name]['id'],'p':ambiguous})
        init_portal_schema(self.engine)
        with self.engine.connect() as c:
            self.assertEqual(c.execute(text('SELECT created_by_id FROM dev_patients WHERE id=:p'),{'p':self.patient}).scalar(),self.users['creator']['id'])
            self.assertIsNone(c.execute(text('SELECT created_by_id FROM dev_patients WHERE id=:p'),{'p':unknown}).scalar())
            self.assertIsNone(c.execute(text('SELECT created_by_id FROM dev_patients WHERE id=:p'),{'p':ambiguous}).scalar())
            self.assertEqual(c.execute(text('SELECT count(*) FROM dev_patients')).scalar(),3)
        self.assertEqual(self.client.get(f'/api/dev/patients/{unknown}').status_code,404)

    def test_original_patient_schema_upgrade_preserves_clinical_rows_and_references(self):
        from dev_portal import SCHEMA
        from database import execute_sqlite_script
        legacy=create_engine('sqlite:///'+self.temp.name+'/legacy-synthetic.db');self.addCleanup(legacy.dispose)
        with legacy.begin() as c:
            execute_sqlite_script(c,SCHEMA.replace(' created_by_id INTEGER REFERENCES dev_users(id),\n',''))
            c.execute(text("INSERT INTO dev_users(id,tenant,username,role,password_hash) VALUES (91,'one','original','clinician','synthetic-hash')"))
            c.execute(text("INSERT INTO dev_patients(id,tenant,name,created_at) VALUES (71,'one','Synthetic preserved','2024-01-01')"))
            c.execute(text("INSERT INTO dev_conversations(id,patient_id,clinician_id,title,created_at) VALUES (61,71,91,'Synthetic conversation','2024-01-01')"))
            c.execute(text("INSERT INTO dev_messages(conversation_id,sender_id,content,created_at) VALUES (61,91,'Synthetic content','2024-01-01')"))
            c.execute(text("INSERT INTO dev_audit(tenant,actor_id,event,resource_type,resource_id,created_at) VALUES ('one',91,'create_patient','patient',71,'2024-01-01')"))
        init_portal_schema(legacy);init_portal_schema(legacy)
        with legacy.connect() as c:
            self.assertEqual(c.execute(text('SELECT created_by_id FROM dev_patients WHERE id=71')).scalar(),91)
            self.assertEqual(c.execute(text('SELECT content FROM dev_messages')).scalar(),'Synthetic content')
            self.assertEqual(c.execute(text('SELECT password_hash FROM dev_users WHERE id=91')).scalar(),'synthetic-hash')
            self.assertEqual(c.execute(text('SELECT count(*) FROM dev_schema_migrations WHERE version=3')).scalar(),1)
            self.assertEqual(c.exec_driver_sql('PRAGMA foreign_key_check').all(),[])

    def test_chart_revoke_cancels_later_ai_tokens_and_hides_cached_chart_thread(self):
        from dev_intelligence import init_intelligence_schema,install_intelligence
        init_intelligence_schema(self.engine)
        patient=TestClient(self.app);self.addCleanup(patient.close)
        patient.post('/api/dev/login',json={'tenant':'one','username':'patient','password':self.password}).raise_for_status()
        def provider(messages,model):
            yield 'Previously authorized synthetic draft '
            patient.post(self.path+'/chart-access/'+str(self.users['other']['id'])+'/revoke',json={}).raise_for_status()
            yield 'MUST-NOT-APPEAR-AFTER-CHART-REVOCATION [fixture]'
        install_intelligence(self.app,self.ctx,token_provider=provider)
        self.login('patient');self.grant();self.login('other')
        thread=self.client.post('/api/dev/intelligence/threads',json={'title':'Synthetic chart-scoped conversation','patient_id':self.patient}).json()['id']
        with patch('dev_knowledge.search_evidence',return_value=[{'chunk_id':'fixture','title':'Synthetic fixture'}]):
            response=self.client.post(f'/api/dev/intelligence/threads/{thread}/stream',json={'content':'Synthetic chart concern'})
        self.assertIn('event: cancelled',response.text);self.assertNotIn('MUST-NOT-APPEAR',response.text)
        self.assertEqual(self.client.get(f'/api/dev/intelligence/threads/{thread}').status_code,404)
        self.assertEqual(self.client.get('/api/dev/intelligence/threads').json(),[])
