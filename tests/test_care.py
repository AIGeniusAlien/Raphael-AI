"""Synthetic consent-scoped care journey; no real facility delivery or model."""
import os
import tempfile
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine,text
from dev_portal import init_portal_schema,install_dev_portal,create_dev_user
from dev_chart import init_chart_schema,install_chart
from dev_care import init_care_schema,install_care
from dev_intelligence import init_intelligence_schema,install_intelligence
from dev_knowledge import init_knowledge_schema


class CareJourneyTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='raphael-care-');self.addCleanup(self.tmp.cleanup)
        self.engine=create_engine('sqlite:///'+self.tmp.name+'/synthetic.db');self.addCleanup(self.engine.dispose)
        flag=patch.dict(os.environ,{'RAPHAEL_DEV_SYNTHETIC_ONLY':'1'});flag.start();self.addCleanup(flag.stop)
        for init in (init_portal_schema,init_chart_schema,init_care_schema,init_knowledge_schema,init_intelligence_schema):init(self.engine)
        self.password='Synthetic-care-test-password'
        for tenant,name,role in [('one','creator','clinician'),('one','receiver','clinician'),('one','other','clinician'),('one','patient','patient'),('one','otherpatient','patient'),('two','foreign','clinician'),('one','admin','org_admin')]:create_dev_user(self.engine,tenant,name,self.password,role)
        self.app=FastAPI();ctx=install_dev_portal(self.app,self.engine);install_chart(self.app,ctx);install_care(self.app,ctx)
        self.prompts=[]
        def tokens(messages,model):self.prompts.append(messages);yield 'Synthetic [fixture]'
        self.tokens=tokens
        install_intelligence(self.app,ctx,token_provider=lambda m,v:self.tokens(m,v))
        self.client=TestClient(self.app);self.addCleanup(self.client.close);self.login('creator')
        self.patient=self.client.post('/api/dev/patients',json={'name':'Synthetic patient','patient_username':'patient'}).json()['id']
        self.client.post(f'/api/dev/patients/{self.patient}/entries',json={'kind':'medication','label':'Synthetic medication','dose':'Entered fixture dose'}).raise_for_status()
        self.client.post(f'/api/dev/patients/{self.patient}/entries',json={'kind':'allergy','label':'Synthetic allergy','reaction':'Fixture reaction'}).raise_for_status()
        self.receiver=self.user_id('receiver');self.login('patient')

    def user_id(self,name):
        with self.engine.connect() as c:return c.execute(text('SELECT id FROM dev_users WHERE username=:u'),{'u':name}).scalar()
    def login(self,name='patient',tenant='one'):
        r=self.client.post('/api/dev/login',json={'tenant':tenant,'username':name,'password':self.password});self.assertEqual(r.status_code,200,r.text)
    def intake(self,kind='pre_visit'):
        r=self.client.post(f'/api/dev/care/patients/{self.patient}/intakes',json={'kind':kind,'symptoms':'Synthetic reported concern'});self.assertEqual(r.status_code,200,r.text);return r.json()['id']
    def share(self,identifier,ai=False):
        r=self.client.post(f'/api/dev/care/intakes/{identifier}/authorize',json={'recipient_id':self.receiver,'purpose':'pre_visit_review','expires_in_minutes':60,'allow_ai':ai});self.assertEqual(r.status_code,200,r.text);return r

    def test_complete_patient_receiver_summary_and_patient_visibility(self):
        r=self.client.post(f'/api/dev/care/patients/{self.patient}/reports',json={'history':'Synthetic patient history','symptoms':'Synthetic current concern','contacts':[{'name':'Synthetic contact','relationship':'family','phone':'TEST-000'}]});self.assertEqual(r.status_code,200)
        identifier=self.intake();path=f'/api/dev/care/intakes/{identifier}'
        self.login('receiver');self.assertEqual(self.client.get(path).status_code,404)
        self.login();self.share(identifier);self.login('receiver')
        self.assertEqual(self.client.get(f'/api/dev/patients/{self.patient}').status_code,404)  # Snapshot consent never grants the underlying chart.
        r=self.client.get(path);self.assertEqual(r.status_code,200);snapshot=r.json()['snapshot'];self.assertEqual(snapshot['patient_report']['history'],'Synthetic patient history');self.assertEqual({e['kind'] for e in snapshot['chart_entries']},{'medication','allergy'});self.assertEqual(r.json()['transport_status'],'not_transmitted_externally')
        self.client.post(path+'/summary',json={'expected_version':0,'content':'Synthetic draft summary','followup':'Synthetic followup'}).raise_for_status()
        self.login();self.assertEqual(self.client.get(path).json()['summaries'],[])
        self.login('receiver');self.client.post(path+'/summary',json={'expected_version':1,'content':'Clinician reviewed synthetic summary','followup':'Reviewed synthetic followup'}).raise_for_status()
        self.assertEqual(self.client.post(path+'/approve',json={'expected_version':1}).status_code,409)
        self.client.post(path+'/approve',json={'expected_version':2}).raise_for_status();self.assertEqual(self.client.post(path+'/approve',json={'expected_version':2}).status_code,409)
        self.login();self.assertEqual(self.client.get(path).json()['summaries'][0]['content'],'Clinician reviewed synthetic summary');self.assertIn('Reviewed synthetic followup',self.client.get(f'/api/dev/patients/{self.patient}/notes').json()[0]['content']);self.assertTrue(self.client.get(path+'/audit').json())

    def test_consent_revocation_and_expiration_deny_recipient_without_deleting_patient_copy(self):
        identifier=self.intake();path=f'/api/dev/care/intakes/{identifier}';self.share(identifier);self.login('receiver');self.assertEqual(self.client.get(path).status_code,200)
        with self.engine.begin() as c:c.execute(text("UPDATE dev_care_intakes SET expires_at='2000-01-01T00:00:00+00:00' WHERE id=:id"),{'id':identifier})
        self.assertEqual(self.client.get(path).status_code,404);self.assertEqual(self.client.get('/api/dev/care/intakes').json(),[])
        self.login();self.assertEqual(self.client.get(path).status_code,200);self.client.post(path+'/revoke',json={}).raise_for_status();self.login('receiver');self.assertEqual(self.client.get(path).status_code,404);self.assertEqual(self.client.post(path+'/summary',json={'expected_version':0,'content':'Synthetic','followup':'Synthetic'}).status_code,404)

    def test_other_patient_clinician_tenant_and_admin_cannot_access_shared_snapshot(self):
        identifier=self.intake();self.share(identifier)
        for name,tenant in [('other','one'),('creator','one'),('otherpatient','one'),('foreign','two'),('admin','one')]:
            self.login(name,tenant);self.assertEqual(self.client.get(f'/api/dev/care/intakes/{identifier}').status_code,404)
        self.login('otherpatient');self.assertEqual(self.client.get(f'/api/dev/care/patients/{self.patient}/emergency').status_code,404)
        self.login('receiver');self.assertEqual(self.client.get(f'/api/dev/care/patients/{self.patient}/emergency').status_code,403)

    def test_sharing_recipient_purpose_and_synthetic_states_are_validated(self):
        identifier=self.intake()
        for recipient,purpose in [(self.user_id('foreign'),'pre_visit_review'),(self.user_id('admin'),'pre_visit_review'),(self.receiver,'urgent_care_review')]:
            r=self.client.post(f'/api/dev/care/intakes/{identifier}/authorize',json={'recipient_id':recipient,'purpose':purpose,'expires_in_minutes':60});self.assertIn(r.status_code,(400,422))
        self.share(identifier);self.assertEqual(self.client.post(f'/api/dev/care/intakes/{identifier}/authorize',json={'recipient_id':self.receiver,'purpose':'pre_visit_review','expires_in_minutes':60}).status_code,409)

    def test_immutable_snapshot_preserves_provenance_and_emergency_view_is_not_delivery(self):
        identifier=self.intake();before=self.client.get(f'/api/dev/care/intakes/{identifier}').json()['snapshot']
        self.client.post(f'/api/dev/care/patients/{self.patient}/reports',json={'history':'New synthetic history'}).raise_for_status()
        after=self.client.get(f'/api/dev/care/intakes/{identifier}').json()['snapshot'];self.assertEqual(before,after)
        packet=self.client.get(f'/api/dev/care/patients/{self.patient}/emergency').json();self.assertEqual(packet['patient_report']['history'],'New synthetic history');self.assertTrue(packet['chart_entries'][0]['provenance']);self.assertIn('Not a complete',packet['limitations'])
        init_care_schema(self.engine);self.assertEqual(self.client.get(f'/api/dev/care/intakes/{identifier}').json()['snapshot'],before)

    def test_ai_intake_scope_requires_separate_consent_and_stops_after_revoke(self):
        identifier=self.intake();self.share(identifier,ai=False);self.login('receiver')
        self.assertEqual(self.client.post('/api/dev/intelligence/threads',json={'title':'Synthetic','intake_id':identifier}).status_code,404)
        self.login();second=self.intake();self.share(second,ai=True);self.login('receiver')
        self.assertEqual(self.client.post('/api/dev/intelligence/threads',json={'title':'Synthetic','intake_id':second,'patient_id':self.patient}).status_code,422)
        thread=self.client.post('/api/dev/intelligence/threads',json={'title':'Consented synthetic intake','intake_id':second}).json()['id']
        with patch('dev_knowledge.search_evidence',return_value=[{'chunk_id':'fixture','title':'Synthetic nonclinical fixture'}]):
            r=self.client.post(f'/api/dev/intelligence/threads/{thread}/stream',json={'content':'Synthetic question'});self.assertIn('event: complete',r.text)
        self.assertIn('consented_snapshot_only',self.prompts[0][0]['content']);self.assertIn('Synthetic reported concern',self.prompts[0][0]['content'])
        self.login();self.client.post(f'/api/dev/care/intakes/{second}/revoke',json={}).raise_for_status();self.login('receiver')
        self.assertEqual(self.client.get(f'/api/dev/intelligence/threads/{thread}').status_code,404);self.assertEqual(self.client.get('/api/dev/intelligence/threads').json(),[])

    def test_revocation_during_stream_prevents_later_tokens(self):
        identifier=self.intake();self.share(identifier,ai=True);self.login('receiver')
        thread=self.client.post('/api/dev/intelligence/threads',json={'title':'Synthetic revocable stream','intake_id':identifier}).json()['id']
        patient=TestClient(self.app);self.addCleanup(patient.close)
        patient.post('/api/dev/login',json={'tenant':'one','username':'patient','password':self.password}).raise_for_status()
        def tokens(messages,model):
            yield 'Previously authorized draft'
            patient.post(f'/api/dev/care/intakes/{identifier}/revoke',json={}).raise_for_status()
            yield 'MUST-NOT-APPEAR-AFTER-REVOCATION'
        self.tokens=tokens
        with patch('dev_knowledge.search_evidence',return_value=[{'chunk_id':'fixture','title':'Synthetic fixture'}]):
            r=self.client.post(f'/api/dev/intelligence/threads/{thread}/stream',json={'content':'Synthetic'})
        self.assertIn('event: cancelled',r.text);self.assertNotIn('MUST-NOT-APPEAR',r.text)
        with self.engine.connect() as c:self.assertEqual(c.execute(text("SELECT status FROM dev_ai_messages WHERE thread_id=:id AND role='assistant'"),{'id':thread}).scalar(),'cancelled')

    def completed_intake_response(self,identifier):
        self.login('receiver')
        thread=self.client.post('/api/dev/intelligence/threads',json={'title':'Synthetic handoff','intake_id':identifier}).json()['id']
        with patch('dev_knowledge.search_evidence',return_value=[{'chunk_id':'fixture','title':'Nonclinical synthetic fixture','text':'Test-owned fixture content','url':'https://example.org/test'}]):
            self.assertIn('event: complete',self.client.post(f'/api/dev/intelligence/threads/{thread}/stream',json={'content':'Synthetic fixture question'}).text)
        response=self.client.get(f'/api/dev/intelligence/threads/{thread}').json()['messages'][-1]
        return thread,response

    def test_ai_summary_handoff_preserves_source_and_requires_review_and_approval(self):
        identifier=self.intake();self.share(identifier,ai=True);thread,response=self.completed_intake_response(identifier)
        self.assertEqual(response['evidence'][0]['chunk_id'],'fixture')
        path=f'/api/dev/care/intakes/{identifier}'
        body={'expected_version':0,'content':response['content'],'followup':'Clinician entered synthetic followup','source_message_id':response['id']}
        self.client.post(path+'/summary',json=body).raise_for_status()
        draft=self.client.get(path).json()['summaries'][0]
        self.assertEqual(draft['status'],'draft');self.assertEqual(draft['ai_provenance']['source_thread_id'],thread);self.assertFalse(draft['ai_provenance']['clinician_edited'])
        self.login();self.assertEqual(self.client.get(path).json()['summaries'],[])
        self.login('receiver')
        self.client.post(path+'/summary',json={'expected_version':1,'content':'Clinician edited synthetic output','followup':'Reviewed followup'}).raise_for_status()
        provenance=self.client.get(path).json()['summaries'][-1]['ai_provenance'];self.assertTrue(provenance['clinician_edited']);self.assertEqual(provenance['source_message_id'],response['id'])
        self.assertEqual(self.client.post(path+'/approve',json={'expected_version':1}).status_code,409)
        self.client.post(path+'/approve',json={'expected_version':2}).raise_for_status()
        self.login();summary=self.client.get(path).json()['summaries'][0];self.assertEqual(summary['content'],'Clinician edited synthetic output');self.assertEqual(summary['ai_provenance']['evidence'][0]['chunk_id'],'fixture')
        self.assertIn('link_ai_draft_to_care_summary',[e['event'] for e in self.client.get(path+'/audit').json()])
        init_care_schema(self.engine);init_intelligence_schema(self.engine)
        self.assertEqual(self.client.get(path).json()['summaries'][0]['ai_provenance']['generation_id'],response['generation_id'])
        # Reopen the file with a new engine/application, as on process restart.
        from dev_runtime import migrate
        restarted_engine=create_engine(self.engine.url);self.addCleanup(restarted_engine.dispose);migrate(restarted_engine)
        restarted_app=FastAPI();restarted_ctx=install_dev_portal(restarted_app,restarted_engine);install_care(restarted_app,restarted_ctx)
        with TestClient(restarted_app) as restarted:
            restarted.post('/api/dev/login',json={'tenant':'one','username':'patient','password':self.password}).raise_for_status()
            self.assertEqual(restarted.get(path).json()['summaries'][0]['ai_provenance']['generation_id'],response['generation_id'])
            self.assertIn('Reviewed followup',restarted.get(f'/api/dev/patients/{self.patient}/notes').json()[0]['content'])


    def test_handoff_denies_wrong_intake_owner_status_and_absent_evidence(self):
        identifier=self.intake();self.share(identifier,ai=True);thread,response=self.completed_intake_response(identifier)
        path=f'/api/dev/care/intakes/{identifier}'
        body={'expected_version':0,'content':'Edited synthetic draft','followup':'Synthetic followup','source_message_id':response['id']}
        self.login();other=self.intake();self.share(other,ai=True);self.login('receiver')
        self.assertEqual(self.client.post(f'/api/dev/care/intakes/{other}/summary',json=body).status_code,404)
        self.login('other');self.assertEqual(self.client.post(path+'/summary',json=body).status_code,404)
        self.login('receiver')
        with self.engine.begin() as c:c.execute(text("UPDATE dev_ai_messages SET status='failed' WHERE id=:id"),{'id':response['id']})
        self.assertEqual(self.client.post(path+'/summary',json=body).status_code,404)
        with self.engine.begin() as c:
            c.execute(text("UPDATE dev_ai_messages SET status='complete' WHERE id=:id"),{'id':response['id']})
            c.execute(text("DELETE FROM dev_ai_generation_evidence WHERE generation_id=:g"),{'g':response['generation_id']})
        self.assertEqual(self.client.post(path+'/summary',json=body).status_code,409)
        self.assertEqual(self.client.get(path).json()['summaries'],[])

    def test_handoff_rechecks_ai_consent_and_expiry(self):
        identifier=self.intake();self.share(identifier,ai=True);thread,response=self.completed_intake_response(identifier)
        path=f'/api/dev/care/intakes/{identifier}'
        body={'expected_version':0,'content':'Synthetic','followup':'Synthetic','source_message_id':response['id']}
        with self.engine.begin() as c:c.execute(text('UPDATE dev_care_intakes SET allow_ai=0 WHERE id=:id'),{'id':identifier})
        self.assertEqual(self.client.post(path+'/summary',json=body).status_code,404)
        with self.engine.begin() as c:c.execute(text("UPDATE dev_care_intakes SET allow_ai=1,expires_at='2000-01-01T00:00:00+00:00' WHERE id=:id"),{'id':identifier})
        self.assertEqual(self.client.post(path+'/summary',json=body).status_code,404)

    def test_revocation_during_retrieval_prevents_initial_provider_transmission(self):
        identifier=self.intake();self.share(identifier,ai=True);self.login('receiver')
        thread=self.client.post('/api/dev/intelligence/threads',json={'title':'Synthetic revoke before inference','intake_id':identifier}).json()['id']
        def retrieved(*args):
            with self.engine.begin() as c:c.execute(text("UPDATE dev_care_intakes SET state='REVOKED' WHERE id=:id"),{'id':identifier})
            return [{'chunk_id':'fixture','title':'Synthetic fixture'}]
        with patch('dev_knowledge.search_evidence',side_effect=retrieved):
            result=self.client.post(f'/api/dev/intelligence/threads/{thread}/stream',json={'content':'Synthetic question'})
        self.assertIn('event: error',result.text);self.assertEqual(self.prompts,[])
