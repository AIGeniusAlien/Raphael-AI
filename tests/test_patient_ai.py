"""Patient education isolation/consent/provenance with labeled nonclinical fixtures."""
import os
import tempfile
import unittest
from unittest.mock import patch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine,text
from dev_portal import init_portal_schema,install_dev_portal,create_dev_user
from dev_chart import init_chart_schema
from dev_care import init_care_schema
from dev_knowledge import init_knowledge_schema
from dev_patient_ai import init_patient_ai_schema,install_patient_ai,export_patient_ai,population_for


class PatientAITests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='raphael-patient-ai-');self.addCleanup(self.tmp.cleanup)
        self.engine=create_engine('sqlite:///'+self.tmp.name+'/synthetic.db');self.addCleanup(self.engine.dispose)
        flags=patch.dict(os.environ,{'RAPHAEL_DEV_SYNTHETIC_ONLY':'1','RAPHAEL_VERIFY_CLAIMS':'0'});flags.start();self.addCleanup(flags.stop)
        for init in (init_portal_schema,init_chart_schema,init_care_schema,init_knowledge_schema,init_patient_ai_schema):init(self.engine)
        self.password='Synthetic-test-only-patient-ai'
        for tenant,name,role in [('one','doctor','clinician'),('one','patient','patient'),('one','other','patient'),('two','foreign','patient')]:create_dev_user(self.engine,tenant,name,self.password,role)
        self.app=FastAPI();self.ctx=install_dev_portal(self.app,self.engine)
        self.seen=[];self.evidence=[{'chunk_id':'fixture-immutable','title':'Nonclinical fixture','text':'Test-owned synthetic educational fixture','url':'https://example.org/test','source_date':'2024-01-01','content_sha256':'PROVENANCE-HASH-MUST-NOT-ENTER-MODEL'}]
        def tokens(messages,model):self.seen.append(messages);yield 'Synthetic educational explanation [E1]'
        self.tokens=tokens;self.retrieval=lambda user,question:self.evidence
        install_patient_ai(self.app,self.ctx,token_provider=lambda m,v:self.tokens(m,v),evidence_provider=lambda u,q:self.retrieval(u,q))
        self.client=TestClient(self.app);self.addCleanup(self.client.close);self.login('doctor')
        self.patient=self.client.post('/api/dev/patients',json={'name':'Synthetic patient','dob':'2000-01-01','patient_username':'patient'}).json()['id']
        self.other=self.client.post('/api/dev/patients',json={'name':'Synthetic other','patient_username':'other'}).json()['id']
        self.login()
    def login(self,name='patient',tenant='one'):
        response=self.client.post('/api/dev/login',json={'tenant':tenant,'username':name,'password':self.password});self.assertEqual(response.status_code,200,response.text)
    def thread(self,context=False):
        r=self.client.post('/api/dev/patient-ai/threads',json={'patient_id':self.patient,'title':'Synthetic education','consent_ai':True,'include_approved_context':context});self.assertEqual(r.status_code,200,r.text);return r.json()['id']
    def ask(self,identifier,question='Synthetic general educational question'):
        return self.client.post(f'/api/dev/patient-ai/threads/{identifier}/stream',json={'content':question})

    def test_explicit_consent_patient_ownership_and_role_boundaries(self):
        body={'patient_id':self.patient,'title':'Synthetic','consent_ai':False}
        self.assertEqual(self.client.post('/api/dev/patient-ai/threads',json=body).status_code,422)
        body.update(consent_ai=True,patient_id=self.other);self.assertEqual(self.client.post('/api/dev/patient-ai/threads',json=body).status_code,404)
        identifier=self.thread()
        for name,tenant,status in [('other','one',404),('foreign','two',404),('doctor','one',403)]:
            self.login(name,tenant);self.assertEqual(self.client.get(f'/api/dev/patient-ai/threads/{identifier}').status_code,status);self.assertEqual(self.ask(identifier).status_code,status)

    def test_approved_context_and_evidence_persist_without_private_drafts(self):
        with self.engine.begin() as c:
            doctor=c.execute(text("SELECT id FROM dev_users WHERE username='doctor'")).scalar();patient=c.execute(text("SELECT id FROM dev_users WHERE username='patient'")).scalar()
            for status,content in [('draft','PRIVATE-DRAFT-MUST-NOT-LEAK'),('approved','Approved synthetic instruction')]:c.execute(text('INSERT INTO dev_notes(patient_id,author_id,content,status,created_at) VALUES (:p,:u,:v,:s,:at)'),{'p':self.patient,'u':doctor,'v':content,'s':status,'at':'2024-01-01'})
            c.execute(text("INSERT INTO dev_patient_reports(patient_id,author_id,history,symptoms,contacts,created_at) VALUES (:p,:u,'Self-reported fixture','Reported symptom','[]','2024-01-01')"),{'p':self.patient,'u':patient})
        identifier=self.thread(True);result=self.ask(identifier);self.assertIn('event: canonical',result.text);self.assertIn('event: complete',result.text)
        self.assertIn('Approved synthetic instruction',self.seen[0][0]['content']);self.assertIn('Self-reported fixture',self.seen[0][0]['content']);self.assertNotIn('PRIVATE-DRAFT',str(self.seen));self.assertNotIn('PROVENANCE-HASH-MUST-NOT-ENTER-MODEL',str(self.seen))
        record=self.client.get(f'/api/dev/patient-ai/threads/{identifier}').json();self.assertEqual(record['messages'][-1]['content'],'Synthetic educational explanation [fixture-immutable]');self.assertEqual(record['messages'][-1]['status'],'complete')
        self.assertEqual(self.client.get(f'/api/dev/patient-ai/threads/{identifier}/evidence/fixture-immutable').status_code,200)
        self.assertEqual(self.client.get(f'/api/dev/patient-ai/threads/{identifier}/evidence/unknown').status_code,404)
        init_patient_ai_schema(self.engine);self.assertEqual(len(self.client.get(f'/api/dev/patient-ai/threads/{identifier}').json()['messages']),2)
        with self.engine.connect() as c:
            user=dict(c.execute(text("SELECT * FROM dev_users WHERE username='patient'")).mappings().first());export=export_patient_ai(c,user)
        self.assertEqual(len(export['threads']),1);self.assertNotIn('password',str(export));self.assertEqual(len(export['threads'][0]['messages']),2)

    def test_context_is_excluded_unless_selected(self):
        identifier=self.thread(False);self.ask(identifier);self.assertIn('general_education_no_chart_context',str(self.seen));self.assertNotIn('approved_notes',str(self.seen))

    def test_unknown_citation_and_provider_error_fail_closed_without_secret_echo(self):
        identifier=self.thread();self.tokens=lambda m,v:iter(['Unsupported [E999]'])
        response=self.ask(identifier);self.assertIn('event: error',response.text);self.assertNotIn('event: complete',response.text)
        def broken(m,v):raise RuntimeError('SECRET-FIXTURE-MUST-NOT-LEAK')
        self.tokens=broken;response=self.ask(identifier);self.assertNotIn('SECRET-FIXTURE',response.text);self.assertIn('event: error',response.text)
        statuses=[m['status'] for m in self.client.get(f'/api/dev/patient-ai/threads/{identifier}').json()['messages'] if m['role']=='assistant'];self.assertEqual(statuses,['failed','failed'])

    def test_missing_and_failed_retrieval_do_not_call_model(self):
        identifier=self.thread();self.evidence=[];response=self.ask(identifier);self.assertIn('event: abstained',response.text);self.assertEqual(self.seen,[])
        def broken(u,q):raise RuntimeError('SECRET-EVIDENCE-FIXTURE')
        self.retrieval=broken;response=self.ask(identifier);self.assertEqual(response.status_code,503);self.assertNotIn('SECRET-EVIDENCE',response.text);self.assertEqual(self.seen,[])

    def test_phrase_notices_are_not_model_inference_or_validated_triage(self):
        identifier=self.thread()
        for question in ('I have chest pain now','What dose should I use?'):
            response=self.ask(identifier,question);self.assertIn('event: safety_notice',response.text);self.assertNotIn('event: complete',response.text)
        self.assertEqual(self.seen,[])
        self.assertTrue(all(m['origin']=='safety_rule' and m['model'] is None for m in self.client.get(f'/api/dev/patient-ai/threads/{identifier}').json()['messages'] if m['role']=='assistant'))

    def test_revoke_before_and_during_generation_denies_future_transmission(self):
        identifier=self.thread()
        def retrieve(user,question):
            self.client.post(f'/api/dev/patient-ai/threads/{identifier}/revoke',json={}).raise_for_status();return self.evidence
        self.retrieval=retrieve;response=self.ask(identifier);self.assertIn('event: cancelled',response.text);self.assertEqual(self.seen,[])
        self.assertEqual(self.ask(identifier).status_code,409);self.assertEqual(self.client.get(f'/api/dev/patient-ai/threads/{identifier}').status_code,200)
        self.retrieval=lambda u,q:self.evidence;identifier=self.thread()
        def tokens(m,v):
            yield 'Previously authorized'
            self.client.post(f'/api/dev/patient-ai/threads/{identifier}/revoke',json={}).raise_for_status()
            yield 'MUST-NOT-APPEAR'
        self.tokens=tokens;response=self.ask(identifier);self.assertIn('event: cancelled',response.text);self.assertNotIn('MUST-NOT-APPEAR',response.text)

    def test_population_comes_from_valid_authorized_dob(self):
        self.assertEqual(population_for('2000-01-01'),'adult');self.assertEqual(population_for('2020-01-01'),'pediatric');self.assertEqual(population_for('not-a-date'),'unknown');self.assertEqual(population_for('2999-01-01'),'unknown')

    def test_logout_during_retrieval_revokes_cached_stream_session(self):
        identifier=self.thread()
        def retrieve(user,question):self.client.post('/api/dev/logout',json={}).raise_for_status();return self.evidence
        self.retrieval=retrieve;response=self.ask(identifier)
        self.assertIn('event: cancelled',response.text);self.assertEqual(self.seen,[])

    def test_claim_support_screen_blocks_or_fails_without_success(self):
        identifier=self.thread()
        with patch.dict(os.environ,{'RAPHAEL_VERIFY_CLAIMS':'1'}),patch('rag.claim_support.screen_claims',return_value={'status':'blocked','checks':[{'support':'unsupported'}]}):
            response=self.ask(identifier)
        self.assertIn('event: safety_blocked',response.text);self.assertNotIn('event: complete',response.text)
        latest=self.client.get(f'/api/dev/patient-ai/threads/{identifier}').json()['messages'][-1]
        self.assertEqual(latest['status'],'blocked');self.assertEqual(latest['claim_screen']['checks'][0]['support'],'unsupported')
        with patch.dict(os.environ,{'RAPHAEL_VERIFY_CLAIMS':'1'}),patch('rag.claim_support.screen_claims',side_effect=RuntimeError('SECRET-VERIFIER-FIXTURE')):
            response=self.ask(identifier)
        self.assertIn('event: error',response.text);self.assertNotIn('SECRET-VERIFIER',response.text);self.assertNotIn('event: complete',response.text)

    def test_revoke_between_tokens_and_claim_screen_prevents_checker_request(self):
        identifier=self.thread()
        def tokens(m,v):
            yield 'Synthetic explanation [E1]'
            self.client.post(f'/api/dev/patient-ai/threads/{identifier}/revoke',json={}).raise_for_status()
        self.tokens=tokens
        with patch.dict(os.environ,{'RAPHAEL_VERIFY_CLAIMS':'1'}),patch('rag.claim_support.screen_claims') as checker:
            response=self.ask(identifier)
        checker.assert_not_called();self.assertIn('event: cancelled',response.text);self.assertNotIn('event: complete',response.text)

    def test_revoke_during_claim_screen_prevents_completion(self):
        identifier=self.thread()
        def verify(*args,**kwargs):
            self.client.post(f'/api/dev/patient-ai/threads/{identifier}/revoke',json={}).raise_for_status()
            return {'status':'passed','checks':[]}
        with patch.dict(os.environ,{'RAPHAEL_VERIFY_CLAIMS':'1'}),patch('rag.claim_support.screen_claims',side_effect=verify):
            response=self.ask(identifier)
        self.assertIn('event: cancelled',response.text);self.assertNotIn('event: complete',response.text)
        self.assertEqual(self.client.get(f'/api/dev/patient-ai/threads/{identifier}').json()['messages'][-1]['status'],'cancelled')
