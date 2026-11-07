"""Actual local vocabulary/FTS provenance tests;nonclinical synthetic text only."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine,text
from dev_knowledge import init_knowledge_schema,install_knowledge
from dev_portal import init_portal_schema,create_dev_user,install_dev_portal

class MedicalTermsTests(unittest.TestCase):
    def setUp(self):
        directory=tempfile.TemporaryDirectory();self.addCleanup(directory.cleanup)
        flag=patch.dict(os.environ,{'RAPHAEL_DEV_SYNTHETIC_ONLY':'1'});flag.start();self.addCleanup(flag.stop)
        self.engine=create_engine('sqlite:///'+str(Path(directory.name)/'synthetic.db'));self.addCleanup(self.engine.dispose)
        init_portal_schema(self.engine);init_knowledge_schema(self.engine)
        self.password='Synthetic-only-terms-password'
        for tenant,name,role in [('a','doctor','clinician'),('a','patient','patient'),('a','admin','org_admin'),('b','outsider','clinician')]:create_dev_user(self.engine,tenant,name,self.password,role)
        app=FastAPI();ctx=install_dev_portal(app,self.engine);install_knowledge(app,ctx)
        self.client=TestClient(app);self.addCleanup(self.client.close);self.login()
        self.source=self.new_source()
        self.client.post(f"/api/dev/knowledge/sources/{self.source['id']}/approve",json={}).raise_for_status()
    def login(self,name='doctor',tenant='a'):
        self.client.post('/api/dev/login',json={'tenant':tenant,'username':name,'password':self.password}).raise_for_status()
    def new_source(self,**changes):
        data={'title':'Nonclinical vocabulary fixture','url':'https://example.org/local-vocabulary','content':'Violet calibrator and amber baseline are nonclinical synthetic calibration fixtures. No medical guidance.','license_declaration':'Test-owned nonclinical synthetic fixture.','permission_attested':True,'org':'Synthetic fixture','tier':'D'};data.update(changes)
        r=self.client.post('/api/dev/knowledge/sources',json=data);r.raise_for_status();return r.json()
    def term(self,**changes):
        data={'source_id':self.source['id'],'source_version_sha256':self.source['version_sha256'],'canonical':'violet calibrator','aliases':['violetwand'],'specialty':'Synthetic calibration','kind':'test'};data.update(changes)
        return self.client.post('/api/dev/knowledge/terms',json=data)
    def review(self,term,target='review',**changes):
        data={'content_sha256':term['content_sha256'],'source_version_sha256':term['source_version_sha256'],'content_reviewed':True};data.update(changes)
        return self.client.post(f"/api/dev/knowledge/terms/{term['id']}/{target}",json=data)
    def search(self,query='violetwand',use_terms=True):return self.client.post('/api/dev/knowledge/search',json={'query':query,'use_terms':use_terms}).json()
    def test_pending_review_search_citation_and_idempotence(self):
        first=self.term();self.assertEqual(first.status_code,200,first.text);term=first.json()
        self.assertTrue(self.term().json()['deduplicated']);self.assertFalse(self.search()['results'])
        self.assertEqual(self.review(term).status_code,200);self.assertFalse(self.search(use_terms=False)['results'])
        result=self.search();self.assertEqual(result['normalization']['status'],'expanded');self.assertEqual(len(result['results']),1)
        self.assertEqual(result['normalization']['mappings'][0]['source_version_sha256'],self.source['version_sha256'])
        citation=self.client.get(result['results'][0]['citation_path']);self.assertEqual(citation.status_code,200);self.assertIn('Violet calibrator',citation.json()['body'])
        self.assertFalse(self.client.get('/api/dev/knowledge/terms/coverage').json()['comprehensive_specialty_coverage'])
        init_knowledge_schema(self.engine);self.assertEqual(len(self.client.get('/api/dev/knowledge/terms').json()['terms']),1)
    def test_source_revocation_and_supersession_disable_aliases(self):
        term=self.term().json();self.review(term).raise_for_status()
        new=self.new_source(content='Violet calibrator revised fixture contains no medical instructions.')
        self.client.post(f"/api/dev/knowledge/sources/{new['id']}/approve",json={}).raise_for_status()
        self.assertEqual(self.search()['normalization']['status'],'no_reviewed_alias_match')
        self.assertEqual(self.review(term).status_code,409)
        self.source=new;term=self.term().json();self.review(term).raise_for_status()
        self.client.post(f"/api/dev/knowledge/sources/{new['id']}/revoke",json={}).raise_for_status()
        self.assertFalse(self.search()['results']);self.assertFalse(self.client.get('/api/dev/knowledge/terms/coverage').json()['facets'])
    def test_conflicting_meanings_abstain_without_expansion(self):
        for canonical in ['violet calibrator','amber baseline']:self.review(self.term(canonical=canonical).json()).raise_for_status()
        result=self.search();self.assertEqual(result['normalization']['status'],'ambiguous_no_expansion');self.assertFalse(result['results']);self.assertFalse(result['normalization']['mappings'])
    def test_permissions_tenant_isolation_and_logout(self):
        term=self.term().json()
        for name in ['patient','admin']:
            self.login(name);self.assertEqual(self.client.get('/api/dev/knowledge/terms').status_code,403);self.assertEqual(self.review(term).status_code,403)
        self.login('outsider','b');self.assertEqual(self.review(term).status_code,404);self.assertEqual(self.term().status_code,404);self.assertEqual(self.client.get('/api/dev/knowledge/terms').json()['terms'],[])
        self.client.post('/api/dev/logout',json={});self.assertEqual(self.client.get('/api/dev/knowledge/terms/coverage').status_code,401)
    def test_exact_version_review_and_immutable_correction(self):
        term=self.term().json();self.assertEqual(self.review(term,content_sha256='0'*64).status_code,409);self.assertEqual(self.review(term,content_reviewed=False).status_code,409)
        self.review(term).raise_for_status();self.review(term,'revoke').raise_for_status();self.assertFalse(self.search()['results'])
        self.assertEqual(self.review(term).status_code,409);self.assertEqual(self.review(term,'revoke').status_code,200)
        corrected=self.term(aliases=['violetgauge']).json();self.assertNotEqual(corrected['id'],term['id']);self.review(corrected).raise_for_status();self.assertTrue(self.search('violetgauge')['results'])
    def test_source_and_input_validation(self):
        self.assertEqual(self.term(canonical='invented medical concept').status_code,422)
        self.assertEqual(self.term(source_version_sha256='0'*64).status_code,409)
        self.assertEqual(self.term(aliases=['\n']).status_code,422);self.assertEqual(self.term(aliases=['a'*121]).status_code,422)
        self.assertEqual(self.term(kind='SNOMED-certified').status_code,422)
        pending=self.new_source(url='https://example.org/pending')
        self.assertEqual(self.term(source_id=pending['id'],source_version_sha256=pending['version_sha256']).status_code,409)
    def test_exact_phrases_and_query_budget(self):
        term=self.term(aliases=['violet wand']).json();self.review(term).raise_for_status()
        self.assertEqual(self.search('ultraviolet wand')['normalization']['status'],'no_reviewed_alias_match')
        self.assertEqual(self.search('violet wand')['normalization']['status'],'expanded')
        query='violet wand '+' '.join('word'+str(i) for i in range(20))
        self.assertEqual(self.search(query)['normalization']['status'],'query_budget_exceeded')
        self.assertFalse(self.search(query)['normalization']['mappings'])
