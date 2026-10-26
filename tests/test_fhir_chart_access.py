"""FHIR fixture fetch must not outlive patient-controlled chart authorization."""
import os
import tempfile
import unittest
from unittest.mock import patch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine,text
from dev_portal import init_portal_schema,install_dev_portal,create_dev_user
from dev_imaging import init_imaging_schema,install_imaging


class FhirChartAccessTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='raphael-fhir-chart-access-');self.addCleanup(self.temp.cleanup)
        self.engine=create_engine('sqlite:///'+self.temp.name+'/synthetic.db');self.addCleanup(self.engine.dispose)
        flag=patch.dict(os.environ,{'RAPHAEL_DEV_SYNTHETIC_ONLY':'1'});flag.start();self.addCleanup(flag.stop)
        init_portal_schema(self.engine);init_imaging_schema(self.engine)
        self.password='Synthetic-FHIR-chart-password'
        self.doctor=create_dev_user(self.engine,'synthetic','doctor',self.password,'clinician')
        create_dev_user(self.engine,'synthetic','patient',self.password,'patient')
        self.app=FastAPI();ctx=install_dev_portal(self.app,self.engine);install_imaging(self.app,ctx)
        self.client=TestClient(self.app);self.addCleanup(self.client.close)
        self.client.post('/api/dev/login',json={'tenant':'synthetic','username':'doctor','password':self.password}).raise_for_status()
        self.patient_id=self.client.post('/api/dev/patients',json={'name':'Synthetic FHIR chart','patient_username':'patient'}).json()['id']
        self.path=f'/api/dev/patients/{self.patient_id}'
        self.patient=TestClient(self.app);self.addCleanup(self.patient.close)
        self.patient.post('/api/dev/login',json={'tenant':'synthetic','username':'patient','password':self.password}).raise_for_status()

    def test_revocation_during_synthetic_fixture_fetch_prevents_record_disclosure(self):
        calls=[]
        async def fetch(resource_id):
            calls.append(resource_id)
            self.patient.post(self.path+'/chart-access/'+str(self.doctor['id'])+'/revoke',json={}).raise_for_status()
            return {'status':'sandbox_preview','patient':{'id':resource_id,'synthetic_marker':'MUST-NOT-RETURN-AFTER-REVOCATION'},'resources':[]}
        with patch('dev_imaging.preview_patient',side_effect=fetch):
            response=self.client.get(self.path+'/fhir-preview',params={'resource_id':'synthetic-patient'})
        self.assertEqual(calls,['synthetic-patient']);self.assertEqual(response.status_code,404)
        self.assertNotIn('MUST-NOT-RETURN',response.text)
        with self.engine.connect() as c:self.assertEqual(c.execute(text("SELECT count(*) FROM dev_audit WHERE event='preview_synthetic_fhir'")).scalar(),0)

    def test_active_chart_authorization_retains_supported_fixture_preview(self):
        async def fetch(resource_id):return {'status':'sandbox_preview','patient':{'id':resource_id},'resources':[]}
        with patch('dev_imaging.preview_patient',side_effect=fetch):
            response=self.client.get(self.path+'/fhir-preview',params={'resource_id':'synthetic-patient'})
        self.assertEqual(response.status_code,200);self.assertEqual(response.json()['patient']['id'],'synthetic-patient')
        with self.engine.connect() as c:self.assertEqual(c.execute(text("SELECT count(*) FROM dev_audit WHERE event='preview_synthetic_fhir'")).scalar(),1)
