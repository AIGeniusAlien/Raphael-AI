import unittest
from fastapi import FastAPI,HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request
from dev_security import RequestBudget

class RequestSecurityTests(unittest.TestCase):
    def request(self,path='/api/dev/patients',client='127.0.0.1',headers=()):
        return Request({'type':'http','scheme':'http','path':path,'method':'POST','headers':headers,'server':('test',80),'client':(client,1),'query_string':b''})
    def test_model_budget_is_separate_bounded_and_forwarded_headers_cannot_bypass(self):
        for path in ('/api/dev/patients/1/assessment','/api/dev/intelligence/threads/1/assessment','/api/dev/conversations/1/draft-assist'):
            endpoint_budget=RequestBudget(expensive_limit=1)
            endpoint_budget.check(self.request(path))
            with self.assertRaises(HTTPException):endpoint_budget.check(self.request(path))
        budget=RequestBudget(limit=4,expensive_limit=1)
        budget.check(self.request('/api/dev/patient-ai/threads/1/stream'))
        with self.assertRaises(HTTPException) as error:budget.check(self.request('/api/dev/patient-ai/threads/2/stream',headers=[(b'x-forwarded-for',b'other')]))
        self.assertEqual(error.exception.status_code,429);self.assertIn('Retry-After',error.exception.headers)
        budget.check(self.request());budget.check(self.request());budget.check(self.request())
        with self.assertRaises(HTTPException):budget.check(self.request())
    def test_expiry_releases_capacity_without_evicting_active_clients(self):
        now=[0];budget=RequestBudget(limit=1,max_clients=1,clock=lambda:now[0])
        budget.check(self.request())
        with self.assertRaises(HTTPException):budget.check(self.request(client='other'))
        now[0]=61;budget.check(self.request(client='other'));self.assertEqual(len(budget.clients),1)
    def test_fetch_metadata_and_malformed_origin_fail_before_mutation(self):
        import tempfile,os
        from unittest.mock import patch
        from sqlalchemy import create_engine
        from dev_portal import init_portal_schema,install_dev_portal
        with tempfile.TemporaryDirectory() as directory,patch.dict(os.environ,{'RAPHAEL_DEV_SYNTHETIC_ONLY':'1'}):
            engine=create_engine('sqlite:///'+directory+'/synthetic.db')
            try:
                init_portal_schema(engine);app=FastAPI();install_dev_portal(app,engine)
                with TestClient(app) as client:
                    for headers in ({'sec-fetch-site':'cross-site'},{'origin':'https://['}):
                        response=client.post('/api/dev/login',json={'tenant':'test','username':'none','password':'not-a-real-password'},headers=headers)
                        self.assertEqual(response.status_code,403)
            finally:engine.dispose()
