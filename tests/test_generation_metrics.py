"""Real local SSE/SQLite measurements;SDK fixtures do not prove live usage."""
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from sqlalchemy import text
from tests import test_intelligence as fixture
from generation_metrics import summary,usage_values

class GenerationMetricsTests(unittest.TestCase):
    login=fixture.IntelligenceTests.login
    thread=fixture.IntelligenceTests.thread
    def setUp(self):fixture.IntelligenceTests.setUp(self)
    def stream(self,**changes):
        identifier=self.thread();return self.client.post(f'/api/dev/intelligence/threads/{identifier}/stream',json={'content':'Synthetic fixture question',**changes})
    def test_actual_local_timings_unknown_fixture_usage_and_owner_boundaries(self):
        response=self.stream();self.assertIn('event: complete',response.text)
        result=self.client.get('/api/dev/intelligence/performance').json();self.assertEqual(result['sample_count'],1)
        self.assertEqual(result['generation_usage_reports'],0);self.assertIsNone(result['generation_tokens']);self.assertIsNone(result['estimated_cost'])
        for key in ('retrieval_ms','total_ms','first_delta_ms','provider_ms'):
            self.assertEqual(result['timings'][key]['samples'],1);self.assertGreaterEqual(result['timings'][key]['p50'],0)
        self.assertNotIn('Synthetic fixture question',json.dumps(result));self.assertNotIn('generation_id',json.dumps(result))
        self.login('second');self.assertEqual(self.client.get('/api/dev/intelligence/performance').json()['sample_count'],0)
        self.login('patient');self.assertEqual(self.client.get('/api/dev/intelligence/performance').status_code,403)
    def test_provider_failure_records_observation_without_fabricated_usage(self):
        def broken(messages,model):raise RuntimeError('Synthetic provider failure');yield ''
        self.provider=broken;self.assertIn('event: error',self.stream().text)
        result=self.client.get('/api/dev/intelligence/performance').json();self.assertEqual(result['sample_count'],1)
        self.assertEqual(result['timings']['first_delta_ms']['samples'],0);self.assertEqual(result['generation_usage_reports'],0)
    def test_provider_usage_validation_and_nearest_rank_small_samples(self):
        valid=SimpleNamespace(prompt_tokens=7,completion_tokens=3,total_tokens=10)
        self.assertEqual(usage_values(valid)['total_tokens'],10)
        for values in [(True,3,4),(7,3,11),(-1,3,2),(7,None,7),(7,3,10000001)]:
            self.assertEqual(usage_values(SimpleNamespace(**dict(zip(('prompt_tokens','completion_tokens','total_tokens'),values)))),{})
        rows=[{'total_ms':n,'total_tokens':None} for n in range(1,21)]
        result=summary(rows);self.assertEqual(result['timings']['total_ms'],{'samples':20,'p50':10.,'p95':19.,'p99':20.});self.assertIsNone(result['generation_tokens'])
    def test_actual_sdk_stream_usage_event_is_consumed_and_close_preserved(self):
        from dev_intelligence import openai_tokens
        from unittest.mock import MagicMock
        usage=SimpleNamespace(prompt_tokens=7,completion_tokens=3,total_tokens=10)
        events=[SimpleNamespace(usage=None,choices=[SimpleNamespace(delta=SimpleNamespace(content='Synthetic chunk'))]),SimpleNamespace(usage=usage,choices=[])]
        client=MagicMock();client.__enter__.return_value=client
        stream=MagicMock();stream.__enter__.return_value=iter(events);client.chat.completions.create.return_value=stream
        captured=[]
        with patch.dict('os.environ',{'LLM_BASE_URL':'https://synthetic.invalid/v1','LLM_API_KEY':'synthetic-not-a-key'}),patch('openai.OpenAI',return_value=client):
            self.assertEqual(list(openai_tokens([], 'synthetic-model',captured.append)),['Synthetic chunk'])
        self.assertEqual(captured,[{'prompt_tokens':7,'completion_tokens':3,'total_tokens':10}])
        self.assertEqual(client.chat.completions.create.call_args.kwargs['stream_options'],{'include_usage':True});self.assertTrue(stream.__exit__.called)
    def test_usage_persistence_aggregates_reported_counts_only(self):
        from dev_intelligence import install_intelligence
        from unittest.mock import MagicMock
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        transport_app=FastAPI();install_intelligence(transport_app,self.ctx)
        transport_client=TestClient(transport_app);transport_client.cookies.update(self.client.cookies)
        self.addCleanup(transport_client.close);self.client=transport_client
        client=MagicMock();client.__enter__.return_value=client
        stream=MagicMock();stream.__enter__.return_value=iter([
            SimpleNamespace(usage=None,choices=[SimpleNamespace(delta=SimpleNamespace(content='Synthetic response [fixture-1]'))]),
            SimpleNamespace(usage=SimpleNamespace(prompt_tokens=7,completion_tokens=3,total_tokens=10),choices=[])])
        client.chat.completions.create.return_value=stream
        with patch.dict('os.environ',{'LLM_BASE_URL':'https://synthetic.invalid/v1','LLM_API_KEY':'synthetic-not-a-key'}),patch('openai.OpenAI',return_value=client):
            self.assertIn('event: complete',self.stream().text)
        result=self.client.get('/api/dev/intelligence/performance').json()
        self.assertEqual(result['generation_tokens'],10);self.assertEqual(result['generation_usage_reports'],1)
        with self.engine.connect() as c:
            self.assertEqual(tuple(c.execute(text('SELECT prompt_tokens,completion_tokens,total_tokens FROM dev_ai_generation_metrics')).one()),(7,3,10))
