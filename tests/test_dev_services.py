"""Connectivity diagnostics inspect names only and redact provider errors."""
import json
import os
import unittest
from unittest.mock import patch
from dev_services import check_services,failure

class ServiceDiagnosticTests(unittest.TestCase):
 def test_missing_bindings_are_external_blockers_not_authentication_failures(self):
  with patch.dict(os.environ,{},clear=True):result=check_services(True)
  self.assertEqual(result['openai']['status'],'blocked_missing_runtime_binding');self.assertEqual(result['qdrant']['status'],'blocked_missing_runtime_binding')
 def test_default_has_no_paid_or_network_requests_and_never_prints_keys(self):
  with patch.dict(os.environ,{'LLM_API_KEY':'SYNTHETIC-SECRET','QDRANT_API_KEY':'OTHER-SYNTHETIC-SECRET','LLM_BASE_URL':'https://example.invalid/v1','QDRANT_URL':'https://cluster.invalid'}):result=check_services(False)
  output=json.dumps(result);self.assertNotIn('SYNTHETIC-SECRET',output);self.assertNotIn('cluster.invalid',output);self.assertEqual(result['openai']['status'],'not_run')
 def test_error_details_cannot_reflect_secret_or_endpoint(self):
  self.assertNotIn('SECRET-TEST',json.dumps(failure(RuntimeError('SECRET-TEST endpoint'))))

 def test_qdrant_key_required_before_any_cloud_request(self):
  with patch.dict(os.environ,{'QDRANT_URL':'https://cluster.invalid'},clear=True), patch('qdrant_client.QdrantClient') as client:
   result=check_services(True)
  client.assert_not_called();self.assertEqual(result['qdrant']['status'],'blocked_missing_runtime_binding')

 def test_https_and_url_credentials_are_checked_before_sdk_calls(self):
  for url in ('http://example.invalid','https://user:password@example.invalid','not-a-url'):
   with self.subTest(url=url),patch.dict(os.environ,{'LLM_BASE_URL':url,'LLM_API_KEY':'synthetic','LLM_MODEL':'synthetic','QDRANT_URL':url,'QDRANT_API_KEY':'synthetic'},clear=True),patch('openai.OpenAI') as llm,patch('qdrant_client.QdrantClient') as qdrant:
    result=check_services(True)
   llm.assert_not_called();qdrant.assert_not_called()
   self.assertEqual(result['openai']['status'],'blocked_unsafe_provider_url');self.assertEqual(result['qdrant']['status'],'blocked_unsafe_provider_url')

 def test_empty_inference_and_stream_cannot_report_success(self):
  from types import SimpleNamespace as NS
  with patch.dict(os.environ,{'LLM_BASE_URL':'https://example.invalid/v1','LLM_API_KEY':'synthetic','LLM_MODEL':'synthetic'},clear=True),patch('openai.OpenAI') as cls:
   client=cls.return_value.__enter__.return_value
   from unittest.mock import MagicMock
   stream=MagicMock();stream.__enter__.return_value=iter([])
   client.chat.completions.create.side_effect=[NS(choices=[NS(message=NS(content=''))],model='synthetic'),stream]
   result=check_services(True)
   self.assertEqual(cls.call_args.kwargs['max_retries'],0);self.assertEqual(cls.call_args.kwargs['timeout'],25)
  self.assertEqual(result['openai']['status'],'failed_empty_response')

 def test_provider_errors_are_redacted_and_not_retried(self):
  with patch.dict(os.environ,{'LLM_BASE_URL':'https://example.invalid/v1','LLM_API_KEY':'synthetic','LLM_MODEL':'synthetic'},clear=True),patch('openai.OpenAI') as cls:
   create=cls.return_value.__enter__.return_value.chat.completions.create
   create.side_effect=RuntimeError('SECRET-MUST-NOT-APPEAR')
   result=check_services(True)
  self.assertEqual(create.call_count,1);self.assertNotIn('SECRET-MUST-NOT-APPEAR',json.dumps(result));self.assertEqual(result['openai']['status'],'failed')

 def test_remote_vector_schema_must_match_current_unnamed_cosine_index(self):
  from types import SimpleNamespace as NS
  from qdrant_client.http import models as qm
  for vectors,compatible in ((qm.VectorParams(size=384,distance=qm.Distance.COSINE),True),(qm.VectorParams(size=1536,distance=qm.Distance.COSINE),False),(qm.VectorParams(size=384,distance=qm.Distance.DOT),False),({'named':qm.VectorParams(size=384,distance=qm.Distance.COSINE)},False)):
   with self.subTest(vectors=vectors),patch.dict(os.environ,{'QDRANT_URL':'https://cluster.invalid','QDRANT_API_KEY':'synthetic'},clear=True),patch('qdrant_client.QdrantClient') as cls:
    client=cls.return_value;client.get_collections.return_value=NS(collections=[NS(name='raphael_evidence_v1')]);client.get_collection.return_value=NS(config=NS(params=NS(vectors=vectors)));client.count.return_value=NS(count=2)
    result=check_services(True)['qdrant']
   self.assertEqual(result['schema_compatible'],compatible);self.assertFalse(result['embedding_model_verified']);self.assertEqual(result['status'],'passed' if compatible else 'blocked_incompatible_vectors')
   client.create_collection.assert_not_called();client.upsert.assert_not_called();client.close.assert_called_once()
