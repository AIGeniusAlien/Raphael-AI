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
