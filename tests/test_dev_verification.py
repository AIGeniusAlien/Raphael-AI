import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from dev_verify import isolated_environment,execute,counts
from dev_start import runtime_environment

class VerificationTests(unittest.TestCase):
    def test_default_children_cannot_inherit_database_or_provider_bindings(self):
        source={'PATH':'preserved','DATABASE_URL':'never-touch.db','LLM_API_KEY':'secret','QDRANT_API_KEY':'secret','RAPHAEL_VERIFY_CLAIMS':'1','RAPHAEL_EMBEDDING_BACKEND':'openai'}
        safe=isolated_environment(source)
        self.assertNotIn('DATABASE_URL',safe);self.assertNotIn('LLM_API_KEY',safe);self.assertNotIn('QDRANT_API_KEY',safe);self.assertNotIn('RAPHAEL_EMBEDDING_BACKEND',safe)
        self.assertEqual(safe['RAPHAEL_VERIFY_CLAIMS'],'0');self.assertEqual(safe['PATH'],'preserved')
        with patch.dict(os.environ,source):
            env=runtime_environment('sqlite:///synthetic.db')
            self.assertEqual(env['DATABASE_URL'],'sqlite:///synthetic.db');self.assertEqual(env['RAPHAEL_DEV_PORTAL'],'1');self.assertNotIn('LLM_API_KEY',env)
    def test_failures_timeouts_and_secret_redaction_are_recorded(self):
        with tempfile.TemporaryDirectory() as directory,patch.dict(os.environ,{'LLM_API_KEY':'Synthetic-redaction-fixture'}):
            path=Path(directory)
            result=execute('failure',[sys.executable,'-c',"print('Synthetic-redaction-fixture');raise SystemExit(3)"],isolated_environment(),path)
            self.assertEqual(result['status'],'failed');self.assertEqual(result['exit_code'],3);self.assertNotIn('Synthetic-redaction-fixture',Path(result['log']).read_text())
            result=execute('timeout',[sys.executable,'-c','import time;time.sleep(10)'],isolated_environment(),path,timeout=.05)
            self.assertTrue(result['timeout']);self.assertEqual(result['status'],'failed')
    def test_summary_does_not_count_skips_as_passes(self):
        self.assertEqual(counts('Ran 4 tests in 1s\nOK (skipped=2)'),{'tests':4,'skipped':2})
        self.assertEqual(counts('ℹ tests 3\nℹ pass 2\nℹ fail 0\nℹ skipped 1'),{'tests':3,'pass':2,'fail':0,'skipped':1})
    def test_ci_failure_annotation_redacts_secrets_and_escapes_untrusted_commands(self):
        import subprocess
        with tempfile.TemporaryDirectory() as directory:
            log=Path(directory)/'native.log';log.write_text('error: Synthetic-CI-secret\n::notice::untrusted injected line\n')
            env=isolated_environment();env['LLM_API_KEY']='Synthetic-CI-secret'
            result=subprocess.run([sys.executable,'.github/scripts/publish_diagnostics.py',str(log)],env=env,text=True,capture_output=True,check=True)
            self.assertNotIn('Synthetic-CI-secret',result.stdout)
            self.assertEqual(len(result.stdout.splitlines()),1)
            self.assertTrue(result.stdout.startswith('::error title=Raphael failure diagnostics::'))
            self.assertIn('[REDACTED]',result.stdout)

    def test_safety_protocol_has_explicit_unvalidated_medical_categories(self):
        from eval.safety_eval import evaluate
        result=evaluate();self.assertEqual(result['status'],'passed');self.assertFalse(result['clinical_validation']);self.assertEqual(result['live_model_calls'],0)
        self.assertTrue(all(not c['model_executed'] for c in result['medical_evaluation']));self.assertEqual(result['threshold']['actual_protocol_pass_rate'],1.0)
