"""Synthetic CI logging protocol checks;no native execution or real credentials."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


class NativeCILoggingTests(unittest.TestCase):
    def test_private_fixture_password_is_redacted_from_console_and_artifact_log(self):
        with tempfile.TemporaryDirectory() as directory:
            password='Synthetic-ephemeral-test-password-123'
            Path(directory,'password').write_text(password)
            result=subprocess.run([sys.executable,str(ROOT/'.github/scripts/redact_native_log.py')],input='Synthetic typed value '+password+'\n',capture_output=True,text=True,cwd=directory,env={**os.environ,'RAPHAEL_NATIVE_CI_PRIVATE':directory},timeout=10)
            self.assertEqual(result.returncode,0,result.stderr)
            for output in (result.stdout,Path(directory,'native-build.log').read_text()):
                self.assertNotIn(password,output);self.assertIn('[SYNTHETIC PASSWORD REDACTED]',output)

    def test_missing_redaction_binding_withholds_raw_output(self):
        with tempfile.TemporaryDirectory() as directory:
            environment={key:value for key,value in os.environ.items() if key!='RAPHAEL_NATIVE_CI_PRIVATE'}
            result=subprocess.run([sys.executable,str(ROOT/'.github/scripts/redact_native_log.py')],input='Sensitive synthetic raw log',capture_output=True,text=True,cwd=directory,env=environment,timeout=10)
            self.assertNotEqual(result.returncode,0)
            self.assertNotIn('Sensitive synthetic raw log',result.stdout+result.stderr)

    def test_annotation_limits_do_not_truncate_late_failure_or_expose_fixture_secret(self):
        with tempfile.TemporaryDirectory() as directory:
            password='Synthetic-annotation-test-secret-123'
            path=Path(directory,'native.log')
            path.write_text(('Routine failed simulator warning '+'x'*200+'\n')*70+'source.swift:7: error: LATE_AUTHENTICATION_FAILURE '+password+'\n')
            result=subprocess.run([sys.executable,str(ROOT/'.github/scripts/publish_diagnostics.py'),str(path)],capture_output=True,text=True,env={**os.environ,'LLM_API_KEY':password},timeout=10)
            self.assertEqual(result.returncode,0,result.stderr)
            annotations=[line.split('::',2)[2] for line in result.stdout.splitlines() if line.startswith('::error')]
            self.assertGreater(len(annotations),1)
            self.assertTrue(all(len(item)<=4096 for item in annotations))
            self.assertIn('LATE_AUTHENTICATION_FAILURE',result.stdout)
            self.assertNotIn(password,result.stdout)
