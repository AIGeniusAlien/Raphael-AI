"""Private native CI inputs must not escape streamed XCTest diagnostics."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class NativeLogRedactionTests(unittest.TestCase):
    def test_all_dynamic_fixture_credentials_are_redacted_in_stdout_and_artifact(self):
        script=Path(__file__).resolve().parents[1]/'.github/scripts/redact_native_log.py'
        with tempfile.TemporaryDirectory() as folder:
            private=Path(folder);password='SYNTHETIC-PASSWORD-FOR-REDACTION';factor='654321';recovery='SYNTHETIC-RECOVERY-ABCDEF'
            (private/'password').write_text(password);(private/'redactions.json').write_text(json.dumps([password,factor,recovery]))
            result=subprocess.run([sys.executable,str(script)],cwd=folder,env={'RAPHAEL_NATIVE_CI_PRIVATE':folder},input=f'Type {password}\nType {factor}\nType {recovery}\n',text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)
            for value in (password,factor,recovery):
                self.assertNotIn(value,result.stdout);self.assertNotIn(value,(private/'native-build.log').read_text())

    def test_missing_extended_bindings_withhold_raw_logs(self):
        script=Path(__file__).resolve().parents[1]/'.github/scripts/redact_native_log.py'
        with tempfile.TemporaryDirectory() as folder:
            Path(folder,'password').write_text('SYNTHETIC-PASSWORD-FOR-REDACTION')
            result=subprocess.run([sys.executable,str(script)],cwd=folder,env={'RAPHAEL_NATIVE_CI_PRIVATE':folder},input='PRIVATE-INPUT',text=True,capture_output=True)
            self.assertNotEqual(result.returncode,0);self.assertNotIn('PRIVATE-INPUT',result.stdout+result.stderr)
