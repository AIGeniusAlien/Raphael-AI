"""Private native CI inputs must not escape streamed XCTest diagnostics."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class NativeLogRedactionTests(unittest.TestCase):
    def test_abbreviated_xctest_typed_arguments_are_never_printed(self):
        script=Path(__file__).resolve().parents[1]/'.github/scripts/redact_native_log.py'
        with tempfile.TemporaryDirectory() as folder:
            private=Path(folder);password='SYNTHETIC-UNIQUE-PASSWORD-LONG-ENOUGH'
            (private/'password').write_text(password);(private/'redactions.json').write_text(json.dumps([password]))
            result=subprocess.run([sys.executable,str(script)],cwd=folder,env={'RAPHAEL_NATIVE_CI_PRIVATE':folder},input="Type '"+password[:17]+"...' into native-mfa-password SecureTextField\n",text=True,capture_output=True,timeout=10)
            self.assertEqual(result.returncode,0,result.stderr)
            for output in (result.stdout,(private/'native-build.log').read_text()):
                self.assertNotIn(password[:12],output);self.assertIn('native-mfa-password',output);self.assertIn('[PRIVATE INPUT WITHHELD]',output)
    def test_late_enrollment_and_unregistered_totp_are_withheld_during_stream(self):
        script=Path(__file__).resolve().parents[1]/'.github/scripts/redact_native_log.py'
        with tempfile.TemporaryDirectory() as folder:
            private=Path(folder);password='SYNTHETIC-PRIVATE-NATIVE-PASSWORD';late='SYNTHETIC-NEW-ENROLLMENT-KEY'
            (private/'password').write_text(password);(private/'redactions.json').write_text(json.dumps([password]))
            runner=private/'runner.py'
            runner.write_text('import runpy,sys,json\nfrom pathlib import Path\ndef lines():\n yield "Before enrollment\\n"\n Path("redactions.json").write_text(json.dumps('+repr([password,late])+'))\n yield '+repr(late+' 135790\n')+'\nsys.stdin=lines()\nrunpy.run_path('+repr(str(script))+',run_name="__main__")\n')
            result=subprocess.run([sys.executable,str(runner)],cwd=folder,env={'RAPHAEL_NATIVE_CI_PRIVATE':folder},text=True,capture_output=True,timeout=10)
            self.assertEqual(result.returncode,0,result.stderr)
            for output in (result.stdout,(private/'native-build.log').read_text()):
                self.assertIn('Before enrollment',output);self.assertNotIn(late,output);self.assertNotIn('135790',output)

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

    def test_timeout_progress_preserves_last_case_and_redacts_bounded_annotation(self):
        script=Path(__file__).resolve().parents[1]/'.github/scripts/redact_native_log.py'
        with tempfile.TemporaryDirectory() as folder:
            private=Path(folder);secret='SYNTHETIC-NATIVE-PROGRESS-SECRET'
            (private/'password').write_text(secret);(private/'redactions.json').write_text(json.dumps([secret]))
            (private/'native-build.log').write_text('Older output\n'*10000+"Test Case '-[SyntheticTests testAwaitingActualUI]' started.\n"+(' t = 12.3s Wait '+secret+' 100%\n')*30)
            result=subprocess.run([sys.executable,str(script),'--progress'],cwd=folder,env={'RAPHAEL_NATIVE_CI_PRIVATE':folder},text=True,capture_output=True,timeout=10)
            self.assertEqual(result.returncode,0,result.stderr);self.assertNotIn(secret,result.stdout+result.stderr)
            self.assertIn('testAwaitingActualUI',result.stdout);self.assertIn('%0A',result.stdout);self.assertIn('%25',result.stdout)
            self.assertLessEqual(len(result.stdout.strip().split('::',2)[2]),4096)
            self.assertIn(secret,(private/'native-build.log').read_text()) # no mutation of the preserved input
