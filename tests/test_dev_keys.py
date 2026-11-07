"""Owner-only POSIX development keys;not production KMS or Windows ACL proof."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from cryptography.fernet import Fernet
from dev_keys import create_key_file,load_key_file
from dev_start import runtime_environment

@unittest.skipUnless(os.name=='posix','POSIX permission semantics required;use secret manager elsewhere.')
class DevelopmentKeyTests(unittest.TestCase):
    def test_private_key_creation_restart_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'development.key';create_key_file(path);before=path.read_bytes()
            self.assertEqual(path.stat().st_mode & 0o777,0o600)
            first=load_key_file(path);token=Fernet(first.encode()).encrypt(b'SYNTHETIC-PAYLOAD')
            self.assertEqual(Fernet(load_key_file(path).encode()).decrypt(token),b'SYNTHETIC-PAYLOAD')
            with self.assertRaises(FileExistsError):create_key_file(path)
            self.assertTrue(path.read_bytes()==before,'An existing key was overwritten.')

    def test_public_symlink_fifo_oversized_and_invalid_key_files_are_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'private.key';create_key_file(path);path.chmod(0o644)
            with self.assertRaises(ValueError):load_key_file(path)
            path.chmod(0o600);link=Path(folder)/'link.key';link.symlink_to(path)
            with self.assertRaises(OSError):load_key_file(link)
            fifo=Path(folder)/'pipe.key';os.mkfifo(fifo,0o600)
            with self.assertRaises(ValueError):load_key_file(fifo)
            for value in (b'x'*257,b'not-a-valid-key'):
                path.write_bytes(value)
                with self.assertRaises(ValueError):load_key_file(path)

    def test_launcher_uses_distinct_explicit_files_and_refuses_ambiguous_sources(self):
        with tempfile.TemporaryDirectory() as folder,patch.dict(os.environ,{'RAPHAEL_DOCUMENT_ENCRYPTION_KEY':'','RAPHAEL_MFA_ENCRYPTION_KEY':''}):
            document=Path(folder)/'document.key';mfa=Path(folder)/'mfa.key';create_key_file(document);create_key_file(mfa)
            env=runtime_environment('sqlite:///synthetic.db',document_key_file=document,mfa_key_file=mfa)
            self.assertTrue(env['RAPHAEL_DOCUMENT_ENCRYPTION_KEY']==load_key_file(document));self.assertTrue(env['RAPHAEL_MFA_ENCRYPTION_KEY']==load_key_file(mfa));self.assertNotIn('LLM_API_KEY',env)
            with self.assertRaises(ValueError):runtime_environment('sqlite:///synthetic.db',document_key_file=document,mfa_key_file=document)
            with patch.dict(os.environ,{'RAPHAEL_DOCUMENT_ENCRYPTION_KEY':'SYNTHETIC-EXISTING-BINDING'}),self.assertRaises(ValueError):runtime_environment('sqlite:///synthetic.db',document_key_file=document)

    def test_cli_does_not_print_key_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'new.key'
            command=[sys.executable,'-m','dev_keys','create','--output',str(path)]
            first=subprocess.run(command,capture_output=True,text=True,timeout=10);self.assertEqual(first.returncode,0)
            key=load_key_file(path);self.assertTrue(key not in first.stdout+first.stderr,'Key material escaped CLI output.')
            again=subprocess.run(command,capture_output=True,text=True,timeout=10);self.assertNotEqual(again.returncode,0);self.assertTrue(key not in again.stdout+again.stderr);self.assertTrue(load_key_file(path)==key)
