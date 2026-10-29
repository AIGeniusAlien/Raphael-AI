import os
from pathlib import Path
import tempfile,unittest
from unittest.mock import patch
from cryptography.fernet import Fernet
from sqlalchemy import create_engine,text
from dev_runtime import migrate
from dev_portal import create_dev_user
from dev_backup import encrypted_backup,restore_encrypted_backup

class EncryptedBackupTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
  self.env=patch.dict(os.environ,{'RAPHAEL_DEV_SYNTHETIC_ONLY':'1','RAPHAEL_BACKUP_ENCRYPTION_KEY':Fernet.generate_key().decode()});self.env.start();self.addCleanup(self.env.stop)
  self.engine=create_engine('sqlite:///'+str(self.root/'synthetic.sqlite'));self.addCleanup(self.engine.dispose);migrate(self.engine)
  create_dev_user(self.engine,'synthetic','Synthetic-preserved-user','Synthetic-only-password','patient')
 def test_encrypted_roundtrip_preserves_rows_and_produces_private_files(self):
  archive=self.root/'backup.enc';result=encrypted_backup(self.engine,archive);self.assertTrue(result['encrypted']);self.assertNotIn(b'Synthetic-preserved-user',archive.read_bytes())
  restored=self.root/'restored.sqlite';self.assertTrue(restore_encrypted_backup(archive,restored)['integrity_checked'])
  other=create_engine('sqlite:///'+str(restored))
  try:
   with other.connect() as c:self.assertEqual(c.execute(text('SELECT username FROM dev_users')).scalar(),'Synthetic-preserved-user')
  finally:other.dispose()
  if os.name=='posix':self.assertEqual(archive.stat().st_mode&0o777,0o600);self.assertEqual(restored.stat().st_mode&0o777,0o600)
 def test_wrong_key_and_tampering_never_create_restore_destination(self):
  archive=self.root/'backup.enc';encrypted_backup(self.engine,archive);target=self.root/'never-created.sqlite'
  with patch.dict(os.environ,{'RAPHAEL_BACKUP_ENCRYPTION_KEY':Fernet.generate_key().decode()}):
   with self.assertRaisesRegex(ValueError,'authentication failed'):restore_encrypted_backup(archive,target)
  archive.write_bytes(archive.read_bytes()[:-5]+b'xxxxx')
  with self.assertRaises(ValueError):restore_encrypted_backup(archive,target)
  self.assertFalse(target.exists())
 def test_existing_backup_and_restore_are_never_overwritten(self):
  archive=self.root/'backup.enc';encrypted_backup(self.engine,archive);original=archive.read_bytes()
  with self.assertRaises(FileExistsError):encrypted_backup(self.engine,archive)
  self.assertEqual(archive.read_bytes(),original)
  target=self.root/'existing.sqlite';target.write_bytes(b'preserve-existing')
  with self.assertRaises(FileExistsError):restore_encrypted_backup(archive,target)
  self.assertEqual(target.read_bytes(),b'preserve-existing')
 def test_restore_scope_and_missing_key_fail_closed(self):
  archive=self.root/'backup.enc';encrypted_backup(self.engine,archive)
  with patch.dict(os.environ,{'RAPHAEL_DEV_SYNTHETIC_ONLY':'0'}):
   with self.assertRaises(ValueError):restore_encrypted_backup(archive,self.root/'not-created')
  with patch.dict(os.environ,{'RAPHAEL_BACKUP_ENCRYPTION_KEY':''}):
   with self.assertRaises(ValueError):encrypted_backup(self.engine,self.root/'no-key.enc')
