"""Actual SQLCipher/SQLite data,FTS,rotation,backup and authenticated API tests."""
import hashlib,json,os,sqlite3,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from cryptography.fernet import Fernet
from sqlalchemy import create_engine,text
from fastapi import FastAPI
from fastapi.testclient import TestClient
from dev_keys import create_key_file,load_key_file
from dev_runtime import migrate,backup
from dev_portal import create_dev_user,install_dev_portal
from dev_account import install_account
from protected_database import application_engine,migrate_copy,storage_status

class ProtectedDatabaseTests(unittest.TestCase):
    def setUp(self):
        if __import__('importlib').util.find_spec('sqlcipher3') is None:self.skipTest('SQLCipher runtime unavailable on this platform;no protected-storage proof.')
        self.temp=tempfile.TemporaryDirectory(prefix='raphael-cipher-test-');self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        environment=patch.dict(os.environ,{'RAPHAEL_DEV_SYNTHETIC_ONLY':'1','RAPHAEL_DATABASE_ENCRYPTED':'0','RAPHAEL_DATABASE_ENCRYPTION_KEY':''});environment.start();self.addCleanup(environment.stop)
        self.source=self.root/'source.db';self.destination=self.root/'encrypted.db';self.key=self.root/'data.key';self.new_key=self.root/'new.key'
        create_key_file(self.key);create_key_file(self.new_key)
        self.password='Synthetic-only-private-password'
        e=create_engine('sqlite:///'+str(self.source));migrate(e);create_dev_user(e,'one','doctor',self.password,'clinician')
        with e.begin() as c:
            c.exec_driver_sql('CREATE TABLE dev_cipher_docs(id INTEGER PRIMARY KEY AUTOINCREMENT,body TEXT,original BLOB)')
            c.exec_driver_sql('CREATE VIRTUAL TABLE dev_cipher_search USING fts5(body)')
            c.exec_driver_sql('CREATE TRIGGER dev_cipher_insert AFTER INSERT ON dev_cipher_docs BEGIN INSERT INTO dev_cipher_search(rowid,body) VALUES(new.id,new.body); END')
            c.exec_driver_sql('CREATE TRIGGER dev_cipher_delete AFTER DELETE ON dev_cipher_docs BEGIN DELETE FROM dev_cipher_search WHERE rowid=old.id; END')
            c.execute(text('INSERT INTO dev_cipher_docs(id,body,original) VALUES(5,:body,:blob),(9,:body,:blob)'),{'body':'SYNTHETIC-PRIVATE-SQLCIPHER-MARKER','blob':b'SYNTHETIC-ORIGINAL-DOCUMENT-BYTES'})
            c.exec_driver_sql('DELETE FROM dev_cipher_docs WHERE id=9')
            c.exec_driver_sql('PRAGMA user_version=17')
        e.dispose();self.source_hash=hashlib.sha256(self.source.read_bytes()).digest()
    def engine(self,path=None,key=None):
        e=application_engine('sqlite:///'+str(path or self.destination),encrypted=True,encryption_key=load_key_file(key or self.key));self.addCleanup(e.dispose);return e
    def copied(self):
        result=migrate_copy(self.source,self.destination,self.key);self.assertTrue(result['sqlite_pages_encrypted']);return self.engine()
    def test_full_copy_preserves_rows_blobs_fts_triggers_sequence_and_source(self):
        e=self.copied();migrate(e);migrate(e)
        with e.begin() as c:
            self.assertEqual(c.exec_driver_sql('SELECT rowid FROM dev_cipher_search WHERE dev_cipher_search MATCH "MARKER"').all(),[(5,)])
            self.assertEqual(c.exec_driver_sql('SELECT original FROM dev_cipher_docs').scalar(),b'SYNTHETIC-ORIGINAL-DOCUMENT-BYTES')
            self.assertEqual(c.exec_driver_sql('PRAGMA user_version').scalar(),17)
            self.assertEqual(c.exec_driver_sql("INSERT INTO dev_cipher_docs(body) VALUES('new synthetic text') RETURNING id").scalar(),10)
            self.assertEqual(c.exec_driver_sql('SELECT rowid FROM dev_cipher_search WHERE dev_cipher_search MATCH "new"').all(),[(10,)])
            c.exec_driver_sql('DELETE FROM dev_cipher_docs WHERE id=10');self.assertEqual(c.exec_driver_sql('SELECT count(*) FROM dev_cipher_search').scalar(),1)
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).digest(),self.source_hash)
        self.assertFalse(self.destination.read_bytes().startswith(b'SQLite format 3'))
        self.assertFalse(b'SYNTHETIC-PRIVATE-SQLCIPHER-MARKER' in self.destination.read_bytes())
        self.assertEqual(self.destination.stat().st_mode&0o777,0o600)
        self.assertTrue(storage_status(e)['temporary_tables_in_memory'])
    def test_missing_wrong_key_and_plaintext_fallback_refused(self):
        self.copied()
        with sqlite3.connect(self.destination) as c:
            with self.assertRaises(sqlite3.DatabaseError):c.execute('SELECT * FROM dev_users').fetchall()
        wrong=self.engine(key=self.new_key)
        with self.assertRaises(ValueError):wrong.connect()
        with patch.dict(os.environ,{'RAPHAEL_DATABASE_ENCRYPTED':'1','RAPHAEL_DATABASE_ENCRYPTION_KEY':''}):
            with self.assertRaises(ValueError):application_engine('sqlite:///'+str(self.root/'must-not-create.db'))
        self.assertFalse((self.root/'must-not-create.db').exists())
        with patch.dict(os.environ,{'RAPHAEL_DATABASE_ENCRYPTION_KEY':load_key_file(self.key)}):
            with self.assertRaises(ValueError):application_engine('sqlite:///'+str(self.source))
        with self.assertRaises(ValueError):self.engine(path=self.source).connect()
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).digest(),self.source_hash)
    def test_rotation_is_new_verified_copy_old_key_and_source_preserved(self):
        self.copied();rotated=self.root/'rotated.db';before=self.destination.read_bytes()
        result=migrate_copy(self.destination,rotated,self.new_key,self.key);self.assertTrue(result['rotation_copy']);self.assertEqual(before,self.destination.read_bytes())
        with self.engine(path=rotated,key=self.new_key).connect() as c:self.assertEqual(c.exec_driver_sql('SELECT count(*) FROM dev_users').scalar(),1)
        with self.assertRaises(ValueError):self.engine(path=rotated,key=self.key).connect()
        with self.engine().connect() as c:self.assertEqual(c.exec_driver_sql('SELECT count(*) FROM dev_users').scalar(),1)
        with self.assertRaises(ValueError):migrate_copy(self.destination,self.root/'same-key.db',self.key,self.key)
    def test_no_overwrite_symlink_or_partial_output_on_failed_verification(self):
        occupied=self.root/'occupied.db';occupied.write_bytes(b'EXISTING SYNTHETIC FILE')
        with self.assertRaises(FileExistsError):migrate_copy(self.source,occupied,self.key)
        self.assertEqual(occupied.read_bytes(),b'EXISTING SYNTHETIC FILE')
        link=self.root/'link.db';link.symlink_to(occupied)
        with self.assertRaises(FileExistsError):migrate_copy(self.source,link,self.key)
        from protected_database import fingerprint
        calls=[]
        def mismatched(*args,**kwargs):
            value=fingerprint(*args,**kwargs);calls.append(True);return value if len(calls)==1 else ([],{})
        with patch('protected_database.fingerprint',side_effect=mismatched):
            with self.assertRaises(ValueError):migrate_copy(self.source,self.destination,self.key)
        self.assertFalse(self.destination.exists());self.assertEqual(list(self.root.glob('.raphael-protected-*')),[])
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).digest(),self.source_hash)
    def test_backup_and_outer_encryption_restore_keep_sqlcipher_pages(self):
        from dev_backup import encrypted_backup,restore_encrypted_backup
        e=self.copied();snapshot=self.root/'snapshot.db';backup(e,snapshot)
        with self.engine(path=snapshot).connect() as c:self.assertEqual(c.exec_driver_sql('SELECT count(*) FROM dev_cipher_docs').scalar(),1)
        outer=self.root/'snapshot.fernet';restored=self.root/'restored.db'
        with patch.dict(os.environ,{'RAPHAEL_BACKUP_ENCRYPTION_KEY':Fernet.generate_key().decode(),'RAPHAEL_DATABASE_ENCRYPTED':'1','RAPHAEL_DATABASE_ENCRYPTION_KEY':load_key_file(self.key)}):
            result=encrypted_backup(e,outer);self.assertTrue(result['live_database_encrypted']);self.assertTrue(restore_encrypted_backup(outer,restored)['sqlite_pages_encrypted'])
        with self.engine(path=restored).connect() as c:self.assertEqual(c.exec_driver_sql('SELECT count(*) FROM dev_cipher_docs').scalar(),1)
        self.assertFalse(restored.read_bytes().startswith(b'SQLite format 3'))
    def test_tampered_ciphertext_and_missing_driver_fail_closed(self):
        self.copied();damaged=self.root/'damaged.db';raw=bytearray(self.destination.read_bytes());raw[50]^=0x20;damaged.write_bytes(raw);damaged.chmod(0o600)
        with self.assertRaises(ValueError):self.engine(path=damaged).connect()
        with patch('protected_database.driver',side_effect=ValueError('Unavailable')):
            with self.assertRaises(ValueError):self.engine(path=self.root/'unavailable.db')
        self.assertFalse((self.root/'unavailable.db').exists())
    def test_actual_authenticated_api_and_content_free_storage_status(self):
        e=self.copied();app=FastAPI();ctx=install_dev_portal(app,e);install_account(app,ctx)
        with TestClient(app) as client:
            self.assertEqual(client.get('/api/dev/account/storage').status_code,401)
            self.assertEqual(client.post('/api/dev/login',json={'tenant':'one','username':'doctor','password':self.password}).status_code,200)
            response=client.get('/api/dev/account/storage');self.assertEqual(response.status_code,200,response.text);self.assertTrue(response.json()['sqlite_pages_encrypted'])
            self.assertFalse(load_key_file(self.key) in response.text);self.assertNotIn(str(self.root),response.text)
            created=client.post('/api/dev/patients',json={'name':'Synthetic protected API patient'});self.assertEqual(created.status_code,200,created.text)
            self.assertIn('Synthetic protected API patient',client.get('/api/dev/patients').text)
            client.post('/api/dev/logout',json={});self.assertEqual(client.get('/api/dev/account/storage').status_code,401)
    def test_launcher_distinct_keys_and_no_implicit_encryption_mode(self):
        from dev_start import runtime_environment
        environment=runtime_environment('sqlite:///'+str(self.root/'new.db'),database_key_file=self.key)
        self.assertEqual(environment['RAPHAEL_DATABASE_ENCRYPTED'],'1');self.assertEqual(environment['RAPHAEL_DATABASE_ENCRYPTION_KEY'],load_key_file(self.key))
        with self.assertRaises(ValueError):runtime_environment('sqlite:///'+str(self.root/'new.db'),database_key_file=self.key,document_key_file=self.key)
        with patch.dict(os.environ,{'RAPHAEL_DATABASE_ENCRYPTION_KEY':load_key_file(self.key)}):
            with self.assertRaises(ValueError):runtime_environment('sqlite:///'+str(self.root/'new.db'))
