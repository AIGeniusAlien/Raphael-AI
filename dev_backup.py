"""Bounded authenticated-encryption backup/restore for isolated synthetic SQLite.

The operator owns external key storage/rotation. This does not encrypt the live
SQLite database, establish a production retention policy or create cloud backups.
"""
import argparse
import os
from pathlib import Path
import sqlite3
import tempfile
from cryptography.fernet import Fernet,InvalidToken
from sqlalchemy import create_engine
from dev_runtime import backup,schema_status
from dev_portal import _require_synthetic

MAX_DATABASE_BYTES=16*1024*1024
MAX_ENCRYPTED_BYTES=24*1024*1024

def cipher():
    try:
        value=os.environ.get('RAPHAEL_BACKUP_ENCRYPTION_KEY','')
        if not value:raise ValueError()
        return Fernet(value.encode('ascii'))
    except (ValueError,UnicodeError):raise ValueError('Protected backup encryption key unavailable or invalid.') from None

def private_write(destination,data):
    fd=os.open(destination,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    try:
        with os.fdopen(fd,'wb') as output:output.write(data);output.flush();os.fsync(output.fileno())
    except BaseException:
        Path(destination).unlink(missing_ok=True);raise

def encrypted_backup(engine,destination):
    encryption=cipher();_require_synthetic(engine)
    with engine.connect() as connection:
        pages=connection.exec_driver_sql('PRAGMA page_count').scalar()
        size=connection.exec_driver_sql('PRAGMA page_size').scalar()
        if pages*size>MAX_DATABASE_BYTES:raise ValueError('Synthetic encrypted backup exceeds the bounded limit.')
    # A consistent SQLite backup, never a byte-copy of a concurrently written DB.
    with tempfile.TemporaryDirectory(prefix='raphael-private-backup-') as private:
        source=Path(private)/'snapshot.sqlite';backup(engine,source)
        if source.stat().st_size>MAX_DATABASE_BYTES:raise ValueError('Synthetic encrypted backup exceeds the bounded limit.')
        encrypted=encryption.encrypt(source.read_bytes());private_write(destination,encrypted)
    return {'encrypted':True,'scope':'synthetic_sqlite','live_database_encrypted':False,'key_stored_in_backup':False}

def restore_encrypted_backup(source,destination):
    if os.getenv('RAPHAEL_DEV_SYNTHETIC_ONLY')!='1':raise ValueError('Restore requires explicit synthetic-only scope.')
    source=Path(source)
    if source.stat().st_size>MAX_ENCRYPTED_BYTES:raise ValueError('Encrypted backup exceeds bounded limit.')
    try:plaintext=cipher().decrypt(source.read_bytes())
    except InvalidToken:raise ValueError('Backup authentication failed. Restore refused; verify the protected key and file integrity.') from None
    if len(plaintext)>MAX_DATABASE_BYTES or not plaintext.startswith(b'SQLite format 3\x00'):raise ValueError('Not a bounded SQLite backup.')
    # Validate privately before exposing the restored file at its final path.
    with tempfile.TemporaryDirectory(prefix='raphael-private-restore-') as private:
        candidate=Path(private)/'candidate.sqlite';private_write(candidate,plaintext)
        with sqlite3.connect('file:'+str(candidate)+'?mode=ro',uri=True) as connection:
            if connection.execute('PRAGMA integrity_check').fetchone()!=('ok',) or connection.execute('PRAGMA foreign_key_check').fetchone():raise ValueError('Restored database integrity check failed.')
        engine=create_engine('sqlite:///'+str(candidate))
        try:
            _require_synthetic(engine)
            if not schema_status(engine)['ready']:raise ValueError('Backup schema does not match this application; no restore performed.')
        finally:engine.dispose()
        private_write(destination,plaintext)
    return {'restored':True,'scope':'synthetic_sqlite','overwritten':False,'integrity_checked':True}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('backup','restore'));parser.add_argument('--database-url');parser.add_argument('--source',type=Path);parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
    try:
        if args.command=='backup':
            if not args.database_url:parser.error('Backup requires the dedicated synthetic --database-url.')
            engine=create_engine(args.database_url)
            try:result=encrypted_backup(engine,args.output)
            finally:engine.dispose()
        else:
            if not args.source:parser.error('Restore requires --source.')
            result=restore_encrypted_backup(args.source,args.output)
        import json
        print(json.dumps(result));return 0
    except Exception as error:
        # Never reflect key,encrypted bytes,DB content or provider configuration.
        print('Backup operation refused: '+type(error).__name__+'. Verify key, synthetic scope, size, integrity and a new output path.');return 1

if __name__=='__main__':raise SystemExit(main())
