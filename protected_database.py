"""Opt-in SQLCipher for isolated synthetic SQLite;no automatic migration or rekey.

Existing plaintext sources/exports/backups remain operator-controlled copies.
Stop writers and plan cutover explicitly;this is a snapshot,not live replication.
No production KMS,secure erasure,HIPAA or cross-platform certification is implied.
"""
import argparse
import base64
import hashlib
import hmac
import os
from pathlib import Path
import sqlite3
import stat
import tempfile
from sqlalchemy import create_engine,event
from sqlalchemy.engine import make_url

MAX_BYTES=64*1024*1024
MAX_ROWS=200000


def key_bytes(value):
    from cryptography.fernet import Fernet
    if not isinstance(value,str):raise ValueError('Protected database key unavailable or invalid;no value disclosed.')
    try:
        value=value.strip();Fernet(value.encode('ascii'));raw=base64.urlsafe_b64decode(value)
        if len(raw)!=32:raise ValueError()
        return raw
    except (ValueError,TypeError,UnicodeError):raise ValueError('Protected database key unavailable or invalid;no value disclosed.') from None


def driver():
    try:from sqlcipher3 import dbapi2
    except ImportError:raise ValueError('SQLCipher runtime unavailable;no plaintext fallback is permitted.') from None
    return dbapi2


def unlock(connection,key):
    try:
        if key is not None:connection.execute('PRAGMA key="x\''+key.hex()+'\'"')
        version=connection.execute('PRAGMA cipher_version').fetchone()
        if not version or not version[0]:raise ValueError()
        connection.execute('PRAGMA cipher_memory_security=ON')
        connection.execute('PRAGMA temp_store=MEMORY')
        connection.execute('SELECT count(*) FROM sqlite_master').fetchone()
    except Exception:raise ValueError('Protected SQLite open refused;verify key and database format. No fallback or migration occurred.') from None


def private_target(path):
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0),0o600)
    os.close(fd)


def application_engine(url,*,encrypted=None,encryption_key=None,**kwargs):
    mode=os.getenv('RAPHAEL_DATABASE_ENCRYPTED','0')
    if encrypted is None:
        if mode not in ('0','1'):raise ValueError('Database protection mode must be explicitly0or1.')
        encrypted=mode=='1'
        if not encrypted and os.getenv('RAPHAEL_DATABASE_ENCRYPTION_KEY'):raise ValueError('Database key supplied without encrypted mode;plaintext startup refused.')
    if not encrypted:return create_engine(url,**kwargs)
    if os.name!='posix' or os.getenv('RAPHAEL_DEV_SYNTHETIC_ONLY')!='1':raise ValueError('Protected database currently requires POSIX and explicit synthetic-only scope.')
    parsed=make_url(url)
    if parsed.drivername!='sqlite' or not parsed.database or parsed.database==':memory:' or parsed.query:raise ValueError('Protected SQLite requires a dedicated file URL without query credentials.')
    raw=key_bytes(encryption_key or os.getenv('RAPHAEL_DATABASE_ENCRYPTION_KEY',''))
    for name in ('RAPHAEL_MFA_ENCRYPTION_KEY','RAPHAEL_DOCUMENT_ENCRYPTION_KEY','RAPHAEL_BACKUP_ENCRYPTION_KEY'):
        try:other=key_bytes(os.getenv(name,''))
        except ValueError:continue
        if hmac.compare_digest(raw,other):raise ValueError('Use distinct component encryption keys.')
    dbapi=driver()
    path=Path(parsed.database).absolute()
    if path.is_symlink():raise ValueError('Protected database symlinks are refused.')
    if not path.exists():private_target(path)
    info=path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid!=os.getuid() or info.st_mode&0o077:raise ValueError('Protected database must be an owner-only regular file.')
    kwargs.setdefault('hide_parameters',True)
    engine=create_engine(url,module=dbapi,**kwargs)
    @event.listens_for(engine,'connect')
    def configure(connection,record):unlock(connection,raw)
    engine._raphael_encrypted=True
    def backup_connection(destination):
        connection=dbapi.connect(str(destination));unlock(connection,raw);return connection
    engine._raphael_backup_connection=backup_connection
    return engine


def storage_status(engine):
    encrypted=bool(getattr(engine,'_raphael_encrypted',False))
    with engine.connect() as c:
        version=c.exec_driver_sql('PRAGMA cipher_version').scalar() if encrypted else None
        memory=c.exec_driver_sql('PRAGMA temp_store').scalar()==2 if encrypted else False
    return {'sqlite_pages_encrypted':bool(encrypted and version),'cipher_version':version,'temporary_tables_in_memory':memory,'scope':'Synthetic SQLite database pages, including stored original-document bytes and clinical text.' if encrypted else 'This database uses plaintext SQLite;attachment payload encryption is a separate control.','limitations':'Existing source copies, exports, upload spooling, OS caches and separately restored files are outside database-page protection. Operator key escrow, rotation, retention, secure erasure and independent production review remain required. No HIPAA compliance claim.'}


def quoted(name):return '"'+name.replace('"','""')+'"'


def fingerprint(connection,schema='main'):
    objects=connection.execute(f'SELECT type,name,sql FROM {schema}.sqlite_master WHERE sql IS NOT NULL ORDER BY type,name').fetchall()
    data={};total=0
    for kind,name,sql in objects:
        if kind!='table':continue
        prefix='rowid,' if sql.upper().startswith('CREATE VIRTUAL TABLE') else ''
        hashes=[]
        for row in connection.execute(f'SELECT {prefix}* FROM {schema}.'+quoted(name)):
            total+=1
            if total>MAX_ROWS:raise ValueError('Migration row budget exceeded;no partial output retained.')
            hashes.append(hashlib.sha256(repr(tuple(row)).encode()).digest())
        data[name]=(len(hashes),hashlib.sha256(b''.join(sorted(hashes))).digest())
    return objects,data


def migrate_copy(source,destination,target_key_file,source_key_file=None):
    """Read-only consistent source snapshot;new0600 output only;verify all objects/rows."""
    from dev_keys import load_key_file
    from dev_portal import _require_synthetic
    if os.getenv('RAPHAEL_DEV_SYNTHETIC_ONLY')!='1':raise ValueError('Explicit synthetic-only migration scope required.')
    source=Path(source).absolute();destination=Path(destination).absolute()
    if source.is_symlink() or not source.is_file() or source==destination:raise ValueError('Use distinct regular source and new destination files.')
    if source.stat().st_size>MAX_BYTES:raise ValueError('Migration exceeds bounded synthetic size.')
    target_key=key_bytes(load_key_file(target_key_file));source_key=key_bytes(load_key_file(source_key_file)) if source_key_file else None
    if source_key==target_key:raise ValueError('Rotation requires a distinct new key;source is never rekeyed in place.')
    validation=application_engine('sqlite:///'+str(source),encrypted=source_key is not None,encryption_key=load_key_file(source_key_file) if source_key_file else None)
    try:
        _require_synthetic(validation)
        from dev_runtime import schema_status
        if not schema_status(validation)['ready']:raise ValueError('Source schema is not ready for this runtime;no copy published.')
    finally:validation.dispose()
    if destination.exists() or destination.is_symlink():raise FileExistsError('Migration destination already exists.')
    try:
        with tempfile.TemporaryDirectory(prefix='.raphael-protected-',dir=destination.parent) as private:
            staged=Path(private)/'verified.sqlite';private_target(staged)
            connection=driver().connect(source.as_uri()+'?mode=ro',uri=True)
            try:
                unlock(connection,source_key)
                connection.execute('ATTACH DATABASE ? AS protected KEY "x\''+target_key.hex()+'\'"',(str(staged),))
                connection.execute('BEGIN')
                if int(connection.execute('PRAGMA page_count').fetchone()[0])*int(connection.execute('PRAGMA page_size').fetchone()[0])>MAX_BYTES:raise ValueError('Snapshot exceeds migration size budget.')
                before=fingerprint(connection)
                user_version=connection.execute('PRAGMA user_version').fetchone()[0];application_id=connection.execute('PRAGMA application_id').fetchone()[0]
                connection.execute("SELECT sqlcipher_export('protected')")
                connection.execute('PRAGMA protected.user_version='+str(user_version));connection.execute('PRAGMA protected.application_id='+str(application_id))
                connection.commit();connection.execute('DETACH DATABASE protected')
            finally:connection.close()
            # SQLCipher export installs virtual-table definitions via writable
            # schema. Reopen after commit so FTS modules load their new schema.
            check=driver().connect(str(staged))
            try:
                unlock(check,target_key)
                if check.execute('PRAGMA cipher_integrity_check').fetchall() or check.execute('PRAGMA integrity_check').fetchone()!=('ok',) or check.execute('PRAGMA foreign_key_check').fetchall() or fingerprint(check)!=before:raise ValueError('Encrypted copy authentication/data verification failed.')
            finally:check.close()
            # Same-filesystem exclusive atomic publication;never replace a file
            # created by another process during verification. Stage stays0700.
            os.link(staged,destination,follow_symlinks=False)
            return {'copied':True,'sqlite_pages_encrypted':True,'rotation_copy':source_key is not None,'source_modified':False,'overwritten':False,'objects_verified':len(before[0]),'tables_verified':len(before[1]),'source_copies_retained':True,'automatic_cutover':False,'clinical_validation':False}
    except FileExistsError:raise
    except Exception:raise ValueError('Protected copy refused;verify keys,synthetic source,integrity,budgets and new output. Source retained;no key or content disclosed.') from None


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--target-key-file',type=Path,required=True);parser.add_argument('--source-key-file',type=Path);args=parser.parse_args()
    try:
        import json
        print(json.dumps(migrate_copy(args.source,args.output,args.target_key_file,args.source_key_file)));return 0
    except Exception:
        print('Protected database copy refused. Check synthetic scope,private keys and a new output path;source retained.');return 1
if __name__=='__main__':raise SystemExit(main())
