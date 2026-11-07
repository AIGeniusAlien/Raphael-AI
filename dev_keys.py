"""Explicit local development key files. Never prints key material or overwrites.

These files are not a production KMS,rotation service or independent ACL review.
Windows users must use a protected secret-manager environment binding instead.
"""
import argparse
import os
from pathlib import Path
import stat
from cryptography.fernet import Fernet

def create_key_file(path):
    if os.name!='posix':raise ValueError('Use a protected secret-manager environment binding on this platform.')
    destination=Path(path)
    fd=os.open(destination,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0),0o600)
    try:
        os.fchmod(fd,0o600)
        with os.fdopen(fd,'wb') as stream:stream.write(Fernet.generate_key()+b'\n');stream.flush();os.fsync(stream.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return destination

def load_key_file(path):
    if os.name!='posix':raise ValueError('Use a protected secret-manager environment binding on this platform.')
    fd=os.open(Path(path),os.O_RDONLY|getattr(os,'O_NOFOLLOW',0)|getattr(os,'O_NONBLOCK',0))
    with os.fdopen(fd,'rb') as stream:
        info=os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid!=os.getuid() or info.st_mode & 0o077:raise ValueError('Key file must be a regular owner-only file with mode0600 or stricter.')
        raw=stream.read(257)
    if len(raw)>256:raise ValueError('Invalid development key file.')
    try:value=raw.strip().decode('ascii');Fernet(value.encode('ascii'))
    except (ValueError,UnicodeError):raise ValueError('Invalid development key file;no value disclosed.') from None
    return value

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('operation',choices=['create']);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    try:create_key_file(args.output)
    except (OSError,ValueError):parser.error('Key file creation refused. Choose a new path in protected storage;existing files are never overwritten.')
    print('Local development key file created. Keep a separate protected backup;do not commit or paste its contents. No value was printed.')
if __name__=='__main__':main()
