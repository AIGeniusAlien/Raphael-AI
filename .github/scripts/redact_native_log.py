"""Stream native diagnostics without exposing the ephemeral test password."""
import os
import json
from pathlib import Path
import sys


def main():
    private=os.getenv('RAPHAEL_NATIVE_CI_PRIVATE')
    if not private:raise SystemExit('Private native fixture is missing;raw log output withheld.')
    password=(Path(private)/'password').read_text()
    if not 20<=len(password)<=128:raise SystemExit('Invalid native redaction binding;raw log output withheld.')
    values=json.loads((Path(private)/'redactions.json').read_text())
    if not isinstance(values,list) or password not in values or any(not isinstance(v,str) or not v for v in values):raise SystemExit('Private extended redaction bindings missing;raw logs withheld.')
    with Path('native-build.log').open('w') as output:
        for line in sys.stdin:
            safe=line
            for value in sorted(values,key=len,reverse=True):safe=safe.replace(value,'[PRIVATE CI INPUT REDACTED]')
            output.write(safe);output.flush();sys.stdout.write(safe);sys.stdout.flush()


if __name__=='__main__':main()
