"""Stream native diagnostics without exposing the ephemeral test password."""
import os
from pathlib import Path
import sys


def main():
    private=os.getenv('RAPHAEL_NATIVE_CI_PRIVATE')
    if not private:raise SystemExit('Private native fixture is missing;raw log output withheld.')
    password=(Path(private)/'password').read_text()
    if not 20<=len(password)<=128:raise SystemExit('Invalid native redaction binding;raw log output withheld.')
    with Path('native-build.log').open('w') as output:
        for line in sys.stdin:
            safe=line.replace(password,'[SYNTHETIC PASSWORD REDACTED]')
            output.write(safe);output.flush();sys.stdout.write(safe);sys.stdout.flush()


if __name__=='__main__':main()
