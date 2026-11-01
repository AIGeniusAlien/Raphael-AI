"""Stream native diagnostics without exposing the ephemeral test password."""
import os
import json
from pathlib import Path
import sys


def main():
    if sys.argv[1:] not in ([],['--progress']):raise SystemExit('Unsupported native log mode;raw output withheld.')
    private=os.getenv('RAPHAEL_NATIVE_CI_PRIVATE')
    if not private:raise SystemExit('Private native fixture is missing;raw log output withheld.')
    password=(Path(private)/'password').read_text()
    if not 20<=len(password)<=128:raise SystemExit('Invalid native redaction binding;raw log output withheld.')
    values=json.loads((Path(private)/'redactions.json').read_text())
    if not isinstance(values,list) or password not in values or any(not isinstance(v,str) or not v for v in values):raise SystemExit('Private extended redaction bindings missing;raw logs withheld.')
    if sys.argv[1:]==['--progress']:
        # Native timeout/cancellation can leave an incomplete xcresult. Publish
        # bounded last progress from the already-redacted log,redacting again.
        source=Path('native-build.log')
        with source.open('rb') as stream:
            offset=max(0,source.stat().st_size-65536);stream.seek(offset);raw=stream.read().decode('utf-8',errors='replace')
        if offset:raw=raw.partition('\n')[2]
        for value in sorted(values,key=len,reverse=True):raw=raw.replace(value,'[PRIVATE CI INPUT REDACTED]')
        lines=raw.splitlines();started=[line for line in lines if line.startswith('Test Case ') and ' started' in line]
        tail=('Last started case: '+started[-1][:400]+'\n' if started else '')+'\n'.join(lines[-20:])[-1500:]
        encoded=tail.replace('%','%25').replace('\r','%0D').replace('\n','%0A')
        print('::notice title=Raphael bounded native progress::'+encoded[:4096])
        return
    with Path('native-build.log').open('w') as output:
        for line in sys.stdin:
            safe=line
            for value in sorted(values,key=len,reverse=True):safe=safe.replace(value,'[PRIVATE CI INPUT REDACTED]')
            output.write(safe);output.flush();sys.stdout.write(safe);sys.stdout.flush()


if __name__=='__main__':main()
