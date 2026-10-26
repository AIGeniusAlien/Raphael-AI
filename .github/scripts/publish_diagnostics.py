"""Publish bounded CI failure evidence without requiring artifact-download access."""
import json
import os
from pathlib import Path
import sys

def escape(value):return value.replace('%','%25').replace('\r','%0D').replace('\n','%0A')
def redact(value):
    for key in ('LLM_API_KEY','QDRANT_API_KEY','OPENAI_API_KEY','GITHUB_TOKEN'):
        secret=os.getenv(key,'')
        if len(secret)>8:value=value.replace(secret,'[REDACTED]')
    return value

def main():
    path=Path(sys.argv[1]);messages=[]
    if path.is_dir():
        report=path/'results.json'
        if report.exists():
            for check in json.loads(report.read_text()).get('checks',[]):
                if check['status']=='failed':
                    log=Path(check.get('log',''))
                    messages.append(check['name']+': '+(log.read_text(errors='replace')[-4500:] if log.is_file() else check.get('reason','failed')))
        else:messages.append('Verification stopped before a machine report was written.')
    elif path.is_file():
        lines=path.read_text(errors='replace').splitlines()
        errors=[line for line in lines if 'error:' in line.lower() or 'failed' in line.lower() or 'Testing failed' in line]
        messages.append('\n'.join(errors[-30:])[-7000:] if errors else '\n'.join(lines[-45:])[-7000:])
    else:messages.append('No diagnostic log was produced.')
    for message in messages:print('::error title=Raphael failure diagnostics::'+escape(redact(message)))
if __name__=='__main__':main()
