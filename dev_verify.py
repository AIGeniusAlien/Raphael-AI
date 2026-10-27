"""One-command,credential-isolated development verification:python -m dev_verify --all."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parent
SECRET_NAMES=('LLM_API_KEY','QDRANT_API_KEY','OPENAI_API_KEY','API_KEY','JWT_SECRET','JWT_SECRET_KEY')
SERVICE_NAMES=SECRET_NAMES+('LLM_BASE_URL','LLM_MODEL','QDRANT_URL','RAPHAEL_ALLOW_LOCAL_LLM')

def isolated_environment(source=None):
    env=dict(os.environ if source is None else source)
    for name in list(env):
        if name in SERVICE_NAMES or name.startswith('RAPHAEL_') or name in ('DATABASE_URL','EMBED_MODEL','EMBED_DIM','STATIC_PATH'):env.pop(name,None)
    env.update(PYTHONDONTWRITEBYTECODE='1',RAPHAEL_VERIFY_CLAIMS='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    return env

def counts(output):
    result={}
    unit=re.search(r'Ran (\d+) tests?',output)
    if unit:result['tests']=int(unit[1])
    for key in ('tests','pass','fail','skipped'):
        match=re.search(r'(?:ℹ|#) '+key+r' (\d+)',output)
        if match:result[key]=int(match[1])
    skips=re.search(r'skipped=(\d+)',output)
    if skips:result['skipped']=int(skips[1])
    return result

def execute(name,command,env,directory,timeout=600):
    started=time.monotonic();path=directory/(name+'.log')
    secrets=[os.environ.get(key,'') for key in SECRET_NAMES if len(os.environ.get(key,''))>8]
    with tempfile.TemporaryFile() as stream:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=os.name!='nt')
        timed_out=False
        try:process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out=True
            if os.name=='nt':subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            else:os.killpg(process.pid,signal.SIGKILL)
            process.wait()
        stream.seek(0);output=stream.read().decode('utf-8',errors='replace')
    for value in secrets:output=output.replace(value,'[REDACTED]')
    path.write_text(output,encoding='utf-8')
    return dict(name=name,status='failed' if process.returncode or timed_out else 'passed',exit_code=process.returncode,timeout=timed_out,duration_seconds=round(time.monotonic()-started,3),counts=counts(output),log=str(path))

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all',action='store_true',help='Run all locally supported checks;does not enable paid/live services.')
    parser.add_argument('--live',action='store_true',help='Explicitly run actual development OpenAI/Qdrant/Chromium checks using existing bindings.')
    parser.add_argument('--require-browser',action='store_true')
    parser.add_argument('--output',type=Path,help='New or existing results directory outside patient databases.')
    args=parser.parse_args(argv)
    directory=(args.output or Path(tempfile.mkdtemp(prefix='raphael-verification-'))).resolve();directory.mkdir(parents=True,exist_ok=True)
    env=isolated_environment();checks=[]
    env['RAPHAEL_VISUAL_OUTPUT']=str(directory/'browser-screenshots')
    def run(name,command,available=True,reason=None,required=True,timeout=600):
        if available:checks.append(execute(name,command,env,directory,timeout))
        else:checks.append(dict(name=name,status='failed' if required else 'skipped',reason=reason))
        print(checks[-1]['name']+': '+checks[-1]['status'],flush=True)
    run('python',[sys.executable,'-m','unittest','discover','-s','tests','-v'])
    run('frontend',['node','--test',*[str(p.relative_to(ROOT)) for p in sorted((ROOT/'tests').glob('*.test.cjs'))]],bool(shutil.which('node')),'Node is unavailable.')
    browser=bool(shutil.which('chromium') or os.getenv('CHROMIUM_EXECUTABLE'))
    run('chromium',[sys.executable,'-m','unittest','discover','-s','e2e','-v'],browser,'Install Playwright and Chromium;set CHROMIUM_EXECUTABLE.',args.require_browser)
    run('mobile-structure',[sys.executable,'mobile/scripts/project_audit.py'])
    run('mobile-tests',[sys.executable,'-m','unittest','discover','-s','mobile/tests','-v'])
    run('traceability',[sys.executable,'-m','scripts.build_engineering_ledger','--check'])
    run('safety-evaluation',[sys.executable,'-m','eval.safety_eval','--output',str(directory/'safety-evaluation.json')])
    native=platform.system()=='Darwin' and bool(shutil.which('xcodebuild')) and bool(os.getenv('RAPHAEL_SIMULATOR_DESTINATION'))
    run('ios-native',['bash','mobile/scripts/native_checks.sh'],native,'macOS,Xcode and RAPHAEL_SIMULATOR_DESTINATION required;no native build ran.',False,1800)
    if args.live:
        needed=('LLM_API_KEY','LLM_BASE_URL','LLM_MODEL','QDRANT_URL','QDRANT_API_KEY')
        if not all(os.getenv(k) for k in needed):checks.append(dict(name='live-services',status='failed',reason='One or more secure development provider bindings are absent;no request made.'))
        else:
            live_env=dict(os.environ,RAPHAEL_EMBEDDING_BACKEND='openai',RAPHAEL_STREAM_EVIDENCE_BACKEND='qdrant',RAPHAEL_VERIFY_CLAIMS='1',EMBED_DIM='384')
            checks.append(execute('live-services',[sys.executable,'-m','scripts.verify_live_medical','--live','--output',str(directory/'live-result.json')],live_env,directory,300))
    else:checks.append(dict(name='live-services',status='skipped',reason='Requires explicit --live;no provider credentials transmitted. Historical live evidence is not rerun evidence.'))
    checks.append(dict(name='dependency-advisory-audit',status='skipped',reason='Separate CI pip-audit gate queries advisory databases;no fresh audit implied by local tests.'))
    report=dict(schema_version=1,generated_at=datetime.now(timezone.utc).isoformat(),platform=platform.system(),python=platform.python_version(),status='failed' if any(c['status']=='failed' for c in checks) else 'passed',checks=checks,clinical_validation=False,production_data_used=False)
    (directory/'results.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':report['status'],'passed':sum(c['status']=='passed' for c in checks),'failed':sum(c['status']=='failed' for c in checks),'skipped':sum(c['status']=='skipped' for c in checks),'report':str(directory/'results.json')}))
    return int(report['status']=='failed')
if __name__=='__main__':raise SystemExit(main())
