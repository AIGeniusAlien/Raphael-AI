"""Cross-platform synthetic startup:python -m dev_start [--setup] [--provision]."""
import argparse
import importlib.util
import os
import platform
from pathlib import Path
import subprocess
import sys
import tempfile
import venv

ROOT=Path(__file__).resolve().parent

def runtime_environment(database,live=False):
    from dev_verify import isolated_environment
    env=dict(os.environ) if live else isolated_environment()
    if os.environ.get('RAPHAEL_MFA_ENCRYPTION_KEY'):
        env['RAPHAEL_MFA_ENCRYPTION_KEY']=os.environ['RAPHAEL_MFA_ENCRYPTION_KEY']
    env.update(DATABASE_URL=database,RAPHAEL_DEV_PORTAL='1',RAPHAEL_DEV_SYNTHETIC_ONLY='1')
    if live:env.update(RAPHAEL_EMBEDDING_BACKEND='openai',RAPHAEL_STREAM_EVIDENCE_BACKEND='qdrant',RAPHAEL_VERIFY_CLAIMS='1',EMBED_DIM='384')
    return env

def runtime_requirements():
    # Same lightweight dependency set exercised by actual native backend CI.
    excluded=('sentence-transformers==','transformers==','filelock==','setuptools==')
    return '\n'.join(line for line in (ROOT/'requirements.txt').read_text().splitlines() if not line.startswith(excluded))+'\n'

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--setup',action='store_true');p.add_argument('--without-local-ml',action='store_true',help='Install the hosted/disabled-AI runtime without optional Torch/transformers;automatic on Intel macOS.');p.add_argument('--provision',action='store_true');p.add_argument('--live-services',action='store_true');p.add_argument('--database',type=Path);p.add_argument('--port',type=int,default=8000);args=p.parse_args()
    if not 1<=args.port<=65535:p.error('Port must be1to65535.')
    if sys.version_info[:2]!=(3,12):p.error('Use Python3.12 for the tested dependency set.')
    if args.setup:
        target=ROOT/'.venv'
        if not target.exists():venv.EnvBuilder(with_pip=True).create(target)
        python=target/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
        lightweight=args.without_local_ml or (sys.platform=='darwin' and platform.machine().lower() in ('x86_64','amd64'))
        if lightweight:
            runtime=target/'runtime-requirements.txt';runtime.write_text(runtime_requirements())
            installs=[[str(python),'-m','pip','install','-r',str(runtime),'-c','requirements.lock']]
        else:
            torch=['torch==2.6.0']
            if sys.platform!='darwin':torch+=['--index-url','https://download.pytorch.org/whl/cpu']
            dependencies=['--no-deps','-r','requirements.lock'] if sys.platform.startswith('linux') else ['-r','requirements.txt','-c','requirements.lock']
            installs=[[str(python),'-m','pip','install',*torch],[str(python),'-m','pip','install',*dependencies]]
        installs += [[str(python),'-m','pip','install','-r','requirements-dev.txt'],[str(python),'-m','pip','check']]
        for command in installs:subprocess.run(command,cwd=ROOT,check=True)
        remaining=[a for a in sys.argv[1:] if a!='--setup']
        return subprocess.call([str(python),'-m','dev_start',*remaining],cwd=ROOT)
    if importlib.util.find_spec('sqlalchemy') is None:p.error('Dependencies missing.Run python -m dev_start --setup --provision.')
    database=args.database.resolve() if args.database else Path(tempfile.mkdtemp(prefix='raphael-synthetic-'))/'portal.db'
    database.parent.mkdir(parents=True,exist_ok=True)
    url='sqlite:///'+database.as_posix();env=runtime_environment(url,args.live_services)
    if args.live_services and not all(env.get(k) for k in ('LLM_API_KEY','LLM_BASE_URL','LLM_MODEL','QDRANT_URL','QDRANT_API_KEY')):p.error('Live mode requires all five secure provider bindings;none are supplied by this launcher.')
    subprocess.run([sys.executable,'-m','dev_runtime','migrate','--database-url',url],cwd=ROOT,env=env,check=True)
    if args.provision:
        for username,role in (('clinician','clinician'),('patient','patient')):subprocess.run([sys.executable,'-m','auth.dev_bootstrap','--database-url',url,'--tenant','synthetic','--username',username,'--role',role],cwd=ROOT,env=env,check=True)
    print('Synthetic database: '+str(database),flush=True)
    print('Reuse --database with this exact path to restart.No default passwords.No public deployment.',flush=True)
    print('Provider mode: '+('actual configured development services' if args.live_services else 'disabled;AI safely unavailable unless reviewed local evidence/provider configured'),flush=True)
    return subprocess.call([sys.executable,'-m','uvicorn','main:app','--host','127.0.0.1','--port',str(args.port)],cwd=ROOT,env=env)
if __name__=='__main__':raise SystemExit(main())
