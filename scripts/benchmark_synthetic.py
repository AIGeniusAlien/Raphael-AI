"""Bounded real HTTP/SQLite concurrency probe; never a production capacity claim."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json,os,secrets,signal,socket,subprocess,sys,tempfile,time
from pathlib import Path
import httpx
from sqlalchemy import create_engine
from dev_start import runtime_environment
from dev_runtime import migrate
from dev_portal import create_dev_user


def run(operations,workers):
    with tempfile.TemporaryDirectory(prefix='raphael-concurrency-') as directory:
        database='sqlite:///'+directory+'/synthetic.sqlite';password=secrets.token_urlsafe(25)
        previous=os.environ.get('RAPHAEL_DEV_SYNTHETIC_ONLY');os.environ['RAPHAEL_DEV_SYNTHETIC_ONLY']='1'
        engine=create_engine(database)
        try:migrate(engine);create_dev_user(engine,'load-synthetic','clinician',password,'clinician')
        finally:
            engine.dispose()
            if previous is None:os.environ.pop('RAPHAEL_DEV_SYNTHETIC_ONLY',None)
            else:os.environ['RAPHAEL_DEV_SYNTHETIC_ONLY']=previous
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        base=f'http://127.0.0.1:{port}';env=runtime_environment(database)
        with tempfile.TemporaryFile() as log:
            process=subprocess.Popen([sys.executable,'-m','uvicorn','main:app','--host','127.0.0.1','--port',str(port)],env=env,stdout=log,stderr=log,start_new_session=True)
            try:
                with httpx.Client(base_url=base,trust_env=False,timeout=10) as client:
                    deadline=time.monotonic()+45
                    while True:
                        if process.poll() is not None or time.monotonic()>deadline:raise RuntimeError('Synthetic server readiness failed.')
                        try:
                            if client.get('/readyz').status_code==200:break
                        except httpx.TransportError:pass
                        time.sleep(.1)
                    response=client.post('/api/dev/login',json={'tenant':'load-synthetic','username':'clinician','password':password});response.raise_for_status();cookies=dict(client.cookies)
                    def operation(index):
                        started=time.perf_counter()
                        with httpx.Client(base_url=base,cookies=cookies,trust_env=False,timeout=10) as worker:
                            response=worker.post('/api/dev/patients',json={'name':'Synthetic concurrency record '+str(index)}) if index%2==0 else worker.get('/api/dev/patients')
                        return response.status_code,(time.perf_counter()-started)*1000
                    start=time.perf_counter()
                    with ThreadPoolExecutor(max_workers=workers) as executor:results=list(executor.map(operation,range(operations)))
                    elapsed=time.perf_counter()-start;times=sorted(result[1] for result in results);statuses=Counter(str(result[0]) for result in results)
                    count=len(client.get('/api/dev/patients').json());expected=(operations+1)//2
                    result={'scope':'single local uvicorn process;actual HTTP and SQLite;synthetic only','operations':operations,'workers':workers,'status_counts':dict(statuses),'duration_seconds':round(elapsed,3),'p50_ms':round(times[len(times)//2],3),'p95_ms':round(times[min(len(times)-1,int(len(times)*.95))],3),'expected_records':expected,'persisted_records':count,'provider_requests':0,'production_capacity_validated':False,'status':'passed' if statuses=={'200':operations} and count==expected else 'failed'}
                    client.post('/api/dev/logout',json={}).raise_for_status();return result
            finally:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--operations',type=int,default=40);parser.add_argument('--workers',type=int,default=4);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if not 1<=args.operations<=100 or not 1<=args.workers<=8:parser.error('Use 1–100 operations and 1–8 workers.')
    try:result=run(args.operations,args.workers)
    except Exception as error:result={'status':'failed','error_type':type(error).__name__,'production_data_used':False}
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));return result['status']!='passed'
if __name__=='__main__':raise SystemExit(main())
