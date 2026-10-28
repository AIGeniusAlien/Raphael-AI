"""Actual synthetic backend with bounded,no-local-variable startup stack capture."""
import argparse
import faulthandler
import os
import json
import re
from pathlib import Path
import signal
import sys


def main():
    if os.getenv('GITHUB_ACTIONS')!='true' or os.getenv('RAPHAEL_DEV_SYNTHETIC_ONLY')!='1':
        raise SystemExit('Restricted to synthetic GitHub Actions backend diagnostics.')
    parser=argparse.ArgumentParser();parser.add_argument('--key',required=True);parser.add_argument('--cert',required=True);args=parser.parse_args()
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    faulthandler.enable()
    faulthandler.dump_traceback_later(25,repeat=False)
    signal.signal(signal.SIGUSR1,lambda *_:faulthandler.cancel_dump_traceback_later())
    print('Synthetic CI backend interpreter started;loading actual main:app.',flush=True)
    import uvicorn
    from main import app
    @app.middleware('http')
    async def trace_authentication(request, call_next):
        response=await call_next(request)
        if re.fullmatch(r'/api/dev/(?:login|session|logout|patients(?:/\d+(?:/notes)?)?)',request.url.path):
            # Fixed route/status metadata only: no queries,headers,cookies or bodies.
            path=re.sub(r'/\d+', '/:id', request.url.path)
            print('RAPHAEL_CI_HTTP '+json.dumps({'method':request.method,'path':path,'status':response.status_code}),flush=True)
        return response
    uvicorn.run(app,host='127.0.0.1',port=8443,ssl_keyfile=args.key,ssl_certfile=args.cert,access_log=False)


if __name__=='__main__':main()
