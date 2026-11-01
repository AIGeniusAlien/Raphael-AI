"""Authenticated fault control for disposable native CI only,never installed in main.

No arbitrary SQL,accounts,paths,command execution,or application auth bypass.
The fixed expiry operation affects only the dedicated synthetic expiry account.
"""
import asyncio
from datetime import datetime,timezone
import hmac
import json
import os
from pathlib import Path
import sqlite3
from starlette.responses import JSONResponse


class FixtureControl:
    def __init__(self,app,private):
        self.private=Path(private)
        if os.getenv('GITHUB_ACTIONS')!='true' or os.getenv('RAPHAEL_DEV_SYNTHETIC_ONLY')!='1' or os.getenv('DATABASE_URL')!='sqlite:///'+str(self.private/'synthetic.db'):
            raise RuntimeError('Disposable synthetic CI database required.')
        self.token=(self.private/'control-token').read_text()
        if not 32<=len(self.token)<=128:raise RuntimeError('Private CI control binding required.')
        self.app=app;self.readiness='healthy'

    async def __call__(self,scope,receive,send):
        if scope['type']=='http' and scope.get('path')=='/_ci/native-control':
            headers=dict(scope.get('headers',[]));supplied=headers.get(b'x-raphael-ci-control',b'')
            if scope.get('method')!='POST' or not hmac.compare_digest(supplied,self.token.encode()):
                return await JSONResponse({'detail':'Not found.'},404)(scope,receive,send)
            body=bytearray()
            while True:
                event=await receive()
                if event['type']!='http.request':return
                body.extend(event.get('body',b''))
                if len(body)>256:return await JSONResponse({'detail':'Bounded control body required.'},413)(scope,receive,send)
                if not event.get('more_body'):break
            try:
                value=json.loads(body)
                if not isinstance(value,dict) or set(value)!={'operation'}:raise ValueError()
                operation=value['operation']
                if operation in ('healthy','unavailable','slow'):self.readiness=operation
                elif operation=='expire_fixture_session':
                    with sqlite3.connect(self.private/'synthetic.db') as c:
                        c.execute("UPDATE dev_sessions SET expires_at='2000-01-01T00:00:00+00:00' WHERE user_id IN (SELECT id FROM dev_users WHERE tenant='native-ci' AND username='native-expiry-patient')")
                else:raise ValueError()
            except (ValueError,TypeError):return await JSONResponse({'detail':'Unsupported fixture operation.'},422)(scope,receive,send)
            return await JSONResponse({'applied':True,'scope':'disposable_synthetic_fixture'})(scope,receive,send)
        if scope['type']=='http' and scope.get('path') in ('/readyz','/api/dev/session'):
            if self.readiness=='unavailable':return await JSONResponse({'status':'not_ready','scope':'synthetic-development'},503)(scope,receive,send)
            if self.readiness=='slow':await asyncio.sleep(16)
        await self.app(scope,receive,send)
