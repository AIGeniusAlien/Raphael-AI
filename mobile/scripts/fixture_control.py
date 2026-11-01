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
        if scope['type']=='http' and scope.get('path') in ('/readyz','/api/dev/session','/api/dev/account/export'):
            if self.readiness=='unavailable':return await JSONResponse({'status':'not_ready','scope':'synthetic-development'},503)(scope,receive,send)
            if self.readiness=='slow':await asyncio.sleep(16)
        if scope['type']=='http' and scope.get('path') in ('/api/dev/account/mfa/enroll','/api/dev/account/mfa/confirm'):
            # Register dynamic test-only secrets BEFORE exposing the actual
            # response to UIKit/XCTest. Never log or return registry contents.
            messages=[];size=0
            async def hold(message):
                nonlocal size
                messages.append(message)
                size+=len(message.get('body',b''))
                if size>65536:raise ValueError('Bounded private fixture response required.')
            await self.app(scope,receive,hold)
            try:
                if messages[0].get('status')==200:
                    value=json.loads(b''.join(m.get('body',b'') for m in messages))
                    private_values=([value['secret']] if 'secret' in value else value.get('recovery_codes',[]))
                    if not private_values or any(not isinstance(v,str) or not 6<=len(v)<=128 for v in private_values):raise ValueError()
                    registry=self.private/'redactions.json'
                    values=json.loads(registry.read_text())
                    if not isinstance(values,list):raise ValueError()
                    updated=self.private/'redactions.next'
                    updated.write_text(json.dumps(list(dict.fromkeys(values+private_values))));updated.chmod(0o600);updated.replace(registry)
            except (ValueError,KeyError,TypeError,OSError):
                return await JSONResponse({'detail':'Private fixture response withheld.'},503)(scope,receive,send)
            for message in messages:await send(message)
            return
        await self.app(scope,receive,send)
