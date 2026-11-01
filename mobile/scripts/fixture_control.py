"""Authenticated fault control for disposable native CI only,never installed in main.

No arbitrary SQL,accounts,paths,commands,or application auth bypass.
The fixed expiry operation affects only the dedicated synthetic expiry account.
"""
import asyncio
from datetime import datetime,timezone
import hmac
import json
import os
import platform
import re
import subprocess
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

    def verify_export_clipboard(self):
        """Observe only the selected disposable simulator;never return its bytes.

        Reading another app's pasteboard synchronously from XCTest can block on
        an OS paste-permission prompt. The host observes the real clipboard via
        simctl instead. This does not deliver data to or authenticate the app.
        """
        destination=os.getenv('RAPHAEL_SIMULATOR_DESTINATION','')
        match=re.fullmatch(r'platform=iOS Simulator,id=([0-9A-Fa-f-]{36})',destination)
        if platform.system()!='Darwin' or not match:raise ValueError()
        output=subprocess.run(['/usr/bin/xcrun','simctl','pbpaste',match[1]],capture_output=True,timeout=3,check=True).stdout
        if not 0<len(output)<=5*1024*1024:raise ValueError()
        value=json.loads(output)
        if not isinstance(value,dict) or not isinstance(value.get('account'),dict) or not isinstance(value.get('patients'),list) or any(not isinstance(row,dict) for row in value['patients']):raise ValueError()
        with sqlite3.connect(self.private/'synthetic.db') as c:
            owner=c.execute("SELECT id FROM dev_users WHERE tenant='native-ci' AND username='native-patient' AND role='patient'").fetchone()
            if not owner:raise ValueError()
            ids=[row[0] for row in c.execute("SELECT id FROM dev_patients WHERE tenant='native-ci' AND patient_user_id=? ORDER BY id",owner)]
        account=value.get('account',{})
        checks={
            'format_matches':value.get('format')=='raphael-synthetic-patient-export-v1',
            'own_account_matches':account.get('id')==owner[0] and account.get('username')=='native-patient' and account.get('tenant')=='native-ci' and account.get('role')=='patient',
            'own_records_match':len(ids)==1 and [row.get('id') for row in value.get('patients',[])]==ids and value['patients'][0].get('name')=='Synthetic native patient',
            'no_unlinked_record':b'Synthetic native unlinked record' not in output,
            'no_fixture_password':(self.private/'password').read_bytes() not in output,
            'no_password_hash':b'password_hash' not in output,
        }
        if not all(checks.values()):raise ValueError()
        return checks

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
                elif operation=='verify_export_clipboard':
                    try:checks=await asyncio.to_thread(self.verify_export_clipboard)
                    except (ValueError,TypeError,KeyError,OSError,subprocess.SubprocessError):
                        return await JSONResponse({'detail':'Synthetic export clipboard verification failed;no clipboard data disclosed.'},422)(scope,receive,send)
                    return await JSONResponse({'scope':'disposable_synthetic_fixture','checks':checks})(scope,receive,send)
                elif operation=='expire_fixture_session':
                    with sqlite3.connect(self.private/'synthetic.db') as c:
                        c.execute("UPDATE dev_sessions SET expires_at='2000-01-01T00:00:00+00:00' WHERE user_id IN (SELECT id FROM dev_users WHERE tenant='native-ci' AND username='native-expiry-patient')")
                else:raise ValueError()
            except (ValueError,TypeError):return await JSONResponse({'detail':'Unsupported fixture operation.'},422)(scope,receive,send)
            return await JSONResponse({'applied':True,'scope':'disposable_synthetic_fixture'})(scope,receive,send)
        if scope['type']=='http' and scope.get('path') in ('/readyz','/api/dev/session','/api/dev/account/export'):
            if self.readiness=='unavailable':return await JSONResponse({'status':'not_ready','scope':'synthetic-development'},503)(scope,receive,send)
            if self.readiness=='slow':await asyncio.sleep(16)
        await self.app(scope,receive,send)
