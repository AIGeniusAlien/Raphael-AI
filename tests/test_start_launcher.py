"""Actual cross-platform launcher path exercised on the current host,not a native-OS claim."""
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import httpx
from dev_verify import isolated_environment

class LauncherTests(unittest.TestCase):
    def test_synthetic_launcher_health_security_headers_and_restart_without_credentials(self):
        with tempfile.TemporaryDirectory(prefix='raphael-launcher-') as directory:
            db=Path(directory)/'synthetic.db';env=isolated_environment()
            for attempt in range(2):
                with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
                with tempfile.TemporaryFile() as log:
                    process=subprocess.Popen([sys.executable,'-m','dev_start','--database',str(db),'--port',str(port)],env=env,stdout=log,stderr=log,start_new_session=os.name!='nt')
                    try:
                        with httpx.Client(base_url=f'http://127.0.0.1:{port}',trust_env=False,timeout=5) as client:
                            deadline=time.monotonic()+35
                            while True:
                                if process.poll() is not None or time.monotonic()>deadline:self.fail('Synthetic launcher failed readiness.')
                                try:
                                    if client.get('/healthz').status_code==200:break
                                except httpx.TransportError:pass
                                time.sleep(.05)
                            self.assertEqual(client.get('/readyz').json()['scope'],'synthetic-development')
                            denied=client.get('/api/patients');self.assertEqual(denied.status_code,403)
                            self.assertEqual(denied.headers['referrer-policy'],'no-referrer');self.assertEqual(denied.headers['x-frame-options'],'DENY');self.assertEqual(denied.headers['cache-control'],'no-store')
                            self.assertNotIn('strict-transport-security',denied.headers)
                            self.assertEqual(client.get('/api/dev/session').status_code,401)
                            self.assertEqual(client.get('/api/dev/session',headers={'sec-fetch-site':'cross-site'}).status_code,403)
                    finally:
                        if os.name=='nt':subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                        else:os.killpg(process.pid,signal.SIGTERM)
                        process.wait(timeout=10)
            self.assertTrue(db.is_file())
