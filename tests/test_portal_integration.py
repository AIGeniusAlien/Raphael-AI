"""Full main:app integration over HTTP with a private synthetic-only database."""
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import httpx
from sqlalchemy import create_engine
from dev_portal import create_dev_user, init_portal_schema


class PortalIntegrationTests(unittest.TestCase):
    def test_main_portal_auth_isolation_and_explicit_unavailable_inference(self):
        with tempfile.TemporaryDirectory(prefix="raphael-portal-http-") as directory:
            url = "sqlite:///" + directory + "/synthetic.db"
            engine = create_engine(url)
            try:
                with patch.dict(os.environ, {"RAPHAEL_DEV_SYNTHETIC_ONLY": "1"}):
                    init_portal_schema(engine)
                    create_dev_user(engine, "synthetic", "clinician", "Synthetic-test-password", "clinician")
            finally:
                engine.dispose()
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            env = dict(os.environ, DATABASE_URL=url, RAPHAEL_DEV_PORTAL="1", RAPHAEL_DEV_SYNTHETIC_ONLY="1",
                       QDRANT_URL="", LLM_BASE_URL="", LLM_API_KEY="", QDRANT_API_KEY="", STATIC_PATH="static",
                       PYTHONDONTWRITEBYTECODE="1", PORT=str(port), XDG_STATE_HOME=directory,
                       PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"])
            with tempfile.TemporaryFile(mode="w+") as log:
                process = subprocess.Popen(["bash", "run_local.sh"], cwd=Path(__file__).resolve().parents[1],
                                           env=env, stdout=log, stderr=log, start_new_session=True)
                try:
                    with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=2, trust_env=False) as client:
                        deadline = time.monotonic() + 45
                        while True:
                            if process.poll() is not None or time.monotonic() > deadline:
                                log.seek(0)
                                self.fail("Portal did not start:\n" + log.read())
                            try:
                                if client.get("/healthz").status_code == 200:
                                    break
                            except httpx.TransportError:
                                pass
                            time.sleep(0.1)
                        self.assertEqual(client.get("/").headers["location"], "/static/portal.html")
                        self.assertEqual(client.get("/static/portal.html").status_code, 200)
                        for path in ("/api/patients", "/api/conversations", "/api/chat", "/api/evidence/ingest"):
                            self.assertEqual(client.get(path).status_code, 403, path)
                        self.assertEqual(client.get("/api/dev/session").status_code, 401)
                        password = "Synthetic-test-password"
                        invalid = client.post("/api/dev/login", json={"tenant": "synthetic", "username": "clinician", "password": password * 20})
                        self.assertEqual(invalid.status_code, 422)
                        self.assertNotIn(password, invalid.text)
                        r = client.post("/api/dev/login", json={"tenant": "synthetic", "username": "clinician", "password": password})
                        self.assertEqual(r.status_code, 200, r.text)
                        patient = client.post("/api/dev/patients", json={"name": "Synthetic Test Patient"}).json()
                        conversation = client.post(f"/api/dev/patients/{patient['id']}/conversations", json={"title": "Synthetic"}).json()
                        r = client.post(f"/api/dev/patients/{patient['id']}/assessment", json={"conversation_id": conversation["id"], "question": "Synthetic"})
                        self.assertEqual(r.status_code, 503)
                        self.assertIn("dedicated development", r.json()["detail"])
                        r = client.get("/api/dev/patients", headers={"Origin": "https://other.invalid"})
                        self.assertNotIn("access-control-allow-origin", r.headers)
                        self.assertEqual(r.headers["cache-control"], "no-store")
                        self.assertIn("frame-ancestors 'none'", r.headers["content-security-policy"])
                        self.assertEqual(client.post("/api/dev/logout", json={}).status_code, 200)
                        self.assertEqual(client.get("/api/dev/patients").status_code, 401)
                finally:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=5)
