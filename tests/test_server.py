"""Real HTTP startup/restart through the supported local entrypoint."""

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


class LiveServerTests(unittest.TestCase):
    def test_local_entrypoint_startup_and_persistence_over_http(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix="raphael-http-test-") as directory:
            env = dict(os.environ, DATABASE_URL="sqlite:///" + directory + "/synthetic.db",
                       QDRANT_URL="", QDRANT_API_KEY="", LLM_BASE_URL="", LLM_API_KEY="",
                       STATIC_PATH="static", PYTHONDONTWRITEBYTECODE="1",
                       RAPHAEL_DEV_PORTAL="0",
                       XDG_STATE_HOME=directory, PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"])
            for iteration in range(2):
                with self.subTest(start=iteration + 1), socket.socket() as sock:
                    sock.bind(("127.0.0.1", 0))
                    port = sock.getsockname()[1]
                env["PORT"] = str(port)
                with tempfile.TemporaryFile(mode="w+") as log:
                    process = subprocess.Popen(["bash", "run_local.sh"], cwd=root, env=env,
                                               stdout=log, stderr=log, start_new_session=True)
                    try:
                        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=2, trust_env=False) as client:
                            deadline = time.monotonic() + 45
                            while True:
                                if process.poll() is not None or time.monotonic() > deadline:
                                    log.seek(0)
                                    self.fail("Server did not start:\n" + log.read())
                                try:
                                    r = client.get("/healthz")
                                    if r.status_code == 200:
                                        break
                                except httpx.TransportError:
                                    pass
                                time.sleep(0.1)
                            self.assertEqual(r.json()["status"], "ok")
                            self.assertEqual(client.get("/").headers["location"], "/static/Raphael.html")
                            self.assertIn('/static/app.js', client.get("/static/Raphael.html").text)
                            self.assertEqual(client.get("/static/app.js").status_code, 200)
                            if iteration == 0:
                                r = client.post("/api/patients", json={"name": "Synthetic HTTP Patient"})
                                r.raise_for_status()
                                patient = r.json()
                                r = client.post("/api/conversations", json={"patient_id": patient["id"], "title": "HTTP test"})
                                r.raise_for_status()
                                conversation = r.json()
                            else:
                                self.assertEqual(client.get("/api/patients").json(), [patient])
                                self.assertEqual(client.get("/api/conversations", params={"patient_id": patient["id"]}).json(), [conversation])
                    finally:
                        # Stop only this test's reload supervisor and child process group.
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait(timeout=5)
