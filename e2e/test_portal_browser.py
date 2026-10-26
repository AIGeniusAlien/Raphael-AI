"""Optional real-browser synthetic portal checks.

Install requirements-dev.txt, provide Chromium (CHROMIUM_EXECUTABLE overrides
/usr/bin/chromium), then run: python -m unittest discover -s e2e -v
No external inference, delivery, real accounts or production database is used.
"""

import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener

from playwright.sync_api import expect, sync_playwright
from sqlalchemy import create_engine

from dev_portal import create_dev_user, init_portal_schema


class PortalBrowserTests(unittest.TestCase):
    def test_two_role_portals_internal_messages_and_explicit_note_approval(self):
        root = Path(__file__).resolve().parents[1]
        password = secrets.token_urlsafe(24)
        with tempfile.TemporaryDirectory(prefix="raphael-browser-synthetic-") as temporary:
            database_url = "sqlite:///" + str(Path(temporary) / "synthetic.db")
            engine = create_engine(database_url)
            try:
                with patch.dict(os.environ, {"RAPHAEL_DEV_SYNTHETIC_ONLY": "1"}):
                    init_portal_schema(engine)
                    create_dev_user(engine, "synthetic", "physician", password, "clinician")
                    create_dev_user(engine, "synthetic", "patient", password, "patient")
            finally:
                engine.dispose()
            with socket.socket() as available:
                available.bind(("127.0.0.1", 0))
                port = available.getsockname()[1]
            base_url = f"http://127.0.0.1:{port}"
            environment = dict(os.environ, DATABASE_URL=database_url,
                RAPHAEL_DEV_PORTAL="1", RAPHAEL_DEV_SYNTHETIC_ONLY="1",
                LLM_BASE_URL="", LLM_API_KEY="", QDRANT_URL="", QDRANT_API_KEY="",
                STATIC_PATH="static", PYTHONDONTWRITEBYTECODE="1")
            with tempfile.TemporaryFile(mode="w+") as log:
                process = subprocess.Popen([sys.executable, "-m", "uvicorn", "main:app",
                    "--host", "127.0.0.1", "--port", str(port)], cwd=root, env=environment,
                    stdout=log, stderr=log, start_new_session=True)
                try:
                    opener = build_opener(ProxyHandler({}))
                    deadline = time.monotonic() + 45
                    while True:
                        if process.poll() is not None or time.monotonic() > deadline:
                            log.seek(0)
                            self.fail("Synthetic portal did not start:\n" + log.read())
                        try:
                            with opener.open(base_url + "/healthz", timeout=2) as response:
                                if response.status == 200:
                                    break
                        except (URLError, TimeoutError):
                            pass
                        time.sleep(0.1)
                    with sync_playwright() as playwright:
                        browser = playwright.chromium.launch(
                            executable_path=os.getenv("CHROMIUM_EXECUTABLE", "/usr/bin/chromium"),
                            headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
                        try:
                            physician_context = browser.new_context(base_url=base_url,
                                viewport={"width": 1280, "height": 900})
                            patient_context = browser.new_context(base_url=base_url,
                                viewport={"width": 390, "height": 844})
                            physician = physician_context.new_page()
                            patient = patient_context.new_page()
                            page_errors = []
                            physician.on("pageerror", lambda error: page_errors.append(str(error)))
                            patient.on("pageerror", lambda error: page_errors.append(str(error)))

                            def login(page, username):
                                page.goto("/")
                                expect(page.locator("#login-panel")).to_be_visible()
                                page.locator("#tenant").fill("synthetic")
                                page.locator("#username").fill(username)
                                page.locator("#password").fill(password)
                                page.locator("#login-form button").click()
                                expect(page.locator("#workspace")).to_be_visible()
                                expect(page.locator("#password")).to_have_value("")

                            self.assertEqual(patient_context.request.get("/api/dev/session").status, 401)
                            login(physician, "physician")
                            name = '<img src=x onerror="window.syntheticXss=true">'
                            thread = "Synthetic internal conversation"
                            physician.locator("#patient-name").fill(name)
                            physician.locator("#patient-account").fill("patient")
                            physician.locator("#patient-form button").click()
                            expect(physician.locator("#chart-heading")).to_have_text(name)
                            self.assertEqual(physician.locator("#patients img").count(), 0)
                            physician.locator("#conversation-title").fill(thread)
                            physician.locator("#conversation-form button").click()
                            expect(physician.locator("#thread-heading")).to_have_text(thread)
                            first_message = "Synthetic physician message <img src=x onerror=alert(1)>"
                            physician.locator("#message-content").fill(first_message)
                            physician.locator("#message-form button").click()
                            expect(physician.locator("#messages .message-content")).to_have_text([first_message])
                            expect(physician.locator("#status")).to_contain_text("No external message was delivered")
                            self.assertEqual(physician.locator("#messages img").count(), 0)

                            login(patient, "patient")
                            expect(patient.locator("#chart-heading")).to_have_text(name)
                            expect(patient.locator("#patient-form")).to_be_hidden()
                            expect(patient.locator("#assessment-panel")).to_be_hidden()
                            patient.get_by_role("button", name=thread, exact=True).click()
                            expect(patient.locator("#messages .message-content")).to_have_text([first_message])
                            second_message = "Synthetic patient reply"
                            patient.locator("#message-content").fill(second_message)
                            patient.locator("#message-form button").click()
                            expect(patient.locator("#messages .message-content")).to_have_text([first_message, second_message])
                            self.assertEqual(patient_context.request.post("/api/dev/patients",
                                data={"name": "Unauthorized synthetic record"}).status, 403)
                            self.assertEqual(physician_context.request.get("/api/patients").status, 403)
                            chart_response = patient_context.request.get("/api/dev/patients")
                            self.assertEqual(chart_response.headers.get("cache-control"), "no-store")
                            self.assertLessEqual(patient.evaluate("document.documentElement.scrollWidth"),
                                                 patient.evaluate("document.documentElement.clientWidth"))

                            physician.reload()
                            expect(physician.locator("#chart-heading")).to_have_text(name)
                            physician.get_by_role("button", name=thread, exact=True).click()
                            expect(physician.locator("#messages .message-content")).to_have_text([first_message, second_message])
                            physician.locator("#draft-note").click()
                            expect(physician.locator("#notes .note-content")).to_contain_text("Transcript-derived draft; not AI inference")
                            patient.reload()
                            expect(patient.locator("#notes")).to_contain_text("No clinician-approved notes")
                            self.assertEqual(patient.locator("#notes .note-content").count(), 0)
                            physician.get_by_role("button", name="Approve this note for the patient account", exact=True).click()
                            expect(physician.locator("#notes h4")).to_have_text("Clinician-approved note (test workflow)")
                            patient.reload()
                            expect(patient.locator("#notes .note-content")).to_contain_text(second_message)
                            expect(patient.locator("#notes h4")).to_have_text("Clinician-approved note (test workflow)")

                            physician.locator("#assessment-question").fill("Synthetic assessment question")
                            physician.locator("#assessment-form button").click()
                            expect(physician.locator("#status")).to_contain_text("Configure dedicated development LLM and evidence")
                            self.assertEqual(physician.locator("#assessment-output").text_content(), "")
                            expect(physician.locator("#messages .message-content")).to_have_text([first_message, second_message])
                            physician.locator("#logout").click()
                            expect(physician.locator("#workspace")).to_be_hidden()
                            self.assertEqual(physician_context.request.get("/api/dev/session").status, 401)
                            self.assertEqual(page_errors, [])
                            physician_context.close()
                            patient_context.close()
                        finally:
                            browser.close()
                finally:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait(timeout=5)
