"""Real-browser synthetic admin, inbox and DICOM development workflows.

No clinical findings, external message delivery, real accounts or live-provider
success is asserted. Uses the installed Chromium and optional Playwright.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from test_dev_imaging import synthetic_dicom


class PlatformBrowserTests(unittest.TestCase):
    def test_internal_reply_approval_admin_and_actual_dicom_view(self):
        root = Path(__file__).resolve().parents[1]
        password = secrets.token_urlsafe(24)
        with tempfile.TemporaryDirectory(prefix="raphael-platform-browser-") as temporary:
            database_url = "sqlite:///" + str(Path(temporary) / "synthetic.db")
            engine = create_engine(database_url)
            try:
                with patch.dict(os.environ, {"RAPHAEL_DEV_SYNTHETIC_ONLY": "1"}):
                    init_portal_schema(engine)
                    for username, role in (("physician", "clinician"), ("patient", "patient"), ("administrator", "org_admin")):
                        create_dev_user(engine, "synthetic", username, password, role)
            finally:
                engine.dispose()
            with socket.socket() as available:
                available.bind(("127.0.0.1", 0)); port = available.getsockname()[1]
            base = f"http://127.0.0.1:{port}"
            environment = dict(os.environ, DATABASE_URL=database_url, RAPHAEL_DEV_PORTAL="1",
                RAPHAEL_DEV_SYNTHETIC_ONLY="1", LLM_BASE_URL="", LLM_API_KEY="", QDRANT_URL="",
                QDRANT_API_KEY="", FHIR_SANDBOX_BASE_URL="", STATIC_PATH="static", PYTHONDONTWRITEBYTECODE="1")
            with tempfile.TemporaryFile(mode="w+") as log:
                process = subprocess.Popen([sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", str(port)],
                    cwd=root, env=environment, stdout=log, stderr=log, start_new_session=True)
                try:
                    opener = build_opener(ProxyHandler({})); deadline = time.monotonic() + 45
                    while True:
                        if process.poll() is not None or time.monotonic() > deadline:
                            log.seek(0); self.fail("Platform did not start:\n" + log.read())
                        try:
                            with opener.open(base + "/healthz", timeout=2) as response:
                                if response.status == 200: break
                        except (URLError, TimeoutError): pass
                        time.sleep(0.1)
                    with sync_playwright() as playwright:
                        browser = playwright.chromium.launch(executable_path=os.getenv("CHROMIUM_EXECUTABLE", "/usr/bin/chromium"),
                            headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
                        try:
                            doctor_context = browser.new_context(base_url=base, viewport={"width":1280,"height":900})
                            patient_context = browser.new_context(base_url=base, viewport={"width":390,"height":844})
                            admin_context = browser.new_context(base_url=base)
                            doctor, patient, admin = doctor_context.new_page(), patient_context.new_page(), admin_context.new_page()
                            errors = []
                            for page in (doctor,patient,admin): page.on("pageerror", lambda error: errors.append(str(error)))
                            def login(context, username):
                                r = context.request.post('/api/dev/login', data={"tenant":"synthetic","username":username,"password":password})
                                self.assertEqual(r.status, 200)
                            for context, name in ((doctor_context,"physician"),(patient_context,"patient"),(admin_context,"administrator")): login(context,name)
                            # Administrator manages own-tenant accounts without medical access.
                            admin.goto('/static/admin.html')
                            expect(admin.locator('#workspace')).to_be_visible()
                            admin.locator('#username').fill('synthetic-new-user')
                            admin.locator('#password').fill(secrets.token_urlsafe(24))
                            admin.locator('#role').select_option('patient')
                            admin.locator('#create-user button').click()
                            expect(admin.locator('#status')).to_contain_text('Synthetic account created')
                            expect(admin.locator('#password')).to_have_value('')
                            account = admin.locator('#users article').filter(has=admin.get_by_role('heading',name='synthetic-new-user',exact=True))
                            expect(account).to_contain_text('Active')
                            account.get_by_role('button',name='Disable account and revoke sessions').click()
                            expect(account).to_contain_text('Disabled')
                            self.assertEqual(admin_context.request.get('/api/dev/patients').status,403)
                            self.assertEqual(patient_context.request.get('/api/dev/admin/users').status,403)
                            # Create synthetic chart/thread through the real existing physician interface.
                            doctor.goto('/')
                            expect(doctor.locator('#workspace')).to_be_visible()
                            doctor.locator('#patient-name').fill('Synthetic platform fixture')
                            doctor.locator('#patient-account').fill('patient')
                            doctor.locator('#patient-form button').click()
                            expect(doctor.locator('#chart-heading')).to_have_text('Synthetic platform fixture')
                            doctor.locator('#conversation-title').fill('Synthetic response approval thread')
                            doctor.locator('#conversation-form button').click()
                            expect(doctor.locator('#thread-heading')).to_have_text('Synthetic response approval thread')
                            charts = doctor_context.request.get('/api/dev/patients').json()
                            patient_id = charts[0]['id']
                            conversations = doctor_context.request.get(f'/api/dev/patients/{patient_id}/conversations').json()
                            conversation_id = conversations[0]['id']
                            # Patient A sends, physician B drafts/edits/approves/separately sends.
                            patient.goto('/static/inbox.html')
                            patient.get_by_role('button',name='Synthetic response approval thread',exact=False).click()
                            patient.locator('#message').fill('Synthetic routine patient question <script>not executed</script>')
                            patient.locator('#message-form button').click()
                            expect(patient.locator('#messages li').first).to_contain_text('Synthetic routine patient question')
                            self.assertEqual(patient.locator('#messages script').count(),0)
                            doctor.goto('/static/inbox.html')
                            expect(doctor.locator('#counts')).to_contain_text('1 unread internal messages')
                            doctor.get_by_role('button',name='Synthetic response approval thread',exact=False).click()
                            doctor.locator('#draft-content').fill('Synthetic unapproved response')
                            doctor.locator('#draft-form button[type=submit]').click()
                            expect(doctor.locator('#drafts article')).to_contain_text('Synthetic unapproved response')
                            doctor.get_by_role('button',name='Edit draft',exact=True).click()
                            doctor.locator('#draft-content').fill('Synthetic edited and approved response')
                            doctor.locator('#draft-form button[type=submit]').click()
                            expect(doctor.locator('#drafts article')).to_contain_text('Synthetic edited and approved response')
                            self.assertEqual(len(patient_context.request.get(f'/api/dev/inbox/{conversation_id}').json()['messages']),1)
                            doctor.get_by_role('button',name='Approve draft (does not send)',exact=True).click()
                            expect(doctor.get_by_role('button',name='Send approved draft internally',exact=True)).to_be_visible()
                            self.assertEqual(len(patient_context.request.get(f'/api/dev/inbox/{conversation_id}').json()['messages']),1)
                            doctor.get_by_role('button',name='Send approved draft internally',exact=True).click()
                            expect(doctor.locator('#status')).to_contain_text('No external message was delivered')
                            expect(doctor.locator('#drafts article')).to_contain_text('Already stored internally')
                            draft_id=doctor_context.request.get(f'/api/dev/inbox/{conversation_id}').json()['drafts'][0]['id']
                            self.assertEqual(doctor_context.request.post(f'/api/dev/message-drafts/{draft_id}/send',data={}).status,409)
                            patient.reload()
                            patient.get_by_role('button',name='Synthetic response approval thread',exact=False).click()
                            expect(patient.locator('#messages')).to_contain_text('Synthetic edited and approved response')
                            # Real DICOM decode/pixel preview, not a generated diagnostic image.
                            doctor.goto(f'/static/imaging.html?patient={patient_id}')
                            expect(doctor.locator('#upload-form')).to_be_visible()
                            doctor.locator('#dicom-file').set_input_files({"name":"synthetic.dcm","mimeType":"application/dicom","buffer":synthetic_dicom()})
                            doctor.locator('#upload-form button').click()
                            expect(doctor.locator('#studies button')).to_have_count(1)
                            doctor.locator('#studies button').click()
                            expect(doctor.locator('#metadata')).to_contain_text('"columns": 3')
                            doctor.wait_for_function("() => document.querySelector('#current-image').complete && document.querySelector('#current-image').naturalWidth===3")
                            doctor.locator('#findings').fill('Synthetic manually entered finding. No image inference.')
                            doctor.locator('#impression').fill('Synthetic fixture report, not a clinical interpretation.')
                            doctor.locator('#report-form button').click()
                            expect(doctor.locator('#reports')).to_contain_text('Draft — clinician review required')
                            patient.goto(f'/static/imaging.html?patient={patient_id}')
                            patient.locator('#studies button').click()
                            expect(patient.locator('#reports')).to_contain_text('No visible reports')
                            doctor.get_by_role('button',name='Approve report for patient visibility',exact=True).click()
                            expect(doctor.locator('#reports')).to_contain_text('Clinician-approved report')
                            patient.reload();patient.locator('#studies button').click()
                            expect(patient.locator('#reports')).to_contain_text('Synthetic manually entered finding')
                            patient.wait_for_function("() => document.querySelector('#current-image').complete && document.querySelector('#current-image').naturalWidth===3")
                            self.assertLessEqual(patient.evaluate('document.documentElement.scrollWidth'),patient.evaluate('document.documentElement.clientWidth'))
                            self.assertEqual(errors,[])
                            doctor_context.close();patient_context.close();admin_context.close()
                        finally: browser.close()
                finally:
                    if process.poll() is None:
                        os.killpg(process.pid,signal.SIGTERM)
                        try: process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
