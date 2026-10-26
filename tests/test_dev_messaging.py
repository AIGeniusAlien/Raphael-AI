"""Internal synthetic message workflow tests; no external transport is used."""
import io
import os
from pathlib import Path
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pypdf import PdfWriter
from sqlalchemy import create_engine, event, text

from dev_portal import create_dev_user, init_portal_schema, install_dev_portal
from dev_messaging import init_messaging_schema, install_messaging, LIMIT


class MessagingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="raphael-inbox-test-")
        self.addCleanup(self.directory.cleanup)
        flag = patch.dict(os.environ, {"RAPHAEL_DEV_SYNTHETIC_ONLY": "1"})
        flag.start(); self.addCleanup(flag.stop)
        self.engine = create_engine("sqlite:///" + str(Path(self.directory.name) / "synthetic.db"))
        self.addCleanup(self.engine.dispose)
        init_portal_schema(self.engine); init_messaging_schema(self.engine)
        self.password = "Synthetic-test-password"
        for tenant, name, role in (("alpha", "doctor", "clinician"), ("alpha", "doctor2", "clinician"),
                                   ("alpha", "patient", "patient"), ("alpha", "other", "patient"),
                                   ("beta", "outsider", "clinician")):
            create_dev_user(self.engine, tenant, name, self.password, role)
        self.app = FastAPI()
        ctx = install_dev_portal(self.app, self.engine)
        install_messaging(self.app, ctx)
        self.client = TestClient(self.app); self.addCleanup(self.client.close)
        self.login()
        self.patient = self.client.post("/api/dev/patients", json={"name": "Synthetic", "patient_username": "patient"}).json()
        self.thread = self.client.post(f"/api/dev/patients/{self.patient['id']}/conversations", json={"title": "Synthetic routine thread"}).json()
        self.cid = self.thread["id"]
        self.base = f"/api/dev/conversations/{self.cid}"

    def login(self, username="doctor", tenant="alpha", client=None):
        c = client or self.client
        r = c.post("/api/dev/login", json={"tenant": tenant, "username": username, "password": self.password})
        self.assertEqual(r.status_code, 200, r.text)
        return r

    def send(self, content="Synthetic internal message"):
        r = self.client.post(self.base + "/messages", json={"content": content})
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()

    def draft(self, content="Synthetic draft"):
        r = self.client.post(self.base + "/drafts", json={"content": content})
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()

    def test_inbox_receipts_and_unread_notifications(self):
        message = self.send()
        self.login("patient")
        box = self.client.get("/api/dev/inbox").json()
        self.assertEqual(box["unread_count"], 1)
        self.assertFalse(box["external_delivery_available"])
        r = self.client.post(self.base + "/read", json={"last_read_id": message["id"]})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertFalse(r.json()["external_delivery_confirmed"])
        self.assertEqual(self.client.get("/api/dev/inbox").json()["unread_count"], 0)
        self.send("Synthetic patient reply")
        self.login()
        thread = self.client.get(f"/api/dev/inbox/{self.cid}").json()
        self.assertTrue(thread["messages"][0]["read_by_recipient"])
        self.assertEqual(self.client.get("/api/dev/inbox").json()["unread_count"], 1)

    def test_draft_edit_approval_and_explicit_send_once(self):
        draft = self.draft()
        path = f"/api/dev/message-drafts/{draft['id']}"
        self.assertEqual(self.client.post(path + "/send", json={}).status_code, 409)
        r = self.client.patch(path, json={"content": "Synthetic revised draft", "due_at": "2000-01-01T00:00:00Z"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(self.client.get("/api/dev/inbox").json()["due_followups"], 1)
        self.assertEqual(self.client.post(path + "/approve", json={}).json()["status"], "approved")
        self.assertEqual(self.client.patch(path, json={"content": "Not allowed after approval"}).status_code, 409)
        r = self.client.post(path + "/send", json={})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["status"], "stored_in_portal")
        self.assertFalse(r.json()["external_delivery_confirmed"])
        self.assertEqual(self.client.post(path + "/send", json={}).status_code, 409)
        self.assertEqual(self.client.get("/api/dev/inbox").json()["due_followups"], 0)
        self.login("patient")
        thread = self.client.get(f"/api/dev/inbox/{self.cid}").json()
        self.assertEqual(thread["drafts"], [])
        self.assertEqual([m["content"] for m in thread["messages"]], ["Synthetic revised draft"])

    def test_concurrent_send_creates_one_internal_message(self):
        draft = self.draft(); path = f"/api/dev/message-drafts/{draft['id']}"
        self.client.post(path + "/approve", json={}).raise_for_status()
        clients = [TestClient(self.app), TestClient(self.app)]
        for c in clients:
            self.login(client=c); self.addCleanup(c.close)
        barrier = threading.Barrier(2)
        def send(c):
            barrier.wait(timeout=5)
            return c.post(path + "/send", json={}).status_code
        with ThreadPoolExecutor(max_workers=2) as pool:
            codes = list(pool.map(send, clients))
        self.assertEqual(sorted(codes), [200, 409])
        with self.engine.connect() as c:
            self.assertEqual(c.execute(text("SELECT count(*) FROM dev_messages")).scalar(), 1)
            self.assertEqual(c.execute(text("SELECT count(*) FROM dev_audit WHERE event='send_approved_internal_draft'")).scalar(), 1)

    def test_send_cannot_use_content_read_before_concurrent_edit_and_approval(self):
        draft = self.draft("Synthetic original unapproved content")
        path = f"/api/dev/message-drafts/{draft['id']}"
        sender, editor = TestClient(self.app), TestClient(self.app)
        for client in (sender, editor):
            self.login(client=client); self.addCleanup(client.close)
        paused, resume = threading.Event(), threading.Event()
        def pause_send(connection, cursor, statement, parameters, context, executemany):
            if statement.startswith("UPDATE dev_message_drafts SET status='sent'") and not paused.is_set():
                paused.set()
                if not resume.wait(timeout=5):
                    raise RuntimeError("Synthetic send-race coordination timed out")
        event.listen(self.engine, "before_cursor_execute", pause_send)
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                sending = pool.submit(lambda: sender.post(path + "/send", json={}).status_code)
                self.assertTrue(paused.wait(timeout=5))
                def revise():
                    r = editor.patch(path, json={"content": "Synthetic reviewed revision"})
                    self.assertEqual(r.status_code, 200, r.text)
                    return editor.post(path + "/approve", json={}).status_code
                editing = pool.submit(revise)
                # The sender owns a SQLite write lock before reading content.
                # The editor waits until send rejects this unapproved draft.
                resume.set()
                self.assertEqual(sending.result(timeout=10), 409)
                self.assertEqual(editing.result(timeout=10), 200)
        finally:
            resume.set()
            event.remove(self.engine, "before_cursor_execute", pause_send)
        self.assertEqual(self.client.post(path + "/send", json={}).status_code, 200)
        thread = self.client.get(f"/api/dev/inbox/{self.cid}").json()
        self.assertEqual([message["content"] for message in thread["messages"]], ["Synthetic reviewed revision"])

    def test_role_patient_thread_and_tenant_authorization(self):
        message = self.send(); draft = self.draft()
        for name, tenant in (("other", "alpha"), ("doctor2", "alpha"), ("outsider", "beta")):
            self.login(name, tenant)
            self.assertEqual(self.client.get("/api/dev/inbox").json()["threads"], [])
            self.assertEqual(self.client.get(f"/api/dev/inbox/{self.cid}").status_code, 404)
            for path, body in ((self.base + "/read", {"last_read_id": message["id"]}),
                               (self.base + "/drafts", {"content": "Synthetic"}),
                               (f"/api/dev/message-drafts/{draft['id']}/approve", {})):
                expected = 403 if name == "other" and "draft" in path else 404
                self.assertEqual(self.client.post(path, json=body).status_code, expected, path)
        self.login("patient")
        self.assertEqual(self.client.post(self.base + "/drafts", json={"content": "Synthetic"}).status_code, 403)
        self.assertEqual(self.client.post(f"/api/dev/message-drafts/{draft['id']}/send", json={}).status_code, 403)

    def test_invalid_receipt_and_routine_only_policy(self):
        self.assertEqual(self.client.post(self.base + "/read", json={"last_read_id": 99999}).status_code, 400)
        self.assertEqual(self.client.post(self.base + "/drafts", json={"content": "Synthetic", "urgency": "urgent"}).status_code, 422)
        self.assertEqual(self.client.post(self.base + "/drafts", json={"content": "Synthetic", "due_at": "2030-01-01"}).status_code, 422)
        self.assertEqual(self.client.post(self.base + "/drafts", json={"content": "Synthetic"}, headers={"Origin": "https://other.invalid"}).status_code, 403)

    def test_ai_drafting_failure_never_sends_or_fabricates_success(self):
        r = self.client.post(self.base + "/draft-assist", json={"instruction": "Synthetic instruction"})
        self.assertEqual(r.status_code, 503)
        self.assertEqual(self.client.get(f"/api/dev/inbox/{self.cid}").json()["messages"], [])

    def test_configured_draft_provider_only_creates_unapproved_draft(self):
        app = FastAPI(); ctx = install_dev_portal(app, self.engine)
        calls = []
        def provider(**kwargs):
            calls.append(kwargs)
            return "Synthetic provider fixture, not clinical advice"
        install_messaging(app, ctx, draft_generator=provider)
        with TestClient(app) as client:
            self.login(client=client)
            r = client.post(self.base + "/draft-assist", json={"instruction": "Synthetic test instruction"})
            self.assertEqual(r.status_code, 200, r.text)
            self.assertEqual(r.json()["status"], "draft")
            self.assertEqual(r.json()["source"], "configured_provider")
            self.assertEqual(client.get(f"/api/dev/inbox/{self.cid}").json()["messages"], [])
        self.assertEqual(calls[0]["instruction"], "Synthetic test instruction")

    def test_configured_draft_provider_errors_are_not_reflected_or_sent(self):
        for provider, expected in ((lambda **kwargs: 42, 502),
                                   (lambda **kwargs: (_ for _ in ()).throw(RuntimeError("synthetic-private-provider-error")), 503)):
            app = FastAPI(); ctx = install_dev_portal(app, self.engine)
            install_messaging(app, ctx, draft_generator=provider)
            with TestClient(app) as client:
                self.login(client=client)
                r = client.post(self.base + "/draft-assist", json={"instruction": "Synthetic"})
                self.assertEqual(r.status_code, expected)
                self.assertNotIn("synthetic-private-provider-error", r.text)
                self.assertEqual(client.get(f"/api/dev/inbox/{self.cid}").json()["messages"], [])

    def test_safe_attachment_storage_download_and_authorization(self):
        message = self.send()
        upload = f"/api/dev/messages/{message['id']}/attachments"
        r = self.client.post(upload, files={"file": ("../../synthetic.txt", b"<script>synthetic fixture</script>")})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["filename"], "synthetic.txt")
        attachment_id = r.json()["id"]
        path = f"/api/dev/message-attachments/{attachment_id}/content"
        self.login("patient")
        r = self.client.get(path)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.content, b"<script>synthetic fixture</script>")
        self.assertIn("attachment", r.headers["content-disposition"])
        self.assertEqual(r.headers["x-content-type-options"], "nosniff")
        self.assertEqual(r.headers["cache-control"], "no-store")
        self.assertEqual(self.client.post(upload, files={"file": ("other.txt", b"Synthetic")}).status_code, 403)
        self.login("other")
        self.assertEqual(self.client.get(path).status_code, 404)
        self.login("outsider", "beta")
        self.assertEqual(self.client.get(path).status_code, 404)

    def test_attachment_limits_type_validation_and_pdf(self):
        message = self.send(); path = f"/api/dev/messages/{message['id']}/attachments"
        for name, data, code in (("large.txt", b"x" * (LIMIT + 1), 413), ("fake.pdf", b"not a pdf", 400),
                                  ("bad.txt", b"\xff\xfe", 400), ("bad.html", b"Synthetic", 415), ("empty.txt", b"", 400)):
            self.assertEqual(self.client.post(path, files={"file": (name, data)}).status_code, code, name)
        writer = PdfWriter(); writer.add_blank_page(width=72, height=72); buffer = io.BytesIO(); writer.write(buffer)
        r = self.client.post(path, files={"file": ("synthetic.pdf", buffer.getvalue())})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(self.client.post(path, files={"file": ("synthetic.txt", b"Synthetic")}, headers={"Origin": "https://other.invalid"}).status_code, 403)

    def test_attachment_envelope_is_bounded_before_multipart_parser(self):
        message = self.send(); path = f"/api/dev/messages/{message['id']}/attachments"
        r = self.client.post(path, content=b"ignored", headers={"Content-Type": "multipart/form-data; boundary=synthetic", "Content-Length": str(LIMIT + 65537)})
        self.assertEqual(r.status_code, 413)
        def chunks():
            for _ in range(35):
                yield b"x" * 65536
        r = self.client.post(path, content=chunks(), headers={"Content-Type": "multipart/form-data; boundary=synthetic"})
        self.assertEqual(r.status_code, 413)
        with self.engine.connect() as c:
            self.assertEqual(c.execute(text("SELECT count(*) FROM dev_message_attachments")).scalar(), 0)

    def test_reinitialization_preserves_threads_drafts_receipts_and_attachments(self):
        message = self.send(); draft = self.draft()
        self.client.post(self.base + "/read", json={"last_read_id": message["id"]}).raise_for_status()
        self.client.post(f"/api/dev/messages/{message['id']}/attachments", files={"file": ("synthetic.txt", b"Synthetic")}).raise_for_status()
        self.engine.dispose(); init_messaging_schema(self.engine)
        thread = self.client.get(f"/api/dev/inbox/{self.cid}").json()
        self.assertEqual(thread["drafts"][0]["id"], draft["id"])
        self.assertEqual(len(thread["messages"][0]["attachments"]), 1)
        self.assertEqual(self.client.get("/api/dev/inbox").json()["unread_count"], 0)
