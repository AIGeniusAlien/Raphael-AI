"""Synthetic portal regression coverage. No network, real accounts or real patient data."""

import hashlib
from concurrent.futures import ThreadPoolExecutor
import threading
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, text

from dev_portal import COOKIE_NAME, create_dev_user, init_portal_schema, install_dev_portal


class DevPortalTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="raphael-portal-test-")
        self.addCleanup(self.directory.cleanup)
        self.engine = create_engine("sqlite:///" + str(Path(self.directory.name) / "synthetic.db"))
        self.addCleanup(self.engine.dispose)
        flag = patch.dict(os.environ, {"RAPHAEL_DEV_SYNTHETIC_ONLY": "1"})
        flag.start()
        self.addCleanup(flag.stop)
        init_portal_schema(self.engine)
        self.password = "Synthetic-only-test-pass"
        for tenant, name, role in (("alpha", "clinician", "clinician"), ("alpha", "second-clinician", "clinician"),
                                   ("alpha", "patient", "patient"), ("alpha", "other-patient", "patient"),
                                   ("beta", "outsider", "clinician")):
            create_dev_user(self.engine, tenant, name, self.password, role)
        self.app = FastAPI()
        self.assessment = Mock(return_value={"confidence": "Synthetic test only", "ranked_differential": []})
        install_dev_portal(self.app, self.engine, assessment_generator=self.assessment)
        self.client = TestClient(self.app)
        self.addCleanup(self.client.close)

    def login(self, username="clinician", tenant="alpha", client=None):
        client = client or self.client
        r = client.post("/api/dev/login", json={"tenant": tenant, "username": username, "password": self.password})
        self.assertEqual(r.status_code, 200, r.text)
        return r

    def patient(self, username="patient"):
        r = self.client.post("/api/dev/patients", json={"name": "Synthetic Patient", "patient_username": username})
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()

    def conversation(self, patient):
        r = self.client.post(f"/api/dev/patients/{patient['id']}/conversations", json={"title": "Synthetic thread"})
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()

    def test_requires_explicit_synthetic_opt_in_and_sqlite(self):
        with patch.dict(os.environ, {"RAPHAEL_DEV_SYNTHETIC_ONLY": "0"}):
            with self.assertRaisesRegex(RuntimeError, "SYNTHETIC_ONLY"):
                install_dev_portal(FastAPI(), self.engine)
            with self.assertRaises(RuntimeError):
                init_portal_schema(self.engine)
        wrong_engine = Mock()
        wrong_engine.dialect.name = "postgresql"
        with self.assertRaisesRegex(RuntimeError, "SQLite"):
            init_portal_schema(wrong_engine)

    def test_populated_legacy_tables_are_rejected_before_portal_schema_writes(self):
        for table in ("patients", "conversations", "messages", "documents", "feedback"):
            with self.subTest(table=table):
                occupied = create_engine("sqlite:///" + str(Path(self.directory.name) / (table + "-occupied.db")))
                try:
                    with occupied.begin() as c:
                        c.exec_driver_sql(f'CREATE TABLE "{table}" (id INTEGER PRIMARY KEY)')
                        c.exec_driver_sql(f'INSERT INTO "{table}" (id) VALUES (1)')
                    with self.assertRaisesRegex(RuntimeError, "legacy records"):
                        init_portal_schema(occupied)
                    with self.assertRaisesRegex(RuntimeError, "legacy records"):
                        install_dev_portal(FastAPI(), occupied)
                    with occupied.connect() as c:
                        self.assertEqual(c.exec_driver_sql(f'SELECT COUNT(*) FROM "{table}"').scalar(), 1)
                        self.assertEqual(c.exec_driver_sql("SELECT COUNT(*) FROM sqlite_master WHERE name='dev_users'").scalar(), 0)
                finally:
                    occupied.dispose()

    def test_empty_legacy_tables_do_not_block_synthetic_schema(self):
        empty = create_engine("sqlite:///" + str(Path(self.directory.name) / "empty-legacy.db"))
        try:
            with empty.begin() as c:
                c.exec_driver_sql("CREATE TABLE patients(id INTEGER PRIMARY KEY)")
            init_portal_schema(empty)
            with empty.connect() as c:
                self.assertEqual(c.exec_driver_sql("SELECT COUNT(*) FROM sqlite_master WHERE name='dev_users'").scalar(), 1)
                self.assertEqual(c.exec_driver_sql("SELECT COUNT(*) FROM patients").scalar(), 0)
        finally:
            empty.dispose()

    def test_password_hashing_validation_and_no_default_users(self):
        with self.engine.connect() as c:
            stored = c.execute(text("SELECT password_hash FROM dev_users WHERE username='patient'")).scalar()
        self.assertNotIn(self.password, stored)
        for password in ("short", "x" * 129):
            with self.assertRaises(ValueError):
                create_dev_user(self.engine, "alpha", "invalid", password, "patient")
        with self.assertRaises(ValueError):
            create_dev_user(self.engine, "alpha", "invalid", self.password, "admin")
        for name in ("unknown", "patient"):
            r = self.client.post("/api/dev/login", json={"tenant": "alpha", "username": name, "password": "wrong"})
            self.assertEqual(r.status_code, 401)
            self.assertEqual(r.json()["detail"], "Invalid development credentials.")

    def test_cookie_session_expiry_and_logout_revocation(self):
        self.assertEqual(self.client.get("/api/dev/session").status_code, 401)
        r = self.login()
        self.assertEqual(set(r.json()), {"id", "tenant", "username", "role"})
        self.assertIn("HttpOnly", r.headers["set-cookie"])
        self.assertIn("SameSite=strict", r.headers["set-cookie"])
        token = self.client.cookies.get(COOKIE_NAME)
        with self.engine.connect() as c:
            stored = c.execute(text("SELECT token_hash FROM dev_sessions")).scalar()
        self.assertEqual(stored, hashlib.sha256(token.encode()).hexdigest())
        self.assertNotEqual(stored, token)
        self.assertEqual(self.client.get("/api/dev/session").json()["role"], "clinician")
        self.assertEqual(self.client.post("/api/dev/logout", json={}).status_code, 200)
        self.client.cookies.set(COOKIE_NAME, token)
        self.assertEqual(self.client.get("/api/dev/session").status_code, 401)
        self.client.cookies.clear()
        self.login()
        with self.engine.begin() as c:
            c.execute(text("UPDATE dev_sessions SET expires_at='2000-01-01T00:00:00+00:00'"))
        self.assertEqual(self.client.get("/api/dev/session").status_code, 401)

    def test_same_origin_json_protection_including_login(self):
        data = {"tenant": "alpha", "username": "clinician", "password": self.password}
        self.assertEqual(self.client.post("/api/dev/login", json=data, headers={"Origin": "https://untrusted.invalid"}).status_code, 403)
        self.assertEqual(self.client.post("/api/dev/login", data=data).status_code, 415)
        self.assertEqual(self.client.post("/api/dev/login", json=data, headers={"Origin": "null"}).status_code, 403)
        self.assertEqual(self.client.post("/api/dev/login", json=data, headers={"Origin": "http://testserver"}).status_code, 200)
        self.assertEqual(self.client.post("/api/dev/patients", json={"name": "Synthetic"}, headers={"Origin": "https://untrusted.invalid"}).status_code, 403)

    def test_patient_role_cannot_create_records_or_read_other_patient(self):
        self.login()
        first = self.patient()
        second = self.patient("other-patient")
        self.login("patient")
        self.assertEqual([p["id"] for p in self.client.get("/api/dev/patients").json()], [first["id"]])
        self.assertEqual(self.client.get(f"/api/dev/patients/{first['id']}").status_code, 200)
        self.assertEqual(self.client.get(f"/api/dev/patients/{second['id']}").status_code, 404)
        self.assertEqual(self.client.post("/api/dev/patients", json={"name": "Synthetic"}).status_code, 403)
        self.assertEqual(self.client.post(f"/api/dev/patients/{first['id']}/conversations", json={}).status_code, 403)

    def test_cross_tenant_access_is_denied_for_every_resource(self):
        self.login()
        p = self.patient()
        conv = self.conversation(p)
        note = self.client.post(f"/api/dev/patients/{p['id']}/notes", json={"content": "Synthetic draft"}).json()
        self.login("outsider", "beta")
        self.assertEqual(self.client.get("/api/dev/patients").json(), [])
        for path in (f"/patients/{p['id']}", f"/patients/{p['id']}/conversations", f"/patients/{p['id']}/notes",
                     f"/conversations/{conv['id']}/messages"):
            self.assertEqual(self.client.get("/api/dev" + path).status_code, 404, path)
        for path, body in ((f"/patients/{p['id']}/conversations", {}),
                           (f"/patients/{p['id']}/notes", {"content": "Synthetic"}),
                           (f"/patients/{p['id']}/note-draft", {"conversation_id": conv['id']}),
                           (f"/conversations/{conv['id']}/messages", {"content": "Synthetic"}),
                           (f"/notes/{note['id']}/approve", {}),
                           (f"/patients/{p['id']}/assessment", {"conversation_id": conv['id'], "question": "Synthetic"})):
            self.assertEqual(self.client.post("/api/dev" + path, json=body).status_code, 404, path)

    def test_internal_two_way_messaging_requires_thread_participation(self):
        self.login()
        p = self.patient()
        conv = self.conversation(p)
        path = f"/api/dev/conversations/{conv['id']}/messages"
        r = self.client.post(path, json={"content": "Synthetic clinician message"})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["delivery_status"], "stored_in_portal")
        self.login("patient")
        self.assertEqual(self.client.get(path).json()[0]["sender_role"], "clinician")
        r = self.client.post(path, json={"content": "Synthetic patient reply"})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["sender_role"], "patient")
        self.assertEqual(self.client.post(path, json={"content": "Synthetic", "recipient": "external@example.invalid"}).status_code, 422)
        self.login("other-patient")
        self.assertEqual(self.client.get(path).status_code, 404)
        self.assertEqual(self.client.post(path, json={"content": "Synthetic"}).status_code, 404)
        self.login("second-clinician")
        self.assertEqual(self.client.get(path).status_code, 404)
        self.assertEqual(self.client.post(path, json={"content": "Synthetic"}).status_code, 404)
        self.login()
        self.assertEqual([m["sender_role"] for m in self.client.get(path).json()], ["clinician", "patient"])

    def test_note_drafting_explicit_approval_and_patient_visibility(self):
        self.login()
        p = self.patient()
        conv = self.conversation(p)
        self.client.post(f"/api/dev/conversations/{conv['id']}/messages", json={"content": "Synthetic transcript"})
        draft = self.client.post(f"/api/dev/patients/{p['id']}/note-draft", json={"conversation_id": conv['id']})
        self.assertEqual(draft.status_code, 200, draft.text)
        note = draft.json()
        self.assertIn("not AI inference", note["content"])
        self.assertIn("Synthetic transcript", note["content"])
        self.assertEqual(note["status"], "draft")
        self.login("patient")
        self.assertEqual(self.client.get(f"/api/dev/patients/{p['id']}/notes").json(), [])
        self.assertEqual(self.client.post(f"/api/dev/notes/{note['id']}/approve", json={}).status_code, 403)
        self.assertEqual(self.client.post(f"/api/dev/patients/{p['id']}/notes", json={"content": "Synthetic"}).status_code, 403)
        self.login()
        r = self.client.post(f"/api/dev/notes/{note['id']}/approve", json={})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertIsNotNone(r.json()["approved_by"])
        self.assertEqual(self.client.post(f"/api/dev/notes/{note['id']}/approve", json={}).status_code, 409)
        self.login("patient")
        notes = self.client.get(f"/api/dev/patients/{p['id']}/notes").json()
        self.assertEqual(notes[0]["status"], "approved")

    def test_concurrent_note_approval_has_one_winner_and_one_audit_event(self):
        self.login()
        p = self.patient()
        note = self.client.post(f"/api/dev/patients/{p['id']}/notes", json={"content": "Synthetic draft"}).json()
        # Collaboration needs the patient's explicit full-chart authorization.
        self.login("patient")
        self.client.post(f"/api/dev/patients/{p['id']}/chart-access",json={"clinician_username":"second-clinician","expires_in_minutes":60,"purpose":"Synthetic concurrent review","authorize_full_chart":True}).raise_for_status()
        self.login()
        second = TestClient(self.app)
        self.addCleanup(second.close)
        self.login("second-clinician", client=second)
        barrier = threading.Barrier(2)
        lock = threading.Lock()
        reads = 0

        def hold_initial_reads(connection, cursor, statement, parameters, context, executemany):
            nonlocal reads
            if statement.startswith("SELECT n.*,u.username AS author_username"):
                with lock:
                    reads += 1
                    first_two = reads <= 2
                if first_two:
                    barrier.wait(timeout=10)

        event.listen(self.engine, "after_cursor_execute", hold_initial_reads)
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(client.post, f"/api/dev/notes/{note['id']}/approve", json={})
                           for client in (self.client, second)]
                responses = [future.result(timeout=20) for future in futures]
        finally:
            event.remove(self.engine, "after_cursor_execute", hold_initial_reads)
        self.assertEqual(sorted(r.status_code for r in responses), [200, 409])
        winner = next(r.json()["approved_by"] for r in responses if r.status_code == 200)
        with self.engine.connect() as c:
            self.assertEqual(c.execute(text("SELECT approved_by FROM dev_notes WHERE id=:id"), {"id": note['id']}).scalar(), winner)
            self.assertEqual(c.execute(text("SELECT COUNT(*) FROM dev_audit WHERE event='approve_note' AND resource_id=:id"),
                                       {"id": note['id']}).scalar(), 1)

    def test_assessment_is_clinician_scoped_and_not_sent_or_approved(self):
        self.login()
        p = self.patient()
        conv = self.conversation(p)
        self.client.post(f"/api/dev/conversations/{conv['id']}/messages", json={"content": "Synthetic context"})
        path = f"/api/dev/patients/{p['id']}/assessment"
        body = {"conversation_id": conv['id'], "question": "Synthetic question"}
        r = self.client.post(path, json=body)
        self.assertEqual(r.status_code, 200, r.text)
        self.assessment.assert_called_once_with(case_context="clinician: Synthetic context", question="Synthetic question", mode="general", tenant="alpha")
        self.assertEqual(len(self.client.get(f"/api/dev/conversations/{conv['id']}/messages").json()), 1)
        self.assertEqual(self.client.get(f"/api/dev/patients/{p['id']}/notes").json(), [])
        self.login("patient")
        self.assertEqual(self.client.post(path, json=body).status_code, 403)

    def test_repeated_initialization_preserves_accounts_records_messages_and_audit(self):
        self.login()
        p = self.patient()
        conv = self.conversation(p)
        path = f"/api/dev/conversations/{conv['id']}/messages"
        self.client.post(path, json={"content": "Synthetic persistent"})
        init_portal_schema(self.engine)
        self.engine.dispose()
        self.assertEqual(self.client.get(path).json()[0]["content"], "Synthetic persistent")
        with self.engine.connect() as c:
            events = c.execute(text("SELECT event FROM dev_audit ORDER BY id")).scalars().all()
            serialized = str(c.execute(text("SELECT * FROM dev_audit")).all())
        self.assertIn("store_internal_message", events)
        self.assertNotIn("Synthetic persistent", serialized)
        self.assertNotIn(self.password, serialized)

    def test_bootstrap_cli_password_stdin_and_duplicate_preservation(self):
        url = "sqlite:///" + str(Path(self.directory.name) / "bootstrap-synthetic.db")
        command = [sys.executable, "-m", "auth.dev_bootstrap", "--database-url", url, "--tenant", "synthetic",
                   "--username", "test-user", "--role", "patient", "--password-stdin"]
        first = subprocess.run(command, input=self.password + "\n", text=True, capture_output=True)
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertNotIn(self.password, first.stdout + first.stderr)
        duplicate = subprocess.run(command, input=self.password + "\n", text=True, capture_output=True)
        self.assertEqual(duplicate.returncode, 2)
        self.assertIn("not overwritten", duplicate.stderr)
