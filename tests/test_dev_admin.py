"""Tenant-scoped local administration, account status and throttle regressions."""
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text, event

from dev_portal import (COOKIE_NAME, LOGIN_MAX_FAILURES, create_dev_user,
                        init_portal_schema, install_dev_portal)


class DevAdministrationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="raphael-admin-test-")
        self.addCleanup(self.directory.cleanup)
        self.engine = create_engine("sqlite:///" + str(Path(self.directory.name) / "synthetic.db"))
        self.addCleanup(self.engine.dispose)
        flag = patch.dict(os.environ, {"RAPHAEL_DEV_SYNTHETIC_ONLY": "1"})
        flag.start()
        self.addCleanup(flag.stop)
        init_portal_schema(self.engine)
        self.password = "Synthetic-account-test-only"
        self.users = {}
        for tenant, username, role in [("alpha", "admin", "org_admin"),
                                       ("alpha", "clinician", "clinician"),
                                       ("alpha", "patient", "patient"),
                                       ("beta", "other-admin", "org_admin")]:
            self.users[username] = create_dev_user(self.engine, tenant, username, self.password, role)
        self.app = FastAPI()
        self.context = install_dev_portal(self.app, self.engine)
        self.client = TestClient(self.app)
        self.addCleanup(self.client.close)

    def login(self, username="admin", tenant="alpha", password=None, client=None):
        return (client or self.client).post("/api/dev/login", json={
            "tenant": tenant, "username": username,
            "password": self.password if password is None else password})

    def test_context_and_tenant_scoped_user_creation_without_secret_disclosure(self):
        self.assertIs(self.context.engine, self.engine)
        self.assertTrue(callable(self.context.user))
        self.assertEqual(self.login().status_code, 200)
        response = self.client.post("/api/dev/admin/users", json={
            "username": "new-patient", "role": "patient", "password": self.password})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.json()["active"])
        self.assertNotIn(self.password, response.text)
        self.assertNotIn("password_hash", response.json())
        listing = self.client.get("/api/dev/admin/users").json()
        self.assertEqual({user["tenant"] for user in listing}, {"alpha"})
        self.assertNotIn(self.password, str(listing))
        self.assertEqual(self.client.post("/api/dev/admin/users", json={
            "username": "new-patient", "role": "patient", "password": self.password}).status_code, 409)
        self.assertEqual(self.client.post("/api/dev/admin/users", json={
            "username": "wrong", "role": "platform_admin", "password": self.password}).status_code, 422)

    def test_nonadmins_and_cross_tenant_mutations_are_denied(self):
        self.assertEqual(self.client.get("/api/dev/admin/users").status_code, 401)
        for name in ("clinician", "patient"):
            self.assertEqual(self.login(name).status_code, 200)
            self.assertEqual(self.client.get("/api/dev/admin/users").status_code, 403)
            self.assertEqual(self.client.post("/api/dev/admin/users", json={
                "username": "escalate", "role": "org_admin", "password": self.password}).status_code, 403)
            self.assertEqual(self.client.patch(f"/api/dev/admin/users/{self.users['admin']['id']}", json={"active": False}).status_code, 403)
        self.login()
        self.assertEqual(self.client.patch(f"/api/dev/admin/users/{self.users['other-admin']['id']}", json={"active": False}).status_code, 404)

    def test_admin_has_no_clinical_record_or_message_privilege(self):
        self.login("clinician")
        patient = self.client.post("/api/dev/patients", json={"name": "Synthetic"}).json()
        conversation = self.client.post(f"/api/dev/patients/{patient['id']}/conversations", json={}).json()
        note = self.client.post(f"/api/dev/patients/{patient['id']}/notes", json={"content": "Synthetic"}).json()
        self.login()
        for path in ("/patients", f"/patients/{patient['id']}", f"/patients/{patient['id']}/conversations",
                     f"/patients/{patient['id']}/notes", f"/conversations/{conversation['id']}/messages"):
            self.assertEqual(self.client.get("/api/dev" + path).status_code, 403, path)
        self.assertEqual(self.client.post(f"/api/dev/notes/{note['id']}/approve", json={}).status_code, 403)

    def test_deactivation_revokes_sessions_and_reactivation_needs_new_login(self):
        patient_client = TestClient(self.app)
        self.addCleanup(patient_client.close)
        self.assertEqual(self.login("patient", client=patient_client).status_code, 200)
        old_token = patient_client.cookies.get(COOKIE_NAME)
        self.login()
        user_id = self.users["patient"]["id"]
        self.assertEqual(self.client.patch(f"/api/dev/admin/users/{user_id}", json={"active": False}).status_code, 200)
        self.assertEqual(patient_client.get("/api/dev/session").status_code, 401)
        self.assertEqual(self.login("patient", client=patient_client).status_code, 401)
        self.login("clinician")
        self.assertEqual(self.client.post("/api/dev/patients", json={"name": "Synthetic", "patient_username": "patient"}).status_code, 400)
        self.login()
        with self.engine.connect() as c:
            self.assertEqual(c.execute(text("SELECT count(*) FROM dev_sessions WHERE user_id=:u"), {"u": user_id}).scalar(), 0)
        self.assertEqual(self.client.patch(f"/api/dev/admin/users/{user_id}", json={"active": True}).status_code, 200)
        patient_client.cookies.set(COOKIE_NAME, old_token)
        self.assertEqual(patient_client.get("/api/dev/session").status_code, 401)
        self.assertEqual(self.login("patient", client=patient_client).status_code, 200)

    def test_self_deactivation_and_untyped_status_are_rejected(self):
        self.login()
        self.assertEqual(self.client.patch(f"/api/dev/admin/users/{self.users['admin']['id']}", json={"active": False}).status_code, 409)
        self.assertEqual(self.client.patch(f"/api/dev/admin/users/{self.users['patient']['id']}", json={"active": "false"}).status_code, 422)
        self.assertEqual(self.client.get("/api/dev/session").status_code, 200)

    def test_admin_creation_and_status_mutations_require_same_origin_json(self):
        self.login()
        body = {"username": "new", "role": "patient", "password": self.password}
        self.assertEqual(self.client.post("/api/dev/admin/users", json=body, headers={"Origin": "https://wrong.invalid"}).status_code, 403)
        self.assertEqual(self.client.patch(f"/api/dev/admin/users/{self.users['patient']['id']}", data={"active": "false"}).status_code, 415)

    def test_throttle_persists_for_unknown_and_known_accounts_and_expires(self):
        for username in ("patient", "unknown"):
            for _ in range(LOGIN_MAX_FAILURES):
                self.assertEqual(self.login(username, password="incorrect").status_code, 401)
            r = self.login(username)
            self.assertEqual(r.status_code, 429)
            self.assertIn("retry-after", r.headers)
        self.engine.dispose()
        self.assertEqual(self.login("patient").status_code, 429)
        with self.engine.begin() as c:
            c.execute(text("UPDATE dev_login_attempts SET reset_at='2000-01-01T00:00:00+00:00'"))
        self.assertEqual(self.login("patient").status_code, 200)
        with self.engine.connect() as c:
            self.assertEqual(c.execute(text("SELECT count(*) FROM dev_login_attempts")).scalar(), 0)

    def test_successful_login_resets_failed_counter_and_stores_no_username(self):
        self.assertEqual(self.login("patient", password="wrong").status_code, 401)
        with self.engine.connect() as c:
            row = dict(c.execute(text("SELECT * FROM dev_login_attempts")).mappings().one())
        self.assertEqual(set(row), {"key_hash", "failures", "reset_at"})
        self.assertEqual(len(row["key_hash"]), 64)
        self.assertEqual(self.login("patient").status_code, 200)
        with self.engine.connect() as c:
            self.assertEqual(c.execute(text("SELECT count(*) FROM dev_login_attempts")).scalar(), 0)


class UserSchemaMigrationTests(unittest.TestCase):
    def test_concurrent_original_schema_startups_upgrade_once_without_losing_users(self):
        with tempfile.TemporaryDirectory(prefix="raphael-migration-test-") as directory, \
             patch.dict(os.environ, {"RAPHAEL_DEV_SYNTHETIC_ONLY": "1"}):
            engine = create_engine("sqlite:///" + str(Path(directory) / "synthetic.db"))
            barrier, seen = threading.Barrier(2), threading.local()
            try:
                with engine.begin() as c:
                    c.exec_driver_sql("CREATE TABLE dev_users(id INTEGER PRIMARY KEY AUTOINCREMENT,tenant TEXT NOT NULL,username TEXT NOT NULL,role TEXT NOT NULL CHECK(role IN ('patient','clinician')),password_hash TEXT NOT NULL, UNIQUE(tenant,username))")
                    c.exec_driver_sql("INSERT INTO dev_users VALUES (7,'alpha','original','clinician','unchanged-hash')")

                def synchronize_old_schema(conn, cursor, statement, parameters, context, executemany):
                    if statement.startswith("SELECT sql FROM sqlite_master WHERE name='dev_users'") and not getattr(seen, "first", False):
                        seen.first = True
                        barrier.wait(timeout=5)

                event.listen(engine, "after_cursor_execute", synchronize_old_schema)
                with ThreadPoolExecutor(max_workers=2) as pool:
                    results = [pool.submit(init_portal_schema, engine) for _ in range(2)]
                    for result in results:
                        result.result(timeout=10)
                event.remove(engine, "after_cursor_execute", synchronize_old_schema)
                with engine.connect() as c:
                    row = c.exec_driver_sql("SELECT id,password_hash,active FROM dev_users").one()
                    self.assertEqual(tuple(row), (7, "unchanged-hash", 1))
                    self.assertEqual(c.exec_driver_sql("SELECT count(*) FROM dev_schema_migrations WHERE version=2").scalar(), 1)
            finally:
                engine.dispose()

    def test_failed_foreign_key_validation_rolls_back_user_rebuild(self):
        with tempfile.TemporaryDirectory(prefix="raphael-migration-test-"), \
             patch.dict(os.environ, {"RAPHAEL_DEV_SYNTHETIC_ONLY": "1"}):
            engine = create_engine("sqlite://")
            try:
                with engine.begin() as c:
                    c.exec_driver_sql("CREATE TABLE dev_users(id INTEGER PRIMARY KEY AUTOINCREMENT,tenant TEXT NOT NULL,username TEXT NOT NULL,role TEXT NOT NULL CHECK(role IN ('patient','clinician')),password_hash TEXT NOT NULL, UNIQUE(tenant,username))")
                    c.exec_driver_sql("INSERT INTO dev_users VALUES (7,'alpha','original','clinician','unchanged-hash')")
                    c.exec_driver_sql("CREATE TABLE dev_sessions(token_hash TEXT PRIMARY KEY,user_id INTEGER NOT NULL REFERENCES dev_users(id),expires_at TEXT NOT NULL)")
                    c.exec_driver_sql("INSERT INTO dev_sessions VALUES ('bad-existing-reference',999,'2099')")
                with self.assertRaisesRegex(RuntimeError, "foreign-key"):
                    init_portal_schema(engine)
                with engine.connect() as c:
                    columns = {row[1] for row in c.exec_driver_sql("PRAGMA table_info(dev_users)")}
                    self.assertNotIn("active", columns)
                    self.assertEqual(c.exec_driver_sql("SELECT password_hash FROM dev_users WHERE id=7").scalar(), "unchanged-hash")
                    self.assertEqual(c.exec_driver_sql("SELECT count(*) FROM sqlite_master WHERE name='dev_users_upgrade'").scalar(), 0)
            finally:
                engine.dispose()

    def test_original_check_upgrade_preserves_ids_passwords_sessions_and_references(self):
        with tempfile.TemporaryDirectory(prefix="raphael-migration-test-") as directory, \
             patch.dict(os.environ, {"RAPHAEL_DEV_SYNTHETIC_ONLY": "1"}):
            engine = create_engine("sqlite:///" + str(Path(directory) / "synthetic.db"))
            try:
                with engine.begin() as c:
                    c.exec_driver_sql("PRAGMA foreign_keys=ON")
                    c.exec_driver_sql("CREATE TABLE dev_users(id INTEGER PRIMARY KEY AUTOINCREMENT,tenant TEXT NOT NULL,username TEXT NOT NULL,role TEXT NOT NULL CHECK(role IN ('patient','clinician')),password_hash TEXT NOT NULL, UNIQUE(tenant,username))")
                    c.exec_driver_sql("CREATE TABLE dev_sessions(token_hash TEXT PRIMARY KEY,user_id INTEGER NOT NULL REFERENCES dev_users(id),expires_at TEXT NOT NULL)")
                    c.exec_driver_sql("INSERT INTO dev_users(id,tenant,username,role,password_hash) VALUES (51,'alpha','deleted','patient','unchanged')")
                    c.exec_driver_sql("DELETE FROM dev_users")
                    c.exec_driver_sql("INSERT INTO dev_users(id,tenant,username,role,password_hash) VALUES (7,'alpha','original','clinician','unchanged-hash')")
                    c.exec_driver_sql("INSERT INTO dev_sessions VALUES ('session-hash',7,'2099-01-01')")
                init_portal_schema(engine)
                init_portal_schema(engine)
                with engine.connect() as c:
                    user = c.exec_driver_sql("SELECT * FROM dev_users WHERE id=7").mappings().one()
                    self.assertEqual(user["password_hash"], "unchanged-hash")
                    self.assertEqual(user["active"], 1)
                    self.assertEqual(c.exec_driver_sql("SELECT user_id FROM dev_sessions WHERE token_hash='session-hash'").scalar(), 7)
                    self.assertEqual(c.exec_driver_sql("PRAGMA foreign_key_check").all(), [])
                    self.assertEqual(c.exec_driver_sql("PRAGMA foreign_keys").scalar(), 1)
                new_user = create_dev_user(engine, "alpha", "admin", "Synthetic-migration-password", "org_admin")
                self.assertGreater(new_user["id"], 51)
            finally:
                engine.dispose()
