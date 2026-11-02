"""Actual SQLite FTS5/provenance tests with nonclinical synthetic source excerpts."""
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text

from dev_knowledge import init_knowledge_schema, install_knowledge, search_evidence
from dev_portal import create_dev_user, init_portal_schema, install_dev_portal


class KnowledgeTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="raphael-knowledge-test-")
        self.addCleanup(directory.cleanup)
        self.engine = create_engine("sqlite:///" + str(Path(directory.name) / "synthetic.db"))
        self.addCleanup(self.engine.dispose)
        flag = patch.dict(os.environ, {"RAPHAEL_DEV_SYNTHETIC_ONLY": "1"})
        flag.start()
        self.addCleanup(flag.stop)
        init_portal_schema(self.engine)
        init_knowledge_schema(self.engine)
        self.password = "Synthetic-only-test-password"
        for tenant, username, role in (("alpha", "clinician", "clinician"), ("alpha", "patient", "patient"),
                                       ("alpha", "admin", "org_admin"), ("beta", "outsider", "clinician")):
            create_dev_user(self.engine, tenant, username, self.password, role)
        app = FastAPI()
        ctx = install_dev_portal(app, self.engine)
        install_knowledge(app, ctx)
        self.client = TestClient(app)
        self.addCleanup(self.client.close)
        self.login()

    def login(self, username="clinician", tenant="alpha"):
        r = self.client.post("/api/dev/login", json={"username": username, "tenant": tenant, "password": self.password})
        self.assertEqual(r.status_code, 200, r.text)

    def source(self, **changes):
        body = {"title": "Nonclinical synthetic source fixture", "url": "https://example.org/synthetic-fixture",
                "content": "Spectral calibrator baseline measurement fixture. The violet calibration marker is synthetic text, not medical advice.",
                "license_declaration": "Test-owned synthetic text; use permitted for this automated fixture only.",
                "permission_attested": True, "org": "Synthetic test", "tier": "D", "source_date": "2024-01-01"}
        body.update(changes)
        r = self.client.post("/api/dev/knowledge/sources", json=body)
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()

    def approve(self, source_id):
        r = self.client.post(f"/api/dev/knowledge/sources/{source_id}/approve", json={})
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()

    def search(self, query="violet calibrator"):
        r = self.client.post("/api/dev/knowledge/search", json={"query": query})
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()["results"]

    def test_pending_source_is_not_retrievable_until_explicit_review(self):
        source = self.source()
        self.assertEqual(source["status"], "pending")
        self.assertEqual(self.search(), [])
        reviewed = self.approve(source["id"])
        self.assertEqual(reviewed["review_status"], "clinician_reviewed")
        self.assertEqual(reviewed["reviewed_by_username"], "clinician")
        results = self.search()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["score"], 1.0)
        self.assertEqual(results[0]["score_kind"], "lexical_query_term_coverage")
        self.assertEqual(results[0]["retrieval_method"], "sqlite_fts5_lexical")
        self.assertIn("not_independently_verified", results[0]["review_status"])

    def test_citation_resolves_exact_stored_excerpt_url_hash_and_reviewer(self):
        source = self.source()
        self.approve(source["id"])
        hit = self.search()[0]
        r = self.client.get(hit["citation_path"])
        self.assertEqual(r.status_code, 200, r.text)
        citation = r.json()
        self.assertEqual(citation["body"], hit["text"])
        self.assertEqual(citation["url"], source["url"])
        self.assertEqual(citation["content_sha256"], hashlib.sha256(source["content"].encode()).hexdigest())
        self.assertEqual(citation["reviewed_by"], "clinician")
        self.assertEqual(citation["version"], 1)
        self.assertEqual(self.client.get("/api/dev/knowledge/citations/unknown").status_code, 404)

    def test_deduplication_versions_and_current_approval(self):
        first = self.source()
        same = self.source()
        self.assertEqual(same["id"], first["id"])
        self.assertTrue(same["deduplicated"])
        self.approve(first["id"])
        old_hit = self.search()[0]
        second = self.source(content="Replacement spectral calibrator. Amber calibration marker is a distinct synthetic fixture.")
        self.assertEqual(second["version"], 2)
        self.assertNotEqual(second["content_sha256"], first["content_sha256"])
        self.assertEqual(self.search()[0]["source_version"], 1)
        self.approve(second["id"])
        self.assertEqual(self.search("amber")[0]["source_version"], 2)
        self.assertEqual(self.search("violet"), [])
        self.assertEqual(self.client.get(old_hit["citation_path"]).status_code, 404)
        self.assertEqual(self.client.get(f"/api/dev/knowledge/sources/{first['id']}").json()["status"], "superseded")
        self.assertEqual(self.client.patch(f"/api/dev/knowledge/sources/{second['id']}", json={"content": "changed"}).status_code, 405)

    def test_revocation_removes_search_and_citation_without_erasing_source(self):
        source = self.source()
        self.approve(source["id"])
        hit = self.search()[0]
        r = self.client.post(f"/api/dev/knowledge/sources/{source['id']}/revoke", json={})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(self.search(), [])
        self.assertEqual(self.client.get(hit["citation_path"]).status_code, 404)
        kept = self.client.get(f"/api/dev/knowledge/sources/{source['id']}").json()
        self.assertEqual(kept["status"], "revoked")
        self.assertEqual(kept["content"], source["content"])
        self.assertEqual(kept["content_sha256"], source["content_sha256"])

    def test_tenant_scoping_and_patient_admin_denial(self):
        source = self.source()
        self.approve(source["id"])
        hit = self.search()[0]
        self.login("outsider", "beta")
        self.assertEqual(self.search(), [])
        for path in (f"/sources/{source['id']}", f"/citations/{hit['chunk_id']}"):
            self.assertEqual(self.client.get("/api/dev/knowledge" + path).status_code, 404)
        for operation in ("approve", "revoke"):
            self.assertEqual(self.client.post(f"/api/dev/knowledge/sources/{source['id']}/{operation}", json={}).status_code, 404)
        for username in ("patient", "admin"):
            self.login(username)
            self.assertEqual(self.client.get("/api/dev/knowledge/sources").status_code, 403)
            self.assertEqual(self.client.post("/api/dev/knowledge/search", json={"query": "violet"}).status_code, 403)
            self.assertEqual(self.client.get(hit["citation_path"]).status_code, 403)

    def test_intake_validation_permission_url_limits_and_metadata(self):
        source = self.source()
        valid = {k: source[k] for k in ("title", "url", "content", "license_declaration", "permission_attested", "org", "tier")}
        for change in ({"permission_attested": False}, {"permission_attested": 1}, {"license_declaration": ""},
                       {"content": "x" * 200001}, {"source_date": "2999-01-01"}, {"tier": "TIER_1"},
                       {"url": "http://example.org/"}, {"url": "https://localhost/foo"},
                       {"url": "https://127.0.0.1/foo"}, {"url": "https://10.0.0.1/foo"},
                       {"url": "https://[::1]/foo"}, {"url": "https://host.internal/foo"},
                       {"url": "https://username:secret@example.org/foo"},
                       {"url": "https://example.org:8443/foo"}, {"url": "https://example.org/?token=secret"}):
            with self.subTest(change=list(change)):
                r = self.client.post("/api/dev/knowledge/sources", json=dict(valid, **change))
                self.assertEqual(r.status_code, 422, r.text)

    def test_fts_sanitization_chunking_and_repeatable_schema(self):
        source = self.source(content=("Synthetic calibrator violet measure. " * 120))
        self.approve(source["id"])
        hits = self.search()
        self.assertGreater(len(hits), 1)
        ids = [hit["chunk_id"] for hit in hits]
        self.assertEqual(ids, [hit["chunk_id"] for hit in self.search()])
        init_knowledge_schema(self.engine)
        self.approve(source["id"])
        self.assertEqual(ids, [hit["chunk_id"] for hit in self.search()])
        self.assertEqual(self.search('" OR * - NEAR(foo) : ; --'), [])
        self.assertEqual(self.search("!!"), [])
        self.assertEqual(len(search_evidence(self.engine, "alpha", "violet", top_k=1)), 1)
        with self.assertRaises(ValueError):
            search_evidence(self.engine, "alpha", "violet", top_k=100)
        with self.engine.connect() as c:
            self.assertEqual(c.exec_driver_sql("PRAGMA integrity_check").scalar(), "ok")
            self.assertEqual(c.execute(text("SELECT count(*) FROM dev_evidence_fts")).scalar(),
                             c.execute(text("SELECT count(*) FROM dev_evidence_chunks")).scalar())

    def test_status_reports_bounded_coverage_and_audit_excludes_excerpt(self):
        source = self.source()
        self.approve(source["id"])
        self.search()
        status = self.client.get("/api/dev/knowledge/status").json()
        self.assertEqual(status["source_versions_by_status"], {"approved": 1})
        self.assertEqual(status["approved_chunks"], 1)
        self.assertEqual(status["source_verification"], "clinician_attestation_only")
        self.assertEqual(status["external_connectors"], "not invoked")
        with self.engine.connect() as c:
            events = c.execute(text("SELECT event,resource_type,resource_id FROM dev_audit")).all()
        self.assertTrue(any(event[0] == "search_evidence" for event in events))
        self.assertNotIn(source["content"], repr(events))
