"""Recovery regression tests: temporary SQLite files and synthetic data only.

Run: python -m unittest discover -s tests -v
No model downloads, external services, production credentials, or production DBs.
"""

import importlib
import asyncio
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

import httpx
import numpy as np
from fastapi import HTTPException
from fastapi.testclient import TestClient
from openai import OpenAI
from qdrant_client import QdrantClient
from sqlalchemy import create_engine

from database import execute_sqlite_script
from corpus.indexer import point_id_for
from evidence.ingest import ingest_text
import pubmed.ingest as pubmed_ingest


# Isolate module import as well as startup from any runtime service bindings.
with patch.dict(os.environ, {
    "DATABASE_URL": "sqlite://", "QDRANT_URL": "", "QDRANT_API_KEY": "",
    "LLM_BASE_URL": "", "LLM_API_KEY": "", "STATIC_PATH": "static",
    "JWT_SECRET": "synthetic-unit-test-key-only",
    "RAPHAEL_DEV_PORTAL": "0",
}):
    main = importlib.import_module("main")
    auth = importlib.import_module("auth.jwt_auth")


class EmbeddingStartupTests(unittest.TestCase):
    def test_application_import_does_not_load_optional_embedding_stack(self):
        import subprocess
        import sys
        environment = dict(os.environ, DATABASE_URL='sqlite://', QDRANT_URL='',
                           QDRANT_API_KEY='', LLM_BASE_URL='', LLM_API_KEY='',
                           RAPHAEL_DEV_PORTAL='0', STATIC_PATH='static')
        result = subprocess.run([sys.executable, '-c',
            "import sys; import main; import rag.embedder; import evidence.ingest; assert 'sentence_transformers' not in sys.modules; assert main._ST_MODEL is None"],
            env=environment, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_first_embedding_loads_once_and_preserves_normalized_vectors(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        import rag.embedder as rag_embedder
        for module, cache in ((main, '_ST_MODEL'), (pubmed_ingest, '_MODEL'), (rag_embedder, '_MODEL')):
            with self.subTest(module=module.__name__):
                model = Mock()
                model.encode.return_value = [[1.0, 0.0]]
                constructor = Mock(return_value=model)
                args = () if module is main else (main.EMBED_MODEL,)
                with patch.object(module, cache, None), patch.dict('sys.modules',
                        {'sentence_transformers': SimpleNamespace(SentenceTransformer=constructor)}):
                    self.assertEqual(module.embed_texts(['synthetic'], *args).tolist(), [[1.0, 0.0]])
                    module.embed_texts(['second synthetic'], *args)
                constructor.assert_called_once_with(main.EMBED_MODEL)
                self.assertEqual(model.encode.call_count, 2)
                model.encode.assert_called_with(['second synthetic'], normalize_embeddings=True)


class SQLiteScriptTests(unittest.TestCase):
    def test_original_split_breaks_triggers_but_complete_script_succeeds(self):
        engine = create_engine("sqlite://")
        try:
            with engine.begin() as c:
                execute_sqlite_script(c, main.DDL)
                broken = main.TRIGGERS.strip().split(";\n")[0]
                with self.assertRaises(sqlite3.OperationalError):
                    c.connection.driver_connection.execute(broken)
                execute_sqlite_script(c, main.TRIGGERS)
                names = set(c.exec_driver_sql(
                    "SELECT name FROM sqlite_master WHERE type='trigger'"
                ).scalars())
                self.assertEqual(names, {"doc_ai", "doc_ad", "doc_au"})
        finally:
            engine.dispose()

    def test_quoted_semicolons_comments_and_multiple_statements(self):
        engine = create_engine("sqlite://")
        try:
            with engine.begin() as c:
                execute_sqlite_script(c, "-- comment;\nCREATE TABLE demo(value TEXT);"
                                      "INSERT INTO demo VALUES ('value;with;semicolons');")
                self.assertEqual(c.exec_driver_sql("SELECT value FROM demo").scalar(),
                                 "value;with;semicolons")
        finally:
            engine.dispose()

    def test_incomplete_script_is_rejected_before_execution(self):
        engine = create_engine("sqlite://")
        try:
            with engine.begin() as c:
                with self.assertRaises(ValueError):
                    execute_sqlite_script(c, "CREATE TABLE demo(x); CREATE TRIGGER unfinished")
                self.assertEqual(c.exec_driver_sql(
                    "SELECT count(*) FROM sqlite_master WHERE name='demo'"
                ).scalar(), 0)
        finally:
            engine.dispose()

    def test_does_not_commit_callers_transaction(self):
        engine = create_engine("sqlite://")
        try:
            with engine.begin() as c:
                c.exec_driver_sql("CREATE TABLE demo(value TEXT)")
            with self.assertRaisesRegex(RuntimeError, "rollback"):
                with engine.begin() as c:
                    c.exec_driver_sql("INSERT INTO demo VALUES ('synthetic')")
                    execute_sqlite_script(c, "CREATE TABLE IF NOT EXISTS demo(value TEXT);")
                    raise RuntimeError("rollback")
            with engine.connect() as c:
                self.assertEqual(c.exec_driver_sql("SELECT count(*) FROM demo").scalar(), 0)
        finally:
            engine.dispose()


class ApplicationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="raphael-test-")
        self.addCleanup(self.directory.cleanup)
        self.url = "sqlite:///" + str(Path(self.directory.name) / "synthetic.db")
        self.engine = create_engine(self.url)
        self.addCleanup(self.engine.dispose)
        for name, value in {"engine": self.engine, "QDRANT_URL": "",
                            "LLM_BASE_URL": "", "CORPUS_RUN_TOKEN": ""}.items():
            p = patch.object(main, name, value)
            p.start()
            self.addCleanup(p.stop)

    def patient_and_conversation(self, client):
        r = client.post("/api/patients", json={"name": "Synthetic Patient", "mrn": "TEST-ONLY"})
        self.assertEqual(r.status_code, 200, r.text)
        patient = r.json()
        r = client.post("/api/conversations", json={"patient_id": patient["id"], "title": "Synthetic encounter"})
        self.assertEqual(r.status_code, 200, r.text)
        return patient, r.json()

    def test_full_startup_health_and_frontend_assets(self):
        with TestClient(main.app) as client:
            self.assertEqual(client.get("/healthz").json()["status"], "ok")
            redirect = client.get("/", follow_redirects=False)
            self.assertEqual(redirect.status_code, 307)
            self.assertEqual(redirect.headers["location"], "/static/Raphael.html")
            for path in ("/static/Raphael.html", "/static/app.js", "/static/styles.css",
                         "/static/logo.png.PNG", "/openapi.json"):
                self.assertEqual(client.get(path).status_code, 200, path)
            self.assertTrue(client.get("/api/radiology/checklist").json()["cant_miss"])
            self.assertTrue(client.get("/api/corpus/sources").json())

    def test_repeated_startup_preserves_records_and_all_schema_objects(self):
        with TestClient(main.app) as client:
            patient, conversation = self.patient_and_conversation(client)
            self.assertEqual(client.post("/api/feedback/submit", json={
                "conversation_id": conversation["id"], "clinician_notes": "synthetic feedback"
            }).status_code, 200)
            self.assertEqual(client.post("/api/docs/upload", data={"title": "Synthetic note"},
                                        files={"file": ("test.txt", b"synthetic searchable content")}).status_code, 200)
        # Dispose the pool and recreate the engine to demonstrate on-disk persistence.
        self.engine.dispose()
        replacement = create_engine(self.url)
        self.addCleanup(replacement.dispose)
        with patch.object(main, "engine", replacement), TestClient(main.app) as client:
            self.assertEqual(client.get("/api/patients").json(), [patient])
            self.assertEqual(client.get("/api/conversations", params={"patient_id": patient["id"]}).json(), [conversation])
            self.assertEqual(client.get("/api/messages", params={"conversation_id": conversation["id"]}).json(), [])
            self.assertEqual(client.get("/api/feedback/list").json()[0]["clinician_notes"], "synthetic feedback")
            self.assertEqual(len(client.get("/api/docs/search", params={"q": "searchable"}).json()), 1)
            with replacement.connect() as c:
                tables = set(c.exec_driver_sql("SELECT name FROM sqlite_master WHERE type='table'").scalars())
                self.assertTrue({"patients", "conversations", "messages", "documents", "doc_fts", "feedback"} <= tables)
                self.assertEqual(c.exec_driver_sql("PRAGMA integrity_check").scalar(), "ok")
                self.assertEqual(c.exec_driver_sql("PRAGMA journal_mode").scalar(), "wal")

    def test_fts_insert_update_delete_triggers_after_repeated_startup(self):
        with TestClient(main.app):
            main.startup()
            with self.engine.begin() as c:
                c.exec_driver_sql("INSERT INTO documents(title, body) VALUES ('Synthetic', 'originaltoken')")
                self.assertEqual(c.exec_driver_sql("SELECT count(*) FROM doc_fts WHERE doc_fts MATCH 'originaltoken'").scalar(), 1)
                c.exec_driver_sql("UPDATE documents SET body='replacementtoken'")
                self.assertEqual(c.exec_driver_sql("SELECT count(*) FROM doc_fts WHERE doc_fts MATCH 'originaltoken'").scalar(), 0)
                self.assertEqual(c.exec_driver_sql("SELECT count(*) FROM doc_fts WHERE doc_fts MATCH 'replacementtoken'").scalar(), 1)
                c.exec_driver_sql("DELETE FROM documents")
                self.assertEqual(c.exec_driver_sql("SELECT count(*) FROM doc_fts WHERE doc_fts MATCH 'replacementtoken'").scalar(), 0)

    def test_validation_and_missing_llm_configuration(self):
        with TestClient(main.app) as client:
            for path, body in [("/api/patients", {}), ("/api/conversations", {}),
                               ("/api/chat", {}), ("/api/raphael/query", {}),
                               ("/api/evidence/ingest", {}), ("/api/corpus/ingest_url", {}),
                               ("/api/feedback/submit", {}), ("/api/pubmed/ingest_pack", {"pack": "unknown"})]:
                self.assertEqual(client.post(path, json=body).status_code, 400, path)
            self.assertEqual(client.get("/api/docs/search?q=x").status_code, 422)
            _, conv = self.patient_and_conversation(client)
            r = client.post("/api/chat", json={"conversation_id": conv["id"], "text": "Synthetic prompt"})
            self.assertEqual(r.status_code, 500)
            self.assertIn("LLM_BASE_URL not configured", r.json()["detail"])

    def test_chat_uses_openai_sdk_and_persists_messages_with_mock_http(self):
        requests = []

        def respond(request):
            requests.append(json.loads(request.content))
            self.assertEqual(request.url.path, "/v1/chat/completions")
            return httpx.Response(200, json={"id": "synthetic", "object": "chat.completion",
                "created": 0, "model": "synthetic", "choices": [{"index": 0,
                "message": {"role": "assistant", "content": "Synthetic mock response"}, "finish_reason": "stop"}]})

        with OpenAI(base_url="https://llm.invalid/v1", api_key="synthetic-test-only",
                    http_client=httpx.Client(transport=httpx.MockTransport(respond))) as llm:
            with patch.object(main, "llm_client", return_value=llm), TestClient(main.app) as client:
                _, conv = self.patient_and_conversation(client)
                r = client.post("/api/chat", json={"conversation_id": conv["id"], "text": "Synthetic prompt"})
                self.assertEqual(r.status_code, 200, r.text)
                self.assertEqual(r.json()["reply"], "Synthetic mock response")
            with TestClient(main.app) as client:
                messages = client.get("/api/messages", params={"conversation_id": conv["id"]}).json()
                self.assertEqual([m["role"] for m in messages], ["user", "assistant"])
                self.assertEqual(messages[1]["content"], "Synthetic mock response")
        self.assertEqual(requests[0]["messages"][-1]["content"], "Synthetic prompt")

    def test_qdrant_local_roundtrip_with_synthetic_embedding(self):
        # Real Qdrant local storage; only the embedding model is mocked.
        qdrant = QdrantClient(location=":memory:")
        self.addCleanup(qdrant.close)
        vector = np.zeros((1, main.EMBED_DIM), dtype=np.float32)
        vector[0, 0] = 1
        with patch.object(main, "qdrant_client", return_value=qdrant), \
             patch.object(main, "embed_texts", return_value=vector), TestClient(main.app) as client:
            r = client.post("/api/evidence/ingest", json={"title": "Synthetic fixture", "text": "Nonclinical test text"})
            self.assertEqual(r.status_code, 200, r.text)
            self.assertEqual(r.json()["chunks_indexed"], 1)
            hits = main.retrieve_evidence("Synthetic question")
            self.assertEqual(hits[0]["title"], "Synthetic fixture")
            # Reingestion should upsert the same point, not duplicate it.
            client.post("/api/evidence/ingest", json={"title": "Synthetic fixture", "text": "Nonclinical test text"}).raise_for_status()
            self.assertEqual(qdrant.count(main.QDRANT_COLLECTION).count, 1)

    def test_insufficient_evidence_abstains_without_calling_llm(self):
        with patch.object(main, "retrieve_evidence", return_value=[]), \
             patch.object(main, "call_llm_json") as llm, TestClient(main.app) as client:
            _, conv = self.patient_and_conversation(client)
            r = client.post("/api/raphael/query", json={"conversation_id": conv["id"], "question": "Synthetic question"})
            self.assertEqual(r.status_code, 200, r.text)
            self.assertEqual(r.json()["ranked_differential"], [])
            self.assertIsNone(r.json()["working_diagnosis"])
            self.assertIn("Low", r.json()["confidence"])
            llm.assert_not_called()

    def test_pubmed_and_legacy_ingestion_accept_qdrant_point_types_and_ids(self):
        qdrant = QdrantClient(location=":memory:")
        self.addCleanup(qdrant.close)
        vector = np.ones((1, 3), dtype=np.float32)
        with patch("evidence.ingest.QdrantClient", return_value=qdrant), \
             patch("evidence.ingest.embed_texts", return_value=vector):
            self.assertEqual(ingest_text(qdrant_url="http://unused.invalid", collection="legacy-test",
                embed_model="synthetic", embed_dim=3, title="Synthetic", tier="TIER_1",
                source_date=None, url=None, section=None, text="Synthetic fixture"), 1)
        with patch.object(pubmed_ingest, "QdrantClient", return_value=qdrant), \
             patch.object(pubmed_ingest, "embed_texts", return_value=vector), \
             patch.object(pubmed_ingest, "esearch_pubmed", new=AsyncMock(return_value=["123"])), \
             patch.object(pubmed_ingest, "esummary_pubmed", new=AsyncMock(return_value={
                 "result": {"123": {"title": "Synthetic fixture", "pubdate": "2025", "pubtype": []}}})):
            result = asyncio.run(pubmed_ingest.ingest_pubmed_query(qdrant_url="http://unused.invalid",
                qdrant_api_key="", collection="pubmed-test", embed_model="synthetic", embed_dim=3,
                query="synthetic", retmax=1))
            self.assertEqual(result["chunks_indexed"], 1)
        for collection in ("legacy-test", "pubmed-test"):
            point = qdrant.scroll(collection)[0][0]
            self.assertEqual(str(point.id), point_id_for(point.payload["chunk_id"]))
            self.assertEqual(len(point.payload["chunk_id"]), 64)

    def test_structured_answer_citations_and_persistence_with_mock_services(self):
        chunks = [{"chunk_id": f"synthetic-{i}", "title": "Synthetic fixture", "tier": "A",
                   "source_type": "test", "text": "Nonclinical fixture", "score": 1.0} for i in range(3)]
        draft = {key: [] for key in main.OUTPUT_KEYS}
        draft.update(problem_representation="Synthetic fixture", working_diagnosis=None,
                     confidence="Low (synthetic test)", ranked_differential=[{
                         "diagnosis": "Synthetic label", "likelihood": "Low", "supports": [],
                         "argues_against": [], "citations": ["synthetic-0"]}])
        with patch.object(main, "retrieve_evidence", return_value=chunks), \
             patch.object(main, "call_llm_json", side_effect=[draft, {"ok": True}]), \
             TestClient(main.app) as client:
            _, conv = self.patient_and_conversation(client)
            r = client.post("/api/raphael/query", json={"conversation_id": conv["id"], "question": "Synthetic question"})
            self.assertEqual(r.status_code, 200, r.text)
            self.assertEqual(r.json()["evidence_used"], chunks)
            self.assertEqual(r.json()["ranked_differential"][0]["citations"], ["synthetic-0"])
            saved = client.get("/api/messages", params={"conversation_id": conv["id"]}).json()
            self.assertEqual(json.loads(saved[0]["content"]), r.json())
        draft["ranked_differential"][0]["citations"] = []
        with self.assertRaises(HTTPException):
            main.enforce_citations(draft)

    def test_auth_helper_and_current_unprotected_route_behavior(self):
        token = auth.create_token("synthetic-user", "synthetic-tenant", "reviewer")
        self.assertEqual(auth.require_user("Bearer " + token)["sub"], "synthetic-user")
        for header in ("", "Bearer invalid"):
            with self.assertRaises(HTTPException) as raised:
                auth.require_user(header)
            self.assertEqual(raised.exception.status_code, 401)
        # Document current behavior, not an assertion of secure API authentication.
        with TestClient(main.app) as client:
            self.assertEqual(client.get("/api/patients").status_code, 200)

    def test_optional_qdrant_outage_does_not_block_core_startup(self):
        with patch.object(main, "QDRANT_URL", "http://qdrant.invalid"), \
             patch.object(main, "ensure_qdrant_collection", side_effect=RuntimeError("synthetic unavailable")), \
             patch("builtins.print") as log, TestClient(main.app) as client:
            self.assertEqual(client.get("/healthz").status_code, 200)
            log.assert_called_with("QDRANT_STARTUP_WARNING", "RuntimeError")

class CollectionCompatibilityTests(unittest.TestCase):
    def test_no_creation_when_actual_model_dimensions_are_wrong_or_nonfinite(self):
        from types import SimpleNamespace as NS
        from unittest.mock import Mock
        for probe in (np.zeros((1, 8)),np.full((1, main.EMBED_DIM),np.nan)):
            with self.subTest(shape=probe.shape):
                client=Mock();client.get_collections.return_value=NS(collections=[])
                with patch.object(main,'qdrant_client',return_value=client),patch.object(main,'embed_texts',return_value=probe):
                    with self.assertRaises(ValueError):main.ensure_qdrant_collection()
                client.create_collection.assert_not_called();client.upsert.assert_not_called()

    def test_existing_wrong_dimension_distance_or_named_vectors_are_preserved(self):
        from types import SimpleNamespace as NS
        from unittest.mock import Mock
        from qdrant_client.http import models as qm
        for vectors in (qm.VectorParams(size=8,distance=qm.Distance.COSINE),qm.VectorParams(size=main.EMBED_DIM,distance=qm.Distance.DOT),{'named':qm.VectorParams(size=main.EMBED_DIM,distance=qm.Distance.COSINE)}):
            client=Mock();client.get_collections.return_value=NS(collections=[NS(name=main.QDRANT_COLLECTION)])
            client.get_collection.return_value=NS(config=NS(params=NS(vectors=vectors)))
            with patch.object(main,'qdrant_client',return_value=client),patch.object(main,'embed_texts') as encode:
                with self.assertRaises(ValueError):main.ensure_qdrant_collection()
            encode.assert_not_called();client.create_collection.assert_not_called();client.recreate_collection.assert_not_called()

    def test_missing_collection_is_created_only_after_actual_dimension_check(self):
        from types import SimpleNamespace as NS
        from unittest.mock import Mock
        client=Mock();client.get_collections.return_value=NS(collections=[])
        def probe(texts):
            client.create_collection.assert_not_called()
            return np.zeros((1,main.EMBED_DIM),dtype=np.float32)
        with patch.object(main,'qdrant_client',return_value=client),patch.object(main,'embed_texts',side_effect=probe):main.ensure_qdrant_collection()
        client.create_collection.assert_called_once()
        self.assertEqual(client.create_collection.call_args.kwargs['vectors_config'].size,main.EMBED_DIM)
