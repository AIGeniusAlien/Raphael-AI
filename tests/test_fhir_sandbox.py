"""FHIR sandbox contract tests; local HTTP fixture is explicitly not a live EHR."""
import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import threading
import unittest
from unittest.mock import patch

import httpx

from integrations.fhir.sandbox import MAX_RESPONSE, RESOURCE_TYPES, SandboxUnavailable, preview_patient, sandbox_base_url


def patient():
    return {"resourceType": "Patient", "id": "synthetic-1", "name": [{"family": "Synthetic"}],
            "meta": {"tag": [{"system": "urn:raphael:development", "code": "synthetic"}]}}


def fixture(path):
    if path.startswith("/fhir/Patient/"):
        return patient()
    kind = path.split("/")[-1].split("?")[0]
    return {"resourceType": "Bundle", "type": "searchset", "entry": [
        {"resource": {"resourceType": kind, "id": "fixture", "subject": {"reference": "Patient/synthetic-1"}}}
    ] if kind == "Observation" else []}


class FhirSandboxTests(unittest.TestCase):
    def setUp(self):
        flags = patch.dict(os.environ, {"FHIR_SANDBOX_BASE_URL": "https://sandbox.invalid/fhir",
                    "FHIR_SANDBOX_SYNTHETIC_ONLY": "1", "FHIR_SANDBOX_ALLOW_LOOPBACK": "0", "FHIR_SANDBOX_TOKEN": ""})
        flags.start(); self.addCleanup(flags.stop)

    def call(self, responder):
        return asyncio.run(preview_patient("synthetic-1", transport=httpx.MockTransport(responder)))

    def test_supported_resource_preview_has_provenance_and_never_imports(self):
        paths = []
        def respond(request):
            paths.append(request.url.path)
            return httpx.Response(200, json=fixture(request.url.path))
        result = self.call(respond)
        self.assertEqual(result["status"], "sandbox_preview")
        self.assertFalse(result["imported"])
        self.assertEqual(result["source_base_url"], "https://sandbox.invalid/fhir")
        self.assertEqual(result["resources"][0]["resourceType"], "Observation")
        self.assertEqual(len(paths), 1 + len(RESOURCE_TYPES))

    def test_missing_config_http_credentials_and_query_are_rejected(self):
        for base in ("", "http://external.invalid/fhir", "https://user:password@sandbox.invalid/fhir", "https://sandbox.invalid/fhir?token=value", "https://sandbox.invalid/#fragment"):
            with self.subTest(base=base), patch.dict(os.environ, {"FHIR_SANDBOX_BASE_URL": base}):
                with self.assertRaises(SandboxUnavailable): sandbox_base_url()
        with patch.dict(os.environ, {"FHIR_SANDBOX_SYNTHETIC_ONLY": "0"}):
            with self.assertRaises(SandboxUnavailable): sandbox_base_url()
        for resource_id in ("../other", ".", "..", "patient%2fother"):
            with self.assertRaises(ValueError):
                asyncio.run(preview_patient(resource_id))

    def test_redirect_failure_and_bad_json_are_not_success(self):
        for response in (httpx.Response(302, headers={"Location": "https://other.invalid"}),
                         httpx.Response(401), httpx.Response(200, text="invalid"), httpx.Response(200, json=[])):
            with self.subTest(status=response.status_code), self.assertRaises(SandboxUnavailable):
                self.call(lambda _: response)

    def test_synthetic_patient_declaration_and_matching_identity_are_required(self):
        for change in ({"id": "other"}, {"meta": {}}, {"meta": "invalid"}, {"resourceType": "Observation"}):
            p = dict(patient(), **change)
            with self.subTest(change=change), self.assertRaises(SandboxUnavailable):
                self.call(lambda _: httpx.Response(200, json=p))

    def test_cross_patient_and_oversized_resources_are_rejected(self):
        def respond(request):
            if "/Patient/" in request.url.path:
                return httpx.Response(200, json=patient())
            return httpx.Response(200, json={"resourceType": "Bundle", "entry": [{"resource": {
                "resourceType": "Observation", "subject": {"reference": "Patient/other"}}}]})
        with self.assertRaises(SandboxUnavailable): self.call(respond)
        with self.assertRaises(SandboxUnavailable):
            self.call(lambda _: httpx.Response(200, content=b"x" * (MAX_RESPONSE + 1)))

    def test_real_local_http_sandbox_fixture_is_read_only(self):
        requests = []
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                requests.append(self.path)
                body = json.dumps(fixture(self.path)).encode()
                self.send_response(200); self.send_header("Content-Type", "application/fhir+json")
                self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
            def log_message(self, *args): pass
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        try:
            with patch.dict(os.environ, {"FHIR_SANDBOX_BASE_URL": f"http://127.0.0.1:{server.server_port}/fhir", "FHIR_SANDBOX_ALLOW_LOOPBACK": "1"}):
                result = asyncio.run(preview_patient("synthetic-1"))
            self.assertEqual(len(requests), 1 + len(RESOURCE_TYPES))
            self.assertTrue(all("patient=synthetic-1" in path for path in requests[1:]))
            self.assertFalse(result["imported"])
        finally:
            server.shutdown(); server.server_close(); thread.join(timeout=3)
