"""Bounded read-only synthetic DICOMweb QIDO-RS response/identity tests."""
import asyncio
import json
import os
import unittest
from unittest.mock import patch

import httpx

from integrations.dicomweb.sandbox import (
    DicomwebUnavailable, preview_studies, sandbox_url,
)


def tag(vr, value):
    return {"vr": vr, "Value": [value]}


def row(patient="SYNTHETIC-PACS-01", uid="1.2.840.10008.5.1", removed="YES"):
    return {
        "00100020": tag("LO", patient),
        "00120062": tag("CS", removed),
        "0020000D": tag("UI", uid),
        "00080060": tag("CS", "CT"),
        "00100010": {"vr": "PN", "Value": [{"Alphabetic": "SYNTHETIC^PERSON"}]},
    }


class DicomwebSandboxTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {
            "DICOMWEB_SANDBOX_BASE_URL": "https://pacs-sandbox.example.test/dicomweb",
            "DICOMWEB_SANDBOX_SYNTHETIC_ONLY": "1",
            "DICOMWEB_SANDBOX_TOKEN": "synthetic-token",
        })
        env.start()
        self.addCleanup(env.stop)

    def call(self, payload, code=200, content_type="application/dicom+json"):
        requests = []
        def respond(request):
            requests.append(request)
            return httpx.Response(code, headers={"Content-Type": content_type},
                                  content=json.dumps(payload).encode())
        output = asyncio.run(preview_studies(
            "SYNTHETIC-PACS-01", transport=httpx.MockTransport(respond)))
        return output, requests

    def test_valid_synthetic_studies_are_bounded_read_only_and_pii_not_echoed(self):
        result, requests = self.call([row()])
        self.assertEqual(result["status"], "synthetic_read_only_dicomweb_preview")
        self.assertEqual(result["studies"], [
            {"study_instance_uid": "1.2.840.10008.5.1", "modality": "CT"}
        ])
        self.assertNotIn("00100010", json.dumps(result))
        self.assertNotIn("SYNTHETIC^PERSON", json.dumps(result))
        self.assertFalse(result["imported"])
        self.assertFalse(result["clinical_use_authorized"])
        self.assertFalse(result["diagnostic_interpretation_performed"])
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].method, "GET")
        self.assertEqual(requests[0].url.params["PatientID"], "SYNTHETIC-PACS-01")
        self.assertEqual(requests[0].url.params["limit"], "25")
        self.assertEqual(requests[0].headers["Authorization"], "Bearer synthetic-token")

    def test_cross_patient_or_non_synthetic_study_fails_closed(self):
        for payload in [
            [row(patient="SYNTHETIC-OTHER")],
            [row(removed="NO")],
            [row(uid="malformed")],
            [row(), row()],
            [{}],
            [row()] * 26,
            {"not": "a DICOMweb JSON array"},
        ]:
            with self.subTest(payload=repr(payload)[:100]), self.assertRaises(DicomwebUnavailable):
                self.call(payload)

    def test_http_redirect_incorrect_content_type_and_large_response_are_rejected(self):
        for status, content in [(302, "application/dicom+json"), (200, "application/json")]:
            def respond(request):
                return httpx.Response(status, headers={"Content-Type": content}, json=[row()])
            with self.assertRaises(DicomwebUnavailable):
                asyncio.run(preview_studies("SYNTHETIC-PACS-01",
                                           transport=httpx.MockTransport(respond)))
        def oversized(request):
            return httpx.Response(200, headers={"Content-Type": "application/dicom+json"},
                                  content=b" " * (1024 * 1024 + 1))
        with self.assertRaises(DicomwebUnavailable):
            asyncio.run(preview_studies("SYNTHETIC-PACS-01",
                                       transport=httpx.MockTransport(oversized)))

    def test_sandbox_configuration_and_patient_key_must_be_explicit(self):
        for key in ["actual-123", "SYNTHETIC-", "../SYNTHETIC-1", 12, None]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                asyncio.run(preview_studies(key))
        for base in ["http://remote.invalid", "https://user:secret@remote.invalid",
                     "https://remote.invalid/?x=1", "https://0.0.0.0/pacs"]:
            with self.subTest(base=base), patch.dict(os.environ, {"DICOMWEB_SANDBOX_BASE_URL": base}):
                with self.assertRaises(DicomwebUnavailable):
                    sandbox_url()
        with patch.dict(os.environ, {"DICOMWEB_SANDBOX_SYNTHETIC_ONLY": "0"}):
            with self.assertRaises(DicomwebUnavailable):
                sandbox_url()


if __name__ == "__main__":
    unittest.main()
