"""Structured-chart tests use temporary synthetic SQLite files and local bytes only."""
import hashlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pypdf import PdfWriter
from sqlalchemy import create_engine, text

from dev_portal import create_dev_user, init_portal_schema, install_dev_portal
from dev_chart import init_chart_schema, install_chart, MAX_DOCUMENT_BYTES


class StructuredChartTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="raphael-chart-test-")
        self.addCleanup(self.tmp.cleanup)
        self.engine = create_engine("sqlite:///" + str(Path(self.tmp.name) / "synthetic.db"))
        self.addCleanup(self.engine.dispose)
        flag = patch.dict(os.environ, {"RAPHAEL_DEV_SYNTHETIC_ONLY": "1"})
        flag.start()
        self.addCleanup(flag.stop)
        init_portal_schema(self.engine)
        init_chart_schema(self.engine)
        self.password = "Synthetic-chart-test-pass"
        for tenant, name, role in (("alpha", "doctor", "clinician"), ("alpha", "otherdoctor", "clinician"),
                                   ("alpha", "patient", "patient"), ("alpha", "otherpatient", "patient"),
                                   ("alpha", "admin", "org_admin"), ("beta", "outsider", "clinician")):
            create_dev_user(self.engine, tenant, name, self.password, role)
        self.app = FastAPI()
        ctx = install_dev_portal(self.app, self.engine)
        install_chart(self.app, ctx)
        self.client = TestClient(self.app)
        self.addCleanup(self.client.close)
        self.login()
        self.patient = self.client.post("/api/dev/patients", json={"name": "Synthetic Chart Patient", "patient_username": "patient"}).json()
        self.other = self.client.post("/api/dev/patients", json={"name": "Other Synthetic", "patient_username": "otherpatient"}).json()
        self.prefix = f"/api/dev/patients/{self.patient['id']}"

    def login(self, name="doctor", tenant="alpha"):
        r = self.client.post("/api/dev/login", json={"tenant": tenant, "username": name, "password": self.password})
        self.assertEqual(r.status_code, 200, r.text)

    def entry(self, **extra):
        body = {"kind": "history", "label": "Synthetic medical history", "text": "Nonclinical synthetic fixture", **extra}
        r = self.client.post(self.prefix + "/entries", json=body)
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()

    def upload(self, content=b"Synthetic document", name="synthetic.txt", visibility="private"):
        return self.client.post(self.prefix + "/documents", data={"title": "Synthetic document", "visibility": visibility},
                                files={"file": (name, content, "application/octet-stream")})

    def test_structured_records_trends_and_entered_reference_flags(self):
        entries = [
            {"kind": "medication", "label": "Synthetic medication", "dose": "Entered test dose; not a prescribing instruction"},
            {"kind": "allergy", "label": "Synthetic allergy", "reaction": "Entered fixture reaction", "severity": "unknown"},
            {"kind": "lab", "label": "Synthetic analyte", "value": 15, "unit": "test-unit", "reference_low": 2, "reference_high": 10,
             "recorded_at": "2030-01-02T12:00:00+00:00"},
            {"kind": "lab", "label": "Synthetic analyte", "value": 5, "unit": "test-unit", "reference_low": 2, "reference_high": 10,
             "recorded_at": "2030-01-01T12:00:00+00:00"},
            {"kind": "vital", "label": "Synthetic observation", "value": 1, "unit": "test-unit", "reference_low": 2},
        ]
        for body in entries:
            self.entry(**body)
        self.login("patient")
        chart = self.client.get(self.prefix + "/chart")
        self.assertEqual(chart.status_code, 200, chart.text)
        values = chart.json()["entries"]
        labs = [entry for entry in values if entry["kind"] == "lab"]
        self.assertEqual([entry["data"]["value"] for entry in labs], [5, 15])
        self.assertEqual([entry["reference_flag"] for entry in labs], ["within_entered_range", "above_entered_range"])
        vital = next(entry for entry in values if entry["kind"] == "vital")
        self.assertEqual(vital["reference_flag"], "below_entered_range")
        self.assertIn("no diagnosis", chart.json()["limitations"])
        self.assertEqual(set(entry["kind"] for entry in values), {"medication", "allergy", "lab", "vital"})

    def test_validation_for_each_record_type_and_timestamps(self):
        bad = [
            {"kind": "lab", "label": "Synthetic"},
            {"kind": "vital", "label": "Synthetic", "value": 5},
            {"kind": "medication", "label": "Synthetic"},
            {"kind": "allergy", "label": "Synthetic"},
            {"kind": "history", "label": "Synthetic", "text": ""},
            {"kind": "care_plan", "label": "Synthetic", "text": ""},
            {"kind": "lab", "label": "Synthetic", "unit": "unit", "value": 1, "reference_low": 10, "reference_high": 1},
            {"kind": "history", "label": "Synthetic", "text": "Synthetic", "recorded_at": "2030-01-01T12:00:00"},
            {"kind": "history", "label": "Synthetic", "text": "Synthetic", "unaccepted": True},
        ]
        for body in bad:
            with self.subTest(body=body):
                self.assertEqual(self.client.post(self.prefix + "/entries", json=body).status_code, 422)
        self.assertEqual(self.client.post(self.prefix + "/appointments", json={"starts_at": "2030-01-01T12:00:00", "purpose": "Synthetic"}).status_code, 422)

    def test_care_plans_and_imaging_reports_require_approval_and_reapproval_after_edit(self):
        for kind in ("care_plan", "imaging_report"):
            entry = self.entry(kind=kind, label="Synthetic draft", text="Synthetic unreviewed text")
            self.assertEqual(entry["status"], "draft")
            self.login("patient")
            self.assertNotIn(entry["id"], [row["id"] for row in self.client.get(self.prefix + "/entries").json()])
            self.assertEqual(self.client.post(f"/api/dev/chart-entries/{entry['id']}/approve", json={"revision": 1}).status_code, 403)
            self.login()
            approved = self.client.post(f"/api/dev/chart-entries/{entry['id']}/approve", json={"revision": 1})
            self.assertEqual(approved.status_code, 200, approved.text)
            self.assertEqual(approved.json()["status"], "approved")
            self.assertEqual(approved.json()["revision"], 2)
            self.assertEqual(self.client.post(f"/api/dev/chart-entries/{entry['id']}/approve", json={"revision": 1}).status_code, 409)
            self.login("patient")
            self.assertIn(entry["id"], [row["id"] for row in self.client.get(self.prefix + "/entries").json()])
            self.login()
            update = dict(entry["data"], revision=2, text="Edited synthetic text")
            changed = self.client.patch(f"/api/dev/chart-entries/{entry['id']}", json=update)
            self.assertEqual(changed.status_code, 200, changed.text)
            self.assertEqual(changed.json()["status"], "draft")
            self.assertIsNone(changed.json()["approved_by"])
            self.assertEqual(self.client.patch(f"/api/dev/chart-entries/{entry['id']}", json=update).status_code, 409)
            self.login("patient")
            self.assertNotIn(entry["id"], [row["id"] for row in self.client.get(self.prefix + "/entries").json()])
            self.login()

    def test_encounters_reject_cross_patient_links_and_closed_record_changes(self):
        encounter = self.client.post(self.prefix + "/encounters", json={"title": "Synthetic visit", "summary": "Fixture"}).json()
        entry = self.entry(encounter_id=encounter["id"])
        other_prefix = f"/api/dev/patients/{self.other['id']}"
        r = self.client.post(other_prefix + "/entries", json={"kind": "history", "label": "Synthetic", "text": "Synthetic", "encounter_id": encounter["id"]})
        self.assertEqual(r.status_code, 404)
        r = self.client.patch(f"/api/dev/encounters/{encounter['id']}", json={"status": "closed", "summary": "Reviewed synthetic visit"})
        self.assertEqual(r.status_code, 200, r.text)
        r = self.client.post(self.prefix + "/entries", json={"kind": "history", "label": "Synthetic", "text": "Synthetic", "encounter_id": encounter["id"]})
        self.assertEqual(r.status_code, 409)
        r = self.client.patch(f"/api/dev/chart-entries/{entry['id']}", json={**entry["data"], "encounter_id": None, "revision": 1})
        self.assertEqual(r.status_code, 409)
        self.login("patient")
        self.assertEqual(self.client.get(self.prefix + "/encounters").json()[0]["status"], "closed")

    def test_appointments_and_followup_are_internal_records_only(self):
        body = {"starts_at": "2030-01-01T12:00:00+01:00", "purpose": "Synthetic follow-up", "follow_up": "Synthetic instructions"}
        r = self.client.post(self.prefix + "/appointments", json=body)
        self.assertEqual(r.status_code, 200, r.text)
        appointment = r.json()
        self.assertEqual(appointment["starts_at"], "2030-01-01T11:00:00+00:00")
        self.login("patient")
        self.assertEqual(self.client.get(self.prefix + "/appointments").json()[0]["follow_up"], "Synthetic instructions")
        self.assertEqual(self.client.patch(f"/api/dev/appointments/{appointment['id']}", json={"status": "completed"}).status_code, 403)
        self.login()
        r = self.client.patch(f"/api/dev/appointments/{appointment['id']}", json={"status": "completed"})
        self.assertEqual(r.json()["status"], "completed")
        self.assertNotIn("delivered", r.json())

    def test_private_document_patient_grant_and_revocation_with_safe_download(self):
        self.login("patient")
        body = b"<script>synthetic fixture only</script>"
        r = self.upload(body, "../../user-supplied.txt")
        self.assertEqual(r.status_code, 200, r.text)
        document = r.json()
        self.assertEqual(document["sha256"], hashlib.sha256(body).hexdigest())
        path = f"/api/dev/documents/{document['id']}/content"
        self.login()
        self.assertEqual(self.client.get(path).status_code, 404)
        self.assertEqual(self.client.get(self.prefix + "/documents").json(), [])
        self.login("patient")
        r = self.client.post(self.prefix + "/document-consents", json={"clinician_username": "doctor", "granted": True})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertIn("not a general chart", r.json()["scope"])
        self.login()
        r = self.client.get(path)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.content, body)
        self.assertEqual(r.headers["content-type"], "application/octet-stream")
        self.assertIn("attachment; filename=\"synthetic-document-", r.headers["content-disposition"])
        self.assertNotIn("user-supplied", r.headers["content-disposition"])
        self.assertEqual(r.headers["x-content-type-options"], "nosniff")
        self.assertEqual(r.headers["cache-control"], "no-store")
        self.login("otherdoctor")
        self.assertEqual(self.client.get(path).status_code, 404)
        self.login("patient")
        self.assertEqual(self.client.post(self.prefix + "/document-consents", json={"clinician_username": "doctor", "granted": False}).status_code, 200)
        self.login()
        self.assertEqual(self.client.get(path).status_code, 404)

    def test_care_team_document_scope_and_invalid_consent_recipients(self):
        self.login("patient")
        r = self.upload(visibility="care_team")
        self.assertEqual(r.status_code, 200, r.text)
        path = f"/api/dev/documents/{r.json()['id']}/content"
        for name in ("doctor", "otherdoctor"):
            self.login(name)
            self.assertEqual(self.client.get(path).status_code, 200)
            self.assertEqual(self.client.post(self.prefix + "/document-consents", json={"clinician_username": "doctor", "granted": True}).status_code, 403)
        self.login("patient")
        for recipient in ("outsider", "admin", "patient", "unknown"):
            self.assertEqual(self.client.post(self.prefix + "/document-consents", json={"clinician_username": recipient, "granted": True}).status_code, 404)
        self.assertEqual(self.client.get(self.prefix + "/document-consents").json(), [])

    def test_document_type_size_pdf_validation_and_header_limits(self):
        for filename, content in (("bad.exe", b"MZ synthetic"), ("bad.txt", b"\xff\x00"), ("bad.pdf", b"%PDF-FAKE\n%%EOF")):
            r = self.upload(content, filename)
            self.assertEqual(r.status_code, 415, r.text)
        r = self.upload(b"x" * (MAX_DOCUMENT_BYTES + 1))
        self.assertEqual(r.status_code, 413, r.text)
        # Transfer-encoding path exercises the middleware even without Content-Length.
        r = self.client.post(self.prefix + "/documents", content=iter([b"x" * (MAX_DOCUMENT_BYTES + 65537)]),
                             headers={"Content-Type": "multipart/form-data; boundary=synthetic"})
        self.assertEqual(r.status_code, 413, r.text)
        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        output = io.BytesIO()
        writer.write(output)
        r = self.upload(output.getvalue(), "valid.pdf")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["file_type"], "pdf")
        writer.encrypt("Synthetic-only-password")
        output = io.BytesIO()
        writer.write(output)
        self.assertEqual(self.upload(output.getvalue(), "encrypted.pdf").status_code, 415)
        with self.engine.connect() as c:
            self.assertEqual(c.execute(text("SELECT count(*) FROM dev_chart_documents")).scalar(), 1)

    def test_cross_tenant_and_other_patient_denied_for_all_chart_resources(self):
        entry = self.entry()
        encounter = self.client.post(self.prefix + "/encounters", json={"title": "Synthetic"}).json()
        appointment = self.client.post(self.prefix + "/appointments", json={"starts_at": "2030-01-01T12:00:00Z", "purpose": "Synthetic"}).json()
        document = self.upload(visibility="care_team").json()
        for name, tenant in (("outsider", "beta"), ("otherpatient", "alpha")):
            self.login(name, tenant)
            for suffix in ("/chart", "/entries", "/encounters", "/appointments", "/documents", "/document-consents"):
                self.assertEqual(self.client.get(self.prefix + suffix).status_code, 404, suffix)
            self.assertEqual(self.client.get(f"/api/dev/documents/{document['id']}/content").status_code, 404)
            if name == "outsider":
                for path, body in ((f"/api/dev/chart-entries/{entry['id']}", {**entry["data"], "revision": 1}),
                                   (f"/api/dev/encounters/{encounter['id']}", {"status": "closed", "summary": "Synthetic"}),
                                   (f"/api/dev/appointments/{appointment['id']}", {"status": "cancelled"})):
                    self.assertEqual(self.client.patch(path, json=body).status_code, 404)
        self.login("admin")
        self.assertEqual(self.client.get(self.prefix + "/chart").status_code, 403)
        self.assertEqual(self.client.get(f"/api/dev/documents/{document['id']}/content").status_code, 403)

    def test_patient_cannot_write_chart_or_schedule_and_csrf_is_enforced(self):
        self.login("patient")
        for path, body in (("/entries", {"kind": "history", "label": "Synthetic", "text": "Synthetic"}),
                           ("/encounters", {"title": "Synthetic"}),
                           ("/appointments", {"starts_at": "2030-01-01T12:00:00Z", "purpose": "Synthetic"})):
            self.assertEqual(self.client.post(self.prefix + path, json=body).status_code, 403)
        r = self.client.post(self.prefix + "/documents", files={"file": ("test.txt", b"Synthetic")}, data={"title": "Synthetic"},
                             headers={"Origin": "https://untrusted.invalid"})
        self.assertEqual(r.status_code, 403)
        self.login()
        self.assertEqual(self.client.post(self.prefix + "/entries", json={"kind": "history", "label": "Synthetic", "text": "Synthetic"},
                                         headers={"Origin": "https://untrusted.invalid"}).status_code, 403)

    def test_repeat_schema_initialization_preserves_records_and_audit_excludes_content(self):
        entry = self.entry()
        self.upload()
        init_chart_schema(self.engine)
        self.engine.dispose()
        chart = self.client.get(self.prefix + "/chart")
        self.assertEqual(chart.status_code, 200, chart.text)
        self.assertEqual(chart.json()["entries"][0]["id"], entry["id"])
        self.assertEqual(len(chart.json()["documents"]), 1)
        with self.engine.connect() as c:
            audit = [tuple(row) for row in c.execute(text("SELECT * FROM dev_audit"))]
            serialized = str(audit)
            self.assertIn("create_chart_entry", serialized)
            self.assertNotIn("Nonclinical synthetic fixture", serialized)
            self.assertNotIn(self.password, serialized)
