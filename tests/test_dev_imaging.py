"""Actual synthetic DICOM parsing, pixel rendering and authorized API regressions."""
import io
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pydicom
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid
from sqlalchemy import create_engine, text

from dev_imaging import MAX_DICOM, init_imaging_schema, install_imaging, preview_png, validate_dicom
from dev_portal import create_dev_user, init_portal_schema, install_dev_portal


def synthetic_dicom(*, frames=1, photo="MONOCHROME2", **fields):
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.StudyInstanceUID, ds.SeriesInstanceUID = generate_uid(), generate_uid()
    ds.PatientID = "SYNTHETIC-FIXTURE"
    ds.PatientName = "SYNTHETIC^FIXTURE"
    ds.PatientIdentityRemoved = "YES"
    ds.Modality = "CT"
    ds.Rows, ds.Columns = 2, 3
    ds.SamplesPerPixel, ds.BitsAllocated, ds.BitsStored, ds.HighBit = 1, 16, 16, 15
    ds.PixelRepresentation, ds.PhotometricInterpretation = 0, photo
    if frames > 1:
        ds.NumberOfFrames = frames
    ds.PixelData = np.arange(6 * frames, dtype="<u2").tobytes()
    for key, value in fields.items():
        setattr(ds, key, value)
    output = io.BytesIO()
    ds.save_as(output, write_like_original=False)
    return output.getvalue()


class PixelTests(unittest.TestCase):
    def test_actual_pixels_render_as_png_and_frames_are_bounded(self):
        data = synthetic_dicom(frames=2)
        _, meta = validate_dicom(data)
        self.assertEqual((meta["rows"], meta["columns"], meta["frames"]), (2, 3, 2))
        image = Image.open(io.BytesIO(preview_png(data, 1)))
        self.assertEqual(image.size, (3, 2))
        self.assertEqual(list(image.getdata()), [0, 51, 102, 153, 204, 255])
        inverted = Image.open(io.BytesIO(preview_png(synthetic_dicom(photo="MONOCHROME1"))))
        self.assertEqual(list(inverted.getdata()), [255, 204, 153, 102, 51, 0])
        with self.assertRaisesRegex(ValueError, "out of range"):
            preview_png(data, 2)

    def test_invalid_and_non_synthetic_datasets_are_rejected(self):
        for data in (b"not dicom", synthetic_dicom(PatientIdentityRemoved="NO"),
                     synthetic_dicom(PatientID="UNDECLARED"), synthetic_dicom(PatientName="UNDECLARED"),
                     synthetic_dicom(Modality="OT"), synthetic_dicom(StudyInstanceUID="invalid"),
                     synthetic_dicom(Rows=2049), synthetic_dicom(SamplesPerPixel=3),
                     synthetic_dicom(PixelData=b"\0\0"), synthetic_dicom(BitsStored=17)):
            with self.subTest(length=len(data)), self.assertRaises(ValueError):
                validate_dicom(data)

    def test_compressed_transfer_syntax_is_rejected_before_decoding(self):
        ds = pydicom.dcmread(io.BytesIO(synthetic_dicom()))
        ds.file_meta.TransferSyntaxUID = "1.2.840.10008.1.2.4.50"
        ds.PixelData = pydicom.encaps.encapsulate([b"not a jpeg"])
        stream = io.BytesIO()
        ds.save_as(stream, write_like_original=False)
        with self.assertRaisesRegex(ValueError, "uncompressed"):
            validate_dicom(stream.getvalue())


class ImagingApiTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="raphael-imaging-test-")
        self.addCleanup(self.directory.cleanup)
        self.engine = create_engine("sqlite:///" + str(Path(self.directory.name) / "synthetic.db"))
        self.addCleanup(self.engine.dispose)
        flag = patch.dict(os.environ, {"RAPHAEL_DEV_SYNTHETIC_ONLY": "1", "FHIR_SANDBOX_BASE_URL": ""})
        flag.start(); self.addCleanup(flag.stop)
        init_portal_schema(self.engine)
        init_imaging_schema(self.engine)
        self.password = "Synthetic-imaging-password"
        for tenant, name, role in (("alpha", "clinician", "clinician"), ("alpha", "patient", "patient"),
                                   ("alpha", "other", "patient"), ("beta", "outsider", "clinician")):
            create_dev_user(self.engine, tenant, name, self.password, role)
        self.app = FastAPI()
        ctx = install_dev_portal(self.app, self.engine)
        install_imaging(self.app, ctx)
        self.client = TestClient(self.app)
        self.addCleanup(self.client.close)
        self.login()
        r = self.client.post("/api/dev/patients", json={"name": "Synthetic Imaging", "patient_username": "patient"})
        self.assertEqual(r.status_code, 200, r.text)
        self.patient_id = r.json()["id"]

    def login(self, name="clinician", tenant="alpha"):
        r = self.client.post("/api/dev/login", json={"tenant": tenant, "username": name, "password": self.password})
        self.assertEqual(r.status_code, 200, r.text)

    def upload(self, data=None, headers=None):
        return self.client.post(f"/api/dev/patients/{self.patient_id}/imaging/studies",
            content=data if data is not None else synthetic_dicom(),
            headers=headers or {"Content-Type": "application/dicom", "X-Raphael-Upload": "synthetic"})

    def test_upload_read_preview_and_repeated_schema_preserve_image(self):
        data = synthetic_dicom()
        r = self.upload(data)
        self.assertEqual(r.status_code, 200, r.text)
        study = r.json()
        self.assertNotIn("dicom", study)
        self.assertNotIn("PatientName", study["metadata"])
        self.assertEqual(self.upload(data).json()["id"], study["id"])
        init_imaging_schema(self.engine)
        self.login("patient")
        rows = self.client.get(f"/api/dev/patients/{self.patient_id}/imaging/studies").json()
        self.assertEqual(len(rows), 1)
        image = self.client.get(f"/api/dev/imaging/studies/{study['id']}/preview.png")
        self.assertEqual(image.status_code, 200)
        self.assertEqual(image.headers["content-type"], "image/png")
        self.assertEqual(Image.open(io.BytesIO(image.content)).size, (3, 2))
        self.assertEqual(self.client.get(f"/api/dev/imaging/studies/{study['id']}/preview.png?frame=2").status_code, 422)
        with self.engine.connect() as c:
            events = set(c.execute(text("SELECT event FROM dev_audit")).scalars())
        self.assertTrue({"upload_synthetic_dicom", "view_synthetic_image"} <= events)

    def test_conflicting_sop_id_preserves_original(self):
        data = synthetic_dicom()
        study = self.upload(data).json()
        ds = pydicom.dcmread(io.BytesIO(data)); ds.PixelData = np.ones(6, dtype="<u2").tobytes()
        stream = io.BytesIO(); ds.save_as(stream, write_like_original=False)
        self.assertEqual(self.upload(stream.getvalue()).status_code, 409)
        with self.engine.connect() as c:
            self.assertEqual(c.execute(text("SELECT dicom FROM dev_imaging_studies WHERE id=:id"), {"id": study["id"]}).scalar(), data)

    def test_concurrent_identical_uploads_store_one_image(self):
        data = synthetic_dicom()
        def send(_):
            with TestClient(self.app) as client:
                login = client.post("/api/dev/login", json={"tenant": "alpha", "username": "clinician", "password": self.password})
                self.assertEqual(login.status_code, 200)
                return client.post(f"/api/dev/patients/{self.patient_id}/imaging/studies", content=data,
                                   headers={"Content-Type": "application/dicom", "X-Raphael-Upload": "synthetic"})
        with ThreadPoolExecutor(max_workers=2) as pool:
            responses = list(pool.map(send, range(2)))
        self.assertEqual([r.status_code for r in responses], [200, 200])
        self.assertEqual(responses[0].json()["id"], responses[1].json()["id"])
        with self.engine.connect() as c:
            self.assertEqual(c.execute(text("SELECT count(*) FROM dev_imaging_studies")).scalar(), 1)

    def test_unauthorized_users_cannot_upload_or_view_other_patient_images(self):
        study = self.upload().json()
        for name, tenant in (("other", "alpha"), ("outsider", "beta")):
            self.login(name, tenant)
            self.assertEqual(self.client.get(f"/api/dev/patients/{self.patient_id}/imaging/studies").status_code, 404)
            self.assertEqual(self.client.get(f"/api/dev/imaging/studies/{study['id']}/preview.png").status_code, 404)
            self.assertIn(self.upload().status_code, (403, 404))
        self.login("patient")
        self.assertEqual(self.upload().status_code, 403)

    def test_invalid_size_type_origin_and_bad_content_fail(self):
        self.assertEqual(self.upload(b"invalid").status_code, 422)
        self.assertEqual(self.upload(b"x" * (MAX_DICOM + 1)).status_code, 413)
        self.assertEqual(self.upload(headers={"Content-Type": "text/plain"}).status_code, 415)
        self.assertEqual(self.upload(headers={"Content-Type": "application/dicom", "X-Raphael-Upload": "synthetic", "Origin": "https://untrusted.invalid"}).status_code, 403)
        with self.engine.connect() as c:
            self.assertEqual(c.execute(text("SELECT count(*) FROM dev_imaging_studies")).scalar(), 0)

    def test_reports_are_drafts_until_explicit_approval(self):
        study = self.upload().json()
        path = f"/api/dev/imaging/studies/{study['id']}/reports"
        r = self.client.post(path, json={"findings": "Synthetic manual fixture", "impression": "Not diagnostic", "recommendation": "Synthetic review"})
        self.assertEqual(r.status_code, 200, r.text)
        report = r.json()
        self.login("patient")
        self.assertEqual(self.client.get(path).json(), [])
        self.assertEqual(self.client.post(f"/api/dev/imaging/reports/{report['id']}/approve", json={}).status_code, 403)
        self.login()
        self.assertEqual(self.client.post(f"/api/dev/imaging/reports/{report['id']}/approve", json={}).status_code, 200)
        self.assertEqual(self.client.post(f"/api/dev/imaging/reports/{report['id']}/approve", json={}).status_code, 409)
        self.login("patient")
        self.assertEqual(self.client.get(path).json()[0]["status"], "approved")
        self.login("outsider", "beta")
        self.assertEqual(self.client.get(path).status_code, 404)

    def test_fhir_absent_configuration_is_explicit_and_does_not_import(self):
        r = self.client.get(f"/api/dev/patients/{self.patient_id}/fhir-preview?resource_id=synthetic")
        self.assertEqual(r.status_code, 503)
        self.assertIn("synthetic FHIR sandbox", r.json()["detail"])
        self.login("patient")
        self.assertEqual(self.client.get(f"/api/dev/patients/{self.patient_id}/fhir-preview?resource_id=synthetic").status_code, 403)
