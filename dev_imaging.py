"""Synthetic-only DICOM storage/viewing and manual report review.

No imaging model or automatic diagnostic interpretation is included.
"""
import hashlib
import io
import json
from datetime import datetime, timezone
from urllib.parse import urlsplit

import numpy as np
import pydicom
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import Response
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import text

from database import execute_sqlite_script
from integrations.fhir.sandbox import SandboxUnavailable, preview_patient

MAX_DICOM = 10 * 1024 * 1024
MAX_PIXELS = 4 * 1024 * 1024
TRANSFER_SYNTAXES = {"1.2.840.10008.1.2", "1.2.840.10008.1.2.1", "1.2.840.10008.1.2.2"}
MODALITIES = {"CT", "MR", "CR", "DX", "US"}
SCHEMA = """
CREATE TABLE IF NOT EXISTS dev_imaging_studies (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 patient_id INTEGER NOT NULL REFERENCES dev_patients(id),
 sop_instance_uid TEXT NOT NULL,
 sha256 TEXT NOT NULL,
 metadata TEXT NOT NULL,
 dicom BLOB NOT NULL,
 uploaded_by INTEGER NOT NULL REFERENCES dev_users(id),
 created_at TEXT NOT NULL,
 UNIQUE(patient_id, sop_instance_uid)
);
CREATE TABLE IF NOT EXISTS dev_imaging_reports (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 study_id INTEGER NOT NULL REFERENCES dev_imaging_studies(id),
 findings TEXT NOT NULL,
 impression TEXT NOT NULL,
 recommendation TEXT NOT NULL,
 author_id INTEGER NOT NULL REFERENCES dev_users(id),
 status TEXT NOT NULL CHECK(status IN ('draft','approved')),
 approved_by INTEGER REFERENCES dev_users(id),
 approved_at TEXT,
 created_at TEXT NOT NULL
);
"""


def init_imaging_schema(engine):
    from dev_portal import _require_synthetic
    _require_synthetic(engine)
    with engine.begin() as c:
        execute_sqlite_script(c, SCHEMA)


def _now():
    return datetime.now(timezone.utc).isoformat()


def validate_dicom(data):
    if not data or len(data) > MAX_DICOM:
        raise ValueError("DICOM must be nonempty and at most 10 MiB.")
    try:
        ds = pydicom.dcmread(io.BytesIO(data), force=False)
        if str(ds.file_meta.TransferSyntaxUID) not in TRANSFER_SYNTAXES:
            raise ValueError("Only uncompressed grayscale DICOM transfer syntaxes are supported.")
        for key in ("SOPInstanceUID", "SOPClassUID", "StudyInstanceUID", "SeriesInstanceUID"):
            value = getattr(ds, key, None)
            if not value or not pydicom.uid.UID(str(value)).is_valid:
                raise ValueError("Required valid DICOM identifiers are missing.")
        if str(getattr(ds, "Modality", "")) not in MODALITIES:
            raise ValueError("Supported modalities are CT, MR, CR, DX and US.")
        if str(getattr(ds, "PatientIdentityRemoved", "")) != "YES":
            raise ValueError("Only explicitly synthetic, identity-removed development DICOM is accepted.")
        if not str(getattr(ds, "PatientID", "")).startswith("SYNTHETIC-") or not str(getattr(ds, "PatientName", "")).upper().startswith("SYNTHETIC"):
            raise ValueError("Use an explicitly labeled SYNTHETIC test patient ID and name.")
        rows, columns = int(ds.Rows), int(ds.Columns)
        frames = int(getattr(ds, "NumberOfFrames", 1))
        if not (1 <= rows <= 2048 and 1 <= columns <= 2048 and 1 <= frames <= 64) or rows * columns * frames > MAX_PIXELS:
            raise ValueError("DICOM dimensions exceed the bounded development viewer.")
        if int(ds.SamplesPerPixel) != 1 or str(ds.PhotometricInterpretation) not in {"MONOCHROME1", "MONOCHROME2"}:
            raise ValueError("Only monochrome single-sample images are supported.")
        bits = int(ds.BitsAllocated)
        stored_bits = int(ds.BitsStored)
        if bits not in (8, 16) or not 1 <= stored_bits <= bits or int(ds.HighBit) != stored_bits - 1 or int(ds.PixelRepresentation) not in (0, 1):
            raise ValueError("Unsupported DICOM pixel representation.")
        expected = rows * columns * frames * (bits // 8)
        actual = len(ds.PixelData)
        if actual not in {expected, expected + (expected % 2)}:
            raise ValueError("DICOM pixel data length does not match its dimensions.")
        # Decode only after validating all allocation bounds and transfer syntax.
        pixels = ds.pixel_array
        if pixels.size != rows * columns * frames:
            raise ValueError("DICOM pixel dimensions are inconsistent.")
    except ValueError:
        raise
    except Exception:
        raise ValueError("Invalid or unsupported DICOM dataset.")
    metadata = {key: str(getattr(ds, key)) for key in (
        "SOPInstanceUID", "SOPClassUID", "StudyInstanceUID", "SeriesInstanceUID", "Modality")}
    metadata.update(rows=rows, columns=columns, frames=frames,
                    photometric_interpretation=str(ds.PhotometricInterpretation),
                    synthetic_declaration=True, viewer_limitation="Display only; no diagnostic interpretation or validated image model.")
    return ds, metadata


def preview_png(data, frame=0):
    ds, metadata = validate_dicom(data)
    if not 0 <= frame < metadata["frames"]:
        raise ValueError("Requested frame is out of range.")
    pixels = ds.pixel_array
    if metadata["frames"] > 1:
        pixels = pixels[frame]
    pixels = pixels.astype(np.float32)
    low, high = float(pixels.min()), float(pixels.max())
    scaled = np.zeros(pixels.shape, dtype=np.uint8) if high == low else np.rint((pixels-low) * 255.0 / (high-low)).astype(np.uint8)
    if metadata["photometric_interpretation"] == "MONOCHROME1":
        scaled = 255 - scaled
    output = io.BytesIO()
    Image.fromarray(scaled, mode="L").save(output, format="PNG")
    return output.getvalue()


class ReportInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    findings: str = Field(min_length=1, max_length=10000)
    impression: str = Field(min_length=1, max_length=10000)
    recommendation: str = Field(default="", max_length=5000)


def install_imaging(app, ctx):
    router = APIRouter(prefix="/api/dev", dependencies=[Depends(ctx.request_guard)], tags=["synthetic imaging and FHIR sandbox"])

    def study_for(c, study_id, user):
        study = c.execute(text("SELECT * FROM dev_imaging_studies WHERE id=:id"), {"id": study_id}).mappings().first()
        if not study:
            raise HTTPException(404, "Imaging study not found.")
        ctx.patient_for(c, study["patient_id"], user)
        return study

    def public_study(study):
        return {"id": study["id"], "patient_id": study["patient_id"], "sha256": study["sha256"],
                "metadata": json.loads(study["metadata"]), "created_at": study["created_at"]}

    @router.get("/patients/{patient_id}/imaging/studies")
    def studies(patient_id: int, user=Depends(ctx.user)):
        with ctx.engine.begin() as c:
            ctx.patient_for(c, patient_id, user)
            rows = c.execute(text("SELECT * FROM dev_imaging_studies WHERE patient_id=:p ORDER BY id DESC"), {"p": patient_id}).mappings().all()
            ctx.audit(c, user, "list_imaging", "patient", patient_id)
            return [public_study(row) for row in rows]

    @router.post("/patients/{patient_id}/imaging/studies")
    async def upload(patient_id: int, request: Request, user=Depends(ctx.clinician)):
        origin = request.headers.get("origin")
        if origin:
            supplied, expected = urlsplit(origin), urlsplit(str(request.url))
            if (supplied.scheme, supplied.netloc) != (expected.scheme, expected.netloc):
                raise HTTPException(403, "Cross-origin imaging upload is forbidden.")
        if request.headers.get("content-type", "").split(";", 1)[0].strip() != "application/dicom" or request.headers.get("x-raphael-upload") != "synthetic":
            raise HTTPException(415, "Use application/dicom and the explicit synthetic upload header.")
        with ctx.engine.begin() as c:
            ctx.patient_for(c, patient_id, user)
        data = bytearray()
        async for chunk in request.stream():
            data.extend(chunk)
            if len(data) > MAX_DICOM:
                raise HTTPException(413, "Development DICOM limit is 10 MiB.")
        try:
            _, metadata = validate_dicom(data)
        except ValueError as error:
            raise HTTPException(422, str(error))
        digest = hashlib.sha256(data).hexdigest()
        with ctx.engine.begin() as c:
            ctx.patient_for(c, patient_id, user)
            result = c.execute(text("INSERT INTO dev_imaging_studies(patient_id,sop_instance_uid,sha256,metadata,dicom,uploaded_by,created_at) "
                                    "VALUES (:p,:uid,:h,:m,:d,:u,:at) ON CONFLICT(patient_id,sop_instance_uid) DO NOTHING"), {"p": patient_id, "uid": metadata["SOPInstanceUID"],
                                    "h": digest, "m": json.dumps(metadata), "d": bytes(data), "u": user["id"], "at": _now()})
            stored = c.execute(text("SELECT * FROM dev_imaging_studies WHERE patient_id=:p AND sop_instance_uid=:uid"),
                               {"p": patient_id, "uid": metadata["SOPInstanceUID"]}).mappings().one()
            if stored["sha256"] != digest:
                raise HTTPException(409, "SOP instance UID already has different content; the original is preserved.")
            if result.rowcount == 1:
                ctx.audit(c, user, "upload_synthetic_dicom", "imaging_study", stored["id"])
            return public_study(stored)

    @router.get("/imaging/studies/{study_id}/preview.png")
    def preview(study_id: int, frame: int = Query(default=0, ge=0), user=Depends(ctx.user)):
        with ctx.engine.begin() as c:
            study = study_for(c, study_id, user)
            ctx.audit(c, user, "view_synthetic_image", "imaging_study", study_id)
            data = study["dicom"]
        try:
            image = preview_png(data, frame)
        except ValueError as error:
            raise HTTPException(422, str(error))
        return Response(image, media_type="image/png", headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"})

    @router.get("/imaging/studies/{study_id}/reports")
    def reports(study_id: int, user=Depends(ctx.user)):
        with ctx.engine.begin() as c:
            study_for(c, study_id, user)
            clause = " AND r.status='approved'" if user["role"] == "patient" else ""
            rows = c.execute(text("SELECT r.*,u.username AS author_username FROM dev_imaging_reports r JOIN dev_users u ON u.id=r.author_id "
                                  "WHERE r.study_id=:s" + clause + " ORDER BY r.id"), {"s": study_id}).mappings().all()
            ctx.audit(c, user, "read_imaging_reports", "imaging_study", study_id)
            return [dict(row) for row in rows]

    @router.post("/imaging/studies/{study_id}/reports", dependencies=[Depends(ctx.same_origin_json)])
    def create_report(study_id: int, body: ReportInput, user=Depends(ctx.clinician)):
        with ctx.engine.begin() as c:
            study_for(c, study_id, user)
            result = c.execute(text("INSERT INTO dev_imaging_reports(study_id,findings,impression,recommendation,author_id,status,created_at) "
                                    "VALUES (:s,:f,:i,:r,:u,'draft',:at)"), {"s": study_id, "f": body.findings,
                                    "i": body.impression, "r": body.recommendation, "u": user["id"], "at": _now()})
            ctx.audit(c, user, "draft_imaging_report", "imaging_report", result.lastrowid)
            return dict(c.execute(text("SELECT * FROM dev_imaging_reports WHERE id=:id"), {"id": result.lastrowid}).mappings().one())

    @router.post("/imaging/reports/{report_id}/approve", dependencies=[Depends(ctx.same_origin_json)])
    def approve_report(report_id: int, user=Depends(ctx.clinician)):
        with ctx.engine.begin() as c:
            report = c.execute(text("SELECT * FROM dev_imaging_reports WHERE id=:id"), {"id": report_id}).mappings().first()
            if not report:
                raise HTTPException(404, "Imaging report not found.")
            study_for(c, report["study_id"], user)
            result = c.execute(text("UPDATE dev_imaging_reports SET status='approved',approved_by=:u,approved_at=:at WHERE id=:id AND status='draft'"),
                               {"u": user["id"], "at": _now(), "id": report_id})
            if result.rowcount != 1:
                raise HTTPException(409, "Only a draft report can be approved.")
            ctx.audit(c, user, "approve_imaging_report", "imaging_report", report_id)
            return dict(c.execute(text("SELECT * FROM dev_imaging_reports WHERE id=:id"), {"id": report_id}).mappings().one())

    @router.get("/patients/{patient_id}/fhir-preview")
    async def fhir_preview(patient_id: int, resource_id: str = Query(min_length=1, max_length=64), user=Depends(ctx.clinician)):
        with ctx.engine.begin() as c:
            ctx.patient_for(c, patient_id, user)
        try:
            result = await preview_patient(resource_id)
        except ValueError:
            raise HTTPException(422, "Invalid FHIR resource ID.")
        except SandboxUnavailable as error:
            raise HTTPException(503, str(error))
        with ctx.engine.begin() as c:
            # A remote preview can outlive the access decision checked before
            # await. Do not disclose fetched records after grant revocation.
            ctx.patient_for(c, patient_id, user)
            ctx.audit(c, user, "preview_synthetic_fhir", "patient", patient_id)
        return result

    app.include_router(router)
