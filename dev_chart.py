"""Structured synthetic charts and bounded patient-controlled document sharing.

These are development records, not clinical interpretation, consent certification,
malware scanning, care delivery, or an EHR integration.
"""
import hashlib
import io
import json
import re
from datetime import datetime, timezone
from typing import Literal
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response
from starlette.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic import StrictBool,StrictInt
from pypdf import PdfReader
from sqlalchemy import text

from database import execute_sqlite_script
from dev_portal import _require_synthetic

MAX_DOCUMENT_BYTES = 5 * 1024 * 1024


class DocumentLimitMiddleware:
    """Bound multipart bodies before the parser can spool an arbitrary file.

    The bounded development limit includes 64 KiB for multipart metadata. It
    also covers chunked bodies, rather than trusting Content-Length alone.
    """
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") != "POST" or not re.fullmatch(
                r"/api/dev/patients/\d+/documents", scope.get("path", "")):
            return await self.app(scope, receive, send)
        cap = MAX_DOCUMENT_BYTES + 65536
        for key, value in scope.get("headers", []):
            if key.lower() == b"content-length":
                try:
                    if int(value) > cap:
                        return await JSONResponse({"detail": "Document request exceeds the 5 MiB file limit."}, 413)(scope, receive, send)
                except ValueError:
                    return await JSONResponse({"detail": "Invalid request length."}, 400)(scope, receive, send)
        chunks, size = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            size += len(chunk)
            if size > cap:
                return await JSONResponse({"detail": "Document request exceeds the 5 MiB file limit."}, 413)(scope, receive, send)
            chunks.append(chunk)
            if not message.get("more_body", False):
                break
        body = b"".join(chunks)
        delivered = False

        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        return await self.app(scope, replay, send)
SCHEMA = """
CREATE TABLE IF NOT EXISTS dev_chart_encounters (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 patient_id INTEGER NOT NULL REFERENCES dev_patients(id),
 author_id INTEGER NOT NULL REFERENCES dev_users(id),
 title TEXT NOT NULL,
 occurred_at TEXT NOT NULL,
 status TEXT NOT NULL CHECK(status IN ('open','closed')),
 summary TEXT NOT NULL,
 created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS dev_chart_entries (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 patient_id INTEGER NOT NULL REFERENCES dev_patients(id),
 encounter_id INTEGER REFERENCES dev_chart_encounters(id),
 author_id INTEGER NOT NULL REFERENCES dev_users(id),
 kind TEXT NOT NULL,
 label TEXT NOT NULL,
 recorded_at TEXT NOT NULL,
 data TEXT NOT NULL,
 status TEXT NOT NULL CHECK(status IN ('recorded','draft','approved')),
 revision INTEGER NOT NULL DEFAULT 1,
 approved_by INTEGER REFERENCES dev_users(id),
 approved_at TEXT,
 created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS dev_chart_entries_patient_time ON dev_chart_entries(patient_id,recorded_at,id);
CREATE TABLE IF NOT EXISTS dev_chart_appointments (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 patient_id INTEGER NOT NULL REFERENCES dev_patients(id),
 author_id INTEGER NOT NULL REFERENCES dev_users(id),
 starts_at TEXT NOT NULL,
 purpose TEXT NOT NULL,
 location TEXT NOT NULL,
 follow_up TEXT NOT NULL,
 status TEXT NOT NULL CHECK(status IN ('scheduled','completed','cancelled')),
 created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS dev_chart_documents (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 patient_id INTEGER NOT NULL REFERENCES dev_patients(id),
 uploaded_by INTEGER NOT NULL REFERENCES dev_users(id),
 title TEXT NOT NULL,
 file_type TEXT NOT NULL CHECK(file_type IN ('txt','pdf')),
 visibility TEXT NOT NULL CHECK(visibility IN ('private','care_team')),
 content BLOB NOT NULL,
 size_bytes INTEGER NOT NULL,
 sha256 TEXT NOT NULL,
 created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS dev_chart_document_consents (
 patient_id INTEGER NOT NULL REFERENCES dev_patients(id),
 clinician_id INTEGER NOT NULL REFERENCES dev_users(id),
 granted INTEGER NOT NULL CHECK(granted IN (0,1)),
 updated_at TEXT NOT NULL,
 PRIMARY KEY(patient_id,clinician_id)
);
CREATE TABLE IF NOT EXISTS dev_chart_observation_imports (
 document_id INTEGER NOT NULL REFERENCES dev_chart_documents(id),
 line_number INTEGER NOT NULL,
 document_sha256 TEXT NOT NULL,
 entry_id INTEGER NOT NULL REFERENCES dev_chart_entries(id),
 approved_by INTEGER NOT NULL REFERENCES dev_users(id),
 created_at TEXT NOT NULL,
 PRIMARY KEY(document_id,line_number)
);
"""


def now():
    return datetime.now(timezone.utc).isoformat()


def timestamp(value):
    if value is None:
        return now()
    if value.tzinfo is None:
        raise ValueError("Timestamp must include a time zone.")
    return value.astimezone(timezone.utc).isoformat()


def init_chart_schema(engine):
    _require_synthetic(engine)
    with engine.begin() as c:
        execute_sqlite_script(c, SCHEMA)


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, allow_inf_nan=False)


class EncounterIn(Input):
    title: str = Field(min_length=1, max_length=200)
    occurred_at: datetime | None = None
    summary: str = Field(default="", max_length=10000)

    @model_validator(mode="after")
    def timezone(self):
        timestamp(self.occurred_at)
        return self


class EncounterUpdate(Input):
    status: Literal["open", "closed"]
    summary: str = Field(max_length=10000)


class EntryIn(Input):
    kind: Literal["history", "medication", "allergy", "lab", "vital", "imaging_report", "care_plan"]
    label: str = Field(min_length=1, max_length=200)
    recorded_at: datetime | None = None
    encounter_id: int | None = Field(default=None, gt=0)
    text: str = Field(default="", max_length=10000)
    value: float | None = None
    unit: str | None = Field(default=None, max_length=100)
    reference_low: float | None = None
    reference_high: float | None = None
    dose: str | None = Field(default=None, max_length=200)
    reaction: str | None = Field(default=None, max_length=1000)
    severity: Literal["mild", "moderate", "severe", "unknown"] | None = None
    source: str = Field(default="Manual synthetic entry", min_length=1, max_length=1000)

    @model_validator(mode="after")
    def complete_entry(self):
        timestamp(self.recorded_at)
        if self.kind in ("lab", "vital") and (self.value is None or not self.unit):
            raise ValueError("Laboratory/vital records need a numeric value and unit.")
        if self.kind == "medication" and not self.dose:
            raise ValueError("Medication records need an entered dose; no dose calculation is performed.")
        if self.kind == "allergy" and not self.reaction:
            raise ValueError("Allergy records need a reaction description.")
        if self.kind in ("history", "imaging_report", "care_plan") and not self.text:
            raise ValueError("This record type needs text.")
        if self.reference_low is not None and self.reference_high is not None and self.reference_low > self.reference_high:
            raise ValueError("Reference interval must be ordered.")
        return self


class EntryUpdate(EntryIn):
    revision: int = Field(gt=0)


class Approve(Input):
    revision: int = Field(gt=0)


class ObservationApproval(Input):
    expected_sha256: str = Field(pattern=r'^[a-f0-9]{64}$')
    row_numbers: list[StrictInt] = Field(min_length=1,max_length=50)
    acknowledged: StrictBool


class AppointmentIn(Input):
    starts_at: datetime
    purpose: str = Field(min_length=1, max_length=2000)
    location: str = Field(default="Synthetic development appointment", max_length=200)
    follow_up: str = Field(default="", max_length=2000)

    @model_validator(mode="after")
    def timezone(self):
        timestamp(self.starts_at)
        return self


class AppointmentUpdate(Input):
    status: Literal["scheduled", "completed", "cancelled"]


class ConsentIn(Input):
    clinician_username: str = Field(min_length=1, max_length=100)
    granted: bool


def install_chart(app, ctx):
    engine = ctx.engine
    router = APIRouter(prefix="/api/dev", dependencies=[Depends(ctx.request_guard)], tags=["synthetic structured chart"])

    def chart_user(user=Depends(ctx.user)):
        if user["role"] not in ("patient", "clinician"):
            raise HTTPException(403, "A patient or clinician chart role is required.")
        return user

    def clinician(user=Depends(chart_user)):
        if user["role"] != "clinician":
            raise HTTPException(403, "Clinician role required.")
        return user

    def encounter_for(c, encounter_id, user):
        row = c.execute(text("SELECT * FROM dev_chart_encounters WHERE id=:id"), {"id": encounter_id}).mappings().first()
        if not row:
            raise HTTPException(404, "Encounter not found.")
        ctx.patient_for(c, row["patient_id"], user)
        return dict(row)

    def entry_for(c, entry_id, user):
        row = c.execute(text("SELECT * FROM dev_chart_entries WHERE id=:id"), {"id": entry_id}).mappings().first()
        if not row:
            raise HTTPException(404, "Chart entry not found.")
        ctx.patient_for(c, row["patient_id"], user)
        if user["role"] == "patient" and row["status"] == "draft":
            raise HTTPException(404, "Chart entry not found.")
        return dict(row)

    def entry_out(row):
        result = dict(row)
        result["data"] = json.loads(result["data"])
        values = result["data"]
        value = values.get("value")
        low, high = values.get("reference_low"), values.get("reference_high")
        result["reference_flag"] = None
        if value is not None and (low is not None or high is not None):
            result["reference_flag"] = "below_entered_range" if low is not None and value < low else (
                "above_entered_range" if high is not None and value > high else "within_entered_range")
        return result

    def entries_for(c, patient_id, user):
        ctx.patient_for(c, patient_id, user)
        clause = " AND status!='draft'" if user["role"] == "patient" else ""
        return [entry_out(row) for row in c.execute(text("SELECT * FROM dev_chart_entries WHERE patient_id=:p" + clause +
                                                       " ORDER BY recorded_at,id"), {"p": patient_id}).mappings()]

    def document_allowed(c, row, user):
        ctx.patient_for(c, row["patient_id"], user)
        if user["role"] == "patient" or row["visibility"] == "care_team" or row["uploaded_by"] == user["id"]:
            return True
        return bool(c.execute(text("SELECT granted FROM dev_chart_document_consents WHERE patient_id=:p AND clinician_id=:u"),
                              {"p": row["patient_id"], "u": user["id"]}).scalar())

    def documents_for(c, patient_id, user):
        ctx.patient_for(c, patient_id, user)
        rows = c.execute(text("SELECT id,patient_id,uploaded_by,title,file_type,visibility,size_bytes,sha256,created_at "
                              "FROM dev_chart_documents WHERE patient_id=:p ORDER BY id"), {"p": patient_id}).mappings()
        return [dict(row) for row in rows if document_allowed(c, row, user)]

    def encounters_for(c, patient_id, user):
        ctx.patient_for(c, patient_id, user)
        return [dict(row) for row in c.execute(text("SELECT * FROM dev_chart_encounters WHERE patient_id=:p ORDER BY occurred_at,id"),
                                             {"p": patient_id}).mappings()]

    def appointments_for(c, patient_id, user):
        ctx.patient_for(c, patient_id, user)
        return [dict(row) for row in c.execute(text("SELECT * FROM dev_chart_appointments WHERE patient_id=:p ORDER BY starts_at,id"),
                                             {"p": patient_id}).mappings()]

    @router.get("/patients/{patient_id}/chart")
    def chart(patient_id: int, user=Depends(chart_user)):
        with engine.begin() as c:
            patient = ctx.patient_for(c, patient_id, user)
            result = {"patient": patient, "encounters": encounters_for(c, patient_id, user),
                      "entries": entries_for(c, patient_id, user), "appointments": appointments_for(c, patient_id, user),
                      "documents": documents_for(c, patient_id, user),
                      "limitations": "Synthetic records only. Range flags compare entered values and ranges; no diagnosis, dosing, reminder delivery or clinical validation is performed. Document consent applies only to private document access, not all chart access."}
            from longitudinal import review
            result['longitudinal']=review(result['entries'])
            ctx.audit(c, user, "read_structured_chart", "patient", patient_id)
            return result

    @router.get("/patients/{patient_id}/encounters")
    def encounters(patient_id: int, user=Depends(chart_user)):
        with engine.begin() as c:
            return encounters_for(c, patient_id, user)

    @router.post("/patients/{patient_id}/encounters", dependencies=[Depends(ctx.same_origin_json)])
    def create_encounter(patient_id: int, body: EncounterIn, user=Depends(clinician)):
        with engine.begin() as c:
            ctx.patient_for(c, patient_id, user)
            result = c.execute(text("INSERT INTO dev_chart_encounters(patient_id,author_id,title,occurred_at,status,summary,created_at) "
                                    "VALUES (:p,:u,:t,:at,'open',:s,:now)"), {"p": patient_id, "u": user["id"], "t": body.title,
                                    "at": timestamp(body.occurred_at), "s": body.summary, "now": now()})
            ctx.audit(c, user, "create_encounter", "encounter", result.lastrowid)
            return encounter_for(c, result.lastrowid, user)

    @router.patch("/encounters/{encounter_id}", dependencies=[Depends(ctx.same_origin_json)])
    def update_encounter(encounter_id: int, body: EncounterUpdate, user=Depends(clinician)):
        with engine.begin() as c:
            encounter_for(c, encounter_id, user)
            c.execute(text("UPDATE dev_chart_encounters SET status=:s,summary=:v WHERE id=:id"),
                      {"s": body.status, "v": body.summary, "id": encounter_id})
            ctx.audit(c, user, "update_encounter", "encounter", encounter_id)
            return encounter_for(c, encounter_id, user)

    @router.get("/patients/{patient_id}/entries")
    def entries(patient_id: int, user=Depends(chart_user)):
        with engine.begin() as c:
            return entries_for(c, patient_id, user)

    @router.post("/patients/{patient_id}/entries", dependencies=[Depends(ctx.same_origin_json)])
    def create_entry(patient_id: int, body: EntryIn, user=Depends(clinician)):
        with engine.begin() as c:
            ctx.patient_for(c, patient_id, user)
            if body.encounter_id:
                encounter = encounter_for(c, body.encounter_id, user)
                if encounter["patient_id"] != patient_id:
                    raise HTTPException(404, "Encounter not found.")
                if encounter["status"] == "closed":
                    raise HTTPException(409, "Reopen the encounter before adding records.")
            data = body.model_dump(mode="json")
            result = c.execute(text("INSERT INTO dev_chart_entries(patient_id,encounter_id,author_id,kind,label,recorded_at,data,status,created_at) "
                                    "VALUES (:p,:e,:u,:k,:l,:at,:data,:s,:now)"), {"p": patient_id, "e": body.encounter_id,
                                    "u": user["id"], "k": body.kind, "l": body.label, "at": timestamp(body.recorded_at),
                                    "data": json.dumps(data), "s": "draft" if body.kind in ("care_plan", "imaging_report") else "recorded", "now": now()})
            ctx.audit(c, user, "create_chart_entry", "chart_entry", result.lastrowid)
            return entry_out(entry_for(c, result.lastrowid, user))

    @router.patch("/chart-entries/{entry_id}", dependencies=[Depends(ctx.same_origin_json)])
    def update_entry(entry_id: int, body: EntryUpdate, user=Depends(clinician)):
        with engine.begin() as c:
            existing = entry_for(c, entry_id, user)
            if body.kind != existing["kind"]:
                raise HTTPException(409, "An existing record's kind cannot change.")
            if existing["encounter_id"] and encounter_for(c, existing["encounter_id"], user)["status"] == "closed":
                raise HTTPException(409, "Reopen the encounter before changing its records.")
            if body.encounter_id:
                encounter = encounter_for(c, body.encounter_id, user)
                if encounter["patient_id"] != existing["patient_id"]:
                    raise HTTPException(404, "Encounter not found.")
                if encounter["status"] == "closed":
                    raise HTTPException(409, "Reopen the encounter before changing records.")
            data = body.model_dump(mode="json", exclude={"revision"})
            provenance=json.loads(existing['data']).get('extraction_provenance')
            if provenance:data['extraction_provenance']=provenance
            result = c.execute(text("UPDATE dev_chart_entries SET label=:l,encounter_id=:e,recorded_at=:at,data=:data,status=:s,"
                                    "revision=revision+1,approved_by=NULL,approved_at=NULL WHERE id=:id AND revision=:r"),
                               {"l": body.label, "e": body.encounter_id, "at": timestamp(body.recorded_at), "data": json.dumps(data),
                                "s": "draft" if provenance or body.kind in ("care_plan", "imaging_report") else "recorded", "id": entry_id, "r": body.revision})
            if result.rowcount != 1:
                raise HTTPException(409, "The chart entry changed; reload before editing.")
            ctx.audit(c, user, "update_chart_entry", "chart_entry", entry_id)
            return entry_out(entry_for(c, entry_id, user))

    @router.post("/chart-entries/{entry_id}/approve", dependencies=[Depends(ctx.same_origin_json)])
    def approve_entry(entry_id: int, body: Approve, user=Depends(clinician)):
        with engine.begin() as c:
            existing = entry_for(c, entry_id, user)
            if existing["status"] != "draft":
                raise HTTPException(409, "Only draft records can be approved.")
            result = c.execute(text("UPDATE dev_chart_entries SET status='approved',approved_by=:u,approved_at=:at,revision=revision+1 "
                                    "WHERE id=:id AND status='draft' AND revision=:r"),
                               {"u": user["id"], "at": now(), "id": entry_id, "r": body.revision})
            if result.rowcount != 1:
                raise HTTPException(409, "The draft changed; reload before approving.")
            ctx.audit(c, user, "approve_chart_entry", "chart_entry", entry_id)
            return entry_out(entry_for(c, entry_id, user))

    @router.get("/patients/{patient_id}/appointments")
    def appointments(patient_id: int, user=Depends(chart_user)):
        with engine.begin() as c:
            return appointments_for(c, patient_id, user)

    @router.post("/patients/{patient_id}/appointments", dependencies=[Depends(ctx.same_origin_json)])
    def create_appointment(patient_id: int, body: AppointmentIn, user=Depends(clinician)):
        with engine.begin() as c:
            ctx.patient_for(c, patient_id, user)
            result = c.execute(text("INSERT INTO dev_chart_appointments(patient_id,author_id,starts_at,purpose,location,follow_up,status,created_at) "
                                    "VALUES (:p,:u,:at,:v,:l,:f,'scheduled',:now)"), {"p": patient_id, "u": user["id"], "at": timestamp(body.starts_at),
                                    "v": body.purpose, "l": body.location, "f": body.follow_up, "now": now()})
            ctx.audit(c, user, "create_appointment", "appointment", result.lastrowid)
            return dict(c.execute(text("SELECT * FROM dev_chart_appointments WHERE id=:id"), {"id": result.lastrowid}).mappings().one())

    @router.patch("/appointments/{appointment_id}", dependencies=[Depends(ctx.same_origin_json)])
    def update_appointment(appointment_id: int, body: AppointmentUpdate, user=Depends(clinician)):
        with engine.begin() as c:
            row = c.execute(text("SELECT * FROM dev_chart_appointments WHERE id=:id"), {"id": appointment_id}).mappings().first()
            if not row:
                raise HTTPException(404, "Appointment not found.")
            ctx.patient_for(c, row["patient_id"], user)
            c.execute(text("UPDATE dev_chart_appointments SET status=:s WHERE id=:id"), {"s": body.status, "id": appointment_id})
            ctx.audit(c, user, "update_appointment", "appointment", appointment_id)
            return dict(c.execute(text("SELECT * FROM dev_chart_appointments WHERE id=:id"), {"id": appointment_id}).mappings().one())

    @router.get("/patients/{patient_id}/documents")
    def documents(patient_id: int, user=Depends(chart_user)):
        with engine.begin() as c:
            return documents_for(c, patient_id, user)

    async def same_origin_upload(request: Request):
        origin = request.headers.get("origin")
        expected = urlsplit(str(request.url))
        if origin:
            actual = urlsplit(origin)
            if (actual.scheme, actual.netloc) != (expected.scheme, expected.netloc):
                raise HTTPException(403, "Cross-origin document upload is forbidden.")
        if request.headers.get("content-type", "").split(";", 1)[0].lower() != "multipart/form-data":
            raise HTTPException(415, "Document upload requires multipart/form-data.")

    @router.post("/patients/{patient_id}/documents", dependencies=[Depends(same_origin_upload)])
    async def upload_document(patient_id: int, title: str = Form(..., min_length=1, max_length=200),
                              visibility: Literal["private", "care_team"] = Form("private"),
                              file: UploadFile = File(...), user=Depends(chart_user)):
        with engine.begin() as c:
            ctx.patient_for(c, patient_id, user)
        try:
            if not title.strip():
                raise HTTPException(422, "Document title must be nonempty.")
            content = await file.read(MAX_DOCUMENT_BYTES + 1)
            if not content or len(content) > MAX_DOCUMENT_BYTES:
                raise HTTPException(413, "Use a nonempty document of at most 5 MiB.")
            name = file.filename or ""
            extension = name.rsplit(".", 1)[-1].lower() if "." in name else ""
            if extension == "txt":
                try:
                    decoded = content.decode("utf-8")
                    if "\x00" in decoded:
                        raise ValueError("NUL bytes")
                except (UnicodeDecodeError, ValueError):
                    raise HTTPException(415, "Text documents must contain UTF-8 text without NUL bytes.")
            elif extension == "pdf":
                try:
                    if not content.startswith(b"%PDF-") or b"%%EOF" not in content[-1024:]:
                        raise ValueError("Missing PDF structure")
                    pdf = PdfReader(io.BytesIO(content), strict=True)
                    if pdf.is_encrypted or not 0 < len(pdf.pages) <= 100:
                        raise ValueError("Encrypted or excessive page count")
                except Exception:
                    raise HTTPException(415, "Use a readable, unencrypted PDF with 1 to 100 pages.")
            else:
                raise HTTPException(415, "Only .txt and .pdf documents are accepted.")
            with engine.begin() as c:
                ctx.patient_for(c, patient_id, user)
                result = c.execute(text("INSERT INTO dev_chart_documents(patient_id,uploaded_by,title,file_type,visibility,content,size_bytes,sha256,created_at) "
                                        "VALUES (:p,:u,:t,:f,:v,:b,:s,:h,:at)"), {"p": patient_id, "u": user["id"], "t": title.strip(),
                                        "f": extension, "v": visibility, "b": content, "s": len(content), "h": hashlib.sha256(content).hexdigest(), "at": now()})
                ctx.audit(c, user, "upload_chart_document", "document", result.lastrowid)
                return next(item for item in documents_for(c, patient_id, user) if item["id"] == result.lastrowid)
        finally:
            await file.close()

    def extraction_source(c,document_id,user):
        row=c.execute(text('SELECT * FROM dev_chart_documents WHERE id=:id'),{'id':document_id}).mappings().first()
        if not row or not document_allowed(c,row,user):raise HTTPException(404,'Document not found.')
        if row['file_type']!='txt':raise HTTPException(422,'Only the documented synthetic tabular text format is supported.')
        content=bytes(row['content'])
        if hashlib.sha256(content).hexdigest()!=row['sha256']:raise HTTPException(409,'Document integrity changed;review the source before extraction.')
        from observation_extraction import preview
        try:result=preview(content,row['patient_id'])
        except ValueError as error:raise HTTPException(422,str(error)) from None
        return row,result

    @router.get('/documents/{document_id}/observation-preview')
    def observation_preview(document_id:int,user=Depends(clinician)):
        with engine.begin() as c:
            row,result=extraction_source(c,document_id,user)
            imports={r['line_number']:r['entry_id'] for r in c.execute(text('SELECT line_number,entry_id FROM dev_chart_observation_imports WHERE document_id=:d'),{'d':document_id}).mappings()}
            for candidate in result['rows']:candidate['imported_entry_id']=imports.get(candidate['line_number'])
            ctx.audit(c,user,'preview_observation_extraction','document',document_id)
            return {**result,'document_id':document_id,'document_sha256':row['sha256']}

    @router.post('/documents/{document_id}/approve-observations',dependencies=[Depends(ctx.same_origin_json)])
    def approve_observations(document_id:int,body:ObservationApproval,user=Depends(clinician)):
        if not body.acknowledged or len(set(body.row_numbers))!=len(body.row_numbers):raise HTTPException(422,'Explicit review and unique source rows are required.')
        with engine.begin() as c:
            c.exec_driver_sql('BEGIN IMMEDIATE')
            row,result=extraction_source(c,document_id,user)
            if body.expected_sha256!=row['sha256']:raise HTTPException(409,'Source hash changed;preview again before approving.')
            candidates={r['line_number']:r for r in result['rows']}
            if set(body.row_numbers)-set(candidates):raise HTTPException(422,'Selected source rows are missing or invalid.')
            identifiers=[];created=0
            for number in body.row_numbers:
                previous=c.execute(text('SELECT entry_id FROM dev_chart_observation_imports WHERE document_id=:d AND line_number=:n'),{'d':document_id,'n':number}).scalar()
                if previous:identifiers.append(previous);continue
                candidate=candidates[number];fields=candidate['fields']
                duplicate=c.execute(text('SELECT id FROM dev_chart_entries WHERE patient_id=:p AND kind=:k AND label=:l AND recorded_at=:at'),{'p':row['patient_id'],'k':fields['kind'],'l':fields['label'],'at':fields['recorded_at']}).first()
                if duplicate:raise HTTPException(409,'An observation at this label/time already exists;reconcile it manually. No records were overwritten.')
                data=EntryIn(**fields,source=f"Reviewed document {document_id}; SHA-256 {row['sha256']}").model_dump(mode='json')
                data['extraction_provenance']={'document_id':document_id,'document_sha256':row['sha256'],'line_number':number,'line_sha256':candidate['line_sha256'],'parser_version':1,'original_fields':fields,'review':'Explicit clinician review;not independent clinical validation'}
                at=now()
                identifier=c.execute(text("INSERT INTO dev_chart_entries(patient_id,author_id,kind,label,recorded_at,data,status,approved_by,approved_at,created_at) VALUES (:p,:u,:k,:l,:recorded,:data,'approved',:u,:at,:at)"),{'p':row['patient_id'],'u':user['id'],'k':fields['kind'],'l':fields['label'],'recorded':fields['recorded_at'],'data':json.dumps(data),'at':at}).lastrowid
                c.execute(text('INSERT INTO dev_chart_observation_imports(document_id,line_number,document_sha256,entry_id,approved_by,created_at) VALUES (:d,:n,:h,:e,:u,:at)'),{'d':document_id,'n':number,'h':row['sha256'],'e':identifier,'u':user['id'],'at':at})
                identifiers.append(identifier);created+=1
            ctx.audit(c,user,'approve_observation_extraction','document',document_id)
            states=[dict(c.execute(text('SELECT id,status,revision FROM dev_chart_entries WHERE id=:id'),{'id':identifier}).mappings().one()) for identifier in identifiers]
            return {'entry_ids':identifiers,'entry_states':states,'entries_created':created,'status':'review_processed','notice':'New approved facts are chart records under full-chart permissions. Original document access remains separately controlled. Previously imported records retain their current state;later edits are not reapproved by retrying import.'}

    @router.get("/documents/{document_id}/content")
    def download_document(document_id: int, user=Depends(chart_user)):
        with engine.begin() as c:
            row = c.execute(text("SELECT * FROM dev_chart_documents WHERE id=:id"), {"id": document_id}).mappings().first()
            if not row or not document_allowed(c, row, user):
                raise HTTPException(404, "Document not found.")
            ctx.audit(c, user, "download_chart_document", "document", document_id)
            return Response(bytes(row["content"]), media_type="application/octet-stream", headers={
                "Content-Disposition": f'attachment; filename="synthetic-document-{document_id}.{row["file_type"]}"',
                "X-Content-Type-Options": "nosniff", "Cache-Control": "no-store"})

    @router.get("/patients/{patient_id}/document-consents")
    def consents(patient_id: int, user=Depends(chart_user)):
        with engine.begin() as c:
            ctx.patient_for(c, patient_id, user)
            if user["role"] != "patient":
                raise HTTPException(403, "Patient account controls private document consent.")
            return [dict(row) for row in c.execute(text("SELECT d.clinician_id,u.username AS clinician_username,d.granted,d.updated_at "
                                                       "FROM dev_chart_document_consents d JOIN dev_users u ON u.id=d.clinician_id WHERE d.patient_id=:p"),
                                                  {"p": patient_id}).mappings()]

    @router.post("/patients/{patient_id}/document-consents", dependencies=[Depends(ctx.same_origin_json)])
    def grant_consent(patient_id: int, body: ConsentIn, user=Depends(chart_user)):
        with engine.begin() as c:
            ctx.patient_for(c, patient_id, user)
            if user["role"] != "patient":
                raise HTTPException(403, "Patient account controls private document consent.")
            clinician_id = c.execute(text("SELECT id FROM dev_users WHERE tenant=:t AND username=:u AND role='clinician' AND active=1"),
                                     {"t": user["tenant"], "u": body.clinician_username}).scalar()
            if not clinician_id:
                raise HTTPException(404, "Active clinician account not found.")
            c.execute(text("INSERT INTO dev_chart_document_consents(patient_id,clinician_id,granted,updated_at) VALUES (:p,:u,:g,:at) "
                           "ON CONFLICT(patient_id,clinician_id) DO UPDATE SET granted=excluded.granted,updated_at=excluded.updated_at"),
                      {"p": patient_id, "u": clinician_id, "g": int(body.granted), "at": now()})
            ctx.audit(c, user, "grant_document_access" if body.granted else "revoke_document_access", "patient", patient_id)
            return {"patient_id": patient_id, "clinician_id": clinician_id, "clinician_username": body.clinician_username,
                    "granted": body.granted, "scope": "Patient-private documents only; not a general chart consent policy."}

    app.include_router(router)
    app.add_middleware(DocumentLimitMiddleware)
