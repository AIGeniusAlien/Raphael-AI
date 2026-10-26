"""Opt-in, isolated synthetic-data portal. This is not production identity or care delivery."""

import hashlib
import hmac
import os
import secrets
from datetime import datetime, timedelta, timezone
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field, SecretStr
from sqlalchemy import text

from database import execute_sqlite_script

COOKIE_NAME = "raphael_dev_session"
SESSION_SECONDS = 8 * 60 * 60

SCHEMA = """
CREATE TABLE IF NOT EXISTS dev_users (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 tenant TEXT NOT NULL,
 username TEXT NOT NULL,
 role TEXT NOT NULL CHECK (role IN ('patient','clinician')),
 password_hash TEXT NOT NULL,
 UNIQUE(tenant, username)
);
CREATE TABLE IF NOT EXISTS dev_sessions (
 token_hash TEXT PRIMARY KEY,
 user_id INTEGER NOT NULL REFERENCES dev_users(id),
 expires_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS dev_patients (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 tenant TEXT NOT NULL,
 name TEXT NOT NULL,
 dob TEXT,
 mrn TEXT,
 patient_user_id INTEGER REFERENCES dev_users(id),
 created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS dev_conversations (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 patient_id INTEGER NOT NULL REFERENCES dev_patients(id),
 clinician_id INTEGER NOT NULL REFERENCES dev_users(id),
 title TEXT NOT NULL,
 created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS dev_messages (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 conversation_id INTEGER NOT NULL REFERENCES dev_conversations(id),
 sender_id INTEGER NOT NULL REFERENCES dev_users(id),
 content TEXT NOT NULL,
 created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS dev_notes (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 patient_id INTEGER NOT NULL REFERENCES dev_patients(id),
 author_id INTEGER NOT NULL REFERENCES dev_users(id),
 content TEXT NOT NULL,
 status TEXT NOT NULL CHECK (status IN ('draft','approved')),
 created_at TEXT NOT NULL,
 approved_by INTEGER REFERENCES dev_users(id),
 approved_at TEXT
);
CREATE TABLE IF NOT EXISTS dev_audit (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 tenant TEXT NOT NULL,
 actor_id INTEGER,
 event TEXT NOT NULL,
 resource_type TEXT,
 resource_id INTEGER,
 created_at TEXT NOT NULL
);
"""


def _now():
    return datetime.now(timezone.utc).isoformat()


def _require_synthetic(engine):
    """Acknowledgement plus a guard against accidentally reusing populated legacy state.

    This cannot prove that data supplied to dev_* tables is synthetic; operators
    remain responsible for using only an isolated synthetic database. Inspect
    names and existence flags only, never legacy patient/message content.
    """
    if os.getenv("RAPHAEL_DEV_SYNTHETIC_ONLY") != "1":
        raise RuntimeError("Portal requires RAPHAEL_DEV_SYNTHETIC_ONLY=1; synthetic data only.")
    if engine.dialect.name != "sqlite":
        raise RuntimeError("The synthetic portal supports only an isolated development SQLite database.")
    legacy_tables = {"patients", "conversations", "messages", "documents", "feedback"}
    with engine.connect() as c:
        existing = set(c.exec_driver_sql("SELECT name FROM sqlite_master WHERE type='table'").scalars())
        for table in sorted(legacy_tables & existing):
            # Table identifiers come exclusively from the static whitelist above.
            if c.exec_driver_sql(f'SELECT EXISTS(SELECT 1 FROM "{table}")').scalar():
                raise RuntimeError("Portal refuses a database containing legacy records; use a new isolated synthetic SQLite file.")


def init_portal_schema(engine):
    _require_synthetic(engine)
    with engine.begin() as c:
        execute_sqlite_script(c, SCHEMA)


def _password_hash(password, salt=None):
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=16384, r=8, p=1, dklen=64)
    return salt.hex() + ":" + digest.hex()


def _password_matches(password, stored):
    try:
        salt, _ = stored.split(":", 1)
        return hmac.compare_digest(_password_hash(password, bytes.fromhex(salt)), stored)
    except (ValueError, TypeError):
        return False


def create_dev_user(engine, tenant, username, password, role):
    """Provision a local synthetic account; no default password or public registration."""
    _require_synthetic(engine)
    if not (tenant.strip() and username.strip()) or len(tenant) > 100 or len(username) > 100:
        raise ValueError("Tenant and username must be nonempty and at most 100 characters.")
    if role not in ("patient", "clinician"):
        raise ValueError("Role must be patient or clinician.")
    if not 12 <= len(password) <= 128:
        raise ValueError("Use a development password between 12 and 128 characters.")
    with engine.begin() as c:
        result = c.execute(text("INSERT INTO dev_users(tenant,username,role,password_hash) VALUES (:t,:u,:r,:p)"),
                           {"t": tenant.strip(), "u": username.strip(), "r": role, "p": _password_hash(password)})
        return {"id": result.lastrowid, "tenant": tenant.strip(), "username": username.strip(), "role": role}


def _audit(c, user, event, resource_type=None, resource_id=None):
    c.execute(text("INSERT INTO dev_audit(tenant,actor_id,event,resource_type,resource_id,created_at) "
                   "VALUES (:t,:a,:e,:rt,:ri,:at)"), {"t": user["tenant"], "a": user["id"], "e": event,
                   "rt": resource_type, "ri": resource_id, "at": _now()})


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Login(Input):
    tenant: str = Field(min_length=1, max_length=100)
    username: str = Field(min_length=1, max_length=100)
    password: SecretStr = Field(min_length=1, max_length=128)


class PatientIn(Input):
    name: str = Field(min_length=1, max_length=200)
    dob: str | None = Field(default=None, max_length=30)
    mrn: str | None = Field(default=None, max_length=100)
    patient_username: str | None = Field(default=None, max_length=100)


class ConversationIn(Input):
    title: str = Field(default="Internal test conversation", min_length=1, max_length=200)


class ContentIn(Input):
    content: str = Field(min_length=1, max_length=20000)


class AssessmentIn(Input):
    conversation_id: int = Field(gt=0)
    question: str = Field(min_length=1, max_length=10000)


class NoteDraftIn(Input):
    conversation_id: int = Field(gt=0)


def install_dev_portal(app, engine, assessment_generator=None):
    """Register routes. The host initializes the schema during normal startup."""
    _require_synthetic(engine)
    router = APIRouter(prefix="/api/dev", tags=["synthetic development portal"])

    async def same_origin_json(request: Request):
        if request.method in ("POST", "PUT", "PATCH", "DELETE"):
            origin = request.headers.get("origin")
            expected = urlsplit(str(request.url))
            if origin:
                supplied = urlsplit(origin)
                if (supplied.scheme, supplied.netloc) != (expected.scheme, expected.netloc):
                    raise HTTPException(403, "Cross-origin portal mutations are forbidden.")
            if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
                raise HTTPException(415, "Portal mutations require application/json.")

    router.dependencies.append(Depends(same_origin_json))

    def current_user(request: Request):
        token = request.cookies.get(COOKIE_NAME, "")
        if not token:
            raise HTTPException(401, "Development login required.")
        with engine.begin() as c:
            row = c.execute(text("SELECT u.id,u.tenant,u.username,u.role FROM dev_sessions s "
                                 "JOIN dev_users u ON u.id=s.user_id "
                                 "WHERE s.token_hash=:h AND s.expires_at>:at"),
                            {"h": hashlib.sha256(token.encode()).hexdigest(), "at": _now()}).mappings().first()
            if not row:
                raise HTTPException(401, "Session expired or invalid.")
            return dict(row)

    def clinician(user=Depends(current_user)):
        if user["role"] != "clinician":
            raise HTTPException(403, "Clinician role required.")
        return user

    def patient_for(c, patient_id, user):
        row = c.execute(text("SELECT p.*,u.username AS patient_username FROM dev_patients p "
                             "LEFT JOIN dev_users u ON u.id=p.patient_user_id "
                             "WHERE p.id=:id AND p.tenant=:t"), {"id": patient_id, "t": user["tenant"]}).mappings().first()
        if not row or (user["role"] == "patient" and row["patient_user_id"] != user["id"]):
            raise HTTPException(404, "Patient not found.")
        return dict(row)

    def conversation_for(c, conversation_id, user):
        row = c.execute(text("SELECT * FROM dev_conversations WHERE id=:id"), {"id": conversation_id}).mappings().first()
        if not row:
            raise HTTPException(404, "Conversation not found.")
        patient_for(c, row["patient_id"], user)
        if user["role"] == "clinician" and row["clinician_id"] != user["id"]:
            raise HTTPException(404, "Conversation not found.")
        return dict(row)

    def note_for(c, note_id, user):
        row = c.execute(text("SELECT n.*,u.username AS author_username FROM dev_notes n "
                             "JOIN dev_users u ON u.id=n.author_id WHERE n.id=:id"), {"id": note_id}).mappings().first()
        if not row:
            raise HTTPException(404, "Note not found.")
        patient_for(c, row["patient_id"], user)
        if user["role"] == "patient" and row["status"] != "approved":
            raise HTTPException(404, "Note not found.")
        return dict(row)

    @router.post("/login")
    def login(body: Login, request: Request, response: Response):
        with engine.begin() as c:
            row = c.execute(text("SELECT * FROM dev_users WHERE tenant=:t AND username=:u"),
                            {"t": body.tenant, "u": body.username}).mappings().first()
            # Keep password hashing on the unknown-user path too.
            stored = row["password_hash"] if row else "00" * 16 + ":" + "00" * 64
            if not _password_matches(body.password.get_secret_value(), stored):
                raise HTTPException(401, "Invalid development credentials.")
            user = {k: row[k] for k in ("id", "tenant", "username", "role")}
            token = secrets.token_urlsafe(32)
            c.execute(text("INSERT INTO dev_sessions(token_hash,user_id,expires_at) VALUES (:h,:u,:e)"),
                      {"h": hashlib.sha256(token.encode()).hexdigest(), "u": row["id"],
                       "e": (datetime.now(timezone.utc) + timedelta(seconds=SESSION_SECONDS)).isoformat()})
            _audit(c, user, "login")
        response.set_cookie(COOKIE_NAME, token, max_age=SESSION_SECONDS, httponly=True,
                            secure=request.url.scheme == "https", samesite="strict", path="/api/dev")
        response.headers["Cache-Control"] = "no-store"
        return user

    @router.get("/session")
    def session(response: Response, user=Depends(current_user)):
        response.headers["Cache-Control"] = "no-store"
        return user

    @router.post("/logout")
    def logout(request: Request, response: Response, user=Depends(current_user)):
        token = request.cookies.get(COOKIE_NAME, "")
        with engine.begin() as c:
            c.execute(text("DELETE FROM dev_sessions WHERE token_hash=:h"),
                      {"h": hashlib.sha256(token.encode()).hexdigest()})
            _audit(c, user, "logout")
        response.delete_cookie(COOKIE_NAME, path="/api/dev", httponly=True, samesite="strict")
        return {"status": "logged_out"}

    @router.get("/patients")
    def patients(user=Depends(current_user)):
        with engine.begin() as c:
            where = "p.tenant=:t" + (" AND p.patient_user_id=:uid" if user["role"] == "patient" else "")
            rows = c.execute(text("SELECT p.*,u.username AS patient_username FROM dev_patients p "
                                  "LEFT JOIN dev_users u ON u.id=p.patient_user_id WHERE " + where + " ORDER BY p.id"),
                             {"t": user["tenant"], "uid": user["id"]}).mappings().all()
            _audit(c, user, "list_patients", "patient")
            return [dict(row) for row in rows]

    @router.post("/patients")
    def create_patient(body: PatientIn, user=Depends(clinician)):
        with engine.begin() as c:
            patient_user_id = None
            if body.patient_username:
                patient_user_id = c.execute(text("SELECT id FROM dev_users WHERE tenant=:t AND username=:u AND role='patient'"),
                                            {"t": user["tenant"], "u": body.patient_username}).scalar()
                if patient_user_id is None:
                    raise HTTPException(400, "Use an existing patient-role account in this tenant.")
            result = c.execute(text("INSERT INTO dev_patients(tenant,name,dob,mrn,patient_user_id,created_at) "
                                    "VALUES (:t,:n,:d,:m,:u,:at)"), {"t": user["tenant"], "n": body.name,
                                    "d": body.dob, "m": body.mrn, "u": patient_user_id, "at": _now()})
            patient = patient_for(c, result.lastrowid, user)
            _audit(c, user, "create_patient", "patient", patient["id"])
            return patient

    @router.get("/patients/{patient_id}")
    def patient(patient_id: int, user=Depends(current_user)):
        with engine.begin() as c:
            result = patient_for(c, patient_id, user)
            _audit(c, user, "read_patient", "patient", patient_id)
            return result

    @router.get("/patients/{patient_id}/conversations")
    def conversations(patient_id: int, user=Depends(current_user)):
        with engine.begin() as c:
            patient_for(c, patient_id, user)
            clause = " AND clinician_id=:uid" if user["role"] == "clinician" else ""
            rows = c.execute(text("SELECT * FROM dev_conversations WHERE patient_id=:p" + clause + " ORDER BY id"),
                             {"p": patient_id, "uid": user["id"]}).mappings().all()
            return [dict(row) for row in rows]

    @router.post("/patients/{patient_id}/conversations")
    def create_conversation(patient_id: int, body: ConversationIn, user=Depends(clinician)):
        with engine.begin() as c:
            patient_for(c, patient_id, user)
            result = c.execute(text("INSERT INTO dev_conversations(patient_id,clinician_id,title,created_at) VALUES (:p,:u,:t,:at)"),
                               {"p": patient_id, "u": user["id"], "t": body.title, "at": _now()})
            conversation = conversation_for(c, result.lastrowid, user)
            _audit(c, user, "create_conversation", "conversation", conversation["id"])
            return conversation

    @router.get("/conversations/{conversation_id}/messages")
    def messages(conversation_id: int, user=Depends(current_user)):
        with engine.begin() as c:
            conversation_for(c, conversation_id, user)
            rows = c.execute(text("SELECT m.*,u.username AS sender_username,u.role AS sender_role "
                                  "FROM dev_messages m JOIN dev_users u ON u.id=m.sender_id "
                                  "WHERE m.conversation_id=:cid ORDER BY m.id"), {"cid": conversation_id}).mappings().all()
            _audit(c, user, "read_messages", "conversation", conversation_id)
            return [dict(row, delivery_status="stored_in_portal") for row in rows]

    @router.post("/conversations/{conversation_id}/messages")
    def send_message(conversation_id: int, body: ContentIn, user=Depends(current_user)):
        with engine.begin() as c:
            conversation_for(c, conversation_id, user)
            result = c.execute(text("INSERT INTO dev_messages(conversation_id,sender_id,content,created_at) VALUES (:c,:u,:v,:at)"),
                               {"c": conversation_id, "u": user["id"], "v": body.content, "at": _now()})
            row = c.execute(text("SELECT * FROM dev_messages WHERE id=:id"), {"id": result.lastrowid}).mappings().first()
            _audit(c, user, "store_internal_message", "message", row["id"])
            return dict(row, sender_username=user["username"], sender_role=user["role"], delivery_status="stored_in_portal")

    @router.get("/patients/{patient_id}/notes")
    def notes(patient_id: int, user=Depends(current_user)):
        with engine.begin() as c:
            patient_for(c, patient_id, user)
            clause = " AND n.status='approved'" if user["role"] == "patient" else ""
            rows = c.execute(text("SELECT n.*,u.username AS author_username FROM dev_notes n "
                                  "JOIN dev_users u ON u.id=n.author_id WHERE n.patient_id=:p" + clause + " ORDER BY n.id"),
                             {"p": patient_id}).mappings().all()
            _audit(c, user, "read_notes", "patient", patient_id)
            return [dict(row) for row in rows]

    def save_note(c, patient_id, content, user):
        result = c.execute(text("INSERT INTO dev_notes(patient_id,author_id,content,status,created_at) VALUES (:p,:u,:v,'draft',:at)"),
                           {"p": patient_id, "u": user["id"], "v": content, "at": _now()})
        note = note_for(c, result.lastrowid, user)
        _audit(c, user, "draft_note", "note", note["id"])
        return note

    @router.post("/patients/{patient_id}/notes")
    def create_note(patient_id: int, body: ContentIn, user=Depends(clinician)):
        with engine.begin() as c:
            patient_for(c, patient_id, user)
            return save_note(c, patient_id, body.content, user)

    @router.post("/patients/{patient_id}/note-draft")
    def draft_note(patient_id: int, body: NoteDraftIn, user=Depends(clinician)):
        with engine.begin() as c:
            patient_for(c, patient_id, user)
            conversation = conversation_for(c, body.conversation_id, user)
            if conversation["patient_id"] != patient_id:
                raise HTTPException(404, "Conversation not found.")
            rows = c.execute(text("SELECT m.content,u.role FROM dev_messages m JOIN dev_users u ON u.id=m.sender_id "
                                  "WHERE m.conversation_id=:cid ORDER BY m.id"), {"cid": body.conversation_id}).all()
            content = "SYNTHETIC DEVELOPMENT DRAFT — clinician review required.\nTranscript-derived draft; not AI inference.\n\n"
            content += "\n".join(f"{role}: {message}" for message, role in rows) or "No messages recorded."
            return save_note(c, patient_id, content, user)

    @router.post("/patients/{patient_id}/assessment")
    def assessment(patient_id: int, body: AssessmentIn, user=Depends(clinician)):
        with engine.begin() as c:
            patient_for(c, patient_id, user)
            conversation = conversation_for(c, body.conversation_id, user)
            if conversation["patient_id"] != patient_id:
                raise HTTPException(404, "Conversation not found.")
            rows = c.execute(text("SELECT m.content,u.role FROM dev_messages m JOIN dev_users u ON u.id=m.sender_id "
                                  "WHERE m.conversation_id=:cid ORDER BY m.id"), {"cid": body.conversation_id}).all()
            context = "\n".join(f"{role}: {message}" for message, role in rows)
            _audit(c, user, "request_assessment", "conversation", body.conversation_id)
        if assessment_generator is None:
            raise HTTPException(503, "Development assessment service is not configured.")
        return assessment_generator(case_context=context, question=body.question, mode="general")

    @router.post("/notes/{note_id}/approve")
    def approve_note(note_id: int, user=Depends(clinician)):
        with engine.begin() as c:
            note = note_for(c, note_id, user)
            if note["status"] != "draft":
                raise HTTPException(409, "Only draft notes can be approved.")
            updated = c.execute(text("UPDATE dev_notes SET status='approved',approved_by=:u,approved_at=:at "
                                     "WHERE id=:id AND status='draft'"),
                                {"u": user["id"], "at": _now(), "id": note_id})
            if updated.rowcount != 1:
                raise HTTPException(409, "Only draft notes can be approved.")
            _audit(c, user, "approve_note", "note", note_id)
            return note_for(c, note_id, user)

    app.include_router(router)
