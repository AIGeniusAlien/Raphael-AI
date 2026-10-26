"""Opt-in, isolated synthetic-data portal. This is not production identity or care delivery."""

import hashlib
import hmac
import os
import secrets
import json
from types import SimpleNamespace
from typing import Literal
from datetime import datetime, timedelta, timezone
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field, SecretStr, StrictBool
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from database import execute_sqlite_script

COOKIE_NAME = "raphael_dev_session"
SESSION_SECONDS = 8 * 60 * 60
LOGIN_MAX_FAILURES = 5
LOGIN_WINDOW_SECONDS = 5 * 60
LOGIN_MAX_TRACKED_KEYS = 10000

SCHEMA = """
CREATE TABLE IF NOT EXISTS dev_users (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 tenant TEXT NOT NULL,
 username TEXT NOT NULL,
 role TEXT NOT NULL CHECK (role IN ('patient','clinician','org_admin')),
 password_hash TEXT NOT NULL,
 active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0,1)),
 UNIQUE(tenant, username)
);
CREATE TABLE IF NOT EXISTS dev_schema_migrations (
 version INTEGER PRIMARY KEY,
 applied_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS dev_login_attempts (
 key_hash TEXT PRIMARY KEY,
 failures INTEGER NOT NULL,
 reset_at TEXT NOT NULL
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
 created_by_id INTEGER REFERENCES dev_users(id),
 created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS dev_chart_access_decisions (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 patient_id INTEGER NOT NULL REFERENCES dev_patients(id),
 clinician_id INTEGER NOT NULL REFERENCES dev_users(id),
 actor_id INTEGER NOT NULL REFERENCES dev_users(id),
 decision TEXT NOT NULL CHECK(decision IN ('grant','revoke')),
 expires_at TEXT,
 purpose TEXT NOT NULL,
 created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS dev_chart_access_latest ON dev_chart_access_decisions(patient_id,clinician_id,id);
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


def _migrate_users(engine):
    """Rebuild the original two-role CHECK without losing references or IDs.

    SQLite cannot alter a CHECK constraint. Foreign-key enforcement is toggled
    only on this connection before an explicit transaction and restored after it;
    the child tables keep references to the final dev_users name throughout.
    """
    with engine.connect() as c:
        schema = c.exec_driver_sql("SELECT sql FROM sqlite_master WHERE name='dev_users' AND type='table'").scalar()
        if schema is None:
            return
        columns = {row[1] for row in c.exec_driver_sql("PRAGMA table_info(dev_users)")}
        if "org_admin" in schema and "active" in columns:
            return
        if columns - {"id", "tenant", "username", "role", "password_hash", "active"}:
            raise RuntimeError("Unknown dev_users schema; refusing a destructive migration.")
        foreign_keys = c.exec_driver_sql("PRAGMA foreign_keys").scalar()
        c.commit()
        c.exec_driver_sql("PRAGMA foreign_keys=OFF")
        c.commit()
        try:
            c.exec_driver_sql("BEGIN IMMEDIATE")
            # A second process may have completed startup while this connection
            # waited for the writer lock. Re-read rather than resetting status
            # using an obsolete pre-lock schema snapshot.
            schema = c.exec_driver_sql("SELECT sql FROM sqlite_master WHERE name='dev_users' AND type='table'").scalar()
            columns = {row[1] for row in c.exec_driver_sql("PRAGMA table_info(dev_users)")}
            if "org_admin" in schema and "active" in columns:
                c.commit()
                return
            if columns - {"id", "tenant", "username", "role", "password_hash", "active"}:
                raise RuntimeError("Unknown dev_users schema; refusing a destructive migration.")
            sequence = c.exec_driver_sql("SELECT seq FROM sqlite_sequence WHERE name='dev_users'").scalar() or 0
            c.exec_driver_sql("""CREATE TABLE dev_users_upgrade (
                id INTEGER PRIMARY KEY AUTOINCREMENT, tenant TEXT NOT NULL,
                username TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('patient','clinician','org_admin')),
                password_hash TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0,1)),
                UNIQUE(tenant,username))""")
            active = "active" if "active" in columns else "1"
            c.exec_driver_sql("INSERT INTO dev_users_upgrade(id,tenant,username,role,password_hash,active) "
                              "SELECT id,tenant,username,role,password_hash," + active + " FROM dev_users")
            c.exec_driver_sql("DROP TABLE dev_users")
            c.exec_driver_sql("ALTER TABLE dev_users_upgrade RENAME TO dev_users")
            c.exec_driver_sql("UPDATE sqlite_sequence SET seq=MAX(seq,?) WHERE name='dev_users'", (sequence,))
            if c.exec_driver_sql("PRAGMA foreign_key_check").first():
                raise RuntimeError("Portal migration found invalid foreign-key references; no changes committed.")
            c.commit()
        except Exception:
            c.rollback()
            raise
        finally:
            c.exec_driver_sql("PRAGMA foreign_keys=" + ("ON" if foreign_keys else "OFF"))
            c.commit()


def init_portal_schema(engine):
    _require_synthetic(engine)
    _migrate_users(engine)
    with engine.begin() as c:
        c.exec_driver_sql('BEGIN IMMEDIATE')
        execute_sqlite_script(c, SCHEMA)
        columns={row[1] for row in c.exec_driver_sql('PRAGMA table_info(dev_patients)')}
        if 'created_by_id' not in columns:
            c.exec_driver_sql('ALTER TABLE dev_patients ADD COLUMN created_by_id INTEGER REFERENCES dev_users(id)')
        if not c.execute(text('SELECT 1 FROM dev_schema_migrations WHERE version=3')).scalar():
            # Only creation audit provenance identifies an originating clinician.
            # Historical reads,notes/conversations or tenant membership do not.
            # Unknown/conflicting provenance stays unassigned;records are retained.
            candidates=c.execute(text("SELECT p.id,MIN(a.actor_id) AS creator FROM dev_patients p JOIN dev_audit a "
                "ON a.resource_id=p.id AND a.resource_type='patient' AND a.event='create_patient' AND a.tenant=p.tenant "
                "WHERE p.created_by_id IS NULL GROUP BY p.id HAVING COUNT(DISTINCT a.actor_id)=1 AND COUNT(*)=COUNT(a.actor_id)")).all()
            for patient_id,creator in candidates:
                c.execute(text("UPDATE dev_patients SET created_by_id=:u WHERE id=:p AND EXISTS "
                    "(SELECT 1 FROM dev_users u WHERE u.id=:u AND u.tenant=dev_patients.tenant AND u.role='clinician')"),{'u':creator,'p':patient_id})
            c.execute(text('INSERT INTO dev_schema_migrations(version,applied_at) VALUES (3,:at)'),{'at':_now()})
        c.execute(text("INSERT OR IGNORE INTO dev_schema_migrations(version,applied_at) VALUES (2,:at)"), {"at": _now()})


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
    if role not in ("patient", "clinician", "org_admin"):
        raise ValueError("Role must be patient, clinician or org_admin.")
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


class UserIn(Input):
    username: str = Field(min_length=1, max_length=100)
    role: Literal["patient", "clinician", "org_admin"]
    password: SecretStr = Field(min_length=12, max_length=128)


class UserStatusIn(Input):
    active: StrictBool


class PatientIn(Input):
    name: str = Field(min_length=1, max_length=200)
    dob: str | None = Field(default=None, max_length=30)
    mrn: str | None = Field(default=None, max_length=100)
    patient_username: str | None = Field(default=None, max_length=100)


class ChartGrantIn(Input):
    clinician_username: str = Field(min_length=1,max_length=100)
    expires_in_minutes: int = Field(ge=1,le=10080)
    purpose: str = Field(min_length=5,max_length=500)
    authorize_full_chart: StrictBool


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

    from dev_security import RequestBudget
    request_budget=RequestBudget()

    async def same_origin_json(request: Request):
        request_budget.check(request)
        if request.headers.get('sec-fetch-site')=='cross-site':
            raise HTTPException(403,'Cross-site portal requests are forbidden.')
        if request.method in ("POST", "PUT", "PATCH", "DELETE"):
            origin = request.headers.get("origin")
            expected = urlsplit(str(request.url))
            if origin:
                try:supplied = urlsplit(origin)
                except ValueError:raise HTTPException(403,'Invalid request origin.') from None
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
                                 "WHERE s.token_hash=:h AND s.expires_at>:at AND u.active=1"),
                            {"h": hashlib.sha256(token.encode()).hexdigest(), "at": _now()}).mappings().first()
            if not row:
                raise HTTPException(401, "Session expired or invalid.")
            return dict(row)

    def clinical_user(user=Depends(current_user)):
        if user["role"] not in ("patient", "clinician"):
            raise HTTPException(403, "Clinical role required.")
        return user

    def administrator(user=Depends(current_user)):
        if user["role"] != "org_admin":
            raise HTTPException(403, "Organization administrator role required.")
        return user

    def clinician(user=Depends(current_user)):
        if user["role"] != "clinician":
            raise HTTPException(403, "Clinician role required.")
        return user

    def can_access_patient(c, patient_id, user):
        row=c.execute(text('SELECT patient_user_id,created_by_id FROM dev_patients WHERE id=:p AND tenant=:t'),{'p':patient_id,'t':user['tenant']}).mappings().first()
        if not row:return False
        if user['role']=='patient':return row['patient_user_id']==user['id']
        if user['role']!='clinician':return False
        decision=c.execute(text('SELECT decision,expires_at FROM dev_chart_access_decisions WHERE patient_id=:p AND clinician_id=:u ORDER BY id DESC LIMIT 1'),{'p':patient_id,'u':user['id']}).mappings().first()
        if decision:return decision['decision']=='grant' and bool(decision['expires_at']) and decision['expires_at']>_now()
        return row['created_by_id']==user['id']

    def patient_for(c, patient_id, user):
        if user["role"] not in ("patient", "clinician"):
            raise HTTPException(403, "Clinical role required.")
        row = c.execute(text("SELECT p.*,u.username AS patient_username FROM dev_patients p "
                             "LEFT JOIN dev_users u ON u.id=p.patient_user_id "
                             "WHERE p.id=:id AND p.tenant=:t"), {"id": patient_id, "t": user["tenant"]}).mappings().first()
        if not row or not can_access_patient(c, patient_id, user):
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
        key = hashlib.sha256(json.dumps([body.tenant, body.username]).encode()).hexdigest()
        # Unknown and disabled accounts share both error and hash work paths.
        with engine.begin() as c:
            c.exec_driver_sql("BEGIN IMMEDIATE")
            c.execute(text("DELETE FROM dev_login_attempts WHERE reset_at<=:at"), {"at": _now()})
            attempt = c.execute(text("SELECT failures,reset_at FROM dev_login_attempts WHERE key_hash=:h"), {"h": key}).mappings().first()
            if attempt and attempt["failures"] >= LOGIN_MAX_FAILURES:
                raise HTTPException(429, "Too many development login attempts. Try again later.", headers={"Retry-After": str(LOGIN_WINDOW_SECONDS)})
            if not attempt and c.execute(text("SELECT count(*) FROM dev_login_attempts")).scalar() >= LOGIN_MAX_TRACKED_KEYS:
                raise HTTPException(429, "Too many development login attempts. Try again later.", headers={"Retry-After": str(LOGIN_WINDOW_SECONDS)})
            row = c.execute(text("SELECT * FROM dev_users WHERE tenant=:t AND username=:u"),
                            {"t": body.tenant, "u": body.username}).mappings().first()
            stored = row["password_hash"] if row else "00" * 16 + ":" + "00" * 64
            valid = _password_matches(body.password.get_secret_value(), stored) and bool(row and row["active"])
            if not valid:
                reset_at = (datetime.now(timezone.utc) + timedelta(seconds=LOGIN_WINDOW_SECONDS)).isoformat()
                c.execute(text("INSERT INTO dev_login_attempts(key_hash,failures,reset_at) VALUES (:h,1,:r) "
                               "ON CONFLICT(key_hash) DO UPDATE SET failures=failures+1"), {"h": key, "r": reset_at})
            else:
                c.execute(text("DELETE FROM dev_login_attempts WHERE key_hash=:h"), {"h": key})
                user = {k: row[k] for k in ("id", "tenant", "username", "role")}
                token = secrets.token_urlsafe(32)
                c.execute(text("INSERT INTO dev_sessions(token_hash,user_id,expires_at) VALUES (:h,:u,:e)"),
                          {"h": hashlib.sha256(token.encode()).hexdigest(), "u": row["id"],
                           "e": (datetime.now(timezone.utc) + timedelta(seconds=SESSION_SECONDS)).isoformat()})
                _audit(c, user, "login")
        # Raise after commit so failed-attempt counters persist.
        if not valid:
            raise HTTPException(401, "Invalid development credentials.")
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
    def patients(user=Depends(clinical_user)):
        with engine.begin() as c:
            where = "p.tenant=:t" + (" AND p.patient_user_id=:uid" if user["role"] == "patient" else "")
            rows = c.execute(text("SELECT p.*,u.username AS patient_username FROM dev_patients p "
                                  "LEFT JOIN dev_users u ON u.id=p.patient_user_id WHERE " + where + " ORDER BY p.id"),
                             {"t": user["tenant"], "uid": user["id"]}).mappings().all()
            _audit(c, user, "list_patients", "patient")
            return [dict(row) for row in rows if can_access_patient(c,row['id'],user)]

    @router.post("/patients")
    def create_patient(body: PatientIn, user=Depends(clinician)):
        with engine.begin() as c:
            patient_user_id = None
            if body.patient_username:
                patient_user_id = c.execute(text("SELECT id FROM dev_users WHERE tenant=:t AND username=:u AND role='patient' AND active=1"),
                                            {"t": user["tenant"], "u": body.patient_username}).scalar()
                if patient_user_id is None:
                    raise HTTPException(400, "Use an existing patient-role account in this tenant.")
            result = c.execute(text("INSERT INTO dev_patients(tenant,name,dob,mrn,patient_user_id,created_by_id,created_at) "
                                    "VALUES (:t,:n,:d,:m,:u,:creator,:at)"), {"t": user["tenant"], "n": body.name,
                                    "d": body.dob, "m": body.mrn, "u": patient_user_id, "creator":user['id'], "at": _now()})
            patient = patient_for(c, result.lastrowid, user)
            _audit(c, user, "create_patient", "patient", patient["id"])
            return patient

    def patient_owner(c,patient_id,user):
        if user['role']!='patient':raise HTTPException(403,'Linked patient account required.')
        return patient_for(c,patient_id,user)

    @router.get('/chart-access/clinicians')
    def chart_clinicians(user=Depends(current_user)):
        if user['role']!='patient':raise HTTPException(403,'Patient account required.')
        with engine.connect() as c:
            return [dict(r) for r in c.execute(text("SELECT id,username FROM dev_users WHERE tenant=:t AND role='clinician' AND active=1 ORDER BY username LIMIT 250"),{'t':user['tenant']}).mappings()]

    @router.get('/patients/{patient_id}/chart-access')
    def chart_access(patient_id:int,user=Depends(current_user)):
        with engine.begin() as c:
            patient=patient_owner(c,patient_id,user)
            rows=c.execute(text("SELECT u.id,u.username,u.active FROM dev_users u WHERE u.tenant=:t AND u.role='clinician' AND "
                "(u.id=:creator OR EXISTS(SELECT 1 FROM dev_chart_access_decisions d WHERE d.patient_id=:p AND d.clinician_id=u.id)) ORDER BY u.username"),{'t':user['tenant'],'creator':patient['created_by_id'],'p':patient_id}).mappings().all()
            result=[]
            for row in rows:
                latest=c.execute(text('SELECT * FROM dev_chart_access_decisions WHERE patient_id=:p AND clinician_id=:u ORDER BY id DESC LIMIT 1'),{'p':patient_id,'u':row['id']}).mappings().first()
                recipient={'id':row['id'],'tenant':user['tenant'],'role':'clinician'}
                result.append({'clinician_id':row['id'],'clinician_username':row['username'],'active':bool(row['active']) and can_access_patient(c,patient_id,recipient),'basis':'patient_decision' if latest else 'originating_clinician','latest_decision':dict(latest) if latest else None})
            _audit(c,user,'review_full_chart_access','patient',patient_id)
            return {'patient_id':patient_id,'scope':'full_current_chart_and_future_updates;clinician_record_and_note_permissions;private_document_controls_also_apply','clinicians':result,
                    'history':[dict(r) for r in c.execute(text('SELECT id,clinician_id,decision,expires_at,purpose,created_at FROM dev_chart_access_decisions WHERE patient_id=:p ORDER BY id'),{'p':patient_id}).mappings()]}

    @router.post('/patients/{patient_id}/chart-access')
    def grant_chart_access(patient_id:int,body:ChartGrantIn,user=Depends(current_user)):
        if not body.authorize_full_chart:raise HTTPException(422,'Explicit full-chart authorization is required;intake sharing does not grant chart access.')
        with engine.begin() as c:
            c.exec_driver_sql('BEGIN IMMEDIATE');patient_owner(c,patient_id,user)
            recipient=c.execute(text("SELECT id FROM dev_users WHERE tenant=:t AND username=:u AND role='clinician' AND active=1"),{'t':user['tenant'],'u':body.clinician_username}).scalar()
            if not recipient:raise HTTPException(404,'Active participating clinician not found.')
            expiry=(datetime.now(timezone.utc)+timedelta(minutes=body.expires_in_minutes)).isoformat()
            c.execute(text("INSERT INTO dev_chart_access_decisions(patient_id,clinician_id,actor_id,decision,expires_at,purpose,created_at) VALUES (:p,:u,:a,'grant',:e,:reason,:at)"),{'p':patient_id,'u':recipient,'a':user['id'],'e':expiry,'reason':body.purpose,'at':_now()})
            _audit(c,user,'grant_full_chart_access','patient',patient_id)
            return {'patient_id':patient_id,'clinician_id':recipient,'decision':'grant','expires_at':expiry,'scope':'full_current_chart_and_future_updates;clinician_record_and_note_permissions;private_document_controls_also_apply'}

    @router.post('/patients/{patient_id}/chart-access/{clinician_id}/revoke')
    def revoke_chart_access(patient_id:int,clinician_id:int,user=Depends(current_user)):
        with engine.begin() as c:
            c.exec_driver_sql('BEGIN IMMEDIATE');patient_owner(c,patient_id,user)
            recipient=c.execute(text("SELECT id FROM dev_users WHERE id=:u AND tenant=:t AND role='clinician'"),{'u':clinician_id,'t':user['tenant']}).scalar()
            if not recipient:raise HTTPException(404,'Participating clinician not found.')
            c.execute(text("INSERT INTO dev_chart_access_decisions(patient_id,clinician_id,actor_id,decision,purpose,created_at) VALUES (:p,:u,:a,'revoke','Patient revoked future full-chart access',:at)"),{'p':patient_id,'u':recipient,'a':user['id'],'at':_now()})
            _audit(c,user,'revoke_full_chart_access','patient',patient_id)
            return {'decision':'revoke','future_access':'denied','scope':'full_chart;separate_intake_snapshot_consents_are_managed_in_connected_care'}

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
        return assessment_generator(case_context=context, question=body.question, mode="general", tenant=user["tenant"])

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

    @router.get("/admin/users")
    def list_users(user=Depends(administrator)):
        with engine.begin() as c:
            rows = c.execute(text("SELECT id,tenant,username,role,active FROM dev_users WHERE tenant=:t ORDER BY id"), {"t": user["tenant"]}).mappings().all()
            _audit(c, user, "list_users", "user")
            return [dict(row, active=bool(row["active"])) for row in rows]

    @router.post("/admin/users")
    def provision_user(body: UserIn, user=Depends(administrator)):
        try:
            created = create_dev_user(engine, user["tenant"], body.username, body.password.get_secret_value(), body.role)
        except IntegrityError:
            raise HTTPException(409, "An account with that username already exists in this tenant.")
        with engine.begin() as c:
            _audit(c, user, "create_user", "user", created["id"])
        return dict(created, active=True)

    @router.patch("/admin/users/{user_id}")
    def user_status(user_id: int, body: UserStatusIn, user=Depends(administrator)):
        with engine.begin() as c:
            # Serialize changes so two admins cannot deactivate each other while
            # independently observing a second active administrator.
            c.exec_driver_sql("BEGIN IMMEDIATE")
            target = c.execute(text("SELECT id,tenant,username,role,active FROM dev_users WHERE id=:id AND tenant=:t"), {"id": user_id, "t": user["tenant"]}).mappings().first()
            if not target:
                raise HTTPException(404, "User not found.")
            if not body.active:
                if user_id == user["id"]:
                    raise HTTPException(409, "Administrators cannot deactivate their own account.")
                if target["role"] == "org_admin" and target["active"]:
                    count = c.execute(text("SELECT count(*) FROM dev_users WHERE tenant=:t AND role='org_admin' AND active=1"), {"t": user["tenant"]}).scalar()
                    if count <= 1:
                        raise HTTPException(409, "The final active administrator cannot be deactivated.")
            c.execute(text("UPDATE dev_users SET active=:a WHERE id=:id"), {"a": int(body.active), "id": user_id})
            if not body.active:
                c.execute(text("DELETE FROM dev_sessions WHERE user_id=:id"), {"id": user_id})
            _audit(c, user, "activate_user" if body.active else "deactivate_user", "user", user_id)
            return dict(target, active=body.active)

    app.include_router(router)
    return SimpleNamespace(engine=engine, user=current_user, clinician=clinician,
                           patient_for=patient_for, conversation_for=conversation_for,
                           note_for=note_for, audit=_audit, same_origin_json=same_origin_json,
                           can_access_patient=can_access_patient)
