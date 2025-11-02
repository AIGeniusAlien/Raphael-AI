import os, json, base64, secrets, httpx
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

from sqlalchemy import (
    create_engine, text
)

# ───────────────── CONFIG
APP_NAME = os.getenv("APP_NAME", "Raphael")
STATIC_PATH = os.getenv("STATIC_PATH", "static")
FRONTEND_URL = os.getenv("FRONTEND_URL", "")

# LLM via OpenAI-compatible server (vLLM/TGI/llama.cpp server)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").rstrip("/")
LLM_MODEL = os.getenv("LLM_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
LLM_API_KEY = os.getenv("LLM_API_KEY", "EMPTY")

# Voice toggle (not implemented in this build)
VOICE_ENABLED = os.getenv("VOICE_ENABLED", "false").lower() == "true"

# FHIR (optional; left idle for v1)
FHIR_BASE_URL = os.getenv("FHIR_BASE_URL", "").rstrip("/")
FHIR_AUTH_URL = os.getenv("FHIR_AUTH_URL", "")
FHIR_TOKEN_URL = os.getenv("FHIR_TOKEN_URL", "")
FHIR_CLIENT_ID = os.getenv("FHIR_CLIENT_ID", "")
FHIR_CLIENT_SECRET = os.getenv("FHIR_CLIENT_SECRET", "")
FHIR_REDIRECT_URI = os.getenv("FHIR_REDIRECT_URI", f"{FRONTEND_URL}/fhir/callback")
FHIR_SCOPES = os.getenv("FHIR_SCOPES", "launch/patient patient/*.read patient/*.write offline_access openid profile fhirUser")

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./raphael.db")

# ───────────────── APP
app = FastAPI(title=APP_NAME)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True
)
app.mount("/static", StaticFiles(directory=STATIC_PATH), name="static")

engine = create_engine(DATABASE_URL, future=True)

DDL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS patients (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  dob TEXT,
  mrn TEXT,
  created_at TEXT
);

CREATE TABLE IF NOT EXISTS conversations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  patient_id INTEGER,
  title TEXT,
  created_at TEXT,
  FOREIGN KEY(patient_id) REFERENCES patients(id)
);

CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  conversation_id INTEGER NOT NULL,
  role TEXT NOT NULL,           -- 'user' | 'assistant' | 'system'
  content TEXT NOT NULL,
  created_at TEXT,
  FOREIGN KEY(conversation_id) REFERENCES conversations(id)
);

-- Simple documents table for guidance/notes
CREATE TABLE IF NOT EXISTS documents (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  title TEXT,
  body TEXT,
  created_at TEXT
);

-- SQLite FTS5 virtual table for fast keyword search over documents
CREATE VIRTUAL TABLE IF NOT EXISTS doc_fts USING fts5(
  title, body, content='documents', content_rowid='id'
);
"""

TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS doc_ai AFTER INSERT ON documents BEGIN
  INSERT INTO doc_fts(rowid, title, body) VALUES (new.id, new.title, new.body);
END;
CREATE TRIGGER IF NOT EXISTS doc_ad AFTER DELETE ON documents BEGIN
  INSERT INTO doc_fts(doc_fts, rowid, title, body) VALUES ('delete', old.id, old.title, old.body);
END;
CREATE TRIGGER IF NOT EXISTS doc_au AFTER UPDATE ON documents BEGIN
  INSERT INTO doc_fts(doc_fts, rowid, title, body) VALUES ('delete', old.id, old.title, old.body);
  INSERT INTO doc_fts(rowid, title, body) VALUES (new.id, new.title, new.body);
END;
"""

@app.on_event("startup")
def startup():
    with engine.begin() as c:
        for stmt in DDL.strip().split(";\n\n"):
            if stmt.strip():
                c.execute(text(stmt))
        for stmt in TRIGGERS.strip().split(";\n"):
            s = stmt.strip().rstrip(";")
            if s:
                c.execute(text(s))

@app.get("/")
def root():
    return RedirectResponse("/static/Raphael.html")

@app.get("/healthz")
def healthz():
    return {"status": "ok", "ts": datetime.utcnow().isoformat()}

@app.get("/api/env")
def read_env():
    return {
        "app": APP_NAME,
        "voice_enabled": VOICE_ENABLED,
        "model": LLM_MODEL
    }

# ───────────────── Patients CRUD
@app.get("/api/patients")
def list_patients(q: Optional[str] = None):
    with engine.begin() as c:
        if q:
            rows = c.execute(text("SELECT * FROM patients WHERE name LIKE :q ORDER BY created_at DESC"),
                             {"q": f"%{q}%"}).mappings().all()
        else:
            rows = c.execute(text("SELECT * FROM patients ORDER BY created_at DESC")).mappings().all()
        return list(rows)

@app.post("/api/patients")
def create_patient(body: dict):
    name = body.get("name")
    dob = body.get("dob")
    mrn = body.get("mrn")
    if not name:
        raise HTTPException(400, "Missing patient name.")
    with engine.begin() as c:
        c.execute(text(
            "INSERT INTO patients(name, dob, mrn, created_at) VALUES (:n,:d,:m,:t)"
        ), {"n": name, "d": dob, "m": mrn, "t": datetime.utcnow().isoformat()})
        row = c.execute(text("SELECT * FROM patients ORDER BY id DESC LIMIT 1")).mappings().first()
    return row

@app.delete("/api/patients/{pid}")
def delete_patient(pid: int):
    with engine.begin() as c:
        c.execute(text("DELETE FROM messages WHERE conversation_id IN (SELECT id FROM conversations WHERE patient_id=:p)"), {"p": pid})
        c.execute(text("DELETE FROM conversations WHERE patient_id=:p"), {"p": pid})
        c.execute(text("DELETE FROM patients WHERE id=:p"), {"p": pid})
    return {"ok": True}

# ───────────────── Conversations & Messages
@app.get("/api/conversations")
def list_conversations(patient_id: int):
    with engine.begin() as c:
        rows = c.execute(text(
            "SELECT * FROM conversations WHERE patient_id=:p ORDER BY created_at DESC"
        ), {"p": patient_id}).mappings().all()
        return list(rows)

@app.post("/api/conversations")
def create_conversation(body: dict):
    pid = body.get("patient_id")
    title = body.get("title") or "New Chat"
    if not pid:
        raise HTTPException(400, "Missing patient_id.")
    with engine.begin() as c:
        c.execute(text(
            "INSERT INTO conversations(patient_id, title, created_at) VALUES (:p,:t,:c)"
        ), {"p": pid, "t": title, "c": datetime.utcnow().isoformat()})
        row = c.execute(text("SELECT * FROM conversations ORDER BY id DESC LIMIT 1")).mappings().first()
    return row

@app.get("/api/messages")
def get_messages(conversation_id: int):
    with engine.begin() as c:
        rows = c.execute(text(
            "SELECT * FROM messages WHERE conversation_id=:c ORDER BY id ASC"
        ), {"c": conversation_id}).mappings().all()
    return list(rows)

# ───────────────── Chat (LLM)
def llm_client() -> OpenAI:
    if not LLM_BASE_URL:
        raise HTTPException(500, "LLM_BASE_URL not configured.")
    return OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

@app.post("/api/chat")
def chat(body: dict):
    conversation_id = body.get("conversation_id")
    user_text = body.get("text", "").strip()
    patient = body.get("patient") or {}

    if not conversation_id:
        raise HTTPException(400, "Missing conversation_id.")
    if not user_text:
        raise HTTPException(400, "Empty prompt.")

    system = (
        "You are Raphael, a multimodal clinical copilot for physicians. "
        "Use clear, cautious, evidence-aligned language. Do not give a definitive diagnosis; "
        "offer differential considerations, clinical reasoning, and next-step guidance. "
        "When you reference evidence or guidelines, be explicit and concise."
    )
    if patient:
        system += f" Patient context: {json.dumps(patient)[:1200]}"

    # Persist user message
    now = datetime.utcnow().isoformat()
    with engine.begin() as c:
        c.execute(text(
            "INSERT INTO messages(conversation_id, role, content, created_at) VALUES (:cid,'user',:ct,:ts)"
        ), {"cid": conversation_id, "ct": user_text, "ts": now})

    # Load last 30 messages for context
    with engine.begin() as c:
        hist = c.execute(text(
            "SELECT role, content FROM messages WHERE conversation_id=:cid ORDER BY id ASC"
        ), {"cid": conversation_id}).all()

    messages = [{"role": "system", "content": system}] + [{"role": r, "content": c} for (r, c) in hist][-30:]

    client = llm_client()
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.2,
        )
        reply = resp.choices[0].message.content
    except Exception as e:
        raise HTTPException(500, f"LLM error: {e}")

    with engine.begin() as c:
        c.execute(text(
            "INSERT INTO messages(conversation_id, role, content, created_at) VALUES (:cid,'assistant',:ct,:ts)"
        ), {"cid": conversation_id, "ct": reply, "ts": datetime.utcnow().isoformat()})
    return {"reply": reply}

# ───────────────── Guidance documents + FTS search (lightweight RAG)
@app.post("/api/docs/upload")
async def upload_doc(title: str = Form(...), file: UploadFile = File(...)):
    text_bytes = await file.read()
    body = text_bytes.decode("utf-8", errors="ignore")
    with engine.begin() as c:
        c.execute(text(
            "INSERT INTO documents(title, body, created_at) VALUES (:t,:b,:ts)"
        ), {"t": title, "b": body, "ts": datetime.utcnow().isoformat()})
    return {"ok": True}

@app.get("/api/docs/search")
def search_docs(q: str = Query(..., min_length=2)):
    # Very fast keyword search using SQLite FTS5
    with engine.begin() as c:
        rows = c.execute(text(
            "SELECT d.id, d.title, snippet(doc_fts, 1, '[', ']', '…', 10) AS snippet "
            "FROM doc_fts JOIN documents d ON d.id = doc_fts.rowid "
            "WHERE doc_fts MATCH :q LIMIT 10"
        ), {"q": q}).mappings().all()
    return list(rows)

# ───────────────── Voice token endpoint placeholder
@app.post("/api/rt-token")
def rt_token():
    if not VOICE_ENABLED:
        raise HTTPException(400, "Voice is disabled in this build.")
    raise HTTPException(501, "Voice backend not implemented yet.")

# ───────────────── (Optional) Minimal SMART-on-FHIR stubs kept for later
# (Routes omitted for brevity; safe to leave out until you enable EHR)
