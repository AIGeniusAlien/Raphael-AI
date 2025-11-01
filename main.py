import os
import json
import secrets
from datetime import datetime, timedelta
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import httpx
from sqlalchemy import text
from sqlalchemy import create_engine

from openai import OpenAI

# ---------- ENV ----------
APP_NAME = os.getenv("APP_NAME", "Raphael")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # text/chat
OPENAI_TTS_VOICE = os.getenv("VOICE_DEFAULT", "verse")   # for future TTS
FRONTEND_URL = os.getenv("FRONTEND_URL", "")
STATIC_PATH = os.getenv("STATIC_PATH", "static")

# SMART-on-FHIR
FHIR_BASE_URL = os.getenv("FHIR_BASE_URL", "").rstrip("/")
FHIR_AUTH_URL = os.getenv("FHIR_AUTH_URL", "")
FHIR_TOKEN_URL = os.getenv("FHIR_TOKEN_URL", "")
FHIR_CLIENT_ID = os.getenv("FHIR_CLIENT_ID", "")
FHIR_CLIENT_SECRET = os.getenv("FHIR_CLIENT_SECRET", "")
FHIR_REDIRECT_URI = os.getenv("FHIR_REDIRECT_URI", f"{FRONTEND_URL.rstrip('/')}/fhir/callback") or ""
FHIR_SCOPES = os.getenv("FHIR_SCOPES", "launch/patient patient/*.read patient/*.write offline_access openid profile fhirUser")

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./raphael.db")
ENABLE_METRICS = os.getenv("ENABLE_METRICS", "false").lower() == "true"

# ---------- APP ----------
app = FastAPI(title=APP_NAME)

# CORS (allow your domain + Render preview)
cors_origins = [FRONTEND_URL] if FRONTEND_URL else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static
static_dir = os.path.abspath(STATIC_PATH)
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# DB
engine = create_engine(DATABASE_URL, future=True)

# ---------- BOOTSTRAP (tables for OAuth tokens & sessions) ----------
DDL = """
CREATE TABLE IF NOT EXISTS oauth_tokens (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  vendor TEXT,
  patient_id TEXT,
  access_token TEXT,
  refresh_token TEXT,
  expires_at TEXT,
  scope TEXT,
  created_at TEXT
);

CREATE TABLE IF NOT EXISTS sessions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_key TEXT UNIQUE,
  name TEXT,
  created_at TEXT
);
"""

@app.on_event("startup")
def startup():
    with engine.begin() as conn:
        for stmt in DDL.strip().split(";"):
            s = stmt.strip()
            if s:
                conn.execute(text(s))

# ---------- HOME (redirect to branded UI) ----------
@app.get("/")
def root():
    return RedirectResponse("/static/raphael.html")

# ---------- OPENAI CHAT (simple JSON request/response) ----------
@app.post("/api/chat")
async def chat_endpoint(body: dict):
    """
    body: { "messages": [{role:'system|user|assistant', content:'...'}],
            "patient": {... optional context ...} }
    """
    messages = body.get("messages") or []
    patient = body.get("patient") or {}

    system_prefix = (
        "You are Raphael, a clinical copilot. Be concise, evidence-aware, and safe. "
        "Never make final diagnoses; provide differentials, red-flags, and guideline-linked suggestions. "
        "If unsure, ask clarifying questions."
    )
    if patient:
        system_prefix += f" Patient brief: {json.dumps(patient)[:1500]}"

    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "system", "content": system_prefix}] + messages,
        temperature=0.2,
    )
    return {"reply": resp.choices[0].message.content}

# ---------- SMART on FHIR OAuth ----------
def save_tokens(vendor: str, tokens: dict, patient_id: Optional[str] = None):
    exp = None
    if "expires_in" in tokens:
        exp_dt = datetime.utcnow() + timedelta(seconds=int(tokens["expires_in"]))
        exp = exp_dt.isoformat()
    with engine.begin() as conn:
        conn.execute(
            text("""INSERT INTO oauth_tokens
                    (vendor, patient_id, access_token, refresh_token, expires_at, scope, created_at)
                    VALUES (:v,:pid,:at,:rt,:ea,:sc,:ca)"""),
            dict(
                v=vendor, pid=patient_id,
                at=tokens.get("access_token"),
                rt=tokens.get("refresh_token"),
                ea=exp, sc=tokens.get("scope"),
                ca=datetime.utcnow().isoformat(),
            ),
        )

def latest_token(vendor: str):
    with engine.begin() as conn:
        row = conn.execute(
            text("""SELECT access_token, refresh_token, expires_at
                    FROM oauth_tokens WHERE vendor=:v
                    ORDER BY id DESC LIMIT 1"""),
            {"v": vendor},
        ).first()
        return dict(row._mapping) if row else None

async def refresh_if_needed(vendor: str):
    tk = latest_token(vendor)
    if not tk:
        raise HTTPException(401, "No EHR token. Visit /fhir/login.")
    if tk["expires_at"]:
        try:
            exp = datetime.fromisoformat(tk["expires_at"])
            if exp - datetime.utcnow() > timedelta(seconds=60):
                return tk["access_token"]
        except Exception:
            pass
    if not tk.get("refresh_token"):
        return tk["access_token"]

    async with httpx.AsyncClient(timeout=30) as client:
        data = {
            "grant_type": "refresh_token",
            "refresh_token": tk["refresh_token"],
            "client_id": FHIR_CLIENT_ID,
        }
        auth = None
        if FHIR_CLIENT_SECRET:
            auth = (FHIR_CLIENT_ID, FHIR_CLIENT_SECRET)
        r = await client.post(FHIR_TOKEN_URL, data=data, auth=auth)
        r.raise_for_status()
        tokens = r.json()
    save_tokens("default", tokens, None)
    return tokens["access_token"]

@app.get("/fhir/login")
def fhir_login(vendor: str = "default"):
    if not all([FHIR_AUTH_URL, FHIR_TOKEN_URL, FHIR_CLIENT_ID, FHIR_REDIRECT_URI, FHIR_BASE_URL]):
        raise HTTPException(400, "FHIR env not configured")
    params = {
        "response_type": "code",
        "client_id": FHIR_CLIENT_ID,
        "redirect_uri": FHIR_REDIRECT_URI,
        "scope": FHIR_SCOPES,
        "state": secrets.token_urlsafe(24),
        "aud": FHIR_BASE_URL,
    }
    return RedirectResponse(f"{FHIR_AUTH_URL}?{httpx.QueryParams(params)}")

@app.get("/fhir/callback")
async def fhir_callback(code: str, state: Optional[str] = None, vendor: str = "default"):
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": FHIR_REDIRECT_URI,
        "client_id": FHIR_CLIENT_ID,
    }
    auth = None
    if FHIR_CLIENT_SECRET:
        auth = (FHIR_CLIENT_ID, FHIR_CLIENT_SECRET)

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(FHIR_TOKEN_URL, data=data, auth=auth)
        if r.status_code >= 400:
            return JSONResponse({"error": r.text}, status_code=r.status_code)
        tokens = r.json()

    save_tokens(vendor, tokens, None)
    return RedirectResponse("/static/raphael.html?ehr=connected")

async def fhir_headers():
    token = await refresh_if_needed("default")
    return {"Authorization": f"Bearer {token}", "Accept": "application/fhir+json", "Content-Type": "application/fhir+json"}

@app.get("/api/fhir/patient/{pid}")
async def fhir_get_patient(pid: str):
    headers = await fhir_headers()
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(f"{FHIR_BASE_URL}/Patient/{pid}", headers=headers)
        return JSONResponse(r.json(), status_code=r.status_code)

@app.post("/api/fhir/patient")
async def fhir_create_patient(resource: dict):
    headers = await fhir_headers()
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(f"{FHIR_BASE_URL}/Patient", headers=headers, json=resource)
        return JSONResponse(r.json(), status_code=r.status_code)

@app.get("/api/fhir/observations")
async def fhir_search_observations(patient: str, code: Optional[str] = None, _count: int = 20):
    headers = await fhir_headers()
    params = {"patient": patient, "_count": str(_count)}
    if code: params["code"] = code
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(f"{FHIR_BASE_URL}/Observation", headers=headers, params=params)
        return JSONResponse(r.json(), status_code=r.status_code)

# ---------- REALTIME VOICE (WebRTC) ----------
# The browser will call this to obtain an ephemeral OpenAI Realtime session token.
@app.post("/api/rt-token")
async def realtime_token():
    if not OPENAI_API_KEY:
        raise HTTPException(400, "OPENAI_API_KEY missing")
    client = OpenAI(api_key=OPENAI_API_KEY)
    # Create ephemeral session (valid ~1 minute)
    session = client.realtime.sessions.create(
        model="gpt-4o-realtime-preview-2024-12-17",   # or latest realtime model
        voice=OPENAI_TTS_VOICE,
        modalities=["audio", "text"],
        # You can pass a default system prompt here:
        instructions="You are Raphael, a clinical copilot. Be helpful, safe, and concise.",
    )
    return session  # contains client_secret.value

# ---------- HEALTHCHECK ----------
@app.get("/healthz")
def healthz():
    return {"ok": True, "ts": datetime.utcnow().isoformat()}
