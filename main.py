import os, json, base64, secrets, httpx
from datetime import datetime, timedelta
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, text
from openai import OpenAI

# ───────────── Config
APP_NAME = os.getenv("APP_NAME", "Raphael")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
VOICE_DEFAULT = os.getenv("VOICE_DEFAULT", "verse")
FRONTEND_URL = os.getenv("FRONTEND_URL", "")
STATIC_PATH = os.getenv("STATIC_PATH", "static")

FHIR_BASE_URL = os.getenv("FHIR_BASE_URL", "").rstrip("/")
FHIR_AUTH_URL = os.getenv("FHIR_AUTH_URL", "")
FHIR_TOKEN_URL = os.getenv("FHIR_TOKEN_URL", "")
FHIR_CLIENT_ID = os.getenv("FHIR_CLIENT_ID", "")
FHIR_CLIENT_SECRET = os.getenv("FHIR_CLIENT_SECRET", "")
FHIR_REDIRECT_URI = os.getenv("FHIR_REDIRECT_URI", f"{FRONTEND_URL}/fhir/callback")
FHIR_SCOPES = os.getenv(
    "FHIR_SCOPES",
    "launch/patient patient/*.read patient/*.write offline_access openid profile fhirUser"
)

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./raphael.db")

# ───────────── App
app = FastAPI(title=APP_NAME)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True
)
app.mount("/static", StaticFiles(directory=STATIC_PATH), name="static")

engine = create_engine(DATABASE_URL, future=True)
DDL = """
CREATE TABLE IF NOT EXISTS oauth_tokens (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  vendor TEXT, patient_id TEXT,
  access_token TEXT, refresh_token TEXT,
  expires_at TEXT, scope TEXT, created_at TEXT
);
"""
@app.on_event("startup")
def startup():
    with engine.begin() as c:
        c.execute(text(DDL))

@app.get("/")
def root():
    # Go to branded UI
    return RedirectResponse("/static/Raphael.html", status_code=307)

@app.get("/healthz")
def healthz():
    return {"status": "ok", "ts": datetime.utcnow().isoformat()}

# ───────────── OpenAI Chat (fallback text chat)
@app.post("/api/chat")
async def chat(req: Dict[str, Any]):
    messages = req.get("messages", [])
    patient = req.get("patient", {})
    sys = (
        "You are Raphael, a multimodal clinical copilot for physicians. "
        "Be cautious, cite guidelines when known, avoid definitive diagnosis; "
        "offer differentials and next steps. Use EHR/FHIR data if provided."
    )
    if patient:
        sys += f" Patient context: {json.dumps(patient)[:1500]}"
    if not OPENAI_API_KEY:
        raise HTTPException(400, "OPENAI_API_KEY missing.")
    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "system", "content": sys}] + messages,
        temperature=0.2,
    )
    return {"reply": resp.choices[0].message.content}

# ───────────── SMART-on-FHIR helpers
def save_tokens(vendor: str, tokens: dict):
    exp = None
    if "expires_in" in tokens:
        exp = (datetime.utcnow() + timedelta(seconds=int(tokens["expires_in"]))).isoformat()
    with engine.begin() as c:
        c.execute(
            text("""INSERT INTO oauth_tokens
                (vendor, access_token, refresh_token, expires_at, scope, created_at)
                VALUES (:v,:at,:rt,:ea,:sc,:ca)"""),
            dict(v=vendor, at=tokens.get("access_token"),
                 rt=tokens.get("refresh_token"),
                 ea=exp, sc=tokens.get("scope"),
                 ca=datetime.utcnow().isoformat())
        )

def latest_token(vendor="default"):
    with engine.begin() as c:
        row = c.execute(
            text("SELECT * FROM oauth_tokens WHERE vendor=:v ORDER BY id DESC LIMIT 1"),
            {"v": vendor}
        ).first()
    return dict(row._mapping) if row else None

async def refresh_if_needed(vendor="default"):
    tok = latest_token(vendor)
    if not tok:
        raise HTTPException(401, "No EHR token. Start at /fhir/login.")
    if tok["expires_at"]:
        try:
            if datetime.fromisoformat(tok["expires_at"]) - datetime.utcnow() > timedelta(seconds=60):
                return tok["access_token"]
        except Exception:
            pass
    if not tok.get("refresh_token"):
        return tok["access_token"]
    async with httpx.AsyncClient() as c:
        data = {
            "grant_type": "refresh_token",
            "refresh_token": tok["refresh_token"],
            "client_id": FHIR_CLIENT_ID,
        }
        auth = (FHIR_CLIENT_ID, FHIR_CLIENT_SECRET) if FHIR_CLIENT_SECRET else None
        r = await c.post(FHIR_TOKEN_URL, data=data, auth=auth)
        r.raise_for_status()
        tokens = r.json()
    save_tokens(vendor, tokens)
    return tokens["access_token"]

async def fhir_headers():
    token = await refresh_if_needed("default")
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/fhir+json",
        "Content-Type": "application/fhir+json",
    }

# ───────────── SMART-on-FHIR endpoints
@app.get("/fhir/login")
def fhir_login():
    if not all([FHIR_BASE_URL, FHIR_AUTH_URL, FHIR_TOKEN_URL, FHIR_CLIENT_ID, FHIR_REDIRECT_URI]):
        return HTMLResponse("<h3>FHIR env vars missing. Set FHIR_* in Render.</h3>", status_code=500)
    params = {
        "response_type": "code",
        "client_id": FHIR_CLIENT_ID,
        "redirect_uri": FHIR_REDIRECT_URI,
        "scope": FHIR_SCOPES,
        "state": secrets.token_urlsafe(16),
        "aud": FHIR_BASE_URL,
    }
    return RedirectResponse(f"{FHIR_AUTH_URL}?{httpx.QueryParams(params)}")

@app.get("/fhir/callback")
async def fhir_callback(code: str):
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": FHIR_REDIRECT_URI,
        "client_id": FHIR_CLIENT_ID,
    }
    auth = (FHIR_CLIENT_ID, FHIR_CLIENT_SECRET) if FHIR_CLIENT_SECRET else None
    async with httpx.AsyncClient() as client:
        r = await client.post(FHIR_TOKEN_URL, data=data, auth=auth)
        r.raise_for_status()
        tokens = r.json()
    save_tokens("default", tokens)
    return RedirectResponse("/static/Raphael.html?ehr=connected")

# ───────────── FHIR CRUD + Summary
@app.get("/api/fhir/patient/{pid}")
async def fhir_get_patient(pid: str):
    headers = await fhir_headers()
    async with httpx.AsyncClient() as c:
        r = await c.get(f"{FHIR_BASE_URL}/Patient/{pid}", headers=headers)
        return JSONResponse(r.json(), status_code=r.status_code)

@app.get("/api/fhir/observations")
async def fhir_obs(patient: str):
    headers = await fhir_headers()
    params = {"patient": patient, "_count": 25, "_sort": "-date"}
    async with httpx.AsyncClient() as c:
        r = await c.get(f"{FHIR_BASE_URL}/Observation", params=params, headers=headers)
        return JSONResponse(r.json(), status_code=r.status_code)

@app.post("/api/fhir/document-reference")
async def create_document_reference(body: dict):
    patient = body.get("patient")
    if not patient:
        raise HTTPException(400, "Missing patient")
    content_type = body.get("content_type", "text/plain")
    title = body.get("title", "Clinical Note")
    data_b64 = body.get("data")
    text_note = body.get("text", "")
    if not data_b64:
        data_b64 = base64.b64encode(text_note.encode("utf-8")).decode("utf-8")
    dr = {
        "resourceType": "DocumentReference",
        "status": "current",
        "type": {"text": title},
        "subject": {"reference": f"Patient/{patient}"},
        "date": datetime.utcnow().isoformat() + "Z",
        "content": [{
            "attachment": {
                "contentType": content_type,
                "data": data_b64,
                "title": title
            }
        }]
    }
    author = body.get("author")
    if author:
        dr["author"] = [{"reference": author}]
    headers = await fhir_headers()
    async with httpx.AsyncClient() as c:
        r = await c.post(f"{FHIR_BASE_URL}/DocumentReference", headers=headers, json=dr)
        return JSONResponse(r.json(), status_code=r.status_code)

@app.get("/api/fhir/summary")
async def patient_summary(patient: str):
    headers = await fhir_headers()
    async with httpx.AsyncClient() as c:
        cond = await c.get(f"{FHIR_BASE_URL}/Condition", params={"patient": patient, "clinical-status": "active", "_count": 50}, headers=headers)
        meds = await c.get(f"{FHIR_BASE_URL}/MedicationStatement", params={"patient": patient, "_count": 50}, headers=headers)
        alg = await c.get(f"{FHIR_BASE_URL}/AllergyIntolerance", params={"patient": patient, "_count": 50}, headers=headers)
        vit = await c.get(f"{FHIR_BASE_URL}/Observation", params={"patient": patient, "category": "vital-signs", "_sort": "-date", "_count": 20}, headers=headers)

    def entries(bundle):
        return [e.get("resource") for e in (bundle.json() if hasattr(bundle, "json") else bundle).get("entry", []) if e.get("resource")]

    def latest_vitals(observations):
        out = {}
        for o in observations:
            code = o.get("code", {}).get("text") or (o.get("code", {}).get("coding", [{}])[0].get("display"))
            if not code:
                continue
            valq = o.get("valueQuantity") or {}
            value = valq.get("value") if valq else None
            unit = valq.get("unit") or valq.get("code") if valq else ""
            if value is None:
                value = o.get("valueString")
            eff = o.get("effectiveDateTime") or o.get("issued")
            if code not in out:
                out[code] = {"value": value, "unit": unit or "", "when": eff}
        return out

    return {
        "conditions": [{"id": x.get("id"), "text": (x.get("code", {}) or {}).get("text")} for x in entries(cond)],
        "medications": [{"id": x.get("id"), "text": (x.get("medicationCodeableConcept", {}) or {}).get("text", "")} for x in entries(meds)],
        "allergies": [{"id": x.get("id"), "text": (x.get("code", {}) or {}).get("text", "")} for x in entries(alg)],
        "vitals": latest_vitals(entries(vit)),
    }

# ───────────── Realtime Voice (WebRTC) session token
@app.post("/api/rt-token")
async def rt_token():
    if not OPENAI_API_KEY:
        raise HTTPException(400, "OPENAI_API_KEY missing.")
    client = OpenAI(api_key=OPENAI_API_KEY)
    session = client.realtime.sessions.create(
        model="gpt-4o-realtime-preview-2024-12-17",
        voice=VOICE_DEFAULT,
        modalities=["audio", "text"],
        instructions="You are Raphael, a calm and precise clinical copilot."
    )
    return session
