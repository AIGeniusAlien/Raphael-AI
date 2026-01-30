import os, json, base64, secrets, httpx, hashlib, io, re
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

from sqlalchemy import create_engine, text

# NEW: RAG + evidence store
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from sentence_transformers import SentenceTransformer
import numpy as np
from pypdf import PdfReader

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
FHIR_SCOPES = os.getenv(
    "FHIR_SCOPES",
    "launch/patient patient/*.read patient/*.write offline_access openid profile fhirUser",
)

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./raphael.db")

# NEW: Evidence + Vector DB (Qdrant)
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333").rstrip("/")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "raphael_evidence_v1").strip()
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_DIM = int(os.getenv("EMBED_DIM", "384"))

TOP_K = int(os.getenv("TOP_K", "12"))
MIN_SOURCES = int(os.getenv("MIN_SOURCES", "3"))
MIN_AVG_SIM = float(os.getenv("MIN_AVG_SIM", "0.25"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "26000"))

# ───────────────── APP
app = FastAPI(title=APP_NAME)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
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

# ───────────────── Internal helpers (NEW: Evidence + RAG)
_ST_MODEL: Optional[SentenceTransformer] = None

def get_st_model() -> SentenceTransformer:
    global _ST_MODEL
    if _ST_MODEL is None:
        _ST_MODEL = SentenceTransformer(EMBED_MODEL)
    return _ST_MODEL

def embed_texts(texts: List[str]) -> np.ndarray:
    model = get_st_model()
    vecs = model.encode(texts, normalize_embeddings=True)
    return np.array(vecs, dtype=np.float32)

def qdrant_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL)

def ensure_qdrant_collection():
    client = qdrant_client()
    existing = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION in existing:
        return
    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=qm.VectorParams(size=EMBED_DIM, distance=qm.Distance.COSINE),
    )

def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def simple_chunk(text: str, max_chars: int = 1200, overlap: int = 150) -> List[str]:
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    chunks = []
    i = 0
    while i < len(text):
        j = min(i + max_chars, len(text))
        chunks.append(text[i:j])
        if j == len(text):
            break
        i = max(0, j - overlap)
    return chunks

def ingest_evidence_text(*, title: str, tier: str, source_date: Optional[str], url: Optional[str], section: Optional[str], text_body: str) -> int:
    ensure_qdrant_collection()
    client = qdrant_client()

    chunks = simple_chunk(text_body)
    if not chunks:
        return 0
    vecs = embed_texts(chunks)

    points = []
    for i, (txt, v) in enumerate(zip(chunks, vecs)):
        chunk_id = sha256(f"{title}|{section}|{i}|{txt[:80]}")
        payload = {
            "chunk_id": chunk_id,
            "title": title,
            "tier": tier,
            "source_date": source_date,
            "url": url,
            "section": section,
            "text": txt,
        }
        points.append({"id": chunk_id, "vector": v.tolist(), "payload": payload})

    client.upsert(collection_name=QDRANT_COLLECTION, points=points)
    return len(points)

def retrieve_evidence(query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    ensure_qdrant_collection()
    client = qdrant_client()
    qv = embed_texts([query])[0].tolist()

    hits = client.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=qv,
        limit=top_k,
        with_payload=True,
    )

    out = []
    for h in hits:
        p = h.payload or {}
        out.append(
            {
                "chunk_id": p.get("chunk_id", ""),
                "title": p.get("title", ""),
                "tier": p.get("tier", "TIER_3"),
                "source_date": p.get("source_date"),
                "url": p.get("url"),
                "section": p.get("section"),
                "text": p.get("text", ""),
                "score": float(h.score),
            }
        )
    return out

def safety_check(chunks: List[Dict[str, Any]]) -> (bool, List[str]):
    notes: List[str] = []
    if len(chunks) < MIN_SOURCES:
        notes.append(f"Insufficient sources retrieved: {len(chunks)} < {MIN_SOURCES}.")
        return False, notes
    avg = sum(c["score"] for c in chunks) / max(1, len(chunks))
    if avg < MIN_AVG_SIM:
        notes.append(f"Low retrieval similarity: avg={avg:.3f} < {MIN_AVG_SIM}.")
        return False, notes
    tiers = {c.get("tier") for c in chunks}
    if "TIER_1" not in tiers:
        notes.append("No Tier 1 evidence retrieved; output must be conservative and low-confidence.")
    return True, notes

def format_evidence(chunks: List[Dict[str, Any]], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    blocks = []
    total = 0
    for c in chunks:
        block = (
            f"[chunk_id={c['chunk_id']} | tier={c['tier']} | title={c['title']} | section={c.get('section')} | date={c.get('source_date')}]\n"
            f"{c['text']}\n"
        )
        if total + len(block) > max_chars:
            break
        blocks.append(block)
        total += len(block)
    return "\n---\n".join(blocks)

def enforce_citations(output_dict: dict):
    # Require citations for each differential item
    for item in output_dict.get("ranked_differential", []):
        if not item.get("citations"):
            raise ValueError("Safety violation: differential item missing citations.")
    # If working diagnosis exists, ensure at least one cited item exists
    if output_dict.get("working_diagnosis"):
        any_cited = any(i.get("citations") for i in output_dict.get("ranked_differential", []))
        if not any_cited:
            raise ValueError("Safety violation: working diagnosis without cited evidence.")

SYSTEM_PROMPT_RAPHAEL = """You are Raphael, a safety-first evidence-grounded clinical AI copilot.
NON-NEGOTIABLE RULES:
- Use ONLY the provided EVIDENCE for medical claims. Do not hallucinate.
- Every clinical claim must cite chunk_id(s). If evidence is insufficient, say so and be conservative.
- Provide diagnostic reasoning: ranked differential (likelihood bands), working diagnosis if justified,
  rule-in/rule-out plan, can't-miss diagnoses, red flags/escalation, missing data, confidence.
- Output MUST be valid JSON matching the OutputV1 schema exactly. No extra keys. No markdown.
"""

CRITIC_SYSTEM = """You are Raphael-SafetyCritic.
Detect unsafe output: missing red flags, missing can't-miss diagnoses, unsupported claims, missing citations, overconfidence.
Return JSON only:
{
  "ok": true/false,
  "issues": ["..."],
  "required_fixes": ["..."]
}
"""

def build_user_prompt(mode: str, case_context: str, question: str, evidence_block: str) -> str:
    return f"""MODE: {mode}

CASE_CONTEXT:
{case_context}

QUESTION:
{question}

EVIDENCE (use only this for medical claims; cite by chunk_id):
{evidence_block}

Return JSON only matching OutputV1.
"""

def build_critic_prompt(draft_json: str, evidence_block: str) -> str:
    return f"""DRAFT_JSON:
{draft_json}

EVIDENCE:
{evidence_block}

Return JSON only.
"""

# Output schema contract (kept lightweight but strict)
OUTPUT_KEYS = [
    "problem_representation",
    "ranked_differential",
    "working_diagnosis",
    "rule_in_out_plan",
    "cant_miss",
    "red_flags_escalation",
    "missing_data",
    "confidence",
    "evidence_used",
    "safety_notes",
]

def llm_client() -> OpenAI:
    if not LLM_BASE_URL:
        raise HTTPException(500, "LLM_BASE_URL not configured.")
    return OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

def call_llm_json(system: str, user: str) -> dict:
    client = llm_client()
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    raw = resp.choices[0].message.content
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(500, "Model did not return valid JSON.")
    return data

def raphael_generate(case_context: str, question: str, mode: str = "radiology") -> dict:
    chunks = retrieve_evidence(question, top_k=TOP_K)
    ok, safety_notes = safety_check(chunks)
    if not ok:
        # Safe refusal output (schema)
        return {
            "problem_representation": "Insufficient evidence retrieved to safely generate a diagnostic output.",
            "ranked_differential": [],
            "working_diagnosis": None,
            "rule_in_out_plan": [
                "Clarify chief complaint, onset, duration, severity, and relevant risk factors.",
                "Collect vitals, focused exam findings, and key labs/imaging based on setting.",
                "Use local protocols; escalate to in-person evaluation if red flags are present."
            ],
            "cant_miss": [],
            "red_flags_escalation": ["If red-flag symptoms/signs are present, seek urgent/emergent evaluation."],
            "missing_data": ["Higher-quality evidence sources and more case context are required."],
            "confidence": "Low (retrieval insufficient).",
            "evidence_used": chunks,
            "safety_notes": safety_notes,
        }

    evidence_block = format_evidence(chunks, MAX_CONTEXT_CHARS)

    # Pass 1: draft
    draft = call_llm_json(
        SYSTEM_PROMPT_RAPHAEL,
        build_user_prompt(mode, case_context, question, evidence_block),
    )

    # Critic pass
    critique = call_llm_json(
        CRITIC_SYSTEM,
        build_critic_prompt(json.dumps(draft), evidence_block),
    )
    if not critique.get("ok", False):
        fixes = "\n".join(f"- {x}" for x in critique.get("required_fixes", []))
        regen_q = question + "\n\nSAFETY_FIXES_REQUIRED:\n" + fixes
        draft = call_llm_json(
            SYSTEM_PROMPT_RAPHAEL,
            build_user_prompt(mode, case_context, regen_q, evidence_block),
        )

    # Enforce schema keys (soft guard) + citations (hard guard)
    for k in OUTPUT_KEYS:
        if k not in draft:
            raise HTTPException(500, f"Model output missing required key: {k}")

    enforce_citations(draft)

    # Attach retrieved evidence + safety notes (ensure included)
    draft["evidence_used"] = chunks
    merged_notes = list(dict.fromkeys((draft.get("safety_notes") or []) + safety_notes + (critique.get("issues") or [])))
    draft["safety_notes"] = merged_notes
    return draft


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

    # Ensure Qdrant exists (if reachable)
    try:
        ensure_qdrant_collection()
    except Exception:
        # Don't fail startup if qdrant isn't up yet (Render/local)
        pass


@app.get("/")
def root():
    return RedirectResponse("/static/Raphael.html")


@app.get("/healthz")
def healthz():
    return {"status": "ok", "ts": datetime.utcnow().isoformat()}


@app.get("/api/env")
def read_env():
    return {"app": APP_NAME, "voice_enabled": VOICE_ENABLED, "model": LLM_MODEL}


# ───────────────── Patients CRUD
@app.get("/api/patients")
def list_patients(q: Optional[str] = None):
    with engine.begin() as c:
        if q:
            rows = c.execute(
                text("SELECT * FROM patients WHERE name LIKE :q ORDER BY created_at DESC"),
                {"q": f"%{q}%"},
            ).mappings().all()
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
        c.execute(
            text("INSERT INTO patients(name, dob, mrn, created_at) VALUES (:n,:d,:m,:t)"),
            {"n": name, "d": dob, "m": mrn, "t": datetime.utcnow().isoformat()},
        )
        row = c.execute(text("SELECT * FROM patients ORDER BY id DESC LIMIT 1")).mappings().first()
    return row


@app.delete("/api/patients/{pid}")
def delete_patient(pid: int):
    with engine.begin() as c:
        c.execute(
            text("DELETE FROM messages WHERE conversation_id IN (SELECT id FROM conversations WHERE patient_id=:p)"),
            {"p": pid},
        )
        c.execute(text("DELETE FROM conversations WHERE patient_id=:p"), {"p": pid})
        c.execute(text("DELETE FROM patients WHERE id=:p"), {"p": pid})
    return {"ok": True}


# ───────────────── Conversations & Messages
@app.get("/api/conversations")
def list_conversations(patient_id: int):
    with engine.begin() as c:
        rows = c.execute(
            text("SELECT * FROM conversations WHERE patient_id=:p ORDER BY created_at DESC"),
            {"p": patient_id},
        ).mappings().all()
        return list(rows)


@app.post("/api/conversations")
def create_conversation(body: dict):
    pid = body.get("patient_id")
    title = body.get("title") or "New Chat"
    if not pid:
        raise HTTPException(400, "Missing patient_id.")
    with engine.begin() as c:
        c.execute(
            text("INSERT INTO conversations(patient_id, title, created_at) VALUES (:p,:t,:c)"),
            {"p": pid, "t": title, "c": datetime.utcnow().isoformat()},
        )
        row = c.execute(text("SELECT * FROM conversations ORDER BY id DESC LIMIT 1")).mappings().first()
    return row


@app.get("/api/messages")
def get_messages(conversation_id: int):
    with engine.begin() as c:
        rows = c.execute(
            text("SELECT * FROM messages WHERE conversation_id=:c ORDER BY id ASC"),
            {"c": conversation_id},
        ).mappings().all()
    return list(rows)


# ───────────────── Chat (LLM) - keep as general chat
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
        "Use clear, cautious, evidence-aligned language. "
        "Offer differential considerations, clinical reasoning, and next-step guidance. "
        "Be explicit and concise when referencing evidence."
    )
    if patient:
        system += f" Patient context: {json.dumps(patient)[:1200]}"

    now = datetime.utcnow().isoformat()
    with engine.begin() as c:
        c.execute(
            text("INSERT INTO messages(conversation_id, role, content, created_at) VALUES (:cid,'user',:ct,:ts)"),
            {"cid": conversation_id, "ct": user_text, "ts": now},
        )

    with engine.begin() as c:
        hist = c.execute(
            text("SELECT role, content FROM messages WHERE conversation_id=:cid ORDER BY id ASC"),
            {"cid": conversation_id},
        ).all()

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
        c.execute(
            text("INSERT INTO messages(conversation_id, role, content, created_at) VALUES (:cid,'assistant',:ct,:ts)"),
            {"cid": conversation_id, "ct": reply, "ts": datetime.utcnow().isoformat()},
        )
    return {"reply": reply}


# ───────────────── NEW: Raphael Clinical Query (evidence-grounded + citations)
@app.post("/api/raphael/query")
def raphael_query(body: dict):
    """
    body:
      - conversation_id (required)
      - question (required)
      - mode: 'radiology'|'general' (optional)
    Uses the conversation history as case_context + evidence from Qdrant.
    """
    conversation_id = body.get("conversation_id")
    question = (body.get("question") or "").strip()
    mode = body.get("mode") or "radiology"

    if not conversation_id:
        raise HTTPException(400, "Missing conversation_id.")
    if not question:
        raise HTTPException(400, "Missing question.")

    # Build case_context from conversation history
    with engine.begin() as c:
        hist = c.execute(
            text("SELECT role, content FROM messages WHERE conversation_id=:cid ORDER BY id ASC"),
            {"cid": conversation_id},
        ).all()

    case_context = "\n".join([f"{r.upper()}: {ct}" for (r, ct) in hist][-50:])

    out = raphael_generate(case_context=case_context, question=question, mode=mode)

    # Persist assistant output as JSON string for auditability
    with engine.begin() as c:
        c.execute(
            text("INSERT INTO messages(conversation_id, role, content, created_at) VALUES (:cid,'assistant',:ct,:ts)"),
            {"cid": conversation_id, "ct": json.dumps(out), "ts": datetime.utcnow().isoformat()},
        )

    return out


# ───────────────── NEW: Evidence ingestion endpoints (Qdrant)
@app.post("/api/evidence/ingest")
def evidence_ingest(body: dict):
    """
    body:
      - title (required)
      - text (required)
      - tier: TIER_1|TIER_2|TIER_3 (optional, default TIER_1)
      - source_date (optional)
      - url (optional)
      - section (optional)
    """
    title = (body.get("title") or "").strip()
    text_body = (body.get("text") or "").strip()
    tier = body.get("tier") or "TIER_1"
    source_date = body.get("source_date")
    url = body.get("url")
    section = body.get("section")

    if not title:
        raise HTTPException(400, "Missing title.")
    if not text_body:
        raise HTTPException(400, "Missing text.")

    n = ingest_evidence_text(title=title, tier=tier, source_date=source_date, url=url, section=section, text_body=text_body)
    return {"status": "ok", "chunks_indexed": n}


@app.post("/api/evidence/ingest_pdf")
async def evidence_ingest_pdf(
    title: str = Form(...),
    tier: str = Form("TIER_1"),
    source_date: str = Form(""),
    url: str = Form(""),
    section: str = Form(""),
    file: UploadFile = File(...),
):
    pdf_bytes = await file.read()
    reader = PdfReader(io.BytesIO(pdf_bytes))
    txt = " ".join([(p.extract_text() or "") for p in reader.pages]).strip()
    if not txt:
        raise HTTPException(400, "Could not extract text from PDF.")
    n = ingest_evidence_text(
        title=title,
        tier=tier,
        source_date=source_date or None,
        url=url or None,
        section=section or None,
        text_body=txt,
    )
    return {"status": "ok", "chunks_indexed": n}


@app.post("/api/evidence/ingest_url")
async def evidence_ingest_url(body: dict):
    """
    body:
      - title (required)
      - url (required)
      - tier (optional)
      - source_date (optional)
      - section (optional)
    Fetches URL; if PDF extracts text; if HTML strips tags.
    """
    title = (body.get("title") or "").strip()
    url = (body.get("url") or "").strip()
    tier = body.get("tier") or "TIER_1"
    source_date = body.get("source_date")
    section = body.get("section")

    if not title:
        raise HTTPException(400, "Missing title.")
    if not url:
        raise HTTPException(400, "Missing url.")

    headers = {"User-Agent": "RaphaelAI/1.0"}
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.get(url, headers=headers, follow_redirects=True)
        r.raise_for_status()
        ctype = (r.headers.get("Content-Type", "") or "").split(";")[0].strip().lower()

        if "application/pdf" in ctype or url.lower().endswith(".pdf"):
            reader = PdfReader(io.BytesIO(r.content))
            txt = " ".join([(p.extract_text() or "") for p in reader.pages]).strip()
        else:
            html = r.text
            html = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
            html = re.sub(r"(?is)<style.*?>.*?</style>", " ", html)
            html = re.sub(r"(?is)<.*?>", " ", html)
            txt = re.sub(r"\s+", " ", html).strip()

    if not txt:
        raise HTTPException(400, "No text extracted from URL.")

    n = ingest_evidence_text(
        title=title,
        tier=tier,
        source_date=source_date,
        url=url,
        section=section,
        text_body=txt,
    )
    return {"status": "ok", "content_type": ctype, "chunks_indexed": n}


# ───────────────── Guidance documents + FTS search (lightweight RAG you already had)
@app.post("/api/docs/upload")
async def upload_doc(title: str = Form(...), file: UploadFile = File(...)):
    text_bytes = await file.read()
    body = text_bytes.decode("utf-8", errors="ignore")
    with engine.begin() as c:
        c.execute(
            text("INSERT INTO documents(title, body, created_at) VALUES (:t,:b,:ts)"),
            {"t": title, "b": body, "ts": datetime.utcnow().isoformat()},
        )
    return {"ok": True}


@app.get("/api/docs/search")
def search_docs(q: str = Query(..., min_length=2)):
    with engine.begin() as c:
        rows = c.execute(
            text(
                "SELECT d.id, d.title, snippet(doc_fts, 1, '[', ']', '…', 10) AS snippet "
                "FROM doc_fts JOIN documents d ON d.id = doc_fts.rowid "
                "WHERE doc_fts MATCH :q LIMIT 10"
            ),
            {"q": q},
        ).mappings().all()
    return list(rows)


# ───────────────── Voice token endpoint placeholder
@app.post("/api/rt-token")
def rt_token():
    if not VOICE_ENABLED:
        raise HTTPException(400, "Voice is disabled in this build.")
    raise HTTPException(501, "Voice backend not implemented yet.")

# ───────────────── (Optional) Minimal SMART-on-FHIR stubs kept for later
# (Routes omitted for brevity; safe to leave out until you enable EHR)
