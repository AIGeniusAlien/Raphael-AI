from __future__ import annotations

import os, json, secrets, httpx, hashlib, io, re
from datetime import datetime
from typing import TYPE_CHECKING, Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query, Depends
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.exception_handlers import request_validation_exception_handler
from starlette.responses import JSONResponse
from pydantic import ValidationError
from openai import OpenAI

from sqlalchemy import create_engine, text
from database import execute_sqlite_script

from qdrant_client import QdrantClient
from qdrant_config import connection_options
from qdrant_client.http import models as qm
if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer
import numpy as np
from pypdf import PdfReader

# New modules you added
from corpus.fetch import fetch_url
from corpus.indexer import build_payload, doc_id_for, point_id_for, sha256 as sha256_hex
from corpus.registry import list_sources
from pubmed.query_packs import QUERY_PACKS
from pubmed.ingest import ingest_pubmed_query
from feedback.store import ensure_feedback_table, insert_feedback, list_feedback
from radiology.cant_miss import get_checklist
from rag.assessment_schema import Assessment, Critique

# ───────────────── CONFIG
APP_NAME = os.getenv("APP_NAME", "Raphael")
STATIC_PATH = os.getenv("STATIC_PATH", "static")
FRONTEND_URL = os.getenv("FRONTEND_URL", "")

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").rstrip("/")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_API_KEY = os.getenv("LLM_API_KEY", "EMPTY")

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./raphael.db")

QDRANT_URL = os.getenv("QDRANT_URL", "").rstrip("/")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip()
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "raphael_evidence_v1").strip()

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_DIM = int(os.getenv("EMBED_DIM", "384"))

TOP_K = int(os.getenv("TOP_K", "12"))
MIN_SOURCES = int(os.getenv("MIN_SOURCES", "3"))
MIN_AVG_SIM = float(os.getenv("MIN_AVG_SIM", "0.25"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "26000"))

# Optional: protect corpus endpoints
CORPUS_RUN_TOKEN = os.getenv("CORPUS_RUN_TOKEN", "").strip()
DEV_PORTAL = os.getenv("RAPHAEL_DEV_PORTAL", "") == "1"

# ───────────────── APP
app = FastAPI(title=APP_NAME)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[] if DEV_PORTAL else ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=not DEV_PORTAL,
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
  role TEXT NOT NULL,
  content TEXT NOT NULL,
  created_at TEXT,
  FOREIGN KEY(conversation_id) REFERENCES conversations(id)
);

CREATE TABLE IF NOT EXISTS documents (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  title TEXT,
  body TEXT,
  created_at TEXT
);

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

# ───────────────── Core helpers
_ST_MODEL: Optional[SentenceTransformer] = None

def get_st_model() -> SentenceTransformer:
    global _ST_MODEL
    if _ST_MODEL is None:
        # Load the numerical/model stack only when embeddings are requested.
        from sentence_transformers import SentenceTransformer
        _ST_MODEL = SentenceTransformer(EMBED_MODEL)
    return _ST_MODEL

def embed_texts(texts: List[str]) -> np.ndarray:
    if os.getenv('RAPHAEL_EMBEDDING_BACKEND') == 'openai':
        from rag.hosted_embeddings import encode
        return encode(texts)
    model = get_st_model()
    vecs = model.encode(texts, normalize_embeddings=True)
    return np.array(vecs, dtype=np.float32)

def qdrant_client() -> QdrantClient:
    if not QDRANT_URL:
        raise HTTPException(500, "QDRANT_URL not set.")
    if QDRANT_API_KEY:
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, **connection_options(QDRANT_URL))
    return QdrantClient(url=QDRANT_URL, **connection_options(QDRANT_URL))

def ensure_qdrant_collection():
    client = qdrant_client()
    existing = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION in existing:
        vectors = client.get_collection(QDRANT_COLLECTION).config.params.vectors
        if not hasattr(vectors, 'size') or vectors.size != EMBED_DIM or str(vectors.distance).lower() != 'cosine':
            raise ValueError('Existing collection is incompatible with unnamed cosine embeddings.')
        if os.getenv('RAPHAEL_EMBEDDING_BACKEND') != 'openai':
            from qdrant_config import verify_legacy_encoder
            verify_legacy_encoder(client,QDRANT_COLLECTION,EMBED_MODEL)
    # Verify actual model output before any creation/upsert path. A configured
    # dimension alone is not proof the selected encoder produces that shape.
    probe = embed_texts(['Raphael nonclinical embedding compatibility probe.'])
    if probe.shape != (1, EMBED_DIM) or not np.isfinite(probe).all():
        raise ValueError('Actual embedding dimensions do not match configured collection dimensions.')
    if os.getenv('RAPHAEL_EMBEDDING_BACKEND') == 'openai':
        from curated_evidence import inventory
        if QDRANT_COLLECTION in existing:
            inventory(client)
    if QDRANT_COLLECTION in existing:
        return
    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=qm.VectorParams(size=EMBED_DIM, distance=qm.Distance.COSINE),
    )

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

# ───────────────── Evidence ingestion with epistemology metadata
def ingest_evidence_text(*,
    title: str,
    tier: str,                 # A/B/C/D
    source_type: str,          # guideline/systematic_review/rct/etc
    org: str,
    jurisdiction: str,
    source_date: Optional[str],
    url: Optional[str],
    section: Optional[str],
    evidence_strength: Optional[str],  # high/moderate/low/very_low
    text_body: str
) -> int:
    if os.getenv('RAPHAEL_EMBEDDING_BACKEND') == 'openai':
        raise HTTPException(409, 'Use the reviewed curated ingestion CLI for the hosted embedding collection; legacy untagged ingestion requires a separate collection.')
    ensure_qdrant_collection()
    client = qdrant_client()

    doc_id = doc_id_for(url or "", title)
    chunks = simple_chunk(text_body)
    if not chunks:
        return 0
    vecs = embed_texts(chunks)

    points = []
    for i, (txt, v) in enumerate(zip(chunks, vecs)):
        chunk_id = sha256_hex(f"{doc_id}|{section}|{i}|{txt[:80]}")
        payload = build_payload(
            chunk_id=chunk_id,
            doc_id=doc_id,
            title=title,
            tier=tier,
            source_type=source_type,
            org=org,
            jurisdiction=jurisdiction,
            source_date=source_date,
            url=url,
            section=section,
            evidence_strength=evidence_strength,
            text=txt,
        )
        payload.update(embedding_model=EMBED_MODEL,embedding_dimensions=EMBED_DIM)
        points.append(qm.PointStruct(id=point_id_for(chunk_id), vector=v.tolist(), payload=payload))

    client.upsert(collection_name=QDRANT_COLLECTION, points=points)
    return len(points)

# ───────────────── Retrieval weighting (Tier + strength + recency)
TIER_WEIGHT = {"A": 3.0, "B": 2.0, "C": 1.2, "D": 0.8}
STRENGTH_WEIGHT = {"high": 1.8, "moderate": 1.4, "low": 1.0, "very_low": 0.7, None: 1.0}

def rerank(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def score(c):
        base = float(c.get("score", 0.0))
        tier = c.get("tier", "C")
        strength = c.get("evidence_strength", None)
        return base * TIER_WEIGHT.get(tier, 1.0) * STRENGTH_WEIGHT.get(strength, 1.0)
    return sorted(chunks, key=score, reverse=True)

def retrieve_evidence(query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    if os.getenv('RAPHAEL_EMBEDDING_BACKEND') == 'openai':
        from curated_evidence import search
        return search(query, min(top_k, 12))
    ensure_qdrant_collection()
    client = qdrant_client()
    qv = embed_texts([query])[0].tolist()

    hits = client.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=qv,
        limit=max(top_k, 20),  # pull more then rerank
        with_payload=True,
    )

    out = []
    for h in hits:
        p = h.payload or {}
        # Bibliographic summaries are discovery records,not clinical claim
        # support. Preserve old records but exclude them and explicit retractions.
        if p.get('retracted') or p.get('is_current') is False or p.get('knowledge_scope')=='bibliographic_metadata_not_full_text_or_claim_support' or (p.get('org')=='PubMed' and p.get('section')=='summary'):
            continue
        out.append({
            "chunk_id": p.get("chunk_id", ""),
            "doc_id": p.get("doc_id", ""),
            "title": p.get("title", ""),
            "tier": p.get("tier", "C"),
            "source_type": p.get("source_type", "unknown"),
            "org": p.get("org", ""),
            "jurisdiction": p.get("jurisdiction", ""),
            "source_date": p.get("source_date"),
            "url": p.get("url"),
            "section": p.get("section"),
            "evidence_strength": p.get("evidence_strength"),
            "text": p.get("text", ""),
            "score": float(h.score),
        })
    out = rerank(out)[:top_k]
    return out

def safety_check(chunks: List[Dict[str, Any]]) -> (bool, List[str]):
    notes: List[str] = []
    sources = {c.get("doc_id") or c.get("url") or c["chunk_id"] for c in chunks}
    if len(sources) < MIN_SOURCES:
        notes.append(f"Insufficient distinct sources retrieved: {len(sources)} < {MIN_SOURCES}.")
        return False, notes
    avg = sum(float(c["score"]) for c in chunks) / max(1, len(chunks))
    if avg < MIN_AVG_SIM:
        notes.append(f"Low retrieval similarity: avg={avg:.3f} < {MIN_AVG_SIM}.")
        return False, notes
    tiers = {c.get("tier") for c in chunks}
    if "A" not in tiers:
        notes.append("No Tier A evidence retrieved; output must be conservative and low-confidence.")
    return True, notes

def format_evidence(chunks: List[Dict[str, Any]], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    blocks = []
    total = 0
    for c in chunks:
        block = (
            f"[chunk_id={c['chunk_id']} | tier={c['tier']} | type={c['source_type']} | strength={c.get('evidence_strength')} | title={c['title']} | org={c.get('org')} | date={c.get('source_date')}]\n"
            f"{c['text']}\n"
        )
        if total + len(block) > max_chars:
            break
        blocks.append(block)
        total += len(block)
    return "\n---\n".join(blocks)

def enforce_citations(output_dict: dict, chunks: Optional[List[dict]] = None):
    allowed = {chunk["chunk_id"] for chunk in chunks} if chunks is not None else None
    # Require citations for each differential item
    for item in output_dict.get("ranked_differential", []):
        if not item.get("citations"):
            raise HTTPException(500, "Safety violation: differential item missing citations.")
        if allowed is not None and any(citation not in allowed for citation in item["citations"]):
            raise HTTPException(502, "Safety violation: citation is not in retrieved evidence.")
    # If working diagnosis exists, require at least one cited evidence
    if output_dict.get("working_diagnosis"):
        any_cited = any(i.get("citations") for i in output_dict.get("ranked_differential", []))
        if not any_cited:
            raise HTTPException(500, "Safety violation: working diagnosis without cited evidence.")

SYSTEM_PROMPT_RAPHAEL = """You are Raphael, a safety-first evidence-grounded clinical AI copilot.
NON-NEGOTIABLE RULES:
- Use ONLY the provided EVIDENCE for medical claims. Do not hallucinate.
- Every clinical claim must cite chunk_id(s). If evidence is insufficient, say so and be conservative.
- Use epistemology: prefer Tier A, then B, then C; describe study type/limitations for Tier C claims.
- Handle contradictions: if evidence disagrees, say so and explain why (population/date/type).
- Scope awareness: if pediatric/pregnancy/immunocompromised context exists, adapt recommendations and add safety notes.
- Output MUST be valid JSON matching OutputV1 schema exactly. No extra keys. No markdown.
"""

CRITIC_SYSTEM = """You are Raphael-SafetyCritic.
Detect unsafe output: missing red flags, missing can't-miss diagnoses, unsupported claims, missing citations, overconfidence, missing scope callouts.
Return JSON only:
{"ok": true/false, "issues": ["..."], "required_fixes": ["..."]}
"""

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
    resp = llm_client().chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.2,
    )
    raw = resp.choices[0].message.content
    try:
        result = json.loads(raw)
        if not isinstance(result, dict):
            raise ValueError("Expected JSON object")
        return result
    except (ValueError, TypeError):
        raise HTTPException(502, "Model did not return a valid JSON object.")

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

def raphael_generate(case_context: str, question: str, mode: str = "radiology", evidence_chunks: Optional[List[dict]] = None) -> dict:
    chunks = retrieve_evidence(question, top_k=TOP_K) if evidence_chunks is None else evidence_chunks
    ok, safety_notes = safety_check(chunks)

    if not ok:
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

    # Draft
    draft = call_llm_json(SYSTEM_PROMPT_RAPHAEL, build_user_prompt(mode, case_context, question, evidence_block))

    # Critic pass
    def critique_draft(candidate):
        result = call_llm_json(CRITIC_SYSTEM, build_critic_prompt(json.dumps(candidate), evidence_block))
        try:
            return Critique.model_validate(result).model_dump()
        except ValidationError:
            raise HTTPException(502, "Model critic output does not match the safety schema.")

    critique = critique_draft(draft)
    if not critique.get("ok", False):
        fixes = "\n".join(f"- {x}" for x in critique.get("required_fixes", []))
        regen_q = question + "\n\nSAFETY_FIXES_REQUIRED:\n" + fixes
        draft = call_llm_json(SYSTEM_PROMPT_RAPHAEL, build_user_prompt(mode, case_context, regen_q, evidence_block))
        critique = critique_draft(draft)
        if not critique["ok"]:
            raise HTTPException(502, "Model critic did not approve the revised assessment.")

    for k in OUTPUT_KEYS:
        if k not in draft:
            raise HTTPException(500, f"Model output missing required key: {k}")

    try:
        draft = Assessment.model_validate(draft).model_dump()
    except ValidationError:
        raise HTTPException(502, "Model output does not match the assessment schema.")
    enforce_citations(draft, chunks)
    draft["evidence_used"] = chunks
    draft["safety_notes"] = list(dict.fromkeys((draft.get("safety_notes") or []) + safety_notes + (critique.get("issues") or [])))

    # Radiology: attach can’t-miss list if user includes study hint
    if mode == "radiology":
        # If user asks about CT head / CXR / CTPE, inject checklist into safety notes (non-invasive)
        q = question.lower()
        if "ct head" in q or "head ct" in q:
            draft["cant_miss"] = list(dict.fromkeys((draft.get("cant_miss") or []) + get_checklist("CT_HEAD")))
        if "cxr" in q or "chest x-ray" in q or "chest xray" in q:
            draft["cant_miss"] = list(dict.fromkeys((draft.get("cant_miss") or []) + get_checklist("CXR")))
        if "ctpe" in q or "pulmonary embol" in q:
            draft["cant_miss"] = list(dict.fromkeys((draft.get("cant_miss") or []) + get_checklist("CTPE")))

    return draft

# ───────────────── Startup
@app.on_event("startup")
def startup():
    with engine.begin() as c:
        execute_sqlite_script(c, DDL)
        execute_sqlite_script(c, TRIGGERS)
    ensure_feedback_table(engine)
    if DEV_PORTAL:
        from dev_runtime import migrate
        migrate(engine)
    try:
        if QDRANT_URL:
            ensure_qdrant_collection()
    except Exception as e:
        # SDK/proxy exception messages can reflect URLs, headers or credentials.
        print("QDRANT_STARTUP_WARNING", type(e).__name__)

# ───────────────── Base routes
@app.get("/")
def root():
    return RedirectResponse("/static/portal.html" if DEV_PORTAL else "/static/Raphael.html")

@app.get("/healthz")
def healthz():
    return {"status": "ok", "ts": datetime.utcnow().isoformat()}


@app.get("/readyz")
def readiness():
    try:
        with engine.connect() as c:
            c.execute(text("SELECT 1"))
        if DEV_PORTAL:
            from dev_runtime import schema_status
            if not schema_status(engine)["ready"]:
                return JSONResponse({"status": "not_ready", "detail": "Development schema migration required."}, status_code=503)
        return {"status": "ready", "scope": "synthetic-development" if DEV_PORTAL else "legacy-development",
                "clinical_validation": "not_established"}
    except Exception:
        return JSONResponse({"status": "not_ready", "detail": "Database readiness check failed."}, status_code=503)

# ───────────────── Patients/Conversations/Messages (unchanged functionality)
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
        c.execute(text("INSERT INTO patients(name, dob, mrn, created_at) VALUES (:n,:d,:m,:t)"),
                  {"n": name, "d": dob, "m": mrn, "t": datetime.utcnow().isoformat()})
        row = c.execute(text("SELECT * FROM patients ORDER BY id DESC LIMIT 1")).mappings().first()
    return row

@app.post("/api/conversations")
def create_conversation(body: dict):
    pid = body.get("patient_id")
    title = body.get("title") or "New Chat"
    if not pid:
        raise HTTPException(400, "Missing patient_id.")
    with engine.begin() as c:
        c.execute(text("INSERT INTO conversations(patient_id, title, created_at) VALUES (:p,:t,:c)"),
                  {"p": pid, "t": title, "c": datetime.utcnow().isoformat()})
        row = c.execute(text("SELECT * FROM conversations ORDER BY id DESC LIMIT 1")).mappings().first()
    return row

@app.get("/api/conversations")
def list_conversations(patient_id: int):
    with engine.begin() as c:
        rows = c.execute(text("SELECT * FROM conversations WHERE patient_id=:p ORDER BY created_at DESC"),
                         {"p": patient_id}).mappings().all()
    return list(rows)

@app.get("/api/messages")
def get_messages(conversation_id: int):
    with engine.begin() as c:
        rows = c.execute(text("SELECT * FROM messages WHERE conversation_id=:c ORDER BY id ASC"),
                         {"c": conversation_id}).mappings().all()
    return list(rows)

# ───────────────── General chat (kept)
@app.post("/api/chat")
def chat(body: dict):
    conversation_id = body.get("conversation_id")
    user_text = (body.get("text") or "").strip()

    if not conversation_id:
        raise HTTPException(400, "Missing conversation_id.")
    if not user_text:
        raise HTTPException(400, "Empty prompt.")

    with engine.begin() as c:
        c.execute(text("INSERT INTO messages(conversation_id, role, content, created_at) VALUES (:cid,'user',:ct,:ts)"),
                  {"cid": conversation_id, "ct": user_text, "ts": datetime.utcnow().isoformat()})

    with engine.begin() as c:
        hist = c.execute(text("SELECT role, content FROM messages WHERE conversation_id=:cid ORDER BY id ASC"),
                         {"cid": conversation_id}).all()
    messages = [{"role": "system", "content": "You are Raphael. Be cautious and concise."}] + [{"role": r, "content": c} for (r, c) in hist][-30:]

    try:
        resp = llm_client().chat.completions.create(model=LLM_MODEL, messages=messages, temperature=0.2)
        reply = resp.choices[0].message.content
    except Exception as e:
        raise HTTPException(500, f"LLM error: {e}")

    with engine.begin() as c:
        c.execute(text("INSERT INTO messages(conversation_id, role, content, created_at) VALUES (:cid,'assistant',:ct,:ts)"),
                  {"cid": conversation_id, "ct": reply, "ts": datetime.utcnow().isoformat()})
    return {"reply": reply}

# ───────────────── Raphael query (core)
@app.post("/api/raphael/query")
def raphael_query(body: dict):
    conversation_id = body.get("conversation_id")
    question = (body.get("question") or "").strip()
    mode = body.get("mode") or "radiology"

    if not conversation_id:
        raise HTTPException(400, "Missing conversation_id.")
    if not question:
        raise HTTPException(400, "Missing question.")

    with engine.begin() as c:
        hist = c.execute(text("SELECT role, content FROM messages WHERE conversation_id=:cid ORDER BY id ASC"),
                         {"cid": conversation_id}).all()
    case_context = "\n".join([f"{r.upper()}: {ct}" for (r, ct) in hist][-50:])
    out = raphael_generate(case_context=case_context, question=question, mode=mode)

    with engine.begin() as c:
        c.execute(text("INSERT INTO messages(conversation_id, role, content, created_at) VALUES (:cid,'assistant',:ct,:ts)"),
                  {"cid": conversation_id, "ct": json.dumps(out), "ts": datetime.utcnow().isoformat()})
    return out

# ───────────────── Evidence ingest endpoints (Tier A/B/D manual or URL)
@app.post("/api/evidence/ingest")
def evidence_ingest(body: dict):
    title = (body.get("title") or "").strip()
    text_body = (body.get("text") or "").strip()
    tier = (body.get("tier") or "A").strip()
    source_type = (body.get("source_type") or "guideline").strip()
    org = (body.get("org") or "Unknown").strip()
    jurisdiction = (body.get("jurisdiction") or "Global").strip()
    source_date = body.get("source_date")
    url = body.get("url")
    section = body.get("section")
    strength = body.get("evidence_strength") or None

    if not title:
        raise HTTPException(400, "Missing title.")
    if not text_body:
        raise HTTPException(400, "Missing text.")

    n = ingest_evidence_text(
        title=title, tier=tier, source_type=source_type, org=org, jurisdiction=jurisdiction,
        source_date=source_date, url=url, section=section, evidence_strength=strength, text_body=text_body
    )
    return {"status": "ok", "chunks_indexed": n}

@app.post("/api/corpus/ingest_url")
async def corpus_ingest_url(body: dict):
    """
    Ingest a public URL as Tier A/B/D depending on metadata you pass.
    """
    if CORPUS_RUN_TOKEN:
        if (body.get("token") or "") != CORPUS_RUN_TOKEN:
            raise HTTPException(403, "Invalid token.")

    title = (body.get("title") or "").strip()
    url = (body.get("url") or "").strip()
    tier = (body.get("tier") or "A").strip()
    source_type = (body.get("source_type") or "guideline").strip()
    org = (body.get("org") or "Unknown").strip()
    jurisdiction = (body.get("jurisdiction") or "Global").strip()
    source_date = body.get("source_date")
    section = body.get("section")
    strength = body.get("evidence_strength") or None

    if not (title and url):
        raise HTTPException(400, "Missing title/url.")

    ctype, txt = await fetch_url(url)
    if not txt:
        raise HTTPException(400, "No text extracted.")

    n = ingest_evidence_text(
        title=title, tier=tier, source_type=source_type, org=org, jurisdiction=jurisdiction,
        source_date=source_date, url=url, section=section, evidence_strength=strength, text_body=txt
    )
    return {"status": "ok", "content_type": ctype, "chunks_indexed": n}

@app.get("/api/corpus/sources")
def corpus_sources():
    return [s.__dict__ for s in list_sources()]

# ───────────────── PubMed ingest endpoints (Tier C)
@app.post("/api/pubmed/ingest_pack")
async def pubmed_ingest_pack(body: dict):
    """
    body:
      - pack: radiology|emergency_medicine|internal_medicine|pediatrics
      - retmax: optional
    """
    pack = (body.get("pack") or "radiology").strip()
    retmax = int(body.get("retmax") or 20)
    queries = QUERY_PACKS.get(pack)
    if not queries:
        raise HTTPException(400, f"Unknown pack: {pack}")

    results = []
    for q in queries:
        res = await ingest_pubmed_query(
            qdrant_url=QDRANT_URL,
            qdrant_api_key=QDRANT_API_KEY,
            collection=QDRANT_COLLECTION,
            embed_model=EMBED_MODEL,
            embed_dim=EMBED_DIM,
            query=q,
            retmax=retmax,
        )
        results.append(res)

    return {"pack": pack, "results": results}

# ───────────────── Feedback loop (clinician-in-the-loop)
@app.post("/api/feedback/submit")
def feedback_submit(body: dict):
    rec = {
        "ts": datetime.utcnow().isoformat(),
        "cid": int(body.get("conversation_id") or 0),
        "q": body.get("question") or "",
        "m": body.get("mode") or "radiology",
        "out": json.dumps(body.get("model_output") or {}),
        "ev": json.dumps(body.get("evidence_used") or []),
        "act": body.get("clinician_action") or "approve",
        "notes": body.get("clinician_notes") or "",
        "corr": json.dumps(body.get("corrected_output") or {}),
    }
    if rec["cid"] <= 0:
        raise HTTPException(400, "Missing/invalid conversation_id.")
    insert_feedback(engine, rec)
    return {"status": "ok"}

@app.get("/api/feedback/list")
def feedback_list(limit: int = 50):
    return list_feedback(engine, limit=limit)

# ───────────────── Radiology helpers
@app.get("/api/radiology/checklist")
def radiology_checklist(study: str = "CT_HEAD"):
    return {"study": study, "cant_miss": get_checklist(study)}

# ───────────────── Docs search remains available
@app.post("/api/docs/upload")
async def upload_doc(title: str = Form(...), file: UploadFile = File(...)):
    text_bytes = await file.read()
    body = text_bytes.decode("utf-8", errors="ignore")
    with engine.begin() as c:
        c.execute(text("INSERT INTO documents(title, body, created_at) VALUES (:t,:b,:ts)"),
                  {"t": title, "b": body, "ts": datetime.utcnow().isoformat()})
    return {"ok": True}

@app.get("/api/docs/search")
def search_docs(q: str = Query(..., min_length=2)):
    with engine.begin() as c:
        rows = c.execute(text(
            "SELECT d.id, d.title, snippet(doc_fts, 1, '[', ']', '…', 10) AS snippet "
            "FROM doc_fts JOIN documents d ON d.id = doc_fts.rowid "
            "WHERE doc_fts MATCH :q LIMIT 10"
        ), {"q": q}).mappings().all()
    return list(rows)


if DEV_PORTAL:
    from dev_portal import install_dev_portal

    @app.exception_handler(RequestValidationError)
    async def safe_portal_validation(request, exc):
        if request.url.path.startswith("/api/dev/"):
            # FastAPI's default validation body includes submitted input values,
            # which could echo a mistyped password. Keep portal errors generic.
            return JSONResponse({"detail": "Invalid portal request. Check required fields and lengths."}, status_code=422)
        return await request_validation_exception_handler(request, exc)

    @app.middleware("http")
    async def isolate_development_portal(request, call_next):
        # Legacy APIs have no role/tenant boundaries. Preserve them in legacy
        # mode, but never expose them alongside authenticated portal sessions.
        if request.url.path.startswith("/api/") and not request.url.path.startswith("/api/dev/"):
            response = JSONResponse({"detail": "Legacy APIs are disabled in development portal mode."}, status_code=403)
        else:
            response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Permissions-Policy"] = "camera=(), geolocation=(), microphone=(self)"
        if request.url.scheme=="https":response.headers["Strict-Transport-Security"]="max-age=31536000"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; frame-ancestors 'none'; base-uri 'self'"
        return response

    def development_assessment(**kwargs):
        tenant = kwargs.pop("tenant", None)
        backend = os.getenv("RAPHAEL_EVIDENCE_BACKEND", "qdrant")
        if backend not in ("sqlite", "qdrant"):
            raise HTTPException(503, "Unsupported development evidence backend.")
        if not (LLM_BASE_URL and LLM_API_KEY and LLM_API_KEY != "EMPTY") or (backend == "qdrant" and not QDRANT_URL):
            raise HTTPException(503, "Configure dedicated development LLM and evidence services before requesting assessments.")
        try:
            if backend == "sqlite":
                from dev_knowledge import search_evidence
                if not tenant:
                    raise HTTPException(403, "Authorized evidence scope is required.")
                kwargs["evidence_chunks"] = search_evidence(engine, tenant, kwargs["question"][:1000], top_k=TOP_K)
            return raphael_generate(**kwargs)
        except HTTPException:
            raise
        except Exception:
            # Never reflect provider errors, URLs or credentials to the portal.
            raise HTTPException(503, "Development assessment services are unavailable.")

    portal_context = install_dev_portal(app, engine, assessment_generator=development_assessment)
    from dev_chart import install_chart
    from dev_imaging import install_imaging
    from dev_messaging import install_messaging
    from dev_knowledge import install_knowledge
    from dev_intelligence import install_intelligence
    from dev_care import install_care
    install_chart(app, portal_context)
    install_imaging(app, portal_context)
    install_messaging(app, portal_context)
    install_knowledge(app, portal_context)
    install_intelligence(app, portal_context, assessment_generator=development_assessment)
    install_care(app, portal_context)
    from dev_patient_ai import install_patient_ai,export_patient_ai
    from dev_account import install_account
    install_patient_ai(app,portal_context)
    install_account(app,portal_context,patient_ai_exporter=export_patient_ai)

    @app.get("/api/dev/status")
    def portal_status(user=Depends(portal_context.user)):
        from dev_runtime import schema_status
        return {"scope": "synthetic-development-only", "database": schema_status(engine),
                "evidence_backend": os.getenv("RAPHAEL_EVIDENCE_BACKEND", "qdrant"),
                "llm": "configured_not_validated" if LLM_BASE_URL and LLM_API_KEY and LLM_API_KEY != "EMPTY" else "not_configured",
                "external_messaging": "disabled", "live_billing": "disabled",
                "clinical_validation": "not_established"}
