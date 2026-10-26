"""Reviewed local evidence intake and real lexical retrieval for synthetic development.

No source is fetched, seeded, or declared medically/legally verified. A clinician
attests permission and reviews the supplied excerpt before it can be retrieved.
This SQLite FTS5 backend is lexical search, not an embedding or hybrid model.
"""

import hashlib
import ipaddress
import json
import re
from datetime import date, datetime, timezone
from typing import Literal
from urllib.parse import parse_qsl, urlsplit, urlunsplit

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator
from sqlalchemy import text

from database import execute_sqlite_script
from dev_portal import _require_synthetic

SCHEMA = """
CREATE TABLE IF NOT EXISTS dev_evidence_sources (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 tenant TEXT NOT NULL,
 url TEXT NOT NULL,
 version INTEGER NOT NULL,
 title TEXT NOT NULL,
 body TEXT NOT NULL,
 content_sha256 TEXT NOT NULL,
 version_sha256 TEXT NOT NULL,
 license_declaration TEXT NOT NULL,
 permission_attested INTEGER NOT NULL CHECK(permission_attested=1),
 tier TEXT NOT NULL,
 source_type TEXT NOT NULL,
 org TEXT NOT NULL,
 jurisdiction TEXT NOT NULL,
 source_date TEXT,
 evidence_strength TEXT,
 status TEXT NOT NULL CHECK(status IN ('pending','approved','revoked','superseded')),
 created_by INTEGER NOT NULL REFERENCES dev_users(id),
 created_at TEXT NOT NULL,
 reviewed_by INTEGER REFERENCES dev_users(id),
 reviewed_at TEXT,
 UNIQUE(tenant,url,version),
 UNIQUE(tenant,url,version_sha256)
);
CREATE TABLE IF NOT EXISTS dev_evidence_chunks (
 chunk_id TEXT PRIMARY KEY,
 source_id INTEGER NOT NULL REFERENCES dev_evidence_sources(id),
 ordinal INTEGER NOT NULL,
 body TEXT NOT NULL,
 UNIQUE(source_id,ordinal)
);
CREATE VIRTUAL TABLE IF NOT EXISTS dev_evidence_fts USING fts5(
 chunk_id UNINDEXED, title, body
);
CREATE INDEX IF NOT EXISTS dev_evidence_sources_tenant ON dev_evidence_sources(tenant,status);
CREATE INDEX IF NOT EXISTS dev_evidence_chunks_source ON dev_evidence_chunks(source_id);
"""


def init_knowledge_schema(engine):
    _require_synthetic(engine)
    with engine.begin() as c:
        execute_sqlite_script(c, SCHEMA)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _url(value):
    if any(ord(character) <= 32 for character in value):
        raise ValueError("Source URL cannot contain whitespace or controls.")
    parsed = urlsplit(value)
    try:
        port = parsed.port
    except ValueError:
        raise ValueError("Invalid source URL port.")
    host = (parsed.hostname or "").lower().rstrip(".")
    if parsed.scheme != "https" or parsed.username is not None or parsed.password is not None or port not in (None, 443):
        raise ValueError("Use a public HTTPS source URL without credentials or custom ports.")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        raise ValueError("Use a public source hostname, not an IP address.")
    labels = host.split(".")
    if len(labels) < 2 or any(not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label) for label in labels):
        raise ValueError("Use a valid public source hostname.")
    if labels[-1] in {"local", "localhost", "internal", "invalid", "test"}:
        raise ValueError("Private/local source hostnames are not accepted.")
    if any(key.lower() in {"key", "api_key", "token", "access_token", "auth", "signature", "password"}
           for key, _ in parse_qsl(parsed.query)):
        raise ValueError("Do not include credentials or signed access tokens in source URLs.")
    return urlunsplit(("https", host, parsed.path or "/", parsed.query, parsed.fragment))


class SourceIn(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    title: str = Field(min_length=1, max_length=300)
    url: str = Field(min_length=8, max_length=2048)
    content: str = Field(min_length=10, max_length=200000)
    license_declaration: str = Field(min_length=8, max_length=2000)
    permission_attested: StrictBool
    tier: Literal["A", "B", "C", "D"] = "D"
    source_type: str = Field(default="manually_supplied_excerpt", min_length=1, max_length=100)
    org: str = Field(min_length=1, max_length=200)
    jurisdiction: str = Field(default="Unspecified", min_length=1, max_length=100)
    source_date: date | None = None
    evidence_strength: Literal["high", "moderate", "low", "very_low"] | None = None

    @field_validator("url")
    @classmethod
    def public_url(cls, value):
        return _url(value)

    @field_validator("permission_attested")
    @classmethod
    def permission_required(cls, value):
        if value is not True:
            raise ValueError("Attest that this supplied excerpt may be stored and used.")
        return value

    @field_validator("source_date")
    @classmethod
    def not_future(cls, value):
        if value is not None and value > datetime.now(timezone.utc).date():
            raise ValueError("Publication date cannot be in the future.")
        return value


class SearchIn(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    query: str = Field(min_length=2, max_length=1000)
    top_k: int = Field(default=12, ge=1, le=30)


def _chunks(content, size=1200, overlap=150):
    normalized = re.sub(r"\s+", " ", content).strip()
    chunks = []
    for start in range(0, len(normalized), size - overlap):
        chunks.append(normalized[start:start + size])
        if start + size >= len(normalized):
            break
    return chunks


def _source(c, source_id, tenant):
    row = c.execute(text("SELECT s.*,u.username AS reviewed_by_username FROM dev_evidence_sources s "
                         "LEFT JOIN dev_users u ON u.id=s.reviewed_by WHERE s.id=:id AND s.tenant=:t"),
                    {"id": source_id, "t": tenant}).mappings().first()
    if not row:
        raise HTTPException(404, "Source not found.")
    return dict(row)


def _source_json(source, include_body=True):
    result = {key: value for key, value in source.items() if key != "body"}
    if include_body:
        result["content"] = source["body"]
    result["permission_attested"] = bool(result["permission_attested"])
    result["review_status"] = "clinician_reviewed" if source["status"] == "approved" else source["status"]
    result["verification_notice"] = "Permission and content are clinician attestations, not independent legal or clinical verification."
    return result


def _tokens(query):
    return list(dict.fromkeys(re.findall(r"\b\w{2,64}\b", query.lower(), flags=re.UNICODE)))[:20]


def search_evidence(engine, tenant, query, top_k=12):
    """Return approved-current lexical matches with provenance and resolvable IDs.

    score is the fraction of query tokens in the chunk/title, not semantic
    similarity, clinical confidence, citation truth, or probability.
    """
    if not isinstance(query, str) or len(query) > 1000 or not 1 <= top_k <= 30:
        raise ValueError("Bounded string query and top_k in 1..30 required.")
    tokens = _tokens(query)
    if not tokens:
        return []
    match = " OR ".join('"' + token + '"' for token in tokens)
    with engine.connect() as c:
        rows = c.execute(text("SELECT ch.chunk_id,ch.ordinal,ch.body AS chunk_body,s.*, "
                              "u.username AS reviewed_by_username,bm25(dev_evidence_fts) AS lexical_rank "
                              "FROM dev_evidence_fts JOIN dev_evidence_chunks ch ON ch.chunk_id=dev_evidence_fts.chunk_id "
                              "JOIN dev_evidence_sources s ON s.id=ch.source_id "
                              "LEFT JOIN dev_users u ON u.id=s.reviewed_by "
                              "WHERE dev_evidence_fts MATCH :q AND s.tenant=:t AND s.status='approved' "
                              "ORDER BY lexical_rank,s.id,ch.ordinal LIMIT :k"),
                         {"q": match, "t": tenant, "k": top_k}).mappings().all()
    results = []
    for row in rows:
        words = set(_tokens(row["title"] + " " + row["chunk_body"]))
        # Do not truncate document tokens when computing term coverage.
        words = set(re.findall(r"\b\w{2,64}\b", (row["title"] + " " + row["chunk_body"]).lower()))
        age = (datetime.now(timezone.utc).date() - date.fromisoformat(row["source_date"])).days if row["source_date"] else None
        results.append({
            "chunk_id": row["chunk_id"], "doc_id": f"local-source:{row['id']}", "title": row["title"],
            "tier": row["tier"], "source_type": row["source_type"], "org": row["org"],
            "jurisdiction": row["jurisdiction"], "source_date": row["source_date"], "url": row["url"],
            "section": f"Supplied excerpt chunk {row['ordinal'] + 1}", "evidence_strength": row["evidence_strength"],
            "text": row["chunk_body"], "score": sum(token in words for token in tokens) / len(tokens),
            "score_kind": "lexical_query_term_coverage", "lexical_rank": row["lexical_rank"],
            "retrieval_method": "sqlite_fts5_lexical", "source_id": row["id"], "source_version": row["version"],
            "content_sha256": row["content_sha256"], "version_sha256": row["version_sha256"],
            "reviewed_by": row["reviewed_by_username"], "reviewed_at": row["reviewed_at"],
            "source_age_days": age, "freshness_status": "date_unknown" if age is None else "older_than_two_years" if age > 730 else "dated_within_two_years",
            "review_status": "clinician_reviewed_not_independently_verified",
            "citation_path": f"/api/dev/knowledge/citations/{row['chunk_id']}",
        })
    return results


def install_knowledge(app, ctx):
    router = APIRouter(prefix="/api/dev/knowledge", tags=["reviewed development evidence"],
                       dependencies=[Depends(ctx.same_origin_json)])
    engine = ctx.engine

    @router.get("/sources")
    def sources(user=Depends(ctx.clinician)):
        with engine.begin() as c:
            rows = c.execute(text("SELECT s.*,u.username AS reviewed_by_username FROM dev_evidence_sources s "
                                  "LEFT JOIN dev_users u ON u.id=s.reviewed_by WHERE s.tenant=:t ORDER BY s.id DESC"),
                             {"t": user["tenant"]}).mappings().all()
            ctx.audit(c, user, "list_evidence_sources", "evidence_source")
            return [_source_json(dict(row), include_body=False) for row in rows]

    @router.post("/sources")
    def intake(body: SourceIn, user=Depends(ctx.clinician)):
        values = body.model_dump(mode="json")
        digest = hashlib.sha256(body.content.encode()).hexdigest()
        version_digest = hashlib.sha256(json.dumps(values, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        with engine.begin() as c:
            c.exec_driver_sql("BEGIN IMMEDIATE")
            existing = c.execute(text("SELECT id FROM dev_evidence_sources WHERE tenant=:t AND url=:u AND version_sha256=:h"),
                                 {"t": user["tenant"], "u": body.url, "h": version_digest}).scalar()
            if existing is not None:
                result = _source_json(_source(c, existing, user["tenant"]))
                result["deduplicated"] = True
                return result
            version = c.execute(text("SELECT COALESCE(MAX(version),0)+1 FROM dev_evidence_sources WHERE tenant=:t AND url=:u"),
                                {"t": user["tenant"], "u": body.url}).scalar()
            row = c.execute(text("INSERT INTO dev_evidence_sources(tenant,url,version,title,body,content_sha256,version_sha256,"
                                 "license_declaration,permission_attested,tier,source_type,org,jurisdiction,source_date,"
                                 "evidence_strength,status,created_by,created_at) VALUES "
                                 "(:t,:url,:v,:title,:content,:h,:vh,:license_declaration,1,:tier,:source_type,:org,:jurisdiction,"
                                 ":source_date,:evidence_strength,'pending',:uid,:at)"),
                            dict(values, t=user["tenant"], v=version, h=digest, vh=version_digest, uid=user["id"], at=_now()))
            source_id = row.lastrowid
            for ordinal, chunk in enumerate(_chunks(body.content)):
                chunk_id = hashlib.sha256(f"{user['tenant']}|{body.url}|{version_digest}|{ordinal}|{chunk}".encode()).hexdigest()
                c.execute(text("INSERT INTO dev_evidence_chunks(chunk_id,source_id,ordinal,body) VALUES (:id,:s,:o,:b)"),
                          {"id": chunk_id, "s": source_id, "o": ordinal, "b": chunk})
            ctx.audit(c, user, "intake_evidence_source", "evidence_source", source_id)
            result = _source_json(_source(c, source_id, user["tenant"]))
            result["deduplicated"] = False
            return result

    @router.get("/sources/{source_id}")
    def source(source_id: int, user=Depends(ctx.clinician)):
        with engine.begin() as c:
            result = _source_json(_source(c, source_id, user["tenant"]))
            ctx.audit(c, user, "read_evidence_source", "evidence_source", source_id)
            return result

    @router.post("/sources/{source_id}/approve")
    def approve(source_id: int, user=Depends(ctx.clinician)):
        with engine.begin() as c:
            c.exec_driver_sql("BEGIN IMMEDIATE")
            current = _source(c, source_id, user["tenant"])
            c.execute(text("DELETE FROM dev_evidence_fts WHERE chunk_id IN (SELECT ch.chunk_id FROM dev_evidence_chunks ch "
                           "JOIN dev_evidence_sources s ON s.id=ch.source_id WHERE s.tenant=:t AND s.url=:u)"),
                      {"t": user["tenant"], "u": current["url"]})
            c.execute(text("UPDATE dev_evidence_sources SET status='superseded' WHERE tenant=:t AND url=:u AND status='approved'"),
                      {"t": user["tenant"], "u": current["url"]})
            c.execute(text("UPDATE dev_evidence_sources SET status='approved',reviewed_by=:uid,reviewed_at=:at WHERE id=:id"),
                      {"id": source_id, "uid": user["id"], "at": _now()})
            c.execute(text("INSERT INTO dev_evidence_fts(chunk_id,title,body) SELECT chunk_id,:title,body "
                           "FROM dev_evidence_chunks WHERE source_id=:id"), {"title": current["title"], "id": source_id})
            ctx.audit(c, user, "approve_evidence_source", "evidence_source", source_id)
            return _source_json(_source(c, source_id, user["tenant"]))

    @router.post("/sources/{source_id}/revoke")
    def revoke(source_id: int, user=Depends(ctx.clinician)):
        with engine.begin() as c:
            c.exec_driver_sql("BEGIN IMMEDIATE")
            _source(c, source_id, user["tenant"])
            c.execute(text("DELETE FROM dev_evidence_fts WHERE chunk_id IN (SELECT chunk_id FROM dev_evidence_chunks WHERE source_id=:id)"),
                      {"id": source_id})
            c.execute(text("UPDATE dev_evidence_sources SET status='revoked' WHERE id=:id"), {"id": source_id})
            ctx.audit(c, user, "revoke_evidence_source", "evidence_source", source_id)
            return _source_json(_source(c, source_id, user["tenant"]))

    @router.post("/search")
    def search(body: SearchIn, user=Depends(ctx.clinician)):
        results = search_evidence(engine, user["tenant"], body.query, body.top_k)
        with engine.begin() as c:
            ctx.audit(c, user, "search_evidence", "evidence_source")
        return {"results": results, "retrieval_method": "sqlite_fts5_lexical",
                "notice": "Only clinician-reviewed supplied excerpts are searched; source truth and permission are not independently verified."}

    @router.get("/citations/{chunk_id}")
    def citation(chunk_id: str, user=Depends(ctx.clinician)):
        with engine.begin() as c:
            row = c.execute(text("SELECT ch.*,s.url,s.title,s.version,s.content_sha256,s.version_sha256,s.reviewed_at,"
                                 "s.license_declaration,u.username AS reviewed_by FROM dev_evidence_chunks ch "
                                 "JOIN dev_evidence_sources s ON s.id=ch.source_id JOIN dev_users u ON u.id=s.reviewed_by "
                                 "WHERE ch.chunk_id=:id AND s.tenant=:t AND s.status='approved'"),
                            {"id": chunk_id, "t": user["tenant"]}).mappings().first()
            if not row:
                raise HTTPException(404, "Approved citation not found.")
            ctx.audit(c, user, "resolve_evidence_citation", "evidence_source", row["source_id"])
            return dict(row, review_status="clinician_reviewed_not_independently_verified")

    @router.get("/status")
    def status(user=Depends(ctx.clinician)):
        with engine.connect() as c:
            counts = dict(c.execute(text("SELECT status,count(*) FROM dev_evidence_sources WHERE tenant=:t GROUP BY status"),
                                    {"t": user["tenant"]}).all())
            chunks = c.execute(text("SELECT count(*) FROM dev_evidence_chunks ch JOIN dev_evidence_sources s ON s.id=ch.source_id "
                                    "WHERE s.tenant=:t AND s.status='approved'"), {"t": user["tenant"]}).scalar()
        return {"retrieval_method": "sqlite_fts5_lexical", "source_versions_by_status": counts,
                "approved_chunks": chunks, "coverage": "Only manually supplied, approved excerpts; no comprehensive coverage claim.",
                "source_verification": "clinician_attestation_only", "external_connectors": "not invoked"}

    app.include_router(router)
