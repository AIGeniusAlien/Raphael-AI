from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from sentence_transformers import SentenceTransformer
import numpy as np
import hashlib

from pubmed.client import esearch_pubmed, esummary_pubmed
from pubmed.classify import infer_study_type, evidence_strength_for

def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

_MODEL = None

def get_model(model_name: str) -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(model_name)
    return _MODEL

def embed_texts(texts: list[str], model_name: str) -> np.ndarray:
    model = get_model(model_name)
    vecs = model.encode(texts, normalize_embeddings=True)
    return np.array(vecs, dtype=np.float32)

def ensure_collection(client: QdrantClient, collection: str, dim: int):
    existing = [c.name for c in client.get_collections().collections]
    if collection in existing:
        return
    client.create_collection(
        collection_name=collection,
        vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
    )

async def ingest_pubmed_query(*,
    qdrant_url: str,
    qdrant_api_key: str,
    collection: str,
    embed_model: str,
    embed_dim: int,
    query: str,
    retmax: int = 20,
    org: str = "PubMed",
    jurisdiction: str = "Global",
) -> dict:
    """
    Ingest PubMed summaries as Tier C.
    Stores title + short metadata. (Abstract pulling is optional; this is a safe first step.)
    """
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key or None)
    ensure_collection(client, collection, embed_dim)

    ids = await esearch_pubmed(query, retmax=retmax)
    summ = await esummary_pubmed(ids)
    result = summ.get("result", {}) if isinstance(summ, dict) else {}

    indexed = 0
    points = []

    for pmid in ids:
        rec = result.get(pmid) or {}
        title = rec.get("title") or ""
        pubdate = rec.get("pubdate") or ""
        pubtypes = rec.get("pubtype") or []

        study_type = infer_study_type(title, pubtypes)
        strength = evidence_strength_for(study_type)

        # Payload text: title + minimal meta
        text = f"PMID: {pmid}. Title: {title}. Publication date: {pubdate}. Publication types: {', '.join(pubtypes)}. Study type: {study_type}."
        chunk_id = sha256(f"pubmed|{pmid}")
        payload = {
            "chunk_id": chunk_id,
            "doc_id": f"pubmed:{pmid}",
            "title": title[:300],
            "tier": "C",
            "source_type": study_type,
            "org": org,
            "jurisdiction": jurisdiction,
            "source_date": pubdate,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "section": "summary",
            "evidence_strength": strength,
            "text": text,
        }
        points.append((chunk_id, text, payload))

    if points:
        vecs = embed_texts([p[1] for p in points], embed_model)
        client.upsert(
            collection_name=collection,
            points=[
                {"id": pid, "vector": v.tolist(), "payload": payload}
                for (pid, _t, payload), v in zip(points, vecs)
            ],
        )
        indexed = len(points)

    return {"query": query, "pmids": len(ids), "chunks_indexed": indexed}
