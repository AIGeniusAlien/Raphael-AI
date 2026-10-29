from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_config import connection_options
from qdrant_client.http import models as qm
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer
import numpy as np
import hashlib
from functools import lru_cache

from pubmed.client import esearch_pubmed, esummary_pubmed
from pubmed.classify import infer_study_type, evidence_strength_for
from corpus.indexer import point_id_for

def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

@lru_cache(maxsize=2)
def get_model(model_name: str) -> SentenceTransformer:
    # Cache by requested model,not a single mutable encoder shared across names.
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)

def embed_texts(texts: list[str], model_name: str) -> np.ndarray:
    model = get_model(model_name)
    vecs = model.encode(texts, normalize_embeddings=True)
    return np.array(vecs, dtype=np.float32)

def ensure_collection(client: QdrantClient, collection: str, dim: int, model_name: str):
    existing = [c.name for c in client.get_collections().collections]
    if collection in existing:
        vectors=client.get_collection(collection).config.params.vectors
        if not hasattr(vectors,'size') or vectors.size!=dim or str(vectors.distance).lower()!='cosine':
            raise ValueError('PubMed target schema is incompatible;existing data preserved.')
        if client.count(collection,exact=True).count>10000:
            raise ValueError('PubMed target exceeds bounded model-space review.')
        offset=None
        while True:
            points,offset=client.scroll(collection,limit=100,offset=offset,with_payload=True,with_vectors=False)
            if any((point.payload or {}).get('embedding_model')!=model_name or (point.payload or {}).get('embedding_dimensions')!=dim for point in points):
                raise ValueError('PubMed target contains a different or untagged embedding space;use a separate collection.')
            if offset is None:break
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
    if not isinstance(query,str) or not 1<=len(query)<=2000 or type(retmax) is not int or not 1<=retmax<=20:
        raise ValueError('PubMed discovery requires a bounded query and1–20records.')
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key or None, **connection_options(qdrant_url))
    try:
        # Review an existing collection before network discovery or any mutation.
        exists=collection in {item.name for item in client.get_collections().collections}
        if exists:
            ensure_collection(client,collection,embed_dim,embed_model)

        ids = await esearch_pubmed(query, retmax=retmax)
        summ = await esummary_pubmed(ids)
        result = summ.get("result", {}) if isinstance(summ, dict) else {}

        indexed = 0
        excluded=[]
        points = []

        for pmid in ids:
            rec = result.get(pmid) or {}
            title = rec.get("title") or ""
            pubdate = rec.get("pubdate") or ""
            pubtypes = rec.get("pubtype") or []
            if any(''.join(kind.lower().split()) in {'retractedpublication','retractionofpublication','expressionofconcern'} for kind in pubtypes):
                if exists:
                    client.set_payload(collection,payload={'retracted':True,'is_current':False,'exclusion_reason':'retraction_or_concern_publication_type'},points=[point_id_for(sha256(f'pubmed|{pmid}'))],wait=True)
                excluded.append({'pmid':pmid,'reason':'retraction_or_concern_publication_type'})
                continue

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
                "embedding_model":embed_model,
                "embedding_dimensions":embed_dim,
                "knowledge_scope":"bibliographic_metadata_not_full_text_or_claim_support",
                "rights_status":"article_full_text_reuse_not_verified",
                "pmid":pmid,
                "doi":next((item.get('value') for item in rec.get('articleids',[]) if item.get('idtype')=='doi'),None),
                "pmcid":next((item.get('value') for item in rec.get('articleids',[]) if item.get('idtype')=='pmc'),None),
            }
            points.append((chunk_id, text, payload))

        if points:
            vecs = embed_texts([p[1] for p in points], embed_model)
            if vecs.shape!=(len(points),embed_dim) or not np.isfinite(vecs).all() or np.any(np.linalg.norm(vecs,axis=1)==0):
                raise ValueError('Actual PubMed embedding dimensions/values are incompatible;no collection write.')
            ensure_collection(client,collection,embed_dim,embed_model)
            client.upsert(
                collection_name=collection,
                points=[
                    qm.PointStruct(id=point_id_for(pid), vector=v.tolist(), payload=payload)
                    for (pid, _t, payload), v in zip(points, vecs)
                ],
            )
            indexed = len(points)

        return {"query": query, "pmids": len(ids), "chunks_indexed": indexed,"excluded_records":excluded,"knowledge_scope":"bibliographic_metadata_not_clinical_claim_support","full_text_reuse_verified":False}

    finally:
        client.close()
