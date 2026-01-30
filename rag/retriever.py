from typing import List
from rag.embedder import embed_texts
from rag.qdrant_store import get_client, ensure_collection
from rag.schemas import EvidenceChunk

def retrieve(query: str, *, qdrant_url: str, collection: str, embed_model: str, embed_dim: int, top_k: int) -> List[EvidenceChunk]:
    client = get_client(qdrant_url)
    ensure_collection(client, collection, embed_dim)

    qv = embed_texts([query], embed_model)[0].tolist()
    hits = client.search(
        collection_name=collection,
        query_vector=qv,
        limit=top_k,
        with_payload=True
    )

    out: List[EvidenceChunk] = []
    for h in hits:
        p = h.payload or {}
        out.append(EvidenceChunk(
            chunk_id=p.get("chunk_id",""),
            title=p.get("title",""),
            tier=p.get("tier","TIER_3"),
            source_date=p.get("source_date"),
            url=p.get("url"),
            section=p.get("section"),
            text=p.get("text",""),
            score=float(h.score)
        ))
    return out
