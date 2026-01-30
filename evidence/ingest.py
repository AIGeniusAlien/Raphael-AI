from qdrant_client import QdrantClient
from rag.chunking import simple_chunk
from rag.embedder import embed_texts
import hashlib

def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def ingest_text(*, qdrant_url: str, collection: str, embed_model: str, embed_dim: int,
                title: str, tier: str, source_date: str | None, url: str | None, section: str | None,
                text: str) -> int:

    client = QdrantClient(url=qdrant_url)

    # create collection if missing
    existing = [c.name for c in client.get_collections().collections]
    if collection not in existing:
        from qdrant_client.http import models as qm
        client.create_collection(
            collection_name=collection,
            vectors_config=qm.VectorParams(size=embed_dim, distance=qm.Distance.COSINE)
        )

    chunks = simple_chunk(text)
    if not chunks:
        return 0

    vecs = embed_texts(chunks, embed_model)

    points = []
    for i, (txt, v) in enumerate(zip(chunks, vecs)):
        chunk_id = _sha(f"{title}|{section}|{i}|{txt[:80]}")
        payload = {
            "chunk_id": chunk_id,
            "title": title,
            "tier": tier,
            "source_date": source_date,
            "url": url,
            "section": section,
            "text": txt
        }
        points.append({"id": chunk_id, "vector": v.tolist(), "payload": payload})

    client.upsert(collection_name=collection, points=points)
    return len(points)
