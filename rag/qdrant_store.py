from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

def get_client(qdrant_url: str) -> QdrantClient:
    return QdrantClient(url=qdrant_url)

def ensure_collection(client: QdrantClient, collection: str, embed_dim: int):
    existing = [c.name for c in client.get_collections().collections]
    if collection in existing:
        return
    client.create_collection(
        collection_name=collection,
        vectors_config=qm.VectorParams(size=embed_dim, distance=qm.Distance.COSINE)
    )
