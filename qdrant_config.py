"""Keep HTTPS URL semantics with the pinned Qdrant client's port fallback."""
from urllib.parse import urlsplit


def connection_options(url: str) -> dict:
    # qdrant-client1.9 treats an omitted/default HTTPS port as6333. Explicit
    # port=443 is needed to use the URL's standard HTTPS route through egress.
    parsed = urlsplit(url)
    return {'port': parsed.port or 443} if parsed.scheme == 'https' else {}


def verify_legacy_encoder(client,collection,expected_model):
    """Preserve legacy untagged data,refuse a known different embedding space."""
    if client.count(collection,exact=True).count>10000:
        raise ValueError('Encoder inventory exceeds bounded development review limit.')
    offset=None
    while True:
        points,offset=client.scroll(collection,offset=offset,limit=100,with_payload=True,with_vectors=False)
        if any((point.payload or {}).get('embedding_model') not in (None,expected_model) for point in points):
            raise ValueError('Existing collection uses another embedding model;select the matching backend or a separate collection.')
        if offset is None:return
