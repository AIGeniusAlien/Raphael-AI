"""Keep HTTPS URL semantics with the pinned Qdrant client's port fallback."""
from urllib.parse import urlsplit


def connection_options(url: str) -> dict:
    # qdrant-client1.9 treats an omitted/default HTTPS port as6333. Explicit
    # port=443 is needed to use the URL's standard HTTPS route through egress.
    parsed = urlsplit(url)
    return {'port': parsed.port or 443} if parsed.scheme == 'https' else {}
