import os
import httpx

NCBI_API_KEY = os.getenv("NCBI_API_KEY", "").strip()
NCBI_EMAIL = os.getenv("NCBI_EMAIL", "").strip()
NCBI_TOOL = os.getenv("NCBI_TOOL", "RaphaelAI").strip()

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

def _params(extra: dict) -> dict:
    p = dict(extra)
    if NCBI_API_KEY:
        p["api_key"] = NCBI_API_KEY
    if NCBI_TOOL:
        p["tool"] = NCBI_TOOL
    if NCBI_EMAIL:
        p["email"] = NCBI_EMAIL
    return p

async def esearch_pubmed(term: str, retmax: int = 20) -> list[str]:
    params = _params({
        "db": "pubmed",
        "term": term,
        "retmode": "json",
        "retmax": str(retmax),
        "sort": "relevance",
    })
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(f"{EUTILS}/esearch.fcgi", params=params)
        r.raise_for_status()
        data = r.json()
        return data.get("esearchresult", {}).get("idlist", [])

async def esummary_pubmed(ids: list[str]) -> dict:
    if not ids:
        return {}
    params = _params({
        "db": "pubmed",
        "id": ",".join(ids),
        "retmode": "json",
    })
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(f"{EUTILS}/esummary.fcgi", params=params)
        r.raise_for_status()
        return r.json()
