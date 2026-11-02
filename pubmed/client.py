import os
import json
import httpx

NCBI_API_KEY = os.getenv("NCBI_API_KEY", "").strip()
NCBI_EMAIL = os.getenv("NCBI_EMAIL", "").strip()
NCBI_TOOL = os.getenv("NCBI_TOOL", "RaphaelAI").strip()

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


async def _get_json(path,params):
    # Fixed authorized origin,no redirects,retries or unlimited response buffering.
    async with httpx.AsyncClient(timeout=30,follow_redirects=False) as client:
        async with client.stream('GET',EUTILS+'/'+path,params=params) as response:
            response.raise_for_status()
            body=bytearray()
            async for part in response.aiter_bytes():
                body.extend(part)
                if len(body)>2_000_000:raise ValueError('NCBI response exceeds bounded metadata limit.')
    result=json.loads(body)
    if not isinstance(result,dict):raise ValueError('NCBI metadata object required.')
    return result

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
    if not isinstance(term,str) or not 1<=len(term)<=2000 or type(retmax) is not int or not 1<=retmax<=20:
        raise ValueError('Bounded PubMed query and1–20records required.')
    params = _params({
        "db": "pubmed",
        "term": term,
        "retmode": "json",
        "retmax": str(retmax),
        "sort": "relevance",
    })
    data=await _get_json('esearch.fcgi',params)
    identifiers=data.get('esearchresult',{}).get('idlist',[])
    if not isinstance(identifiers,list) or len(identifiers)>retmax or any(not isinstance(item,str) or not item.isascii() or not item.isdigit() or not item or len(item)>12 for item in identifiers) or len(set(identifiers))!=len(identifiers):
        raise ValueError('Invalid bounded PMID response.')
    return identifiers

async def esummary_pubmed(ids: list[str]) -> dict:
    if not ids:
        return {}
    if not isinstance(ids,list) or len(ids)>20 or any(not isinstance(item,str) or not item.isascii() or not item.isdigit() or not item or len(item)>12 for item in ids):
        raise ValueError('Bounded numeric PubMed identifiers required.')
    params = _params({
        "db": "pubmed",
        "id": ",".join(ids),
        "retmode": "json",
    })
    return await _get_json('esummary.fcgi',params)
