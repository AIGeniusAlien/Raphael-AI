import os
import httpx

DICOMWEB_BASE_URL = os.getenv("DICOMWEB_BASE_URL", "").rstrip("/")
DICOMWEB_TOKEN = os.getenv("DICOMWEB_TOKEN", "")

async def qido_search_studies(params: dict) -> list:
    if not DICOMWEB_BASE_URL:
        raise ValueError("DICOMWEB_BASE_URL must be set.")
    headers = {"Authorization": f"Bearer {DICOMWEB_TOKEN}"} if DICOMWEB_TOKEN else {}
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(f"{DICOMWEB_BASE_URL}/studies", params=params, headers=headers)
        r.raise_for_status()
        return r.json()

async def wado_retrieve_instance(study_uid: str, series_uid: str, instance_uid: str) -> bytes:
    if not DICOMWEB_BASE_URL:
        raise ValueError("DICOMWEB_BASE_URL must be set.")
    headers = {"Authorization": f"Bearer {DICOMWEB_TOKEN}"} if DICOMWEB_TOKEN else {}
    url = f"{DICOMWEB_BASE_URL}/studies/{study_uid}/series/{series_uid}/instances/{instance_uid}"
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.get(url, headers=headers)
        r.raise_for_status()
        return r.content
