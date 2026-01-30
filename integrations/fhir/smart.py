import os
import httpx
from urllib.parse import urlencode

FHIR_BASE_URL = os.getenv("FHIR_BASE_URL", "").rstrip("/")
FHIR_CLIENT_ID = os.getenv("FHIR_CLIENT_ID", "")
FHIR_CLIENT_SECRET = os.getenv("FHIR_CLIENT_SECRET", "")
FHIR_REDIRECT_URI = os.getenv("FHIR_REDIRECT_URI", "")
FHIR_SCOPES = os.getenv("FHIR_SCOPES", "launch/patient patient/*.read openid profile offline_access")

FHIR_AUTH_URL = os.getenv("FHIR_AUTH_URL", "")
FHIR_TOKEN_URL = os.getenv("FHIR_TOKEN_URL", "")

def build_auth_url(state: str) -> str:
    if not (FHIR_AUTH_URL and FHIR_CLIENT_ID and FHIR_REDIRECT_URI and FHIR_BASE_URL):
        raise ValueError("FHIR_AUTH_URL / FHIR_CLIENT_ID / FHIR_REDIRECT_URI / FHIR_BASE_URL must be set.")
    q = {
        "response_type": "code",
        "client_id": FHIR_CLIENT_ID,
        "redirect_uri": FHIR_REDIRECT_URI,
        "scope": FHIR_SCOPES,
        "state": state,
        "aud": FHIR_BASE_URL,
    }
    return FHIR_AUTH_URL + "?" + urlencode(q)

async def exchange_code(code: str) -> dict:
    if not (FHIR_TOKEN_URL and FHIR_CLIENT_ID and FHIR_CLIENT_SECRET and FHIR_REDIRECT_URI):
        raise ValueError("FHIR_TOKEN_URL / FHIR_CLIENT_ID / FHIR_CLIENT_SECRET / FHIR_REDIRECT_URI must be set.")
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": FHIR_REDIRECT_URI,
        "client_id": FHIR_CLIENT_ID,
        "client_secret": FHIR_CLIENT_SECRET,
    }
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(FHIR_TOKEN_URL, data=data)
        r.raise_for_status()
        return r.json()

async def fhir_get(path: str, token: str) -> dict:
    if not FHIR_BASE_URL:
        raise ValueError("FHIR_BASE_URL must be set.")
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(
            f"{FHIR_BASE_URL}/{path.lstrip('/')}",
            headers={"Authorization": f"Bearer {token}"},
        )
        r.raise_for_status()
        return r.json()
