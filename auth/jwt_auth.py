import os
from datetime import datetime, timedelta, timezone
from fastapi import Header, HTTPException
import jwt

JWT_SECRET = os.getenv("JWT_SECRET", "")
JWT_ALG = "HS256"
JWT_TTL_MIN = int(os.getenv("JWT_TTL_MIN", "720"))

def create_token(sub: str, tenant_id: str, role: str) -> str:
    if not JWT_SECRET or JWT_SECRET == "CHANGE_ME_NOW":
        raise HTTPException(503, "Legacy JWT signing is not configured.")
    exp = datetime.now(timezone.utc) + timedelta(minutes=JWT_TTL_MIN)
    payload = {"sub": sub, "tenant_id": tenant_id, "role": role, "exp": exp}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)

def require_user(authorization: str = Header(default="")):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token.")
    token = authorization.split(" ", 1)[1].strip()
    if not JWT_SECRET or JWT_SECRET == "CHANGE_ME_NOW":
        raise HTTPException(503, "Legacy JWT signing is not configured.")
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG], options={"require": ["exp", "sub", "tenant_id", "role"]})
        return payload
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token.")
