import json
from datetime import datetime

def audit_event(event_type: str, payload: dict):
    """
    v1: log audit events to stdout (Render logs).
    Later: write to DB + immutable storage.
    """
    rec = {"ts": datetime.utcnow().isoformat(), "event": event_type, "payload": payload}
    print("AUDIT_EVENT " + json.dumps(rec))
