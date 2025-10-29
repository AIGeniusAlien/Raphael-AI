
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, List
import os, pathlib

app = FastAPI(title="Raphael — Doctor's Copilot (Prototype)", version="0.6.0")

# CORS permissive for demo; lock down later
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static UIs directly from the same app (zero config)
static_dir = pathlib.Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
def home():
    return RedirectResponse("/static/standard.html")

@app.get("/handsfree")
def handsfree():
    return RedirectResponse("/static/handsfree.html")

@app.get("/fhir-demo")
def fhir_demo():
    return RedirectResponse("/static/fhir-demo.html")

@app.get("/healthz")
def healthz():
    return {"ok": True, "mode": "prototype", "version": "0.6.0"}

# Discovery so frontends auto-learn API base
@app.get("/.well-known/raphael.json")
def well_known(request: Request):
    base = request.url.scheme + "://" + request.headers.get("host","")
    return {"ok": True, "api_base": base}

# --------- INTENTS (placeholder NLU) ---------
class IntentRequest(BaseModel):
    text: str = Field(..., description="Transcript or typed input")

class IntentResponse(BaseModel):
    intent: str
    confidence: float
    entities: Dict[str, Any] = {}
    readback: str

INTENT_KEYS = {
    "start_recording": ["start recording", "scribe on", "begin note"],
    "stop_recording": ["stop recording", "scribe off", "end note"],
    "draft_orders": ["draft orders", "order", "place order"],
    "confirm": ["read back", "confirm"],
}

def detect_intent(t: str) -> str:
    t = (t or "").lower().strip()
    for key, phrases in INTENT_KEYS.items():
        if any(p in t for p in phrases):
            return key
    return "free_text"

@app.post("/v1/intents", response_model=IntentResponse)
def parse_intent(payload: IntentRequest):
    intent = detect_intent(payload.text)
    readback = f"Intent {intent.replace('_',' ')} captured. Say 'Confirm' to proceed."
    return IntentResponse(intent=intent, confidence=0.77, readback=readback)

# --------- VOICE WS (echo scaffold) ---------
@app.websocket("/v1/voice/ws")
async def voice_ws(ws: WebSocket):
    await ws.accept()
    try:
        await ws.send_json({"status":"ready","message":"Send text chunks as JSON; audio pipeline omitted in prototype."})
        async for msg in ws.iter_text():
            await ws.send_text(f"echo:{msg}")
    except WebSocketDisconnect:
        pass

# --------- SMART-on-FHIR scaffold ---------
from fastapi import APIRouter
fhir_router = APIRouter(prefix="/fhir", tags=["fhir"])

@fhir_router.get("/authorize")
def fhir_authorize():
    if not os.getenv("FHIR_AUTH_URL"):
        return JSONResponse({"ok": False, "note": "Set FHIR_* envs to enable real OAuth"}, status_code=200)
    from urllib.parse import urlencode
    q = urlencode({
        "client_id": os.getenv("FHIR_CLIENT_ID"),
        "response_type": "code",
        "redirect_uri": os.getenv("FHIR_REDIRECT_URI"),
        "scope": os.getenv("FHIR_SCOPES","openid fhirUser patient/*.read patient/*.write"),
        "aud": os.getenv("FHIR_BASE_URL","")
    })
    return RedirectResponse(f"{os.getenv('FHIR_AUTH_URL')}?{q}")

@fhir_router.get("/callback")
def fhir_callback(code: str = None):
    if not code:
        return {"ok": False, "error": "no code"}
    return {"ok": True, "token_received": True}

@fhir_router.post("/draft/note")
def draft_note(patient_id: str, content: str):
    return {"ok": True, "action": "DocumentReference.create", "status": "draft", "patient_id": patient_id, "length": len(content)}

@fhir_router.post("/draft/orders")
def draft_orders(patient_id: str, orders: List[str]):
    return {"ok": True, "action": "ServiceRequest.batchCreate", "status": "draft", "patient_id": patient_id, "count": len(orders)}

app.include_router(fhir_router)
