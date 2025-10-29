
# Raphael API Reference (Prototype)
- `GET /healthz`
- `POST /v1/intents` → { intent, confidence, readback }
- `WS /v1/voice/ws` → echo scaffold
- FHIR:
  - `GET /fhir/authorize` (redirect if env set; else returns note)
  - `GET /fhir/callback?code=...`
  - `POST /fhir/draft/note?patient_id=&content=`
  - `POST /fhir/draft/orders` (JSON list body)
