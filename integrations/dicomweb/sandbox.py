"""Read-only, explicitly synthetic DICOMweb QIDO-RS discovery.

This does not import images, interpret scans, connect to a hospital PACS,
validate DICOMweb conformance, or establish de-identification. Only a
trusted operator-configured sandbox is queried using the patient's already
authorized, explicitly synthetic local identifier.
"""
import json
import os
import re
from datetime import datetime, timezone
from urllib.parse import urlsplit

import httpx


MAX_RESPONSE = 1024 * 1024
MAX_RESULTS = 25
UID = re.compile(r"[0-9]+(?:\.[0-9]+)*\Z")
SYNTHETIC_ID = re.compile(r"SYNTHETIC-[A-Za-z0-9_.-]{1,54}\Z")
ACCEPT = "application/dicom+json"


class DicomwebUnavailable(Exception):
    pass


def sandbox_url():
    raw = os.getenv("DICOMWEB_SANDBOX_BASE_URL", "").rstrip("/")
    if not raw or os.getenv("DICOMWEB_SANDBOX_SYNTHETIC_ONLY") != "1":
        raise DicomwebUnavailable("An explicitly synthetic DICOMweb sandbox is not configured.")
    try:
        parsed = urlsplit(raw)
        local = parsed.hostname in {"127.0.0.1", "localhost", "::1"}
        allowed_http = local and os.getenv("DICOMWEB_SANDBOX_ALLOW_LOOPBACK") == "1"
        if (parsed.scheme != "https" and not (parsed.scheme == "http" and allowed_http)):
            raise ValueError("TLS required")
        if not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Invalid destination URL")
        if _looks_like_unsafe_host(parsed.hostname):
            raise ValueError("Unsupported destination")
        _ = parsed.port
    except ValueError:
        raise DicomwebUnavailable("DICOMweb sandbox requires a valid HTTPS URL without credentials or query parameters.")
    return raw


def _looks_like_unsafe_host(host):
    # Loopback is permitted only in the explicit synthetic dev mode above.
    return host.endswith(".onion") or host == "0.0.0.0"


def _one(dataset, tag, vr, maxlen=128):
    value = dataset.get(tag) if isinstance(dataset, dict) else None
    if (not isinstance(value, dict) or value.get("vr") != vr
            or not isinstance(value.get("Value"), list)
            or len(value["Value"]) != 1):
        raise DicomwebUnavailable("DICOMweb sandbox study metadata is missing required bounded fields.")
    token = value["Value"][0]
    if not isinstance(token, str) or not 1 <= len(token) <= maxlen:
        raise DicomwebUnavailable("DICOMweb sandbox returned an invalid metadata value.")
    return token


async def preview_studies(patient_mrn: str, *, transport=None):
    if not isinstance(patient_mrn, str) or not SYNTHETIC_ID.fullmatch(patient_mrn):
        raise ValueError("An explicitly synthetic stored patient ID is required.")
    base = sandbox_url()
    token = os.getenv("DICOMWEB_SANDBOX_TOKEN", "")
    headers = {"Accept": ACCEPT}
    if token:
        headers["Authorization"] = "Bearer " + token
    local = urlsplit(base).hostname in {"127.0.0.1", "localhost", "::1"}
    try:
        async with httpx.AsyncClient(
            timeout=15, follow_redirects=False, transport=transport, trust_env=not local
        ) as client:
            body = bytearray()
            async with client.stream(
                "GET", base + "/studies",
                params={"PatientID": patient_mrn, "limit": MAX_RESULTS, "includefield": "00120062"},
                headers=headers,
            ) as response:
                if response.status_code != 200:
                    raise DicomwebUnavailable("DICOMweb sandbox did not return a successful study search.")
                if not response.headers.get("content-type", "").split(";")[0].strip().lower() == ACCEPT:
                    raise DicomwebUnavailable("DICOMweb sandbox did not return DICOM JSON.")
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > MAX_RESPONSE:
                        raise DicomwebUnavailable("DICOMweb sandbox response exceeds the bounded preview limit.")
            try:
                records = json.loads(body)
            except (ValueError, TypeError):
                raise DicomwebUnavailable("DICOMweb sandbox returned invalid JSON.") from None
            if not isinstance(records, list) or len(records) > MAX_RESULTS:
                raise DicomwebUnavailable("DICOMweb sandbox response exceeds the study limit.")
            studies = []
            seen = set()
            for row in records:
                if not isinstance(row, dict) or len(row) > 64:
                    raise DicomwebUnavailable("DICOMweb sandbox study metadata is malformed.")
                patient = _one(row, "00100020", "LO", 64)
                removed = _one(row, "00120062", "CS", 8)
                study_uid = _one(row, "0020000D", "UI", 64)
                if patient != patient_mrn or removed != "YES":
                    raise DicomwebUnavailable("DICOMweb study is not exclusively declared synthetic for the authorized patient.")
                if not UID.fullmatch(study_uid) or study_uid in seen:
                    raise DicomwebUnavailable("DICOMweb study UID is invalid or duplicated.")
                seen.add(study_uid)
                modality = _one(row, "00080060", "CS", 16) if "00080060" in row else "unknown"
                studies.append({"study_instance_uid": study_uid, "modality": modality})
            return {
                "status": "synthetic_read_only_dicomweb_preview",
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "studies": studies,
                "source_base_url": base,
                "imported": False,
                "diagnostic_interpretation_performed": False,
                "clinical_use_authorized": False,
                "limitations": [
                    "Only explicitly synthetic study metadata is returned; no DICOM pixels or reports are imported.",
                    "Synthetic declarations are not proof of de-identification or clinical validation.",
                    "Read-only QIDO-RS discovery in one operator-configured sandbox; no hospital PACS connection.",
                ],
            }
    except httpx.HTTPError:
        raise DicomwebUnavailable("DICOMweb sandbox request failed; no studies were imported.") from None
