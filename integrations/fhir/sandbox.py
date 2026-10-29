"""Bounded read-only FHIR sandbox preview; never a production EHR connection."""
import json
import os
import re
from datetime import datetime, timezone
from urllib.parse import urlsplit

import httpx


class SandboxUnavailable(Exception):
    """A configured synthetic sandbox is missing, invalid or unavailable."""


RESOURCE_TYPES = (
    "Observation", "Condition", "MedicationRequest", "AllergyIntolerance",
    "DiagnosticReport", "ImagingStudy", "Encounter", "DocumentReference",
)
MAX_RESPONSE = 1024 * 1024
MAX_TOTAL = 5 * 1024 * 1024


def sandbox_base_url():
    base = os.getenv("FHIR_SANDBOX_BASE_URL", "").rstrip("/")
    if not base or os.getenv("FHIR_SANDBOX_SYNTHETIC_ONLY") != "1":
        raise SandboxUnavailable("Configure an explicitly synthetic FHIR sandbox before previewing resources.")
    try:
        parsed = urlsplit(base)
        loopback = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
        permitted_http = loopback and os.getenv("FHIR_SANDBOX_ALLOW_LOOPBACK") == "1"
        if (parsed.scheme != "https" and not (parsed.scheme == "http" and permitted_http)) or not parsed.hostname:
            raise ValueError("HTTPS required")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Credentials, query and fragments forbidden")
        _ = parsed.port
    except ValueError:
        raise SandboxUnavailable("FHIR sandbox configuration must be an HTTPS base URL without embedded credentials.")
    return base


async def preview_patient(resource_id: str, *, transport=None):
    """Read one explicitly tagged synthetic patient plus supported FHIR resources.

    transport is an injectable HTTP transport for tests, never a runtime mock.
    The only destination is operator-configured; callers cannot supply URLs.
    """
    if resource_id in {".", ".."} or not re.fullmatch(r"[A-Za-z0-9.-]{1,64}", resource_id):
        raise ValueError("Invalid FHIR resource ID.")
    base = sandbox_base_url()
    total = 0
    headers = {"Accept": "application/fhir+json"}
    token = os.getenv("FHIR_SANDBOX_TOKEN", "")
    if token:
        headers["Authorization"] = "Bearer " + token

    loopback = urlsplit(base).hostname in {"localhost", "127.0.0.1", "::1"}
    async with httpx.AsyncClient(timeout=15, follow_redirects=False, transport=transport, trust_env=not loopback) as client:
        async def get(path, params=None):
            nonlocal total
            data = bytearray()
            async with client.stream("GET", base + "/" + path, params=params, headers=headers) as response:
                if response.status_code != 200:
                    raise SandboxUnavailable("FHIR sandbox did not confirm a successful resource response.")
                async for chunk in response.aiter_bytes():
                    data.extend(chunk)
                    total += len(chunk)
                    if len(data) > MAX_RESPONSE or total > MAX_TOTAL:
                        raise SandboxUnavailable("FHIR sandbox response exceeds the development preview limit.")
            try:
                result = json.loads(data)
            except (ValueError, TypeError):
                raise SandboxUnavailable("FHIR sandbox returned invalid JSON.")
            if not isinstance(result, dict):
                raise SandboxUnavailable("FHIR sandbox returned an invalid resource.")
            return result

        try:
            patient = await get("Patient/" + resource_id)
            if patient.get("resourceType") != "Patient" or patient.get("id") != resource_id:
                raise SandboxUnavailable("FHIR patient identity does not match the requested sandbox resource.")
            meta = patient.get("meta", {})
            tags = meta.get("tag", []) if isinstance(meta, dict) else []
            if not isinstance(tags, list):
                raise SandboxUnavailable("FHIR patient has invalid synthetic metadata.")
            if not any(isinstance(tag, dict) and tag.get("system") == "urn:raphael:development"
                       and tag.get("code") == "synthetic" for tag in tags):
                raise SandboxUnavailable("FHIR preview requires an explicitly tagged synthetic patient.")
            resources = []
            for kind in RESOURCE_TYPES:
                bundle = await get(kind, {"patient": resource_id, "_count": 25})
                if bundle.get("resourceType") != "Bundle" or not isinstance(bundle.get("entry", []), list):
                    raise SandboxUnavailable("FHIR sandbox search did not return a resource bundle.")
                entries = bundle.get("entry", [])
                if len(entries) > 25:
                    raise SandboxUnavailable("FHIR sandbox search exceeds the preview entry limit.")
                for entry in entries:
                    resource = entry.get("resource") if isinstance(entry, dict) else None
                    if not isinstance(resource, dict) or resource.get("resourceType") != kind:
                        raise SandboxUnavailable("FHIR sandbox search contains an unexpected resource type.")
                    references = [resource.get(key, {}).get("reference") for key in ("subject", "patient")
                                  if isinstance(resource.get(key, {}), dict)]
                    if "Patient/" + resource_id not in references:
                        raise SandboxUnavailable("FHIR resource is not linked to the requested synthetic patient.")
                    resources.append(resource)
        except httpx.HTTPError:
            raise SandboxUnavailable("FHIR sandbox is unavailable; no resources were imported.")
    return {"status": "sandbox_preview", "source_base_url": base,
            "retrieved_at": datetime.now(timezone.utc).isoformat(), "patient": patient,
            "resources": resources, "imported": False,
            "limitations": ["Read-only synthetic sandbox preview; no chart records were overwritten.",
                            "Synthetic tags are operator declarations, not deidentification or clinical validation.",
                            "At most 25 resources of each supported type; this is not a complete clinical record."]}
