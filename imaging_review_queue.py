"""Deterministic synthetic imaging-report review queue.

This inspects clinician-authored chart metadata only: it does not open DICOM,
interpret pixels, assert modality, or make a clinical safety determination.
"""
from collections import Counter


def summarize_imaging_reports(entries, encounters):
    by_id = {row["id"]: row for row in encounters}
    reports = []
    for entry in entries:
        if entry["kind"] != "imaging_report":
            continue
        encounter_id = entry.get("encounter_id")
        encounter = by_id.get(encounter_id) if encounter_id is not None else None
        flags = []
        if entry["status"] == "draft":
            flags.append("clinician_approval_pending")
        if encounter_id is None:
            flags.append("encounter_not_linked")
        elif encounter is None:
            flags.append("linked_encounter_unavailable")
        elif encounter["status"] == "closed" and entry["status"] == "draft":
            flags.append("encounter_closed_for_approval")
        source = entry.get("data", {}).get("source", "")
        if not isinstance(source, str) or not source.strip():
            flags.append("provenance_missing")
        reports.append({
            "entry_id": entry["id"],
            "encounter_id": encounter_id,
            "status": entry["status"],
            "revision": entry["revision"],
            "label": entry["label"],
            "review_flags": flags,
            "action_required": bool(flags),
        })
    counts = Counter(row["status"] for row in reports)
    return {
        "reports": reports,
        "total_reports": len(reports),
        "drafts": counts["draft"],
        "approved": counts["approved"],
        "flagged": sum(row["action_required"] for row in reports),
        "limitations": (
            "Synthetic chart report coordination only. No DICOM/PACS image ingestion, "
            "pixel interpretation, diagnostic completeness assessment, or clinical validation."
        ),
    }
