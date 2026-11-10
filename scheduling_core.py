"""Deterministic scheduling primitives for future Raphael facility workflows.

No appointments are booked by this module. Time slots are half-open UTC
intervals; an AI suggestion must never be treated as a confirmed booking.
"""
from datetime import datetime, timezone


def utc_instant(value: str) -> datetime:
    """Require an explicit UTC offset and normalize for reliable comparisons."""
    instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if instant.tzinfo is None or instant.utcoffset() is None:
        raise ValueError("An explicit time-zone offset is required")
    return instant.astimezone(timezone.utc)


def validate_interval(start: str, end: str) -> tuple[datetime, datetime]:
    beginning, finish = utc_instant(start), utc_instant(end)
    if finish <= beginning:
        raise ValueError("Appointment end must be after its start")
    return beginning, finish


def conflicts(start: str, end: str, existing: list[dict], resources: set[str]) -> list[dict]:
    """Return overlapping bookings sharing a required staff/room/equipment resource.

    Caller must enforce tenant, access control, and transactional persistence.
    Fail closed on malformed intervals or resource records.
    """
    beginning, finish = validate_interval(start, end)
    if not resources or any(not isinstance(r, str) or not r.strip() for r in resources):
        raise ValueError("At least one valid resource is required")
    found = []
    for booking in existing:
        if not isinstance(booking, dict):
            raise ValueError("Invalid booking record")
        booked_start, booked_end = validate_interval(booking["start"], booking["end"])
        booked_resources = booking["resources"]
        if not isinstance(booked_resources, (list, set, tuple)) or any(not isinstance(r, str) for r in booked_resources):
            raise ValueError("Invalid booked resources")
        if beginning < booked_end and booked_start < finish and resources.intersection(booked_resources):
            found.append(booking)
    return found
