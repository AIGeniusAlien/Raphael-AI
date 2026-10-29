"""Retrieval age band only; never a legal age-of-consent determination."""
from datetime import date,datetime,timezone


def from_dob(value):
    try:
        born=date.fromisoformat(value);today=datetime.now(timezone.utc).date()
        if born>today:return 'unknown'
        years=today.year-born.year-((today.month,today.day)<(born.month,born.day))
        return 'pediatric' if years<18 else 'adult'
    except (TypeError,ValueError):return 'unknown'
