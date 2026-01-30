import re
from datetime import datetime

def infer_study_type(title: str, pubtypes: list[str]) -> str:
    t = (title or "").lower()
    pts = [p.lower() for p in (pubtypes or [])]

    # Prefer explicit publication types
    if any("systematic review" in p for p in pts):
        return "systematic_review"
    if any("meta-analysis" in p for p in pts):
        return "meta_analysis"
    if any("randomized controlled trial" in p for p in pts):
        return "rct"
    if any("clinical trial" in p for p in pts):
        return "clinical_trial"
    if any("case reports" in p for p in pts) or "case report" in t:
        return "case_report"

    # Heuristics
    if "meta-analysis" in t:
        return "meta_analysis"
    if "systematic review" in t:
        return "systematic_review"
    if "randomized" in t or "trial" in t:
        return "rct"
    if "cohort" in t:
        return "cohort"
    return "unknown"

def evidence_strength_for(study_type: str) -> str:
    # Rough GRADE-like tiers
    if study_type in ("systematic_review", "meta_analysis"):
        return "high"
    if study_type in ("rct", "clinical_trial"):
        return "high"
    if study_type in ("cohort",):
        return "moderate"
    if study_type in ("case_report",):
        return "very_low"
    return "low"

def recency_score(pubdate: str | None) -> float:
    """
    Returns 0..1. Newer => higher.
    """
    if not pubdate:
        return 0.5
    # pubdate like "2023 Jan 12"
    year = None
    m = re.search(r"(19|20)\d{2}", pubdate)
    if m:
        year = int(m.group(0))
    if not year:
        return 0.5
    now = datetime.utcnow().year
    age = max(0, now - year)
    # 0 years old => 1.0, 10+ => ~0.2
    return max(0.2, 1.0 - (age / 12.0))
