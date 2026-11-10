"""Synthetic physician-review casebook scoring, separate from clinical validation.

Accepts reviewer judgments only; never auto-grades clinical correctness.
No free-text clinical content or model outputs are stored. Identifiers must be synthetic opaque tokens; this tool cannot independently prove that identifiers contain no PHI.
"""
from collections import Counter
import argparse
import json
from pathlib import Path

DOMAINS = frozenset(("radiology", "internal_medicine", "emergency", "documentation", "interoperability", "voice"))
RATINGS = frozenset(("acceptable", "needs_review", "unsafe"))
MAX_CASES = 10000


def summarize(records):
    if not isinstance(records, list) or len(records) > MAX_CASES:
        raise ValueError("Expected a bounded list of review records")
    seen = set()
    case_judgments = {}
    by_domain = {domain: Counter() for domain in sorted(DOMAINS)}
    for item in records:
        if not isinstance(item, dict) or set(item) != {"case_id", "domain", "rating", "reviewer_id"}:
            raise ValueError("Unexpected review fields; do not submit clinical content or PHI")
        case_id, domain, rating, reviewer = (item[k] for k in ("case_id", "domain", "rating", "reviewer_id"))
        if any(not isinstance(v, str) or not v.isascii() or not (1 <= len(v) <= 64) or not all(c.isalnum() or c in "-_" for c in v) for v in (case_id, reviewer)):
            raise ValueError("Case and reviewer identifiers must be bounded opaque tokens")
        if domain not in DOMAINS or rating not in RATINGS:
            raise ValueError("Unknown domain or rating")
        identity = (case_id, reviewer)
        if identity in seen:
            raise ValueError("Duplicate reviewer/case judgment")
        seen.add(identity)
        prior = case_judgments.setdefault(case_id, {"domain": domain, "ratings": set()})
        if prior["domain"] != domain:
            raise ValueError("Conflicting domain assignments for the same case")
        prior["ratings"].add(rating)
        by_domain[domain][rating] += 1
    total = Counter()
    for counts in by_domain.values():
        total.update(counts)
    disagreements = sum(len(item["ratings"]) > 1 for item in case_judgments.values())
    unsafe_cases = sum("unsafe" in item["ratings"] for item in case_judgments.values())
    return {
        "status": "SAFETY_ESCALATION_REQUIRED" if unsafe_cases else "REVIEW_REQUIRED",
        "distinct_cases": len(case_judgments),
        "disagreement_cases": disagreements,
        "unsafe_cases": unsafe_cases,
        "case_reviews": len(records),
        "ratings": {rating: total[rating] for rating in sorted(RATINGS)},
        "domains": {domain: {rating: counts[rating] for rating in sorted(RATINGS)} for domain, counts in by_domain.items()},
        "clinical_validation": False,
        "clinical_use_authorized": False,
        "note": "Reviewer observations are not sensitivity/specificity, independent validation, or clinical authorization.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review_json", type=Path, help="Local synthetic-case judgments; never upload PHI")
    args = parser.parse_args(argv)
    try:
        records = json.loads(args.review_json.read_text(encoding="utf-8"))
        result = summarize(records)
    except (OSError, ValueError) as exc:
        print(json.dumps({"status": "BLOCKED", "error": type(exc).__name__}))
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
