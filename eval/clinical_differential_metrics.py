"""Synthetic-only ranked differential diagnosis evaluation.

Mathematical scoring of opaque labels. This is NOT a clinical benchmark, a
model inference test, a diagnosis, or evidence of medical competence. Real
validation requires independently adjudicated cases and population review.
"""
from collections import defaultdict
import json
import argparse
from pathlib import Path

DOMAINS = frozenset(("radiology", "internal_medicine", "emergency"))
MAX_CASES = 10000
MAX_CANDIDATES = 10


def _token(value):
    return (isinstance(value, str) and 1 <= len(value) <= 64
            and value.isascii()
            and all(char.isalnum() or char in "_-." for char in value))


def _rate(numerator, denominator):
    return None if not denominator else round(numerator / denominator, 6)


def _score(rows):
    n = len(rows)
    answered = [row for row in rows if row["ranked_predictions"]]
    ranked = []
    for row in rows:
        truth = row["reference"]
        candidates = row["ranked_predictions"]
        ranked.append(candidates.index(truth) + 1 if truth in candidates else None)
    top1 = sum(rank == 1 for rank in ranked)
    top3 = sum(rank is not None and rank <= 3 for rank in ranked)
    top10 = sum(rank is not None for rank in ranked)
    return {
        "cases": n,
        "answered_cases": len(answered),
        "abstained_cases": n - len(answered),
        "coverage": _rate(len(answered), n),
        "abstention_rate": _rate(n - len(answered), n),
        "top1_recall_all_cases": _rate(top1, n),
        "top3_recall_all_cases": _rate(top3, n),
        "top10_recall_all_cases": _rate(top10, n),
        "mean_reciprocal_rank_all_cases": _rate(sum(1 / rank for rank in ranked if rank), n),
        "top1_recall_answered_only": _rate(top1, len(answered)),
        "top3_recall_answered_only": _rate(top3, len(answered)),
        "limitations": (
            "All-case scores count abstentions and missed references as zero; "
            "answered-only scores are selection-biased. No case validity, "
            "differential appropriateness, urgency ranking or external validity "
            "has been established."
        ),
    }


def evaluate(cases):
    if not isinstance(cases, list) or len(cases) > MAX_CASES:
        raise ValueError("Expected a bounded list of synthetic evaluation cases")
    seen = set()
    buckets = defaultdict(list)
    for row in cases:
        if not isinstance(row, dict) or set(row) != {
            "case_id", "domain", "reference", "ranked_predictions", "synthetic"
        }:
            raise ValueError("Unexpected fields: no patient narratives or identifiers permitted")
        if not _token(row["case_id"]) or row["case_id"] in seen:
            raise ValueError("Case ID must be unique, opaque and bounded")
        seen.add(row["case_id"])
        if (row["synthetic"] is not True or not isinstance(row["domain"], str)
                or row["domain"] not in DOMAINS or not _token(row["reference"])):
            raise ValueError("Only supported synthetic case labels are accepted")
        predictions = row["ranked_predictions"]
        if (not isinstance(predictions, list) or len(predictions) > MAX_CANDIDATES
                or not all(_token(label) for label in predictions)
                or len(set(predictions)) != len(predictions)):
            raise ValueError("Predictions must be a unique ordered list of opaque labels")
        buckets[row["domain"]].append(row)
    return {
        "status": "REQUIRES_INDEPENDENT_CLINICAL_REVIEW",
        "total_cases": len(cases),
        "overall": _score(cases),
        "by_domain": {name: _score(buckets[name]) for name in sorted(DOMAINS)},
        "labels_are_synthetic": True,
        "clinical_validation": False,
        "clinical_use_authorized": False,
        "source_verified": False,
        "model_executed": False,
        "note": (
            "Metrics cannot establish safe diagnostic reasoning, evidence quality, "
            "clinical significance, demographic performance or physician utility. "
            "A qualified independent clinical evaluation remains required."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases_json", type=Path)
    args = parser.parse_args(argv)
    try:
        result = evaluate(json.loads(args.cases_json.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError) as exc:
        print(json.dumps({"status": "BLOCKED", "reason": type(exc).__name__}))
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
