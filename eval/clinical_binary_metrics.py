"""Bounded diagnostic-performance accounting for synthetic, labeled review cases.

This module computes statistics; it does not establish ground-truth quality,
medical effectiveness, regulatory clearance, or authorization for patient care.
There are no patient identifiers, narrative fields, or diagnostic texts.
"""
from collections import Counter
from math import sqrt
import argparse
import json
from pathlib import Path

DOMAINS = frozenset(("radiology", "internal_medicine", "emergency"))
LABELS = frozenset(("positive", "negative"))
MAX_CASES = 10000


def _wilson(successes: int, trials: int):
    """95% Wilson score interval for an observed binomial proportion."""
    if trials == 0:
        return None
    z = 1.959963984540054
    p = successes / trials
    d = 1 + z * z / trials
    c = (p + z * z / (2 * trials)) / d
    half = z * sqrt((p * (1 - p) + z * z / (4 * trials)) / trials) / d
    return {"estimate": round(p, 6), "lower_95": round(max(0, c - half), 6),
            "upper_95": round(min(1, c + half), 6), "denominator": trials}


def _domain_metrics(rows):
    counts = Counter()
    for row in rows:
        reference = row["reference"]
        predicted = row["prediction"]
        counts[reference + "_" + predicted] += 1
    tp = counts["positive_positive"]
    fn = counts["positive_negative"]
    tn = counts["negative_negative"]
    fp = counts["negative_positive"]
    abstained_positive = counts["positive_abstain"]
    abstained_negative = counts["negative_abstain"]
    pos = tp + fn + abstained_positive
    neg = tn + fp + abstained_negative
    return {
        "cases": len(rows),
        "true_positive": tp, "true_negative": tn,
        "false_positive": fp, "false_negative": fn,
        "abstained_positive": abstained_positive,
        "abstained_negative": abstained_negative,
        "sensitivity_conservative": _wilson(tp, pos),
        "specificity_conservative": _wilson(tn, neg),
        "abstention_rate": _wilson(abstained_positive + abstained_negative, len(rows)),
        "limitations": "Abstentions count as incorrect for conservative rates. Case selection, label provenance, dataset shift and external validity are not established.",
    }


def evaluate(cases):
    if not isinstance(cases, list) or len(cases) > MAX_CASES:
        raise ValueError("Expected at most 10000 bounded synthetic diagnostic cases")
    seen = set()
    by_domain = {domain: [] for domain in sorted(DOMAINS)}
    for case in cases:
        if not isinstance(case, dict) or set(case) != {"case_id", "domain", "reference", "prediction", "synthetic"}:
            raise ValueError("Unexpected fields; no patient details or medical text permitted")
        identity = case["case_id"]
        if (not isinstance(identity, str) or not 1 <= len(identity) <= 64
                or not identity.isascii() or any(not (c.isalnum() or c in "_-") for c in identity)):
            raise ValueError("Case identifier must be a bounded opaque ASCII token")
        if identity in seen:
            raise ValueError("Duplicate case ID; the same case must not increase sample size")
        seen.add(identity)
        if (case["synthetic"] is not True
                or not all(isinstance(case[key], str) for key in ("domain", "reference", "prediction"))
                or case["domain"] not in DOMAINS or case["reference"] not in LABELS
                or case["prediction"] not in LABELS | {"abstain"}):
            raise ValueError("Only supported labeled synthetic cases are accepted")
        by_domain[case["domain"]].append(case)
    result = {
        "status": "REQUIRES_INDEPENDENT_CLINICAL_REVIEW",
        "total_cases": len(cases),
        "overall": _domain_metrics(cases),
        "by_domain": {domain: _domain_metrics(rows) for domain, rows in by_domain.items()},
        "clinical_validation": False,
        "clinical_use_authorized": False,
        "source_verified": False,
        "note": "Mathematical synthetic-case summaries only. Real independent adjudication, prespecified outcomes, bias assessment, specialty evaluation and governance are not supplied or implied.",
    }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases_json", type=Path)
    args = parser.parse_args(argv)
    try:
        data = json.loads(args.cases_json.read_text(encoding="utf-8"))
        report = evaluate(data)
    except (OSError, ValueError, TypeError) as error:
        print(json.dumps({"status": "BLOCKED", "reason": type(error).__name__}))
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
