"""Paired synthetic ranked-differential regression checks.

Strictly a software evaluation gate. It does not establish diagnostic
validity or permit any clinical deployment. Both datasets must contain the
same synthetic cases, labels and specialty; model outputs can be compared
only after independent adjudication in a separate qualified review process.
"""
from collections import defaultdict
from pathlib import Path
import argparse
import json

from eval.clinical_differential_metrics import evaluate as evaluate_differentials

METRICS = (
    "coverage",
    "top1_recall_all_cases",
    "top3_recall_all_cases",
    "top10_recall_all_cases",
    "mean_reciprocal_rank_all_cases",
)


def compare(baseline, candidate):
    base = evaluate_differentials(baseline)
    changed = evaluate_differentials(candidate)
    if not baseline:
        raise ValueError("Empty synthetic cohort cannot establish regression evidence")
    base_cases = {row["case_id"]: row for row in baseline}
    changed_cases = {row["case_id"]: row for row in candidate}
    if base_cases.keys() != changed_cases.keys():
        raise ValueError("Baseline and candidate must have identical case IDs")
    for identifier, old in base_cases.items():
        new = changed_cases[identifier]
        if (old["reference"], old["domain"]) != (new["reference"], new["domain"]):
            raise ValueError("Paired reference and specialty labels must remain unchanged")
    comparisons = {}
    regressions = []
    for domain in ("overall", *sorted(base["by_domain"])):
        previous = base["overall"] if domain == "overall" else base["by_domain"][domain]
        current = changed["overall"] if domain == "overall" else changed["by_domain"][domain]
        if not previous["cases"]:
            comparisons[domain] = {"cases": 0, "status": "NO_CASES", "metrics": {}}
            continue
        metric_changes = {}
        for metric in METRICS:
            before, after = previous[metric], current[metric]
            # All five metrics are exact proportions computed on the same
            # fixed labels. Use a small rounding tolerance; do not silently
            # permit a clinical performance margin.
            decreased = after + 0.0000005 < before
            metric_changes[metric] = {
                "baseline": before,
                "candidate": after,
                "delta": round(after - before, 6),
                "nonregressing": not decreased,
            }
            if decreased:
                regressions.append({"domain": domain, "metric": metric})
        comparisons[domain] = {
            "cases": previous["cases"],
            "status": "REGRESSED" if any(not x["nonregressing"] for x in metric_changes.values()) else "NONREGRESSING",
            "metrics": metric_changes,
        }
    return {
        "status": "REGRESSION_DETECTED" if regressions else "SYNTHETIC_METRIC_NONREGRESSION",
        "cohort_cases": len(baseline),
        "comparisons": comparisons,
        "regressions": regressions,
        "cohort_paired": True,
        "source_verified": False,
        "model_executed": False,
        "clinical_validation": False,
        "clinical_use_authorized": False,
        "release_gate": "NOT_SAFE_FOR_CLINICAL_RELEASE",
        "limitations": "Synthetic paired labels only; unmeasured safety, calibration, bias, evidence truth and population drift remain. Nonregression is not diagnostic competence.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_json", type=Path)
    parser.add_argument("candidate_json", type=Path)
    args = parser.parse_args(argv)
    try:
        baseline = json.loads(args.baseline_json.read_text(encoding="utf-8"))
        candidate = json.loads(args.candidate_json.read_text(encoding="utf-8"))
        report = compare(baseline, candidate)
    except (OSError, ValueError, TypeError) as error:
        print(json.dumps({"status": "BLOCKED", "reason": type(error).__name__, "clinical_validation": False}))
        return 2
    print(json.dumps(report, indent=2))
    return 1 if report["regressions"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
