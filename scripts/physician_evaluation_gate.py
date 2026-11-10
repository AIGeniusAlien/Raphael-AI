"""Fail-closed physician evaluation readiness gate for synthetic-only engineering.

Consumes actual verification results; never substitutes test counts for clinician review.
Usage: python -m scripts.physician_evaluation_gate /path/to/results.json
"""
import argparse
import json
from pathlib import Path

REQUIRED = (
    "python", "frontend", "chromium", "mobile-structure",
    "mobile-tests", "traceability", "platform-acceptance",
    "safety-evaluation",
)


def evaluate(report):
    if not isinstance(report, dict):
        return {"status": "BLOCKED", "reasons": ["Invalid verification report"]}
    checks = report.get("checks")
    if not isinstance(checks, list):
        return {"status": "BLOCKED", "reasons": ["Missing verification checks"]}
    reasons = []
    by_name = {}
    for check in checks:
        if not isinstance(check, dict) or not isinstance(check.get("name"), str):
            reasons.append("Malformed verification check")
            continue
        name = check["name"]
        if name in by_name:
            reasons.append("Duplicate verification check: " + name)
        by_name[name] = check
    for name in REQUIRED:
        if by_name.get(name, {}).get("status") != "passed":
            reasons.append("Required check not passed: " + name)
    if report.get("status") != "passed":
        reasons.append("Overall engineering verification did not pass")
    if report.get("production_data_used") is not False:
        reasons.append("Synthetic-only data provenance not confirmed")
    if report.get("clinical_validation") is not False:
        reasons.append("Clinical validation flag must remain explicitly false")
    if any(check.get("status") == "failed" for check in by_name.values()):
        reasons.append("One or more additional checks failed")
    return {
        "status": "ENGINEERING_GATE_PASS" if not reasons else "BLOCKED",
        "reasons": reasons,
        "physician_review_completed": False,
        "clinical_use_authorized": False,
        "note": "Passing permits planning a supervised synthetic-case review only; clinician, privacy, and institutional approvals remain separate.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("verification_report", type=Path)
    args = parser.parse_args(argv)
    try:
        report = json.loads(args.verification_report.read_text(encoding="utf-8"))
        outcome = evaluate(report)
    except (OSError, ValueError) as exc:
        outcome = {"status": "BLOCKED", "reasons": ["Cannot read valid verification report: " + type(exc).__name__]}
    print(json.dumps(outcome, indent=2))
    return 0 if outcome["status"] == "ENGINEERING_GATE_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
