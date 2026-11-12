"""Fail-closed, evidence-bound Raphael synthetic preview preflight.

No clinical/public distribution approvals can be issued by this program.
This tool evaluates *provided* verification evidence; it never runs checks
by itself or claims the environment, data, or operator is certified.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

REQUIRED_CHECKS = (
    "python", "frontend", "chromium", "mobile-structure", "mobile-tests",
    "traceability", "platform-acceptance", "protected-runtime",
    "safety-evaluation",
)
COMMIT = re.compile(r"[a-fA-F0-9]{40}\Z")


def check_candidate(verification, *, commit, verified_commit, environment):
    """Return a non-sensitive report; require exact-source verification.

    The consumer candidate is an authenticated, synthetic patient records
    preview ONLY. It is not an evaluation release of physician AI, medical
    imaging, or clinical documentation.
    """
    blockers = []
    checks = {}
    if not isinstance(commit, str) or not COMMIT.fullmatch(commit):
        blockers.append("invalid_source_commit")
    if not isinstance(verified_commit, str) or not COMMIT.fullmatch(verified_commit) or verified_commit != commit:
        blockers.append("verification_source_mismatch")
    if not isinstance(environment, dict):
        environment = {}
    for name, expected in (
        ("RAPHAEL_DEV_PORTAL", "1"),
        ("RAPHAEL_DEV_SYNTHETIC_ONLY", "1"),
        ("RAPHAEL_RELEASE_SCOPE", "consumer_candidate"),
    ):
        checks[name] = environment.get(name) == expected
        if not checks[name]:
            blockers.append("incorrect_" + name.lower())
    if environment.get("RAPHAEL_RELEASE_PAUSE_FILE") is None:
        blockers.append("operator_pause_not_configured")
    if not isinstance(verification, dict) or verification.get("schema_version") != 1:
        blockers.append("invalid_verification_report")
        entries = []
    else:
        entries = verification.get("checks")
        if not isinstance(entries, list):
            entries = []
            blockers.append("invalid_verification_checks")
        if verification.get("status") != "passed":
            blockers.append("verification_report_not_passing")
        if verification.get("clinical_validation") is not False or verification.get("production_data_used") is not False:
            blockers.append("verification_scope_unacceptable")
    observed = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            blockers.append("malformed_check_entry")
            continue
        label = entry["name"]
        if label in observed:
            blockers.append("duplicate_check")
        observed[label] = entry.get("status")
    for name in REQUIRED_CHECKS:
        checks[name] = observed.get(name) == "passed"
        if not checks[name]:
            blockers.append("missing_or_failed_" + name.replace("-", "_"))
    # Native simulator and advisory audit are independent CI gates, NOT
    # inferred from local verification where they can be skipped.
    checks["native_ios_ci"] = False
    checks["dependency_advisory_ci"] = False
    blockers.extend(("native_ci_evidence_required", "dependency_ci_evidence_required"))
    return {
        "status": "BLOCKED" if blockers else "SYNTHETIC_PREVIEW_PREFLIGHT_PASSED",
        "source_commit": commit if isinstance(commit, str) and COMMIT.fullmatch(commit) else None,
        "gate": "candidate_synthetic_patient_preview_only",
        "checks": checks,
        "blockers": sorted(set(blockers)),
        "external_service_and_host_review_required": True,
        "operator_authorization_required": True,
        "clinical_validation": False,
        "clinical_production_authorized": False,
        "public_distribution_authorized": False,
        "remarks": (
            "This cannot certify environment isolation, actual synthetic data provenance, "
            "iOS distribution or clinical claims. Independent CI results and accountable "
            "human approval are mandatory."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verification-report", type=Path, required=True)
    parser.add_argument("--commit", required=True, help="Full 40-character tested source commit SHA")
    parser.add_argument("--verified-commit", required=True, help="Full commit SHA recorded for the verification artifact")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        report = json.loads(args.verification_report.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        report = None
    from os import environ
    result = check_candidate(report, commit=args.commit,
                             verified_commit=args.verified_commit,
                             environment=environ)
    # Reports contain only booleans, enum-like codes, and a source SHA.
    document = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(document, encoding="utf-8")
    print(document, end="")
    # No automated approval from a local artifact. 2 is a deliberate
    # fail-closed exit, not a code/test execution error.
    return 2 if result["blockers"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
