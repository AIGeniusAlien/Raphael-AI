"""Source-scoped Raphael full-system engineering acceptance matrix.

This summarizes executed engineering checks and separately marks unsupported
clinical/enterprise validation. It cannot confer production authorization.
"""
import argparse
import json
from pathlib import Path

# Each entry gives the local engineering checks required for that subset.
# These are NOT exhaustive real-world acceptance criteria for the subsystem.
SYSTEMS = {
    "clinical_conversations": ("python", "frontend", "chromium", "safety-evaluation"),
    "charting_and_documentation": ("python", "frontend", "chromium"),
    "voice_and_ambient_consent": ("python", "frontend", "chromium"),
    "radiology_and_dicom": ("python", "frontend", "chromium"),
    "synthetic_fhir_and_dicomweb": ("python", "chromium"),
    "medication_and_care_coordination": ("python", "chromium"),
    "scheduling_and_hospital_operations": ("python", "chromium"),
    "patient_account_and_portal": ("python", "frontend", "chromium"),
    "native_ios": ("mobile-structure", "mobile-tests"),
    "data_protection_and_recovery": ("python", "protected-runtime"),
    "clinical_safety_evaluation": ("safety-evaluation", "python"),
    "requirements_and_release_governance": ("traceability", "platform-acceptance"),
}
CI_JOBS = ("regression", "ios-simulator", "dependency-audit", "engineering-watch")
EXTERNAL_GATES = (
    "clinician_adjudicated_multispecialty_validation",
    "radiology_diagnostic_image_model_and_miss_rate_validation",
    "medical_grade_speech_accuracy_and_emergency_escalation_validation",
    "hospital_ehr_pacs_order_and_pharmacy_integration_acceptance",
    "verified_real_critical_result_delivery_and_receipt",
    "qualified_privacy_security_phi_lifecycle_and_incident_review",
    "independent_regulatory_intended_use_and_quality_management_review",
    "actual_hardware_staging_operational_recovery_and_partner_acceptance",
)


def evaluate(local, ci, expected_sha):
    """No 'fully operational' output; all clinical/environment gates default blocked."""
    source_valid = isinstance(expected_sha, str) and len(expected_sha) == 40 and all(
        ch in "0123456789abcdefABCDEF" for ch in expected_sha
    )
    local_valid = (
        isinstance(local, dict) and local.get("schema_version") == 1
        and local.get("status") == "passed"
        and local.get("production_data_used") is False
        and local.get("clinical_validation") is False
        and isinstance(local.get("checks"), list)
    )
    counts = {}
    malformed = False
    if local_valid:
        for check in local["checks"]:
            if not isinstance(check, dict) or not isinstance(check.get("name"), str):
                malformed = True
                continue
            key = check["name"]
            if key in counts:
                malformed = True
            counts[key] = check.get("status")
    local_valid = local_valid and not malformed
    ci_valid = (
        source_valid and isinstance(ci, dict)
        and ci.get("head_sha") == expected_sha
        and ci.get("source") == "github_actions_api"
        and isinstance(ci.get("jobs"), dict)
        and set(ci["jobs"]) == set(CI_JOBS)
        and all(ci["jobs"].get(job) == "success" for job in CI_JOBS)
    )
    system_results = {}
    for name, requirements in SYSTEMS.items():
        missing = [item for item in requirements if counts.get(item) != "passed"]
        system_results[name] = {
            "engineering_test_gate": "PASSED" if local_valid and ci_valid and not missing else "BLOCKED",
            "required_checks": list(requirements),
            "missing_or_failed_checks": missing if local_valid else list(requirements),
            "independent_clinical_or_production_validation": "NOT_ESTABLISHED",
            "full_operational_acceptance": False,
        }
    engineering_passed = local_valid and ci_valid and all(
        result["engineering_test_gate"] == "PASSED"
        for result in system_results.values()
    )
    return {
        "schema_version": 1,
        "source_commit": expected_sha if source_valid else None,
        "status": "SYNTHETIC_ENGINEERING_CHECKS_PASSED" if engineering_passed else "BLOCKED",
        "systems": system_results,
        "ci_exact_source_passed": bool(ci_valid),
        "local_report_accepted": bool(local_valid),
        "external_gates": {gate: "REQUIRES_INDEPENDENT_EVIDENCE" for gate in EXTERNAL_GATES},
        "all_systems_fully_operational": False,
        "full_clinical_validation": False,
        "public_launch_authorized": False,
        "clinical_deployment_authorized": False,
        "scope": (
            "Local development and CI software checks only. Test names are not "
            "proof of complete feature coverage, real-world clinical correctness, "
            "hospital integrations, production security, or regulatory clearance."
        ),
        "ci_assertion_independently_authenticated_by_this_script": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--ci", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    def load(path):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
    result = evaluate(load(args.local), load(args.ci), args.source_commit)
    message = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(message, encoding="utf-8")
    print(message, end="")
    return 0 if result["status"] == "SYNTHETIC_ENGINEERING_CHECKS_PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
