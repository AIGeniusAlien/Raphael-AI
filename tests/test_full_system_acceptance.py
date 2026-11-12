"""System acceptance is source-scoped, non-clinical and fails closed."""
import unittest
from scripts.full_system_acceptance import evaluate, SYSTEMS, CI_JOBS, EXTERNAL_GATES

SHA = "a" * 40

def sample():
    names = set()
    for requirements in SYSTEMS.values():
        names.update(requirements)
    local = {
        "schema_version": 1,
        "status": "passed",
        "production_data_used": False,
        "clinical_validation": False,
        "checks": [{"name": name, "status": "passed"} for name in sorted(names)],
    }
    ci = {"source": "github_actions_api", "head_sha": SHA,
          "jobs": {job: "success" for job in CI_JOBS}}
    return local, ci

class FullSystemAcceptanceTests(unittest.TestCase):
    def test_valid_synthetic_checks_never_certify_real_operational_clinical_launch(self):
        local, ci = sample()
        result = evaluate(local, ci, SHA)
        self.assertEqual(result["status"], "SYNTHETIC_ENGINEERING_CHECKS_PASSED")
        self.assertEqual(len(result["systems"]), len(SYSTEMS))
        self.assertTrue(all(item["engineering_test_gate"] == "PASSED"
                            for item in result["systems"].values()))
        self.assertFalse(result["all_systems_fully_operational"])
        self.assertFalse(result["full_clinical_validation"])
        self.assertFalse(result["public_launch_authorized"])
        self.assertFalse(result["clinical_deployment_authorized"])
        self.assertFalse(result["ci_assertion_independently_authenticated_by_this_script"])
        self.assertEqual(set(result["external_gates"]), set(EXTERNAL_GATES))
        self.assertTrue(all(status == "REQUIRES_INDEPENDENT_EVIDENCE"
                            for status in result["external_gates"].values()))

    def test_missing_or_failing_local_subsystem_does_not_disappear(self):
        local, ci = sample()
        local["checks"] = [item for item in local["checks"] if item["name"] != "chromium"]
        result = evaluate(local, ci, SHA)
        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(result["systems"]["clinical_conversations"]["engineering_test_gate"], "BLOCKED")
        self.assertEqual(result["systems"]["native_ios"]["engineering_test_gate"], "PASSED")
        self.assertIn("chromium", result["systems"]["radiology_and_dicom"]["missing_or_failed_checks"])

    def test_ci_source_and_four_jobs_are_required(self):
        local, ci = sample()
        for invalid in (
            None,
            dict(ci, head_sha="b" * 40),
            dict(ci, source="manual"),
            dict(ci, jobs={"regression": "success"}),
            dict(ci, jobs=dict(ci["jobs"], ios_simulator="success")),
            dict(ci, jobs=dict(ci["jobs"], regression="failure")),
        ):
            with self.subTest(invalid=invalid):
                result = evaluate(local, invalid, SHA)
                self.assertEqual(result["status"], "BLOCKED")
                self.assertFalse(result["ci_exact_source_passed"])

    def test_duplicate_or_invalid_or_failed_local_checks_block_all(self):
        local, ci = sample()
        for invalid in (
            None, {},
            dict(local, status="failed"),
            dict(local, clinical_validation=True),
            dict(local, production_data_used=True),
            dict(local, checks=local["checks"] + [local["checks"][0]]),
            dict(local, checks=[{"name": "python", "status": "passed"}, "bad"]),
        ):
            with self.subTest(invalid=str(invalid)[:100]):
                result = evaluate(invalid, ci, SHA)
                self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(evaluate(local, ci, "not-a-commit")["status"], "BLOCKED")


if __name__ == "__main__":
    unittest.main()
