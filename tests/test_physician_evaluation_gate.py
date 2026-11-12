"""Unit tests for physician evaluation readiness, without network or clinical data."""
import unittest
from scripts.physician_evaluation_gate import REQUIRED, evaluate


def report():
    checks = [{"name": name, "status": "passed"} for name in REQUIRED]
    return {"status": "passed", "checks": checks, "production_data_used": False, "clinical_validation": False}


class PhysicianEvaluationGateTests(unittest.TestCase):
    def test_complete_synthetic_engineering_report_passes_only_engineering_gate(self):
        result = evaluate(report())
        self.assertEqual(result["status"], "ENGINEERING_GATE_PASS")
        self.assertFalse(result["physician_review_completed"])
        self.assertFalse(result["clinical_use_authorized"])

    def test_missing_required_browser_test_blocks(self):
        source = report()
        source["checks"] = [c for c in source["checks"] if c["name"] != "chromium"]
        self.assertEqual(evaluate(source)["status"], "BLOCKED")

    def test_skipped_safety_evaluation_blocks(self):
        source = report()
        next(c for c in source["checks"] if c["name"] == "safety-evaluation")["status"] = "skipped"
        self.assertEqual(evaluate(source)["status"], "BLOCKED")

    def test_missing_provenance_blocks(self):
        source = report()
        source.pop("production_data_used")
        self.assertEqual(evaluate(source)["status"], "BLOCKED")

    def test_duplicate_check_blocks(self):
        source = report()
        source["checks"].append(source["checks"][0].copy())
        self.assertEqual(evaluate(source)["status"], "BLOCKED")

    def test_failing_optional_check_blocks(self):
        source = report()
        source["checks"].append({"name": "ios-native", "status": "failed"})
        self.assertEqual(evaluate(source)["status"], "BLOCKED")

    def test_malformed_report_blocks(self):
        self.assertEqual(evaluate(None)["status"], "BLOCKED")
        self.assertEqual(evaluate({"checks": "not-a-list"})["status"], "BLOCKED")


if __name__ == "__main__":
    unittest.main()
