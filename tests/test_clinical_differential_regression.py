"""Technical paired synthetic diagnostic regression tests; no clinical data."""
import json
import tempfile
import unittest
from pathlib import Path

from eval.clinical_differential_regression import compare, main


def row(identifier, truth, predictions, specialty="radiology"):
    return {
        "case_id": identifier, "reference": truth, "domain": specialty,
        "ranked_predictions": predictions, "synthetic": True,
    }


class DifferentialRegressionTests(unittest.TestCase):
    def test_nonregressing_same_cohort_reports_no_medical_authorization(self):
        baseline = [row("r1", "target", ["other", "target"]),
                    row("e1", "critical", [], "emergency")]
        candidate = [row("e1", "critical", ["critical"], "emergency"),
                     row("r1", "target", ["target", "other"])]
        result = compare(baseline, candidate)
        self.assertEqual(result["status"], "SYNTHETIC_METRIC_NONREGRESSION")
        self.assertEqual(result["cohort_cases"], 2)
        self.assertEqual(result["regressions"], [])
        self.assertFalse(result["clinical_validation"])
        self.assertFalse(result["model_executed"])
        self.assertFalse(result["clinical_use_authorized"])
        self.assertEqual(result["release_gate"], "NOT_SAFE_FOR_CLINICAL_RELEASE")

    def test_specialty_regression_is_blocked_even_if_other_specialty_improves(self):
        baseline = [row("r1", "dx", ["dx"]),
                    row("e1", "critical", ["other"], "emergency")]
        candidate = [row("r1", "dx", [], "radiology"),
                     row("e1", "critical", ["critical"], "emergency")]
        result = compare(baseline, candidate)
        self.assertEqual(result["status"], "REGRESSION_DETECTED")
        self.assertEqual(result["comparisons"]["radiology"]["status"], "REGRESSED")
        self.assertEqual(result["comparisons"]["emergency"]["status"], "NONREGRESSING")
        self.assertTrue(any(item["domain"] == "radiology" for item in result["regressions"]))
        self.assertFalse(result["clinical_use_authorized"])

    def test_relabeling_cases_or_cherry_picking_is_rejected(self):
        baseline = [row("r1", "a", ["a"]), row("r2", "b", [])]
        bad_candidates = [
            [row("r1", "a", ["a"])],
            [row("r1", "a", ["a"]), row("r3", "b", [])],
            [row("r1", "changed", ["changed"]), row("r2", "b", [])],
            [row("r1", "a", ["a"], "internal_medicine"), row("r2", "b", [])],
            [dict(row("r1", "a", ["a"]), patient_name="unsafe"), row("r2", "b", [])],
        ]
        for candidate in bad_candidates:
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                compare(baseline, candidate)

    def test_empty_or_duplicate_cases_are_not_acceptable(self):
        with self.assertRaises(ValueError):
            compare([], [])
        with self.assertRaises(ValueError):
            compare([row("same", "a", []), row("same", "a", [])],
                    [row("same", "a", []), row("same", "a", [])])

    def test_cli_exit_codes_communicate_regression_and_invalid_comparison(self):
        with tempfile.TemporaryDirectory() as folder:
            original = Path(folder) / "baseline.json"
            changed = Path(folder) / "candidate.json"
            original.write_text(json.dumps([row("r1", "a", ["a"])]), encoding="utf-8")
            changed.write_text(json.dumps([row("r1", "a", [])]), encoding="utf-8")
            self.assertEqual(main([str(original), str(changed)]), 1)
            changed.write_text(json.dumps([row("r2", "a", [])]), encoding="utf-8")
            self.assertEqual(main([str(original), str(changed)]), 2)
            changed.write_text(json.dumps([row("r1", "a", ["a"])]), encoding="utf-8")
            self.assertEqual(main([str(original), str(changed)]), 0)


if __name__ == "__main__":
    unittest.main()
