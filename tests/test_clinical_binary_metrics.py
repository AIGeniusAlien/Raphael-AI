"""Mathematical diagnostic benchmark tests use entirely synthetic labels."""
import json
from pathlib import Path
import tempfile
import unittest

from eval.clinical_binary_metrics import evaluate, main


def case(identifier, reference, prediction, domain="radiology"):
    return {"case_id": identifier, "domain": domain,
            "reference": reference, "prediction": prediction, "synthetic": True}


class BinaryMetricsTests(unittest.TestCase):
    def test_empty_cohort_is_explicitly_unvalidated(self):
        report = evaluate([])
        self.assertEqual(report["total_cases"], 0)
        self.assertIsNone(report["overall"]["sensitivity_conservative"])
        self.assertIsNone(report["overall"]["specificity_conservative"])
        self.assertIsNone(report["overall"]["abstention_rate"])
        self.assertFalse(report["clinical_validation"])
        self.assertFalse(report["clinical_use_authorized"])

    def test_abstentions_are_not_removed_from_sensitivity_denominator(self):
        report = evaluate([
            case("a", "positive", "positive"),
            case("b", "positive", "negative"),
            case("c", "positive", "abstain"),
            case("d", "negative", "negative"),
            case("e", "negative", "positive"),
            case("f", "negative", "abstain"),
        ])
        metrics = report["by_domain"]["radiology"]
        self.assertEqual(metrics["false_negative"], 1)
        self.assertEqual(metrics["false_positive"], 1)
        self.assertEqual(metrics["abstained_positive"], 1)
        self.assertEqual(metrics["abstained_negative"], 1)
        self.assertEqual(metrics["sensitivity_conservative"]["denominator"], 3)
        self.assertAlmostEqual(metrics["sensitivity_conservative"]["estimate"], 1 / 3, places=5)
        self.assertEqual(metrics["specificity_conservative"]["denominator"], 3)
        self.assertAlmostEqual(metrics["specificity_conservative"]["estimate"], 1 / 3, places=5)
        self.assertAlmostEqual(report["overall"]["abstention_rate"]["estimate"], 1 / 3, places=5)
        self.assertGreater(metrics["sensitivity_conservative"]["upper_95"],
                           metrics["sensitivity_conservative"]["lower_95"])
        self.assertFalse(report["source_verified"])

    def test_domain_stratification_does_not_hide_missing_cohorts(self):
        result = evaluate([case("a", "negative", "negative", "emergency")])
        self.assertEqual(result["by_domain"]["emergency"]["cases"], 1)
        self.assertEqual(result["by_domain"]["radiology"]["cases"], 0)
        self.assertIsNone(result["by_domain"]["radiology"]["specificity_conservative"])

    def test_malformed_labels_duplicates_and_unexpected_fields_fail_closed(self):
        examples = [
            [dict(case("case", "positive", "positive"), patient_name="PHI")],
            [case("same", "positive", "positive"), case("same", "negative", "negative")],
            [case("a", "positive", "positive", domain="oncology")],
            [dict(case("a", "positive", "positive"), synthetic=False)],
            [dict(case("a", "positive", "positive"), domain=[])],
            [dict(case("a", "positive", "positive"), reference=None)],
            [dict(case("a", "positive", "positive"), prediction="uncertain")],
            [case("../bad", "positive", "positive")],
        ]
        for rows in examples:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                evaluate(rows)

    def test_cli_reports_math_without_claiming_clinical_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic-metrics.json"
            path.write_text(json.dumps([case("a", "positive", "abstain")]), encoding="utf-8")
            self.assertEqual(main([str(path)]), 0)
            path.write_text(json.dumps([{"patient": "unexpected"}]), encoding="utf-8")
            self.assertEqual(main([str(path)]), 1)


if __name__ == "__main__":
    unittest.main()
