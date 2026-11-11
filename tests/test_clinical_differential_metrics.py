"""All fixtures and ranked labels are synthetic, not medical adjudication."""
import json
from pathlib import Path
import tempfile
import unittest

from eval.clinical_differential_metrics import evaluate, main


def case(identifier, reference, ranked, domain="radiology"):
    return {
        "case_id": identifier, "domain": domain, "reference": reference,
        "ranked_predictions": ranked, "synthetic": True,
    }


class DifferentialMetricsTests(unittest.TestCase):
    def test_empty_cohort_is_not_a_successful_clinical_validation(self):
        result = evaluate([])
        self.assertIsNone(result["overall"]["coverage"])
        self.assertIsNone(result["overall"]["top3_recall_all_cases"])
        self.assertFalse(result["clinical_validation"])
        self.assertFalse(result["clinical_use_authorized"])
        self.assertFalse(result["model_executed"])

    def test_ranked_recall_and_abstention_count_all_cases(self):
        rows = [
            case("a", "code_a", ["code_a", "code_b"]),
            case("b", "code_b", ["code_a", "code_c", "code_b"]),
            case("c", "code_c", []),
            case("d", "code_d", ["code_a", "code_b", "code_c"]),
            case("e", "code_e", ["code_a", "code_b", "code_c", "code_d", "code_e"]),
        ]
        result = evaluate(rows)["by_domain"]["radiology"]
        self.assertEqual(result["cases"], 5)
        self.assertEqual(result["answered_cases"], 4)
        self.assertEqual(result["abstained_cases"], 1)
        self.assertEqual(result["coverage"], 0.8)
        self.assertEqual(result["top1_recall_all_cases"], 0.2)
        self.assertEqual(result["top3_recall_all_cases"], 0.4)
        self.assertEqual(result["top10_recall_all_cases"], 0.6)
        self.assertEqual(result["top3_recall_answered_only"], 0.5)
        self.assertAlmostEqual(result["mean_reciprocal_rank_all_cases"], (1+1/3+1/5)/5, places=6)

    def test_nonpresent_domain_and_abstention_cannot_be_hidden(self):
        result = evaluate([case("one", "x", [], "emergency")])
        self.assertEqual(result["by_domain"]["emergency"]["abstention_rate"], 1.0)
        self.assertEqual(result["by_domain"]["emergency"]["top1_recall_all_cases"], 0.0)
        self.assertIsNone(result["by_domain"]["internal_medicine"]["top1_recall_all_cases"])
        self.assertIsNone(result["by_domain"]["radiology"]["top1_recall_all_cases"])

    def test_bad_labels_duplicates_or_patient_details_fail_closed(self):
        bad = [
            [dict(case("a", "d1", []), name="patient")],
            [case("dup", "a", []), case("dup", "b", [])],
            [case("../identifier", "a", [])],
            [case("a", "a", ["a", "a"])],
            [case("a", "a", ["a"] * 11)],
            [case("a", "a", "a")],
            [case("a", "a", ["p@tient"])],
            [case("a", "a", [], domain=[])],
            [dict(case("a", "a", []), synthetic=False)],
            [dict(case("a", "a", []), reference=None)],
        ]
        for rows in bad:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                evaluate(rows)

    def test_cli_never_claims_independent_clinical_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "synthetic.json"
            path.write_text(json.dumps([case("x", "dx_a", ["dx_a"])]), encoding="utf-8")
            self.assertEqual(main([str(path)]), 0)
            path.write_text(json.dumps([{"patient_name": "rejected"}]), encoding="utf-8")
            self.assertEqual(main([str(path)]), 1)


if __name__ == "__main__":
    unittest.main()
