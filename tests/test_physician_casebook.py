"""Deterministic, no-provider tests for clinician review casebook."""
import unittest
from eval.physician_casebook import summarize


class PhysicianCasebookTests(unittest.TestCase):
    def test_radiology_and_internal_medicine_reviews_remain_separate(self):
        report = summarize([
            {"case_id": "synthetic-1", "reviewer_id": "reviewer-A", "domain": "radiology", "rating": "unsafe"},
            {"case_id": "synthetic-2", "reviewer_id": "reviewer-A", "domain": "internal_medicine", "rating": "acceptable"},
        ])
        self.assertEqual(report["domains"]["radiology"]["unsafe"], 1)
        self.assertEqual(report["domains"]["internal_medicine"]["acceptable"], 1)
        self.assertFalse(report["clinical_validation"])
        self.assertFalse(report["clinical_use_authorized"])

    def test_duplicate_case_reviewer_is_rejected(self):
        row = {"case_id": "synthetic-1", "reviewer_id": "reviewer-A", "domain": "radiology", "rating": "acceptable"}
        with self.assertRaises(ValueError):
            summarize([row, row])

    def test_free_text_patient_details_are_rejected(self):
        with self.assertRaises(ValueError):
            summarize([{"case_id": "synthetic-1", "reviewer_id": "reviewer-A", "domain": "radiology", "rating": "unsafe", "patient_name": "REDACTED"}])

    def test_unknown_ratings_are_rejected(self):
        with self.assertRaises(ValueError):
            summarize([{"case_id": "synthetic-1", "reviewer_id": "reviewer-A", "domain": "radiology", "rating": "perfect"}])

    def test_opaque_identifiers_are_enforced(self):
        with self.assertRaises(ValueError):
            summarize([{"case_id": "patient name", "reviewer_id": "reviewer-A", "domain": "radiology", "rating": "unsafe"}])

    def test_invalid_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            summarize({"cases": []})


if __name__ == "__main__":
    unittest.main()
