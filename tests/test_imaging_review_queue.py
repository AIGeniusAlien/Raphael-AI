"""Synthetic imaging-review derivation tests: no image interpretation or providers."""
import unittest

from imaging_review_queue import summarize_imaging_reports


class ImagingReviewQueueTests(unittest.TestCase):
    def test_empty_queue_is_explicit(self):
        result = summarize_imaging_reports([], [])
        self.assertEqual(result["reports"], [])
        self.assertEqual(result["total_reports"], 0)
        self.assertEqual(result["flagged"], 0)
        self.assertIn("No DICOM/PACS", result["limitations"])

    def test_unlinked_report_is_flagged_without_fabricating_an_encounter(self):
        result = summarize_imaging_reports([
            {"id": 9, "kind": "imaging_report", "encounter_id": None, "status": "draft",
             "revision": 1, "label": "Synthetic only", "data": {"source": "Manual synthetic entry"}},
            {"id": 10, "kind": "lab", "encounter_id": None, "status": "recorded",
             "revision": 1, "label": "Unrelated", "data": {}},
        ], [])
        self.assertEqual(result["total_reports"], 1)
        self.assertEqual(result["drafts"], 1)
        self.assertEqual(result["flagged"], 1)
        self.assertEqual(result["reports"][0]["review_flags"],
                         ["clinician_approval_pending", "encounter_not_linked"])

    def test_closed_encounter_flag_is_specific_to_unapproved_draft(self):
        records = [
            {"id": 7, "kind": "imaging_report", "encounter_id": 5, "status": "approved",
             "revision": 2, "label": "Synthetic", "data": {"source": "Synthetic manual"}},
            {"id": 8, "kind": "imaging_report", "encounter_id": 5, "status": "draft",
             "revision": 1, "label": "Synthetic 2", "data": {"source": "Synthetic manual"}},
        ]
        queue = summarize_imaging_reports(records, [{"id": 5, "status": "closed"}])
        self.assertNotIn("encounter_closed_for_approval", queue["reports"][0]["review_flags"])
        self.assertIn("encounter_closed_for_approval", queue["reports"][1]["review_flags"])
        self.assertEqual(queue["approved"], 1)


if __name__ == "__main__":
    unittest.main()
