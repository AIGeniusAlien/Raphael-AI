"""Synthetic assessment schema and citation-membership regressions."""
import json
import unittest
from unittest.mock import patch

from fastapi import HTTPException
from test_recovery import main


class AssessmentContractTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [{"chunk_id": f"fixture-{i}", "doc_id": f"source-{i}",
                        "title": "Synthetic", "tier": "A", "source_type": "test",
                        "text": "Nonclinical fixture", "score": 1.0} for i in range(3)]
        self.draft = {"problem_representation": "Synthetic", "ranked_differential": [{
            "diagnosis": "Synthetic label", "likelihood": "Low", "supports": [],
            "argues_against": [], "citations": ["fixture-0"]}], "working_diagnosis": None,
            "rule_in_out_plan": [], "cant_miss": [], "red_flags_escalation": [],
            "missing_data": [], "confidence": "Low", "evidence_used": [], "safety_notes": []}

    def generate(self, responses):
        with patch.object(main, "retrieve_evidence", return_value=self.chunks), \
             patch.object(main, "call_llm_json", side_effect=responses):
            return main.raphael_generate("Synthetic context", "Synthetic question")

    def test_unknown_citations_are_rejected(self):
        self.draft["ranked_differential"][0]["citations"] = ["fabricated-id"]
        with self.assertRaises(HTTPException) as raised:
            self.generate([self.draft, {"ok": True}])
        self.assertEqual(raised.exception.status_code, 502)
        self.assertIn("not in retrieved evidence", raised.exception.detail)

    def test_malformed_structure_is_rejected(self):
        for key, value in [("ranked_differential", "not a list"), ("confidence", None),
                           ("safety_notes", [12]), ("unknown", "extra field")]:
            draft = dict(self.draft, **{key: value})
            with self.subTest(key=key), self.assertRaises(HTTPException) as raised:
                self.generate([draft, {"ok": True}])
            self.assertEqual(raised.exception.status_code, 502)

    def test_critic_must_use_boolean_approval(self):
        with self.assertRaises(HTTPException) as raised:
            self.generate([self.draft, {"ok": "false"}])
        self.assertEqual(raised.exception.status_code, 502)

    def test_revised_assessment_requires_critic_approval(self):
        with self.assertRaises(HTTPException) as raised:
            self.generate([self.draft, {"ok": False, "required_fixes": ["Synthetic fix"]},
                           self.draft, {"ok": False}])
        self.assertEqual(raised.exception.status_code, 502)
        out = self.generate([self.draft, {"ok": False}, self.draft, {"ok": True}])
        self.assertEqual(out["evidence_used"], self.chunks)

    def test_duplicate_chunks_do_not_satisfy_distinct_source_gate(self):
        for chunk in self.chunks:
            chunk["doc_id"] = "one-source"
        out = self.generate([])
        self.assertIsNone(out["working_diagnosis"])
        self.assertEqual(out["ranked_differential"], [])
        self.assertIn("distinct sources", out["safety_notes"][0])
