"""Synthetic schema protocol tests; not a clinical competence assessment."""
import unittest

from pydantic import ValidationError
from rag.assessment_schema import Assessment, Critique, Differential


def fixture():
    return {
        "problem_representation": "Synthetic symptom statement; no patient data.",
        "ranked_differential": [{
            "diagnosis": "SYNTHETIC-A", "likelihood": "Low",
            "supports": ["Reported synthetic feature"],
            "argues_against": ["Synthetic evidence unknown"],
            "citations": ["synthetic-source-1"],
        }],
        "working_diagnosis": None,
        "rule_in_out_plan": ["Clinician review only"],
        "cant_miss": [], "red_flags_escalation": [], "missing_data": ["Examination unknown"],
        "confidence": "Unknown; no independent evaluation",
        "evidence_used": [{"chunk_id": "synthetic-source-1"}],
        "safety_notes": ["Provisional and not validated"],
    }


class AssessmentSchemaTests(unittest.TestCase):
    def test_valid_synthetic_draft_retains_explicit_uncertainty(self):
        row = Assessment.model_validate(fixture())
        self.assertIsNone(row.working_diagnosis)
        self.assertEqual(row.ranked_differential[0].likelihood, "Low")

    def test_overlong_prose_and_too_many_hypotheses_fail_closed(self):
        for field in ["problem_representation", "confidence"]:
            data = fixture()
            data[field] = "x" * 4000
            with self.subTest(field=field), self.assertRaises(ValidationError):
                Assessment.model_validate(data)
        data = fixture()
        data["ranked_differential"] = [
            dict(data["ranked_differential"][0], diagnosis="synthetic-" + str(i))
            for i in range(11)
        ]
        with self.assertRaises(ValidationError):
            Assessment.model_validate(data)

    def test_duplicate_differentials_even_with_different_case_are_blocked(self):
        data = fixture()
        data["ranked_differential"].append(dict(data["ranked_differential"][0], diagnosis="synthetic-a"))
        with self.assertRaises(ValidationError):
            Assessment.model_validate(data)

    def test_duplicate_or_unbounded_citations_are_blocked(self):
        for values in (["source", "source"], ["x" * 129], []):
            data = fixture()
            data["ranked_differential"][0]["citations"] = values
            with self.subTest(values=values), self.assertRaises(ValidationError):
                Assessment.model_validate(data)

    def test_unbounded_or_blank_justifications_fail_closed(self):
        for field, value in (("supports", [""]), ("argues_against", ["x" * 1200])):
            data = fixture()
            data["ranked_differential"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValidationError):
                Assessment.model_validate(data)

    def test_large_untrusted_evidence_metadata_fails_closed(self):
        data = fixture()
        data["evidence_used"] = [{"content": "x" * 25000}]
        with self.assertRaises(ValidationError):
            Assessment.model_validate(data)

    def test_critique_remains_strict(self):
        self.assertTrue(Critique.model_validate({"ok": True}).ok)
        with self.assertRaises(ValidationError):
            Critique.model_validate({"ok": True, "extra": "untrusted"})


if __name__ == "__main__":
    unittest.main()
