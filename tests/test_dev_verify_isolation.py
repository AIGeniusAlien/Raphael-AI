"""Regression checks for the synthetic-only verification boundary.

These tests never invoke providers, access patient data, or start services.
"""
import unittest

from dev_verify import counts, isolated_environment


class VerificationIsolationTests(unittest.TestCase):
    def test_verification_removes_provider_credentials_and_clinical_bindings(self):
        source = {
            "OPENAI_API_KEY": "synthetic-openai-secret",
            "LLM_API_KEY": "synthetic-llm-secret",
            "QDRANT_API_KEY": "synthetic-qdrant-secret",
            "JWT_SECRET": "synthetic-jwt-secret",
            "RAPHAEL_DOCUMENT_ENCRYPTION_KEY": "synthetic-document-secret",
            "RAPHAEL_DATABASE_ENCRYPTION_KEY": "synthetic-database-secret",
            "RAPHAEL_MFA_ENCRYPTION_KEY": "synthetic-mfa-secret",
            "RAPHAEL_BACKUP_ENCRYPTION_KEY": "synthetic-backup-secret",
            "RAPHAEL_RELEASE_SCOPE": "clinical",
            "RAPHAEL_VERIFY_CLAIMS": "1",
            "DATABASE_URL": "sqlite:////private/patient.db",
            "STATIC_PATH": "/private/static",
            "QDRANT_URL": "https://synthetic.invalid",
            "LLM_BASE_URL": "https://synthetic.invalid",
            "LLM_MODEL": "synthetic",
            "EMBED_DIM": "384",
            "EMBED_MODEL": "synthetic",
            "UNRELATED_TEST_MARKER": "preserved",
        }
        env = isolated_environment(source)
        for key in source:
            if key not in ("UNRELATED_TEST_MARKER", "RAPHAEL_VERIFY_CLAIMS"):
                self.assertNotIn(key, env, key)
        self.assertEqual(env["UNRELATED_TEST_MARKER"], "preserved")
        self.assertEqual(env["RAPHAEL_VERIFY_CLAIMS"], "0")
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")
        self.assertEqual(env["TRANSFORMERS_OFFLINE"], "1")

    def test_isolation_does_not_modify_input_mapping(self):
        source = {"OPENAI_API_KEY": "synthetic-secret", "UNRELATED_TEST_MARKER": "yes"}
        original = source.copy()
        isolated_environment(source)
        self.assertEqual(source, original)

    def test_test_counts_do_not_infer_success_from_absent_output(self):
        self.assertEqual(counts(""), {})
        self.assertEqual(counts("Ran 12 tests in 1.2s"), {"tests": 12})


if __name__ == "__main__":
    unittest.main()
