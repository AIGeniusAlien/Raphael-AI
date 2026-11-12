"""Synthetic preview preflight remains fail-closed and never authorizes medicine."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.release_preflight import REQUIRED_CHECKS, check_candidate, main

SHA = "a" * 40
OTHER_SHA = "b" * 40


def fixtures():
    local = {
        "schema_version": 1, "status": "passed",
        "clinical_validation": False, "production_data_used": False,
        "checks": [{"name": name, "status": "passed"} for name in REQUIRED_CHECKS]
        + [{"name": "ios-native", "status": "skipped"},
           {"name": "dependency-advisory-audit", "status": "skipped"}],
    }
    ci = {"source": "github_actions_api", "head_sha": SHA,
          "jobs": {name: "success" for name in (
              "regression", "ios-simulator", "dependency-audit", "engineering-watch")}}
    environment = {
        "RAPHAEL_DEV_PORTAL": "1",
        "RAPHAEL_DEV_SYNTHETIC_ONLY": "1",
        "RAPHAEL_RELEASE_SCOPE": "consumer_candidate",
        "RAPHAEL_RELEASE_PAUSE_FILE": "operator-pause-file-configured",
    }
    return local, ci, environment


class SyntheticPreviewPreflightTests(unittest.TestCase):
    def test_full_technical_evidence_is_not_distribution_or_clinical_approval(self):
        local, ci, env = fixtures()
        report = check_candidate(local, commit=SHA, verified_commit=SHA,
                                 environment=env, ci_evidence=ci)
        self.assertEqual(report["status"], "SYNTHETIC_PREVIEW_TECHNICAL_CHECKS_PASSED")
        self.assertEqual(report["blockers"], [])
        self.assertFalse(report["clinical_production_authorized"])
        self.assertFalse(report["public_distribution_authorized"])
        self.assertFalse(report["clinical_validation"])
        self.assertFalse(report["ci_evidence_independently_authenticated"])
        self.assertTrue(report["operator_authorization_required"])

    def test_matching_commit_and_ci_job_results_must_all_be_present(self):
        local, ci, env = fixtures()
        invalid = [
            (OTHER_SHA, ci), (SHA, None),
            (SHA, dict(ci, head_sha=OTHER_SHA)),
            (SHA, dict(ci, source="manual_claim")),
            (SHA, dict(ci, jobs=dict(ci["jobs"], regression="failure"))),
            (SHA, dict(ci, jobs={"regression": "success"})),
            ("bad", ci),
        ]
        for tested_sha, evidence in invalid:
            with self.subTest(tested_sha=tested_sha, evidence=evidence):
                report = check_candidate(local, commit=SHA, verified_commit=tested_sha,
                                         environment=env, ci_evidence=evidence)
                self.assertEqual(report["status"], "BLOCKED")

    def test_missing_skipped_duplicate_or_failed_local_checks_are_blocked(self):
        local, ci, env = fixtures()
        cases = [
            None,
            dict(local, status="failed"),
            dict(local, clinical_validation=True),
            dict(local, production_data_used=True),
            dict(local, checks=local["checks"][:-3]),
            dict(local, checks=local["checks"] + [local["checks"][0]]),
            dict(local, checks=[dict(item, status="skipped")
                                if item["name"] == "chromium" else item
                                for item in local["checks"]]),
        ]
        for report in cases:
            with self.subTest(report=repr(report)[:80]):
                out = check_candidate(report, commit=SHA, verified_commit=SHA,
                                      environment=env, ci_evidence=ci)
                self.assertEqual(out["status"], "BLOCKED")

    def test_restricted_synthetic_environment_and_operator_pause_are_required(self):
        local, ci, env = fixtures()
        for key in env:
            with self.subTest(key=key):
                invalid = dict(env)
                invalid.pop(key)
                self.assertEqual(check_candidate(
                    local, commit=SHA, verified_commit=SHA,
                    environment=invalid, ci_evidence=ci)["status"], "BLOCKED")

    def test_cli_reports_boolean_evidence_without_printing_secrets(self):
        local, ci, env = fixtures()
        with tempfile.TemporaryDirectory() as folder:
            inputs = [Path(folder) / name for name in ("local.json", "ci.json")]
            for file, data in zip(inputs, (local, ci)):
                file.write_text(json.dumps(data), encoding="utf-8")
            with patch.dict("os.environ", dict(env, LLM_API_KEY="TOP_SECRET_VALUE"),
                            clear=True):
                self.assertEqual(main([
                    "--verification-report", str(inputs[0]),
                    "--ci-evidence", str(inputs[1]),
                    "--commit", SHA, "--verified-commit", SHA,
                    "--output", str(Path(folder) / "result.json"),
                ]), 0)
            output = (Path(folder) / "result.json").read_text(encoding="utf-8")
            self.assertNotIn("TOP_SECRET_VALUE", output)
            self.assertNotIn("operator-pause-file-configured", output)
            self.assertFalse(json.loads(output)["public_distribution_authorized"])
            ci["jobs"]["ios-simulator"] = "failure"
            inputs[1].write_text(json.dumps(ci), encoding="utf-8")
            with patch.dict("os.environ", env, clear=True):
                self.assertEqual(main([
                    "--verification-report", str(inputs[0]),
                    "--ci-evidence", str(inputs[1]),
                    "--commit", SHA, "--verified-commit", SHA,
                ]), 2)


if __name__ == "__main__":
    unittest.main()
