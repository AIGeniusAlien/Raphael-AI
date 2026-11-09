"""Safety checks for the bounded Windows coding-agent runner."""
import unittest
from unittest.mock import patch
from scripts import windows_agent


class RunnerTests(unittest.TestCase):
    def test_prompt_includes_safety_boundary(self):
        prompt = windows_agent.build_prompt({"objective": "Implement synthetic account tests only."})
        self.assertIn("No production PHI", prompt)
        self.assertIn("Implement synthetic account tests only.", prompt)

    def test_rejects_unapproved_task(self):
        with self.assertRaisesRegex(RuntimeError, "not explicitly approved"):
            windows_agent.load_task("patient-account-lifecycle")

    def test_rejects_missing_task(self):
        with self.assertRaisesRegex(RuntimeError, "exist exactly once"):
            windows_agent.load_task("not-a-real-task")

    def test_rejects_invalid_duration(self):
        with self.assertRaises(SystemExit):
            windows_agent.main(["--task", "patient-account-lifecycle", "--max-seconds", "5"])

    def test_dry_run_never_invokes_agent(self):
        with patch.object(windows_agent, "verify_checkout", return_value="feature/safe"):
            with self.assertRaisesRegex(RuntimeError, "not explicitly approved"):
                windows_agent.main(["--task", "patient-account-lifecycle"])


if __name__ == "__main__":
    unittest.main()
