"""Static UI contract checks for synthetic account session management."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class AccountSessionUiTests(unittest.TestCase):
    def test_sign_out_other_sessions_is_present_and_connected(self):
        html = (ROOT / "static" / "account.html").read_text(encoding="utf-8")
        js = (ROOT / "static" / "account.js").read_text(encoding="utf-8")
        self.assertIn('id="revoke-other-sessions"', html)
        self.assertIn("$('revoke-other-sessions').onclick", js)
        self.assertIn("api('/sessions/revoke-others',{})", js)
        self.assertIn("await sessions()", js)
        self.assertIn("Your current session remains active.", js)

    def test_user_must_confirm_bulk_revocation(self):
        js = (ROOT / "static" / "account.js").read_text(encoding="utf-8")
        handler = js.split("$('revoke-other-sessions').onclick=", 1)[1].split("async function requests()", 1)[0]
        self.assertIn("if(!confirm(", handler)
        self.assertIn("return;", handler)


if __name__ == "__main__":
    unittest.main()
