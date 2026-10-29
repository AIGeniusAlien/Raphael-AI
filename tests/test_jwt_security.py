"""Legacy JWT compatibility and security regressions; synthetic keys only."""
from datetime import datetime, timedelta, timezone
import unittest
from unittest.mock import patch

import jwt
from fastapi import HTTPException
from auth import jwt_auth as auth


class JWTSecurityTests(unittest.TestCase):
    def test_missing_or_historical_default_key_fails_closed(self):
        for key in ("", "CHANGE_ME_NOW"):
            with self.subTest(key=bool(key)), patch.object(auth, "JWT_SECRET", key):
                with self.assertRaises(HTTPException) as result:
                    auth.create_token("synthetic", "tenant", "clinician")
                self.assertEqual(result.exception.status_code, 503)

    def test_expiry_required_claims_and_allowed_algorithm(self):
        key = "synthetic-only-unit-test-secret-32-bytes"
        with patch.object(auth, "JWT_SECRET", key):
            token = auth.create_token("synthetic", "tenant", "clinician")
            self.assertEqual(auth.require_user("Bearer " + token)["tenant_id"], "tenant")
            expired = jwt.encode({"sub": "synthetic", "tenant_id": "tenant", "role": "clinician",
                                  "exp": datetime.now(timezone.utc) - timedelta(seconds=1)}, key, algorithm="HS256")
            missing = jwt.encode({"sub": "synthetic"}, key, algorithm="HS256")
            wrong_algorithm = jwt.encode({"sub": "synthetic"}, key, algorithm="HS384")
            for invalid in (expired, missing, wrong_algorithm, "malformed"):
                with self.assertRaises(HTTPException) as result:
                    auth.require_user("Bearer " + invalid)
                self.assertEqual(result.exception.status_code, 401)
