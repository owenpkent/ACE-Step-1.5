"""Auth tests for the Gradio-embedded API token verification helper."""

import os
import unittest
from unittest.mock import patch

from fastapi import HTTPException

from acestep.ui.gradio.api.api_routes import set_api_key, verify_token_from_request

_FLAG = "ACESTEP_ALLOW_BODY_API_TOKEN"


class GradioBodyTokenTests(unittest.TestCase):
    """Body ai_token handling is gated by ACESTEP_ALLOW_BODY_API_TOKEN."""

    def setUp(self) -> None:
        """Enable auth with a known key."""

        set_api_key("secret")

    def tearDown(self) -> None:
        """Disable auth again."""

        set_api_key(None)

    def test_body_token_alone_rejected_when_flag_off(self) -> None:
        """Body token is ignored by default, giving the missing-header 401."""

        with patch.dict(os.environ, {}):
            os.environ.pop(_FLAG, None)
            with self.assertRaises(HTTPException) as ctx:
                verify_token_from_request({"ai_token": "secret"}, None)
        self.assertEqual(401, ctx.exception.status_code)
        self.assertIn(_FLAG, ctx.exception.detail)
        self.assertTrue(ctx.exception.detail.startswith("Missing Authorization header."))

    def test_body_token_ignored_when_valid_header_present(self) -> None:
        """With the flag off, header auth succeeds despite a wrong body token."""

        with patch.dict(os.environ, {_FLAG: "0"}):
            token = verify_token_from_request({"ai_token": "wrong"}, "Bearer secret")
        self.assertEqual("secret", token)

    def test_body_token_accepted_and_rejected_when_flag_on(self) -> None:
        """With the flag on, matching body token passes and a mismatch is 401."""

        with patch.dict(os.environ, {_FLAG: "true"}):
            self.assertEqual("secret", verify_token_from_request({"ai_token": "secret"}, None))
            with self.assertRaises(HTTPException) as ctx:
                verify_token_from_request({"ai_token": "wrong"}, None)
        self.assertEqual("Invalid ai_token", ctx.exception.detail)


if __name__ == "__main__":
    unittest.main()
