"""Unit tests for API-key authentication helpers."""

from __future__ import annotations

import asyncio
import os
import unittest
from unittest.mock import patch

from fastapi import HTTPException

from acestep.api.http.auth import (
    MISSING_HEADER_DETAIL,
    body_api_token_allowed,
    set_api_key,
    verify_api_key,
    verify_token_from_request,
)


class AuthHelpersTests(unittest.TestCase):
    """Behavior tests for API-key state and token/header validation helpers."""

    def setUp(self) -> None:
        """Reset auth state before each test."""

        set_api_key(None)

    def tearDown(self) -> None:
        """Reset auth state after each test."""

        set_api_key(None)

    def test_verify_token_returns_none_when_auth_is_disabled(self) -> None:
        """Token verification should allow requests when no API key is configured."""

        token = verify_token_from_request(body={}, authorization=None)
        self.assertIsNone(token)
        asyncio.run(verify_api_key(authorization=None))

    def test_body_token_alone_rejected_when_flag_off(self) -> None:
        """Body ai_token is ignored by default, yielding the missing-header 401."""

        set_api_key("secret")
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ACESTEP_ALLOW_BODY_API_TOKEN", None)
            with self.assertRaises(HTTPException) as ctx:
                verify_token_from_request(body={"ai_token": "secret"}, authorization=None)
        self.assertEqual(401, ctx.exception.status_code)
        self.assertEqual(MISSING_HEADER_DETAIL, ctx.exception.detail)
        self.assertIn("ACESTEP_ALLOW_BODY_API_TOKEN", ctx.exception.detail)

    def test_body_token_ignored_when_valid_header_present(self) -> None:
        """With the flag off, a bad body token is ignored and the header decides."""

        set_api_key("secret")
        with patch.dict(os.environ, {"ACESTEP_ALLOW_BODY_API_TOKEN": "false"}):
            token = verify_token_from_request(
                body={"ai_token": "wrong"}, authorization="Bearer secret"
            )
        self.assertEqual("secret", token)

    def test_flag_parsing_accepts_only_truthy_values(self) -> None:
        """Only 1/true/yes/on (any case) enable the body token path."""

        for value in ("1", "true", "TRUE", "Yes", "on"):
            with patch.dict(os.environ, {"ACESTEP_ALLOW_BODY_API_TOKEN": value}):
                self.assertTrue(body_api_token_allowed(), value)
        for value in ("", "0", "false", "no", "off", "maybe"):
            with patch.dict(os.environ, {"ACESTEP_ALLOW_BODY_API_TOKEN": value}):
                self.assertFalse(body_api_token_allowed(), value)

    def test_verify_token_accepts_ai_token_in_body(self) -> None:
        """Body ai_token is accepted when the flag is on and it matches the key."""

        set_api_key("secret")
        with patch.dict(os.environ, {"ACESTEP_ALLOW_BODY_API_TOKEN": "true"}):
            token = verify_token_from_request(body={"ai_token": "secret"}, authorization=None)
        self.assertEqual("secret", token)

    def test_verify_token_rejects_invalid_ai_token(self) -> None:
        """With the flag on, a body ai_token mismatch returns the legacy 401 detail."""

        set_api_key("secret")
        with patch.dict(os.environ, {"ACESTEP_ALLOW_BODY_API_TOKEN": "true"}):
            with self.assertRaises(HTTPException) as ctx:
                verify_token_from_request(body={"ai_token": "wrong"}, authorization=None)
        self.assertEqual(401, ctx.exception.status_code)
        self.assertEqual("Invalid ai_token", ctx.exception.detail)

    def test_verify_token_accepts_bearer_header(self) -> None:
        """Authorization header should support Bearer-prefix token values."""

        set_api_key("secret")
        token = verify_token_from_request(body={}, authorization="Bearer secret")
        self.assertEqual("secret", token)

    def test_verify_token_requires_token_when_auth_enabled(self) -> None:
        """Missing body token and header should preserve existing 401 error message."""

        set_api_key("secret")
        with self.assertRaises(HTTPException) as ctx:
            verify_token_from_request(body={}, authorization=None)
        self.assertEqual(401, ctx.exception.status_code)
        self.assertEqual(MISSING_HEADER_DETAIL, ctx.exception.detail)

    def test_verify_api_key_rejects_invalid_or_missing_header(self) -> None:
        """Header-only verifier should preserve missing/invalid API key errors."""

        set_api_key("secret")

        with self.assertRaises(HTTPException) as missing_ctx:
            asyncio.run(verify_api_key(authorization=None))
        self.assertEqual(401, missing_ctx.exception.status_code)
        self.assertEqual("Missing Authorization header", missing_ctx.exception.detail)

        with self.assertRaises(HTTPException) as invalid_ctx:
            asyncio.run(verify_api_key(authorization="Bearer wrong"))
        self.assertEqual(401, invalid_ctx.exception.status_code)
        self.assertEqual("Invalid API key", invalid_ctx.exception.detail)

        asyncio.run(verify_api_key(authorization="Bearer secret"))

    def test_empty_key_disables_auth(self) -> None:
        """An empty key (e.g. an unset Compose passthrough) means no auth, not an empty secret."""

        set_api_key("")

        self.assertIsNone(verify_token_from_request(body={}, authorization=None))
        asyncio.run(verify_api_key(authorization=None))


if __name__ == "__main__":
    unittest.main()
