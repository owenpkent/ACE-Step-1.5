"""HTTP tests for API-key enforcement on the OpenRouter-compatible router."""

import unittest
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from acestep.api.http.auth import set_api_key, verify_api_key
from acestep.openrouter_adapter import create_openrouter_router


def _client(with_auth: bool) -> TestClient:
    """Build an app mounting the router the way route_setup does."""

    app = FastAPI()
    app.state = SimpleNamespace(_initialized=False, _init_error="test")
    kwargs = {"verify_api_key": verify_api_key} if with_auth else {}
    app.include_router(create_openrouter_router(lambda: app.state, **kwargs))

    @app.get("/health")
    async def health():
        """Open health route."""

        return {"ok": True}

    return TestClient(app)


class OpenRouterAuthTests(unittest.TestCase):
    """Auth behavior of /v1/models and /v1/chat/completions."""

    def tearDown(self) -> None:
        """Reset auth state."""

        set_api_key(None)

    def test_key_set_rejects_missing_and_wrong_key(self) -> None:
        """Both routes return 401 without a valid key."""

        set_api_key("secret")
        client = _client(True)
        for bad in ({}, {"Authorization": "Bearer nope"}):
            self.assertEqual(401, client.get("/v1/models", headers=bad).status_code)
            self.assertEqual(
                401, client.post("/v1/chat/completions", json={}, headers=bad).status_code
            )
        self.assertEqual(200, client.get("/health").status_code)

    def test_key_set_accepts_valid_key(self) -> None:
        """A valid bearer key passes auth and reaches the handlers."""

        set_api_key("secret")
        client = _client(True)
        headers = {"Authorization": "Bearer secret"}
        self.assertEqual(200, client.get("/v1/models", headers=headers).status_code)
        # Model not initialized -> 503 proves auth was passed.
        self.assertEqual(
            503, client.post("/v1/chat/completions", json={}, headers=headers).status_code
        )

    def test_no_key_configured_leaves_routes_open(self) -> None:
        """With no key configured the routes work without credentials."""

        set_api_key(None)
        client = _client(True)
        self.assertEqual(200, client.get("/v1/models").status_code)
        self.assertEqual(503, client.post("/v1/chat/completions", json={}).status_code)


if __name__ == "__main__":
    unittest.main()
