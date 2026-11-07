"""Tests that api_routes handlers do not leak raw exception text to clients."""

import unittest
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from acestep.ui.gradio.api import api_routes

_SECRET = "secret /path/leak"


def _make_client() -> TestClient:
    """Build a test client with mock handlers that look initialized."""
    app = FastAPI()
    app.include_router(api_routes.router)
    app.state.dit_handler = mock.MagicMock(model=object())
    app.state.llm_handler = mock.MagicMock(llm_initialized=True)
    app.state.api_key = None
    return TestClient(app, raise_server_exceptions=False)


class ApiRoutesErrorDetailTests(unittest.TestCase):
    """Verify 500 responses are generic and the exception is logged."""

    def test_format_input_hides_exception_text(self):
        """format_input returns generic error text and logs the exception."""
        with mock.patch(
            "acestep.inference.format_sample", side_effect=ValueError(_SECRET)
        ), mock.patch.object(api_routes, "logger") as fake_logger:
            resp = _make_client().post("/format_input", json={"prompt": "x"})
        self.assertNotIn(_SECRET, resp.text)
        self.assertEqual(resp.json()["code"], 500)
        self.assertIn("see server log", resp.json()["error"])
        fake_logger.exception.assert_called_once()

    def test_release_task_hides_exception_text(self):
        """release_task returns HTTP 500 with generic detail and logs the exception."""
        with mock.patch(
            "acestep.inference.generate_music", side_effect=ValueError(_SECRET)
        ), mock.patch.object(api_routes, "logger") as fake_logger:
            resp = _make_client().post("/release_task", json={"prompt": "x"})
        self.assertEqual(resp.status_code, 500)
        self.assertNotIn(_SECRET, resp.text)
        self.assertIn("see server log", resp.json()["detail"])
        fake_logger.exception.assert_called_once()

    def test_not_initialized_message_unchanged(self):
        """Already-generic errors are left untouched."""
        client = _make_client()
        client.app.state.llm_handler = None
        resp = client.post("/format_input", json={})
        self.assertEqual(resp.json()["error"], "LLM not initialized")


if __name__ == "__main__":
    unittest.main()
