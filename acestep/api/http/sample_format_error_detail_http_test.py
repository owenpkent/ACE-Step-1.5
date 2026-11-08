"""Tests that sample/format routes hide failed-result and first LM-init details in 500s."""

from __future__ import annotations

import threading
import unittest
from types import SimpleNamespace
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from acestep.api.http import sample_format_routes

_SECRET = "SECRET C:/internal/path"
_ROUTES = (
    ("/format_input", "format_sample", {"prompt": "x"}),
    ("/v1/create_sample", "create_sample", {"query": "x"}),
)


def _wrap(data, code=200, error=None):
    """Return an API-compatible envelope."""
    return {"data": data, "code": code, "error": error}


def _failed_result(*_args, **_kwargs):
    """Return a failed helper result whose fields carry a secret path."""
    return SimpleNamespace(success=False, error=_SECRET, status_message=_SECRET)


def _client(llm: mock.MagicMock, initialized: bool, format_fn=_failed_result) -> TestClient:
    """Build a client whose helpers return failed results unless ``format_fn`` is given."""
    app = FastAPI()
    sample_format_routes.register_sample_format_routes(
        app, lambda body, auth: None, _wrap, [], [], format_fn, lambda: ".",
        lambda path: "", lambda name, root: "", lambda name, default=False: default,
        lambda v, d=None: d, lambda v, d=None: d, _failed_result,
    )
    app.state.llm_handler = llm
    app.state._llm_init_lock = threading.Lock()
    app.state._llm_initialized = initialized
    app.state._llm_init_error = None
    return TestClient(app)


class SampleFormatFailureDetailTests(unittest.TestCase):
    """Synchronous failure branches return generic text and log the detail."""

    def test_failed_helper_result_is_not_echoed(self):
        """A failed format/create result keeps code 500 without its raw fields."""
        client = _client(mock.MagicMock(), initialized=True)
        for path, prefix, body in _ROUTES:
            with self.subTest(path=path), \
                    mock.patch.object(sample_format_routes, "logger") as lg:
                resp = client.post(path, json=body)
            self.assertNotIn(_SECRET, resp.text)
            self.assertEqual(500, resp.json()["code"])
            self.assertEqual(f"{prefix} failed; see server log", resp.json()["error"])
            self.assertIn(_SECRET, lg.error.call_args.args)

    def test_first_lm_init_failure_is_not_echoed(self):
        """A fresh LM init failure returns a generic 500 and records the status."""
        for path, _prefix, body in _ROUTES:
            llm = mock.MagicMock()
            llm.initialize.return_value = (_SECRET, False)
            client = _client(llm, initialized=False)
            with self.subTest(path=path), \
                    mock.patch.object(sample_format_routes, "logger") as lg:
                resp = client.post(path, json=body)
            self.assertEqual(500, resp.status_code)
            self.assertNotIn(_SECRET, resp.text)
            self.assertEqual("LLM init failed; see server log", resp.json()["detail"])
            self.assertIn(_SECRET, lg.error.call_args.args)
            self.assertEqual(_SECRET, client.app.state._llm_init_error)

    def test_successful_format_still_returns_data(self):
        """Non-target: a successful format result is returned unchanged."""
        ok = SimpleNamespace(
            success=True, caption="c", lyrics="l", bpm=120, keyscale="C major",
            timesignature="4", duration=30, language="en",
        )
        client = _client(mock.MagicMock(), initialized=True, format_fn=lambda **_: ok)
        resp = client.post("/format_input", json={"prompt": "x"})
        self.assertEqual(200, resp.json()["code"])
        self.assertEqual("c", resp.json()["data"]["caption"])


if __name__ == "__main__":
    unittest.main()
