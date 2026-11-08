"""Tests that REST routes do not leak raw exception text in synchronous 500 responses."""

from __future__ import annotations

import asyncio
import threading
import unittest
from pathlib import Path
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from acestep.api.http import (
    lora_routes,
    model_service_routes,
    reinitialize_route,
    sample_format_routes,
)

_SECRET = "SECRET C:/internal/path"


def _wrap(data, code=200, error=None):
    """Return an API-compatible envelope."""
    return {"data": data, "code": code, "error": error}


async def _no_auth():
    """Auth dependency that always passes."""


def _boom(*_args, **_kwargs):
    """Raise an error carrying a secret path."""
    raise RuntimeError(_SECRET)


class LoraRoutesErrorDetailTests(unittest.TestCase):
    """LoRA routes return generic 500s."""

    def _client(self) -> TestClient:
        """Build a client whose handler operations all raise."""
        app = FastAPI()
        lora_routes.register_lora_routes(app, _no_auth, _wrap)
        app.state.handler = mock.MagicMock()
        for name in ("load_lora", "unload_lora", "set_use_lora", "set_lora_scale"):
            getattr(app.state.handler, name).side_effect = _boom
        return TestClient(app)

    def test_all_endpoints_hide_exception_text(self):
        """Each LoRA endpoint keeps its 500 shape without the exception text."""
        client = self._client()
        cases = [
            ("/v1/lora/load", {"lora_path": "p"}, "Failed to load LoRA", True),
            ("/v1/lora/unload", None, "Failed to unload LoRA", True),
            ("/v1/lora/toggle", {"use_lora": True}, "Failed to toggle LoRA", False),
            ("/v1/lora/scale", {"scale": 0.5}, "Failed to set LoRA scale", False),
        ]
        for path, payload, prefix, is_http in cases:
            with self.subTest(path=path), \
                    mock.patch.object(lora_routes, "logger") as lg, \
                    mock.patch.object(lora_routes, "resolve_allowed", return_value=Path("x")):
                resp = client.post(path, json=payload)
            self.assertNotIn(_SECRET, resp.text)
            lg.exception.assert_called_once()
            if is_http:
                self.assertEqual(500, resp.status_code)
                self.assertEqual(f"{prefix}; see server log", resp.json()["detail"])
            else:
                self.assertEqual(500, resp.json()["code"])
                self.assertEqual(f"{prefix}; see server log", resp.json()["error"])


class ModelServiceErrorDetailTests(unittest.TestCase):
    """The /v1/init route returns generic 500s but keeps the slot 400 message."""

    def _post(self, exc: Exception):
        """POST /v1/init with initialization raising ``exc``."""
        app = FastAPI()
        model_service_routes.register_model_service_routes(
            app, _no_auth, _wrap, store=mock.MagicMock(), queue_maxsize=1,
            initial_avg_job_seconds=1.0, get_project_root=lambda: ".",
            get_model_name=lambda p: p, ensure_model_downloaded=lambda a, b: "",
            env_bool=lambda *_: False,
        )
        app.state._init_lock = asyncio.Lock()
        app.state._llm_init_lock = threading.Lock()
        app.state.executor = None
        with mock.patch.object(model_service_routes, "logger") as lg, mock.patch.object(
            model_service_routes, "initialize_models_for_request", side_effect=exc
        ):
            return TestClient(app).post("/v1/init", json={}), lg

    def test_runtime_and_generic_errors_hidden(self):
        """RuntimeError and other exceptions both yield the generic message."""
        for exc in (RuntimeError(_SECRET), ValueError(_SECRET)):
            with self.subTest(exc=type(exc).__name__):
                resp, lg = self._post(exc)
                self.assertNotIn(_SECRET, resp.text)
                self.assertEqual(500, resp.json()["code"])
                self.assertEqual(
                    "Model initialization failed; see server log", resp.json()["error"]
                )
                lg.exception.assert_called_once()

    def test_slot_not_available_message_unchanged(self):
        """The client-error 400 for an unavailable slot still returns its message."""
        resp, lg = self._post(RuntimeError("Slot 2 is not available"))
        self.assertEqual(400, resp.json()["code"])
        self.assertEqual("Slot 2 is not available", resp.json()["error"])
        lg.exception.assert_not_called()


class ReinitializeErrorDetailTests(unittest.TestCase):
    """Reinitialize returns a generic 500."""

    def test_hides_exception_text(self):
        """A failing LLM initialize does not leak its message."""
        app = FastAPI()
        reinitialize_route.register_reinitialize_route(
            app, _no_auth, _wrap, env_bool=lambda *_: False, get_project_root=lambda: "."
        )
        app.state.handler = mock.MagicMock(last_init_params=None)
        app.state.llm_handler = mock.MagicMock(llm_initialized=False, last_init_params={})
        app.state.llm_handler.initialize.side_effect = _boom
        with mock.patch.object(reinitialize_route, "logger") as lg:
            resp = TestClient(app).post("/v1/reinitialize")
        self.assertNotIn(_SECRET, resp.text)
        self.assertEqual(500, resp.json()["code"])
        self.assertEqual("Reinitialization failed; see server log", resp.json()["error"])
        lg.exception.assert_called_once()


class SampleFormatErrorDetailTests(unittest.TestCase):
    """format_input and create_sample return generic 500s."""

    def _client(self) -> TestClient:
        """Build a client whose format and create callables raise."""
        app = FastAPI()
        sample_format_routes.register_sample_format_routes(
            app, lambda body, auth: None, _wrap, [], [], _boom, lambda: ".", lambda p: p,
            lambda a, b: "", lambda *_: False, lambda v, d=None: d, lambda v, d=None: d, _boom,
        )
        app.state.llm_handler = mock.MagicMock()
        app.state._llm_init_lock = threading.Lock()
        app.state._llm_initialized = True
        return TestClient(app)

    def test_format_and_create_hide_exception_text(self):
        """Both routes return generic text and log the exception."""
        client = self._client()
        for path, prefix, body in (
            ("/format_input", "format_sample", {"prompt": "x"}),
            ("/v1/create_sample", "create_sample", {"query": "x"}),
        ):
            with self.subTest(path=path), \
                    mock.patch.object(sample_format_routes, "logger") as lg:
                resp = client.post(path, json=body)
            self.assertNotIn(_SECRET, resp.text)
            self.assertEqual(500, resp.json()["code"])
            self.assertEqual(f"{prefix} failed; see server log", resp.json()["error"])
            lg.exception.assert_called_once()


    def test_stored_llm_init_error_is_not_echoed(self):
        """A recorded LM init failure yields a generic 500 and is logged."""
        client = self._client()
        client.app.state._llm_initialized = False
        client.app.state._llm_init_error = _SECRET
        for path, body in (("/format_input", {"prompt": "x"}), ("/v1/create_sample", {"query": "x"})):
            with self.subTest(path=path),                     mock.patch.object(sample_format_routes, "logger") as lg:
                resp = client.post(path, json=body)
            self.assertEqual(500, resp.status_code)
            self.assertNotIn(_SECRET, resp.text)
            self.assertEqual("LLM init failed; see server log", resp.json()["detail"])
            lg.error.assert_called_once()

if __name__ == "__main__":
    unittest.main()
