"""HTTP integration tests for training API route registration."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Optional
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi import FastAPI, Header, HTTPException
from fastapi.testclient import TestClient

from acestep.api.train_api_service import register_training_api_routes


def _wrap_response(data: Any, code: int = 200, error: Optional[str] = None) -> Dict[str, Any]:
    """Return API-compatible response envelope for tests."""

    return {"data": data, "code": code, "error": error}


async def _verify_api_key(authorization: str | None = Header(None)) -> None:
    """Require fixed bearer token for test requests."""

    if authorization != "Bearer test-token":
        raise HTTPException(status_code=401, detail="Unauthorized")


@contextmanager
def _temporary_llm_model(_app: FastAPI, _llm: Any, _lm_model_path: Optional[str]):
    """No-op context manager used by dataset routes during registration."""

    yield


class TrainingApiServiceHttpTests(unittest.TestCase):
    """HTTP tests covering core training service routes."""

    def _build_client(self) -> tuple[TestClient, dict[str, int], FastAPI]:
        """Build app/client pair with deterministic route dependencies."""

        app = FastAPI()
        calls = {"stop_tensorboard": 0}

        def _start_tensorboard(_app: FastAPI, _logdir: str) -> Optional[str]:
            """Return fixed tensorboard URL."""

            return "http://localhost:6006"

        def _stop_tensorboard(_app: FastAPI) -> None:
            """Record stop callback invocations."""

            calls["stop_tensorboard"] += 1

        register_training_api_routes(
            app=app,
            verify_api_key=_verify_api_key,
            wrap_response=_wrap_response,
            start_tensorboard=_start_tensorboard,
            stop_tensorboard=_stop_tensorboard,
            temporary_llm_model=_temporary_llm_model,
            atomic_write_json=lambda _path, _payload: None,
            append_jsonl=lambda _path, _record: None,
        )
        return TestClient(app), calls, app

    def test_training_status_requires_auth(self):
        """GET /v1/training/status should return 401 when auth is missing."""

        client, _calls, _app = self._build_client()
        response = client.get("/v1/training/status")
        self.assertEqual(401, response.status_code)

    def test_training_status_returns_wrapped_payload(self):
        """GET /v1/training/status should return wrapped status payload when authorized."""

        client, _calls, _app = self._build_client()
        response = client.get("/v1/training/status", headers={"Authorization": "Bearer test-token"})
        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertEqual(200, payload["code"])
        self.assertIsNone(payload["error"])
        self.assertIn("is_training", payload["data"])
        self.assertIn("status", payload["data"])

    def test_training_stop_sets_should_stop_and_calls_tensorboard_stop(self):
        """POST /v1/training/stop should set stop flag and call stop callback when training."""

        client, calls, app = self._build_client()
        app.state.training_state = {
            "is_training": True,
            "should_stop": False,
            "status": "Running",
        }
        response = client.post("/v1/training/stop", headers={"Authorization": "Bearer test-token"})
        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertEqual(200, payload["code"])
        self.assertEqual("Stopping training...", payload["data"]["message"])
        self.assertEqual(1, calls["stop_tensorboard"])
        self.assertTrue(app.state.training_state["should_stop"])

    def test_training_stop_returns_idle_message_when_not_training(self):
        """POST /v1/training/stop should return idle message when no training is active."""

        client, calls, app = self._build_client()
        app.state.training_state = {
            "is_training": False,
            "should_stop": False,
            "status": "Idle",
        }
        response = client.post("/v1/training/stop", headers={"Authorization": "Bearer test-token"})
        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertEqual("No training in progress", payload["data"]["message"])
        self.assertEqual(1, calls["stop_tensorboard"])

    def test_load_tensor_info_returns_400_for_missing_dir(self):
        """POST /v1/training/load_tensor_info should return HTTP 400 for missing directory."""

        client, _calls, _app = self._build_client()
        response = client.post(
            "/v1/training/load_tensor_info",
            json={"tensor_dir": "missing-dir"},
            headers={"Authorization": "Bearer test-token"},
        )

        self.assertEqual(400, response.status_code)
        self.assertIn("Path not found", response.json()["detail"])

    def test_load_tensor_info_rejects_path_outside_allowed_roots(self):
        """POST /v1/training/load_tensor_info should return 400 for a directory outside allowed roots."""

        client, _calls, _app = self._build_client()
        with tempfile.TemporaryDirectory() as outside:
            (Path(outside) / "a.pt").write_bytes(b"x")
            with mock.patch.dict(os.environ, {"ACESTEP_ALLOWED_PATHS": "", "ACESTEP_TMPDIR": ""}):
                with mock.patch("acestep.api.http.path_policy._PROJECT_ROOT", Path.cwd() / "nonexistent-root"), \
                        mock.patch("acestep.api.http.path_policy._STARTUP_CWD", Path.cwd() / "nonexistent-root"):
                    response = client.post(
                        "/v1/training/load_tensor_info",
                        json={"tensor_dir": outside},
                        headers={"Authorization": "Bearer test-token"},
                    )
        self.assertEqual(400, response.status_code)
        self.assertIn("allowed directory", response.json()["detail"])
        self.assertNotIn(outside, response.json()["detail"])

    def test_load_tensor_info_accepts_path_inside_allowed_roots(self):
        """POST /v1/training/load_tensor_info should proceed for a directory under an allowed root."""

        client, _calls, _app = self._build_client()
        with tempfile.TemporaryDirectory() as inside:
            (Path(inside) / "a.pt").write_bytes(b"x")
            with mock.patch.dict(os.environ, {"ACESTEP_ALLOWED_PATHS": inside}):
                response = client.post(
                    "/v1/training/load_tensor_info",
                    json={"tensor_dir": inside},
                    headers={"Authorization": "Bearer test-token"},
                )
        self.assertEqual(200, response.status_code)
        self.assertEqual(1, response.json()["data"]["num_samples"])

    def _export_fixture(self, tmp: str) -> Path:
        """Create lora/final/adapter.bin under ``tmp`` and return the resolved root."""

        root = Path(tmp).resolve()
        (root / "lora" / "final").mkdir(parents=True)
        (root / "lora" / "final" / "adapter.bin").write_bytes(b"x")
        return root

    def _post_export(self, client: TestClient, root: Path, export_path: Path):
        """POST /v1/training/export with mocked rmtree/copytree; return (resp, rm, cp)."""

        with mock.patch.dict(os.environ, {"ACESTEP_ALLOWED_PATHS": str(root)}), \
                mock.patch("acestep.api.train_api_service.shutil.rmtree") as rm, \
                mock.patch("acestep.api.train_api_service.shutil.copytree") as cp:
            response = client.post(
                "/v1/training/export",
                json={"lora_output_dir": str(root / "lora"), "export_path": str(export_path)},
                headers={"Authorization": "Bearer test-token"},
            )
        return response, rm, cp

    def test_export_rejects_link_destination(self):
        """POST /v1/training/export must 400 and not rmtree when export_path is a link."""

        client, _calls, _app = self._build_client()
        with tempfile.TemporaryDirectory() as tmp:
            root = self._export_fixture(tmp)
            target = root / "target"
            target.mkdir()
            (target / "keep.txt").write_text("k", encoding="utf-8")
            link = root / "export-link"
            if os.name == "nt":
                import _winapi

                _winapi.CreateJunction(str(target), str(link))
            else:
                try:
                    os.symlink(target, link, target_is_directory=True)
                except (OSError, NotImplementedError):
                    self.skipTest("symlinks not permitted on this platform")
            response, rm, cp = self._post_export(client, root, link)
            self.assertEqual(400, response.status_code)
            rm.assert_not_called()
            cp.assert_not_called()
            self.assertTrue((target / "keep.txt").exists())

    def test_export_rejects_outside_link_pointing_inside(self):
        """An export_path outside the roots that is a link to an inside directory must 400."""

        client, _calls, _app = self._build_client()
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as elsewhere:
            root = self._export_fixture(tmp)
            target = root / "target"
            target.mkdir()
            (target / "keep.txt").write_text("k", encoding="utf-8")
            alias = Path(elsewhere).resolve() / "export-alias"
            if os.name == "nt":
                import _winapi

                _winapi.CreateJunction(str(target), str(alias))
            else:
                try:
                    os.symlink(target, alias, target_is_directory=True)
                except (OSError, NotImplementedError):
                    self.skipTest("symlinks not permitted on this platform")
            response, rm, cp = self._post_export(client, root, alias)
            self.assertEqual(400, response.status_code)
            rm.assert_not_called()
            cp.assert_not_called()
            self.assertTrue((target / "keep.txt").exists())

    @unittest.skipUnless(os.name == "nt", "8.3 names are Windows-only")
    def test_export_to_short_name_destination_succeeds(self):
        """POST /v1/training/export accepts an ordinary directory spelled with its 8.3 alias."""

        import ctypes
        from ctypes import wintypes

        client, _calls, _app = self._build_client()
        with tempfile.TemporaryDirectory() as tmp:
            root = self._export_fixture(tmp)
            export = root / "Long Export Destination"
            export.mkdir()
            buf = ctypes.create_unicode_buffer(1024)
            fn = ctypes.WinDLL("kernel32").GetShortPathNameW
            fn.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
            fn.restype = wintypes.DWORD
            if not fn(str(export), buf, len(buf)) or buf.value.lower() == str(export).lower():
                self.skipTest("8.3 names not available on this volume")
            response, rm, cp = self._post_export(client, root, Path(buf.value))
            self.assertEqual(200, response.status_code, response.text)
            rm.assert_called_once_with(str(export))
            cp.assert_called_once_with(str(root / "lora" / "final"), str(export))
    def test_export_to_plain_new_dir_succeeds(self):
        """POST /v1/training/export copies to a plain non-existent directory."""

        client, _calls, _app = self._build_client()
        with tempfile.TemporaryDirectory() as tmp:
            root = self._export_fixture(tmp)
            export = root / "export"
            response, rm, cp = self._post_export(client, root, export)
            self.assertEqual(200, response.status_code)
            self.assertEqual(200, response.json()["code"])
            rm.assert_not_called()
            cp.assert_called_once_with(str(root / "lora" / "final"), str(export))


if __name__ == "__main__":
    unittest.main()
