"""Tests that training REST routes do not leak raw exception text in 500 responses."""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from acestep.api import (
    train_api_dataset_auto_label_sync_route as auto_label_mod,
    train_api_dataset_preprocess_routes as preprocess_mod,
    train_api_dataset_sample_routes as sample_mod,
    train_api_dataset_scan_load_routes as scan_load_mod,
    train_api_lokr_start_route as lokr_mod,
    train_api_lora_start_route as lora_mod,
    train_api_service as service_mod,
)

_SECRET = "SECRET C:/internal/path"


def _wrap(data, code=200, error=None):
    """Return an API-compatible envelope."""
    return {"data": data, "code": code, "error": error}


async def _no_auth():
    """Auth dependency that always passes."""


def _client(register, **kwargs) -> tuple[TestClient, FastAPI]:
    """Build an app with one registered route group."""
    app = FastAPI()
    register(app=app, verify_api_key=_no_auth, wrap_response=_wrap, **kwargs)
    return TestClient(app), app


def _failing_builder():
    """Dataset builder double whose operations raise a secret-bearing error."""
    builder = mock.MagicMock()
    for name in ("preprocess_to_tensors", "save_dataset", "update_sample", "scan_directory",
                 "load_dataset"):
        getattr(builder, name).side_effect = RuntimeError(_SECRET)
    return builder


class TrainApiErrorDetailTests(unittest.TestCase):
    """Each synchronous 500 is generic, keeps its prefix, and is logged."""

    def _check(self, resp, prefix, fake_logger):
        """Assert generic 500 envelope and logging."""
        self.assertNotIn(_SECRET, resp.text)
        body = resp.json()
        self.assertEqual(500, body["code"])
        self.assertEqual(f"{prefix}; see server log", body["error"])
        fake_logger.exception.assert_called_once()

    def _allowed(self, mod):
        """Patch resolve_allowed in a module."""
        return mock.patch.object(mod, "resolve_allowed", return_value=Path("x"))

    def test_scan_and_load(self):
        """Scan and load return generic 500s (load also in its message field)."""
        client, _ = _client(scan_load_mod.register_training_dataset_scan_load_routes)
        fake_ds = types.SimpleNamespace(
            DatasetBuilder=mock.MagicMock(return_value=_failing_builder())
        )
        mods = {"acestep.training.dataset_builder": fake_ds}
        with self._allowed(scan_load_mod), mock.patch.object(scan_load_mod, "logger") as lg, \
                mock.patch.dict(sys.modules, mods):
            resp = client.post("/v1/dataset/scan", json={"audio_dir": "d"})
        self._check(resp, "Scan failed", lg)
        with self._allowed(scan_load_mod), mock.patch.object(scan_load_mod, "logger") as lg, \
                mock.patch.dict(sys.modules, mods):
            resp = client.post("/v1/dataset/load", json={"dataset_path": "d"})
        self._check(resp, "Load failed", lg)
        self.assertEqual("Load failed; see server log", resp.json()["data"]["message"])

    def test_save_and_update_sample(self):
        """Save and update return generic 500s."""
        client, app = _client(sample_mod.register_training_dataset_sample_routes)
        app.state.dataset_builder = _failing_builder()
        with self._allowed(sample_mod), mock.patch.object(sample_mod, "logger") as lg:
            resp = client.post("/v1/dataset/save", json={"save_path": "p"})
        self._check(resp, "Save failed", lg)
        with mock.patch.object(sample_mod, "logger") as lg:
            resp = client.put("/v1/dataset/sample/0", json={"sample_idx": 0})
        self._check(resp, "Update failed", lg)

    def test_preprocess(self):
        """Preprocess returns a generic 500."""
        client, app = _client(preprocess_mod.register_training_dataset_preprocess_routes)
        app.state.dataset_builder = _failing_builder()
        app.state.handler = mock.MagicMock()
        app.state.llm_handler = mock.MagicMock()
        with self._allowed(preprocess_mod), mock.patch.object(preprocess_mod, "logger") as lg, \
                mock.patch.object(preprocess_mod, "RuntimeComponentManager"):
            resp = client.post("/v1/dataset/preprocess", json={"output_dir": "o"})
        self._check(resp, "Preprocessing failed", lg)

    def test_auto_label_sync(self):
        """Synchronous auto-label returns a generic 500."""
        def boom(*_a, **_k):
            raise RuntimeError(_SECRET)

        client, app = _client(
            auto_label_mod.register_training_dataset_auto_label_sync_route,
            temporary_llm_model=boom, atomic_write_json=mock.MagicMock(),
            append_jsonl=mock.MagicMock(),
        )
        app.state.dataset_builder = mock.MagicMock()
        app.state.handler = mock.MagicMock()
        app.state.llm_handler = mock.MagicMock(llm_initialized=True)
        with mock.patch.object(auto_label_mod, "logger") as lg, \
                mock.patch.object(auto_label_mod, "RuntimeComponentManager"):
            resp = client.post("/v1/dataset/auto_label", json={})
        self._check(resp, "Auto-label failed", lg)

    def test_export(self):
        """Export returns a generic 500 when copying fails."""
        client, _ = _client(
            service_mod.register_training_api_routes,
            start_tensorboard=mock.MagicMock(), stop_tensorboard=mock.MagicMock(),
            temporary_llm_model=mock.MagicMock(), atomic_write_json=mock.MagicMock(),
            append_jsonl=mock.MagicMock(),
        )
        with self._allowed(service_mod), mock.patch.object(service_mod, "logger") as lg, \
                mock.patch.object(service_mod.os.path, "exists", return_value=True), \
                mock.patch.object(service_mod.shutil, "rmtree"), \
                mock.patch.object(service_mod.shutil, "copytree", side_effect=OSError(_SECRET)):
            resp = client.post("/v1/training/export",
                               json={"lora_output_dir": "l", "export_path": "e"})
        self._check(resp, "Export failed", lg)

    def _start(self, mod, register, path, trainer_name, cfg_names):
        """Run a start route whose trainer constructor raises."""
        client, app = _client(register, start_tensorboard=mock.MagicMock())
        app.state.handler = mock.MagicMock()
        app.state.llm_handler = mock.MagicMock()
        cfgs = types.SimpleNamespace(**{n: mock.MagicMock() for n in cfg_names})
        trainer = types.SimpleNamespace(
            **{trainer_name: mock.MagicMock(side_effect=OSError(_SECRET))}
        )
        mods = {"acestep.training.configs": cfgs, "acestep.training.trainer": trainer}
        with self._allowed(mod), mock.patch.object(mod, "logger") as lg, \
                mock.patch.object(mod, "RuntimeComponentManager"), \
                mock.patch.object(mod, "unwrap_module", side_effect=lambda m: m), \
                mock.patch.dict(sys.modules, mods):
            resp = client.post(path, json={"tensor_dir": "t", "lora_output_dir": "o"})
        self.assertFalse(app.state.training_state["is_training"])
        return resp, lg

    def test_lora_start(self):
        """LoRA start returns a generic 500."""
        resp, lg = self._start(lora_mod, lora_mod.register_lora_training_start_route,
                               "/v1/training/start", "LoRATrainer",
                               ["LoRAConfig", "TrainingConfig"])
        self._check(resp, "Failed to start training", lg)

    def test_lokr_start(self):
        """LoKR start returns a generic 500."""
        resp, lg = self._start(lokr_mod, lokr_mod.register_lokr_training_start_route,
                               "/v1/training/start_lokr", "LoKRTrainer",
                               ["LoKRConfig", "TrainingConfig"])
        self._check(resp, "Failed to start LoKR training", lg)


if __name__ == "__main__":
    unittest.main()
