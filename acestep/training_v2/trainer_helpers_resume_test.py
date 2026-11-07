"""Tests that resume state is loaded with weights_only=True."""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch

from acestep.training_v2.optim import build_scheduler
from acestep.training_v2.trainer_helpers import (
    load_training_state_file,
    save_checkpoint,
)


class _Evil:
    def __reduce__(self):
        return (os.getcwd, ())


def _make():
    model = torch.nn.Linear(4, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = build_scheduler(opt, "cosine", total_steps=100, warmup_steps=5, lr=1e-3)
    model(torch.randn(3, 4)).sum().backward()
    opt.step()
    sched.step()
    return opt, sched


class ResumeStateTest(unittest.TestCase):
    def test_roundtrip_under_weights_only(self):
        opt, sched = _make()
        with tempfile.TemporaryDirectory() as d:
            with patch("acestep.training_v2.trainer_helpers.save_adapter_flat"):
                save_checkpoint(MagicMock(), opt, sched, 3, 42, d)
            state = load_training_state_file(os.path.join(d, "training_state.pt"), "cpu")
        self.assertEqual(state["epoch"], 3)
        self.assertEqual(state["global_step"], 42)
        opt2, sched2 = _make()
        opt2.load_state_dict(state["optimizer_state_dict"])
        sched2.load_state_dict(state["scheduler_state_dict"])

    def test_refuses_non_tensor_objects(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "training_state.pt")
            torch.save({"epoch": 1, "evil": _Evil()}, path)
            with self.assertRaises(ValueError) as cm:
                load_training_state_file(path, "cpu")
            self.assertIn("will not be loaded", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
