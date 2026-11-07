"""Unit tests for ``_training_kwargs`` in train_vanilla.py.

Verifies that ``resume_from`` is only forwarded to the LoRA trainer, since
the vanilla LoKR trainer's ``train_from_preprocessed`` does not accept it.
Skipped when the heavy training imports (torch) are unavailable.
"""

import argparse
import unittest
from unittest.mock import patch

try:
    from acestep.training_v2.cli import train_vanilla
except ImportError:  # pragma: no cover - environment without torch
    train_vanilla = None


@unittest.skipIf(train_vanilla is None, "train_vanilla imports unavailable")
class TestTrainingKwargs(unittest.TestCase):
    """Tests for the adapter-aware keyword builder."""

    def _args(self, resume_from=None) -> argparse.Namespace:
        """Return a namespace with the fields the helper reads."""
        return argparse.Namespace(dataset_dir="/data", resume_from=resume_from)

    def test_lora_passes_resume_from(self):
        """LoRA receives tensor_dir and resume_from."""
        kwargs = train_vanilla._training_kwargs("lora", self._args("ckpt"))
        self.assertEqual(kwargs, {"tensor_dir": "/data", "resume_from": "ckpt"})

    def test_lokr_omits_resume_from(self):
        """LoKR never receives resume_from."""
        with patch.object(train_vanilla, "logger") as log:
            kwargs = train_vanilla._training_kwargs("lokr", self._args())
        self.assertEqual(kwargs, {"tensor_dir": "/data"})
        log.warning.assert_not_called()

    def test_lokr_with_resume_from_warns(self):
        """LoKR with resume_from set warns and omits it."""
        with patch.object(train_vanilla, "logger") as log:
            kwargs = train_vanilla._training_kwargs("lokr", self._args("ckpt"))
        self.assertNotIn("resume_from", kwargs)
        log.warning.assert_called_once()


if __name__ == "__main__":
    unittest.main()
