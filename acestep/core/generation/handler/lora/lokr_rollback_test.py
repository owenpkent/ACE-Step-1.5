"""Tests for rolling back a rejected LoKr adapter, including a real LyCORIS round trip."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import torch
import torch.nn as nn

from acestep.core.generation.handler.lora import lifecycle
from acestep.core.generation.handler.lora.lokr_load_check import rollback_lokr_adapter

try:
    import lycoris  # noqa: F401

    HAS_LYCORIS = True
except ImportError:  # pragma: no cover - environment dependent
    HAS_LYCORIS = False


class RollbackHelperTests(unittest.TestCase):
    """Unit coverage for ``rollback_lokr_adapter``."""

    def test_restores_network_and_previous_reference(self):
        """restore() is called and the prior _lycoris_net value is put back."""
        decoder = Mock()
        net = Mock()
        previous = object()
        rollback_lokr_adapter(decoder, net, previous)
        net.restore.assert_called_once_with()
        self.assertIs(decoder._lycoris_net, previous)

    def test_tolerates_network_without_restore(self):
        """A network lacking restore() still has its reference cleared."""
        decoder = Mock()
        net = Mock(spec=[])
        rollback_lokr_adapter(decoder, net, None)
        self.assertIsNone(decoder._lycoris_net)


class _TinyDecoder(nn.Module):
    """Decoder with one Linear named after the default LoKr target module."""

    def __init__(self, target_name: str) -> None:
        super().__init__()
        setattr(self, target_name, nn.Linear(4, 4))
        self._target_name = target_name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the single target projection."""
        return getattr(self, self._target_name)(x)


@unittest.skipUnless(HAS_LYCORIS, "lycoris not installed")
class RealLycorisRollbackTests(unittest.TestCase):
    """Rejected weights leave a real decoder usable, with its forwards unwrapped."""

    def test_rejected_weights_leave_decoder_usable(self):
        """Loading a file with no matching keys raises and fully detaches the adapter."""
        from safetensors.torch import save_file

        target = lifecycle.LoKRConfig().target_modules[0]
        decoder = _TinyDecoder(target).to(torch.bfloat16)
        linear = getattr(decoder, target)
        original_forward = linear.forward
        x = torch.ones(1, 4, dtype=torch.bfloat16)
        expected = decoder(x)

        with tempfile.TemporaryDirectory() as tmp:
            weights = Path(tmp) / lifecycle.LOKR_WEIGHTS_FILENAME
            save_file({"unrelated": torch.zeros(1)}, str(weights))
            with self.assertRaises(ValueError):
                lifecycle._load_lokr_adapter(decoder, str(weights))

        self.assertIsNone(getattr(decoder, "_lycoris_net", None))
        self.assertEqual(getattr(decoder, target).forward, original_forward)
        self.assertEqual(decoder(x).dtype, torch.bfloat16)
        self.assertTrue(torch.equal(decoder(x), expected))


if __name__ == "__main__":
    unittest.main()
