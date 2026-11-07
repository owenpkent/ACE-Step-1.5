"""Tests for LoKr load-result key mismatch checking."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from acestep.core.generation.handler.lora import lokr_load_check as mod


def _net(total: int) -> MagicMock:
    """Build a fake network whose state_dict has ``total`` entries."""
    net = MagicMock()
    net.state_dict.return_value = {f"k{i}": i for i in range(total)}
    return net


class CheckLokrLoadResultTests(unittest.TestCase):
    """Behavior of ``check_lokr_load_result``."""

    def test_all_keys_matched_is_quiet(self):
        """Empty missing/unexpected lists neither warn nor raise."""
        result = SimpleNamespace(missing_keys=[], unexpected_keys=[])
        with patch.object(mod, "logger") as log:
            mod.check_lokr_load_result(result, _net(4), "w.safetensors")
        log.warning.assert_not_called()
        log.debug.assert_called_once()

    def test_partial_match_warns(self):
        """Some missing keys log a warning without raising."""
        result = {"missing keys": ["a"], "unexpected keys": ["b", "c"]}
        with patch.object(mod, "logger") as log:
            mod.check_lokr_load_result(result, _net(4), "w.safetensors")
        log.warning.assert_called_once()
        self.assertIn("w.safetensors", log.warning.call_args.args[0])

    def test_nothing_matched_raises(self):
        """All network keys missing raises ValueError naming the path."""
        result = SimpleNamespace(missing_keys=["a", "b"], unexpected_keys=["x"])
        with self.assertRaises(ValueError) as ctx:
            mod.check_lokr_load_result(result, _net(2), "w.safetensors")
        self.assertIn("w.safetensors", str(ctx.exception))

    def test_none_result_is_noop(self):
        """A None result does nothing and never inspects the network."""
        net = _net(2)
        mod.check_lokr_load_result(None, net, "w.safetensors")
        net.state_dict.assert_not_called()

    def test_unrecognized_result_is_noop(self):
        """Objects without key lists (e.g. mocks) are ignored."""
        mod.check_lokr_load_result(MagicMock(), _net(2), "w.safetensors")


if __name__ == "__main__":
    unittest.main()
