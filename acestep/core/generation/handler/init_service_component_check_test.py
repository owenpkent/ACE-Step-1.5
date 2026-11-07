"""Tests for per-component main-model presence checks."""

import unittest
from pathlib import Path
from unittest.mock import patch

from acestep.core.generation.handler import init_service_component_check as mod

_MOD = "acestep.core.generation.handler.init_service_component_check"


def _present(missing):
    """Build a fake weights check that reports components in ``missing`` as absent."""
    return lambda path: Path(path).name not in missing


class EnsureMainComponentsTests(unittest.TestCase):
    """Verify only missing, required components are downloaded."""

    def _run(self, missing, require_lm=True, result=(True, "ok")):
        with patch(f"{_MOD}._contains_model_weights", side_effect=_present(missing)), patch(
            f"{_MOD}.download_main_component", return_value=result
        ) as dl:
            out = mod.ensure_main_components(Path("ckpt"), None, require_lm)
        return out, dl

    def test_all_present_no_download(self):
        """Nothing is downloaded when every component exists."""
        out, dl = self._run(set())
        self.assertIsNone(out)
        dl.assert_not_called()

    def test_lm_missing_not_required_skips(self):
        """A missing LM is ignored when require_lm is False."""
        out, dl = self._run({"acestep-5Hz-lm-1.7B"}, require_lm=False)
        self.assertIsNone(out)
        dl.assert_not_called()

    def test_lm_missing_required_downloads_only_lm(self):
        """Only the LM is fetched when it is the sole missing component."""
        out, dl = self._run({"acestep-5Hz-lm-1.7B"})
        self.assertIsNone(out)
        dl.assert_called_once()
        self.assertEqual(dl.call_args.args[0], "acestep-5Hz-lm-1.7B")

    def test_dit_missing_downloads_dit(self):
        """A missing default DiT is fetched on its own."""
        out, dl = self._run({"acestep-v15-turbo"}, require_lm=False)
        self.assertIsNone(out)
        dl.assert_called_once()
        self.assertEqual(dl.call_args.args[0], "acestep-v15-turbo")

    def test_failure_returns_error_tuple(self):
        """A failed download yields the ("ERROR: ...", False) tuple."""
        out, _ = self._run({"vae"}, result=(False, "boom"))
        self.assertIsInstance(out, tuple)
        self.assertTrue(out[0].startswith("ERROR: "))
        self.assertIn("boom", out[0])
        self.assertFalse(out[1])


if __name__ == "__main__":
    unittest.main()
