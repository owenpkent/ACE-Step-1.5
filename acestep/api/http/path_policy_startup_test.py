"""Tests for startup logging of the path confinement roots."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from acestep.api.http import path_policy_startup


class IsBroadRootTests(unittest.TestCase):
    """Classification of drive, home and ordinary roots."""

    def test_drive_root_is_broad(self) -> None:
        """A filesystem or drive root is broad."""
        anchor = Path(tempfile.gettempdir()).resolve().anchor
        self.assertTrue(path_policy_startup.is_broad_root(Path(anchor)))

    def test_home_and_its_parents_are_broad(self) -> None:
        """The home directory and every directory above it are broad."""
        home = Path(tempfile.gettempdir()).resolve() / "home" / "someone"
        with patch.object(path_policy_startup, "_home_directory", return_value=home):
            self.assertTrue(path_policy_startup.is_broad_root(home))
            self.assertTrue(path_policy_startup.is_broad_root(home.parent))
            self.assertFalse(path_policy_startup.is_broad_root(home / "projects" / "ace"))

    def test_ordinary_directory_is_not_broad(self) -> None:
        """A project directory is not broad."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve() / "project"
            with patch.object(path_policy_startup, "_home_directory", return_value=None):
                self.assertFalse(path_policy_startup.is_broad_root(root))


class LogAllowedRootsTests(unittest.TestCase):
    """Startup logging emits one info line and a warning per broad root."""

    def test_logs_roots_and_warns_only_for_broad_ones(self) -> None:
        """Ordinary roots are listed without a warning; a broad root warns once."""
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp).resolve() / "project"
            drive = Path(project.anchor)
            with patch.object(path_policy_startup, "allowed_roots", return_value=[project, drive]), \
                    patch.object(path_policy_startup, "_home_directory", return_value=None), \
                    patch.object(path_policy_startup, "logger") as mock_logger:
                roots = path_policy_startup.log_allowed_roots()
        self.assertEqual([project, drive], roots)
        mock_logger.info.assert_called_once()
        self.assertIn(str(project), mock_logger.info.call_args.args[1])
        mock_logger.warning.assert_called_once()
        self.assertEqual(drive, mock_logger.warning.call_args.args[1])

    def test_no_warning_for_ordinary_roots(self) -> None:
        """No warning when every root is an ordinary directory."""
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp).resolve() / "project"
            with patch.object(path_policy_startup, "allowed_roots", return_value=[project]), \
                    patch.object(path_policy_startup, "_home_directory", return_value=None), \
                    patch.object(path_policy_startup, "logger") as mock_logger:
                path_policy_startup.log_allowed_roots()
        mock_logger.warning.assert_not_called()

    def test_warning_names_origin_and_does_not_offer_allowed_paths_as_replacement(self) -> None:
        """A broad startup root tells the operator to restart elsewhere, not to set the additive var."""
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp).resolve() / "home"
            home.mkdir()
            with patch.object(path_policy_startup, "allowed_roots", return_value=[home]),                     patch.object(path_policy_startup, "_home_directory", return_value=home),                     patch.object(path_policy_startup.path_policy, "_STARTUP_CWD", home),                     patch.object(path_policy_startup.path_policy, "_PROJECT_ROOT", home / "proj"),                     patch.dict(os.environ, {"ACESTEP_ALLOWED_PATHS": "", "ACESTEP_TMPDIR": ""}),                     patch.object(path_policy_startup, "logger") as mock_logger:
                path_policy_startup.log_allowed_roots()
        args = mock_logger.warning.call_args.args
        self.assertEqual("startup working directory", args[2])
        self.assertIn("Restart the server from the project directory", args[3])
        self.assertIn("only adds roots and cannot remove this one", args[3])
        self.assertNotIn("instead", args[3].split("only adds roots")[0])

    def test_origin_detection_for_env_roots(self) -> None:
        """Roots that come from ACESTEP_TMPDIR or ACESTEP_ALLOWED_PATHS are attributed to them."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            (base / "tmp").mkdir()
            (base / "data").mkdir()
            with patch.object(path_policy_startup.path_policy, "_STARTUP_CWD", base / "proj"),                     patch.object(path_policy_startup.path_policy, "_PROJECT_ROOT", base / "proj"),                     patch.dict(os.environ, {"ACESTEP_TMPDIR": str(base / "tmp"),
                                            "ACESTEP_ALLOWED_PATHS": str(base / "data")}):
                self.assertEqual(["ACESTEP_TMPDIR"], path_policy_startup.root_origins(base / "tmp"))
                self.assertEqual(
                    ["ACESTEP_ALLOWED_PATHS"], path_policy_startup.root_origins(base / "data")
                )
                self.assertEqual(
                    ["startup working directory", "project directory"],
                    path_policy_startup.root_origins(base / "proj"),
                )


if __name__ == "__main__":
    unittest.main()
