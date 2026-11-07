"""Tests for startup logging of the path confinement roots."""

from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()