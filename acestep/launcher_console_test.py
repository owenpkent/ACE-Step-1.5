"""Tests for the run.py launcher's console handling."""

import unittest
from unittest.mock import patch

from acestep import launcher_console


class HoldWindowOpenTest(unittest.TestCase):
    """hold_window_open pauses only for a double-clicked console or when forced."""

    def test_shared_console_does_not_wait(self):
        with patch.object(launcher_console, "console_is_ours", return_value=False), \
                patch("builtins.input") as read:
            launcher_console.hold_window_open(force=False)
        read.assert_not_called()

    def test_force_waits_and_survives_eof(self):
        with patch("builtins.input", side_effect=EOFError) as read, \
                patch("builtins.print"):
            launcher_console.hold_window_open(force=True)
        read.assert_called_once()

    def test_non_windows_console_is_never_ours(self):
        with patch.object(launcher_console.os, "name", "posix"):
            self.assertFalse(launcher_console.console_is_ours())


if __name__ == "__main__":
    unittest.main()
