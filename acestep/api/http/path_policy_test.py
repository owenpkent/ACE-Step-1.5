"""Unit tests for API path confinement helper."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi import HTTPException

from acestep.api.http import path_policy


class PathPolicyTests(unittest.TestCase):
    """Behavior of allowed_roots and resolve_allowed."""

    def setUp(self) -> None:
        """Create an isolated allowed root and clear env overrides."""

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name).resolve()
        self.root = self.base / "root"
        self.outside = self.base / "outside"
        self.root.mkdir()
        self.outside.mkdir()
        (self.outside / "secret.txt").write_text("s", encoding="utf-8")
        patches = [
            mock.patch.object(path_policy, "_PROJECT_ROOT", self.root),
            mock.patch.object(path_policy, "_STARTUP_CWD", self.root),
            mock.patch.dict(os.environ, {"ACESTEP_ALLOWED_PATHS": "", "ACESTEP_TMPDIR": ""}),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def test_inside_root_is_accepted(self) -> None:
        """A directory and a not-yet-existing file under the root resolve."""

        (self.root / "data").mkdir()
        got = path_policy.resolve_allowed(str(self.root / "data"), must_exist=True, kind="dir")
        self.assertEqual(self.root / "data", got)
        new = path_policy.resolve_allowed(str(self.root / "data" / "new.json"), must_exist=False, kind="file")
        self.assertEqual(self.root / "data" / "new.json", new)

    def test_dotdot_escape_rejected(self) -> None:
        """A relative-looking escape via .. is rejected."""

        with self.assertRaises(HTTPException) as ctx:
            path_policy.resolve_allowed(
                str(self.root / ".." / "outside"), must_exist=False, kind="dir"
            )
        self.assertEqual(400, ctx.exception.status_code)

    def test_absolute_outside_rejected_without_echo(self) -> None:
        """An absolute outside path is rejected and the message does not echo layout."""

        with self.assertRaises(HTTPException) as ctx:
            path_policy.resolve_allowed(str(self.outside), must_exist=True, kind="dir")
        self.assertEqual(400, ctx.exception.status_code)
        self.assertNotIn(str(self.outside), ctx.exception.detail)
        self.assertNotIn(str(self.root), ctx.exception.detail)

    def test_symlink_escape_rejected(self) -> None:
        """A symlink inside the root pointing outside is rejected."""

        link = self.root / "link"
        try:
            os.symlink(self.outside, link, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks not permitted on this platform")
        with self.assertRaises(HTTPException):
            path_policy.resolve_allowed(str(link / "secret.txt"), must_exist=True, kind="file")

    def test_env_var_extends_roots(self) -> None:
        """ACESTEP_ALLOWED_PATHS adds extra roots (pathsep-separated)."""

        with self.assertRaises(HTTPException):
            path_policy.resolve_allowed(str(self.outside), must_exist=True, kind="dir")
        extra = os.pathsep.join([str(self.base / "unused"), str(self.outside)])
        with mock.patch.dict(os.environ, {"ACESTEP_ALLOWED_PATHS": extra}):
            self.assertIn(self.outside, path_policy.allowed_roots())
            got = path_policy.resolve_allowed(str(self.outside), must_exist=True, kind="dir")
        self.assertEqual(self.outside, got)

    def test_missing_and_wrong_kind_rejected(self) -> None:
        """must_exist and kind are enforced."""

        with self.assertRaises(HTTPException):
            path_policy.resolve_allowed(str(self.root / "nope"), must_exist=True, kind="dir")
        with self.assertRaises(HTTPException):
            path_policy.resolve_allowed(str(self.root), must_exist=True, kind="file")

    def test_allow_root_false_rejects_the_root_itself(self) -> None:
        """Destructive callers can refuse a path equal to an allowed root."""

        with self.assertRaises(HTTPException):
            path_policy.resolve_allowed(str(self.root), must_exist=False, kind="dir", allow_root=False)

    def test_empty_and_nul_rejected(self) -> None:
        """Empty strings and NUL bytes are rejected."""

        for bad in ("", "   ", "a\x00b"):
            with self.assertRaises(HTTPException):
                path_policy.resolve_allowed(bad, must_exist=False, kind="any")


if __name__ == "__main__":
    unittest.main()
