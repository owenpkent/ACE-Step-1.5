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

    def _assert_root_rejected(self, path: Path) -> None:
        """Assert ``path`` is rejected with 400 under ``allow_root=False``."""

        with self.assertRaises(HTTPException) as ctx:
            path_policy.resolve_allowed(
                str(path), must_exist=False, kind="dir", allow_root=False
            )
        self.assertEqual(400, ctx.exception.status_code)

    def test_nested_root_rejected_parent_first(self) -> None:
        """A nested root listed after its ancestor is still rejected as a root."""

        nested = self.root / "datasets"
        nested.mkdir()
        with mock.patch.dict(os.environ, {"ACESTEP_ALLOWED_PATHS": str(nested)}):
            self._assert_root_rejected(nested)
            child = path_policy.resolve_allowed(
                str(nested / "a"), must_exist=False, kind="dir", allow_root=False
            )
        self.assertEqual(nested / "a", child)

    def test_nested_root_rejected_child_first(self) -> None:
        """A nested root listed before its ancestor is still rejected as a root."""

        nested = self.root / "datasets"
        nested.mkdir()
        with mock.patch.dict(os.environ, {"ACESTEP_ALLOWED_PATHS": str(self.root)}), \
                mock.patch.object(path_policy, "_PROJECT_ROOT", nested), \
                mock.patch.object(path_policy, "_STARTUP_CWD", nested):
            self._assert_root_rejected(nested)
            self._assert_root_rejected(self.root)
            child = path_policy.resolve_allowed(
                str(nested / "a"), must_exist=False, kind="dir", allow_root=False
            )
        self.assertEqual(nested / "a", child)

    def _assert_link_behavior(self, link: Path) -> None:
        """Check reject_link rejects ``link`` and the default follows it."""

        with self.assertRaises(HTTPException) as ctx:
            path_policy.resolve_allowed(
                str(link), must_exist=False, kind="dir", reject_link=True
            )
        self.assertEqual(400, ctx.exception.status_code)
        self.assertIn("symbolic link", ctx.exception.detail)
        got = path_policy.resolve_allowed(str(link), must_exist=False, kind="dir")
        self.assertEqual(self.root / "target", got)
        plain = path_policy.resolve_allowed(
            str(self.root / "target"), must_exist=False, kind="dir", reject_link=True
        )
        self.assertEqual(self.root / "target", plain)
        missing = path_policy.resolve_allowed(
            str(self.root / "nope"), must_exist=False, kind="dir", reject_link=True
        )
        self.assertEqual(self.root / "nope", missing)

    def test_reject_link_symlink(self) -> None:
        """A directory symlink is rejected with reject_link=True, accepted by default."""

        (self.root / "target").mkdir()
        link = self.root / "link"
        try:
            os.symlink(self.root / "target", link, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks not permitted on this platform")
        self._assert_link_behavior(link)

    @unittest.skipUnless(os.name == "nt", "junctions are Windows-only")
    def test_reject_link_junction(self) -> None:
        """A Windows junction is rejected with reject_link=True, accepted by default."""

        import _winapi

        if not hasattr(_winapi, "CreateJunction"):
            self.skipTest("_winapi.CreateJunction unavailable")
        (self.root / "target").mkdir()
        link = self.root / "junction"
        _winapi.CreateJunction(str(self.root / "target"), str(link))
        self._assert_link_behavior(link)

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
