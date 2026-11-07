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

    def _make_link(self, link: Path, target: Path) -> None:
        """Create a directory symlink, falling back to a junction on Windows, or skip."""

        try:
            os.symlink(target, link, target_is_directory=True)
            return
        except (OSError, NotImplementedError):
            pass
        if os.name == "nt":
            import _winapi

            if hasattr(_winapi, "CreateJunction"):
                _winapi.CreateJunction(str(target), str(link))
                return
        self.skipTest("links not permitted on this platform")

    def test_link_outside_roots_gets_generic_rejection(self) -> None:
        """reject_link must not reveal whether a path outside the roots is a link."""

        (self.outside / "target").mkdir()
        link = self.outside / "link"
        self._make_link(link, self.outside / "target")
        with self.assertRaises(HTTPException) as ctx:
            path_policy.resolve_allowed(str(link), must_exist=False, kind="dir", reject_link=True)
        self.assertEqual(path_policy._REJECTED, ctx.exception.detail)

    @unittest.skipUnless(os.name == "nt", "reserved device names are Windows-only")
    def test_reserved_device_names_rejected(self) -> None:
        """Device names under an allowed root are rejected; look-alikes are not."""

        for name in ("NUL", "con", "COM1.txt", os.path.join("sub", "lpt1"), "aux.wav"):
            with self.assertRaises(HTTPException) as ctx:
                path_policy.resolve_allowed(str(self.root / name), must_exist=False, kind="any")
            self.assertIn("reserved", ctx.exception.detail)
        for name in ("console.txt", "nul_data", "com10", "auxiliary"):
            got = path_policy.resolve_allowed(str(self.root / name), must_exist=False, kind="any")
            self.assertEqual(self.root / name, got)

    @unittest.skipUnless(os.name == "nt", "case-insensitive paths are Windows-only")
    def test_case_variant_of_root_is_accepted(self) -> None:
        """A differently-cased spelling of an allowed root still resolves, link check included."""

        (self.root / "data").mkdir()
        spelled = os.path.join(str(self.root).upper(), "data")
        got = path_policy.resolve_allowed(spelled, must_exist=True, kind="dir", reject_link=True)
        self.assertEqual(self.root / "data", got)


    def test_outside_link_into_root_rejected_for_destructive_use(self) -> None:
        """A link outside the roots that points inside is refused with reject_link, accepted otherwise."""

        target = self.root / "target"
        target.mkdir()
        alias = self.outside / "alias"
        self._make_link(alias, target)
        with self.assertRaises(HTTPException) as ctx:
            path_policy.resolve_allowed(str(alias), must_exist=False, kind="dir", reject_link=True)
        self.assertEqual(path_policy._REJECTED, ctx.exception.detail)
        followed = path_policy.resolve_allowed(str(alias), must_exist=False, kind="dir")
        self.assertEqual(target, followed)

    @unittest.skipUnless(os.name == "nt", "case-insensitive paths are Windows-only")
    def test_case_variant_under_not_yet_created_root(self) -> None:
        """A differently cased path under an allowed root that does not exist yet is accepted."""

        new_root = self.base / "NewRoot"
        with mock.patch.dict(os.environ, {"ACESTEP_ALLOWED_PATHS": str(new_root)}):
            spelled = os.path.join(str(self.base).lower(), "newroot", "Run1")
            got = path_policy.resolve_allowed(spelled, must_exist=False, kind="dir", reject_link=True)
        self.assertEqual(new_root / "Run1", got)
        self.assertEqual(str(new_root / "Run1"), str(got))

    def _short_name(self, path: Path) -> str:
        """Return the 8.3 spelling of ``path`` via GetShortPathNameW, or skip if unavailable."""

        import ctypes
        from ctypes import wintypes

        buf = ctypes.create_unicode_buffer(1024)
        fn = ctypes.WinDLL("kernel32").GetShortPathNameW
        fn.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
        fn.restype = wintypes.DWORD
        if not fn(str(path), buf, len(buf)) or buf.value.lower() == str(path).lower():
            self.skipTest("8.3 names not available on this volume")
        return buf.value

    @unittest.skipUnless(os.name == "nt", "8.3 names are Windows-only")
    def test_short_name_leaf_is_not_a_link(self) -> None:
        """An 8.3 alias of an ordinary directory is accepted under reject_link."""

        target = self.root / "Long Export Destination"
        target.mkdir()
        short = self._short_name(target)
        got = path_policy.resolve_allowed(short, must_exist=False, kind="dir", reject_link=True)
        self.assertEqual(target, got)

    @unittest.skipUnless(os.name == "nt", "8.3 names are Windows-only")
    def test_short_name_root_spelling_is_accepted(self) -> None:
        """A path whose root portion is spelled with 8.3 names is accepted under reject_link."""

        short_root = self._short_name(self.root)
        got = path_policy.resolve_allowed(
            os.path.join(short_root, "new-export"), must_exist=False, kind="dir", reject_link=True
        )
        self.assertEqual(self.root / "new-export", got)

if __name__ == "__main__":
    unittest.main()
