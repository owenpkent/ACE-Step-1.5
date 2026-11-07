"""Filesystem path confinement for API routes that accept caller-chosen paths.

Training, dataset and LoRA routes take directories and files from the request
body. Every such path must pass through :func:`resolve_allowed` before it is
read, written, scanned or deleted.

The checks are written so a static analyser can follow them: the caller path is
canonicalised with ``os.path.realpath`` and only reaches a filesystem call after a
``str.startswith`` test against an allowed root has passed.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import List, Literal, Optional

from fastapi import HTTPException

ALLOWED_PATHS_ENV = "ACESTEP_ALLOWED_PATHS"

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_STARTUP_CWD = Path.cwd().resolve()

_REJECTED = (
    "Path is not within an allowed directory. "
    f"Move the data under the project directory or add its parent to {ALLOWED_PATHS_ENV}."
)
_LINK_REJECTED = "Path must not be a symbolic link or junction"
_RESERVED_REJECTED = "Path uses a reserved device name"

# Windows device names; the OS opens the device for these in any directory, with or
# without an extension (``COM1.txt`` still means the serial port).
_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in "123456789\u00b9\u00b2\u00b3"}
    | {f"LPT{i}" for i in "123456789\u00b9\u00b2\u00b3"}
)


def _safe_resolve(raw: str) -> Optional[Path]:
    """Resolve a configured root, or return None when the string is not a usable path."""
    try:
        return Path(raw).expanduser().resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        return None


def _canonical(raw: str) -> Optional[str]:
    """Return the canonical absolute form of ``raw``: symlinks followed, case as on disk."""
    try:
        return os.path.realpath(os.path.expanduser(raw))
    except (OSError, RuntimeError, ValueError):
        return None


def _lexical(raw: str) -> str:
    """Normalise ``raw`` without following links, case-folded for prefix comparison."""
    return os.path.normpath(os.path.normcase(os.path.abspath(os.path.expanduser(raw))))


def _is_link(path: Path) -> bool:
    """Return True if the final component of ``path`` is a symlink or Windows junction.

    The path is not resolved. A missing or unreadable path is not a link.
    """

    try:
        st = os.lstat(path)
    except (OSError, ValueError):
        return False
    if stat.S_ISLNK(st.st_mode):
        return True
    mount_point = getattr(stat, "IO_REPARSE_TAG_MOUNT_POINT", -1)
    return getattr(st, "st_reparse_tag", 0) == mount_point


def _has_reserved_name(path: str) -> bool:
    """Return True on Windows if any component of ``path`` names a reserved device."""
    if os.name != "nt":
        return False
    for part in Path(path).parts:
        stem = part.split(".", 1)[0].rstrip(" .").upper()
        if stem in _RESERVED_NAMES:
            return True
    return False


def allowed_roots() -> List[Path]:
    """Return the resolved directories the API may read or write for training data.

    Includes the project root, the working directory at startup, the configured
    temp dir (``ACESTEP_TMPDIR``), and any ``ACESTEP_ALLOWED_PATHS`` entries
    (separated by ``os.pathsep``). The project root already covers the default
    ``checkpoints``, ``gradio_outputs`` and ``.cache/acestep/tmp`` directories.
    """

    candidates: List[str] = [str(_PROJECT_ROOT), str(_STARTUP_CWD)]
    tmpdir = (os.getenv("ACESTEP_TMPDIR") or "").strip()
    if tmpdir:
        candidates.append(tmpdir)
    candidates.extend(
        part.strip()
        for part in (os.getenv(ALLOWED_PATHS_ENV) or "").split(os.pathsep)
        if part.strip()
    )
    roots: List[Path] = []
    for raw in candidates:
        resolved = _safe_resolve(raw)
        if resolved is not None and resolved not in roots:
            roots.append(resolved)
    return roots


def _reject_link_before_resolution(raw: str, prefixes: List[str]) -> None:
    """Raise if ``raw`` lexically sits under an allowed root and ends in a link.

    This runs before symlinks are followed. A path outside every root is left for
    the generic containment rejection, so the link check reveals nothing about it.
    """

    lexical = _lexical(raw)
    for prefix in prefixes:
        base = os.path.normcase(prefix).rstrip(os.sep) + os.sep
        if lexical.startswith(base):
            if _is_link(Path(lexical)):
                raise HTTPException(status_code=400, detail=_LINK_REJECTED)
            return


def _check_kind(resolved: Path, must_exist: bool, kind: str) -> Path:
    """Apply the reserved-name, existence and kind checks to a contained path."""

    if _has_reserved_name(str(resolved)):
        raise HTTPException(status_code=400, detail=_RESERVED_REJECTED)
    if must_exist and not resolved.exists():
        raise HTTPException(status_code=400, detail="Path not found")
    if resolved.exists():
        if kind == "dir" and not resolved.is_dir():
            raise HTTPException(status_code=400, detail="Path is not a directory")
        if kind == "file" and not resolved.is_file():
            raise HTTPException(status_code=400, detail="Path is not a file")
    return resolved


def resolve_allowed(
    path: str,
    *,
    must_exist: bool,
    kind: Literal["dir", "file", "any"],
    allow_root: bool = True,
    reject_link: bool = False,
) -> Path:
    """Resolve ``path`` (following symlinks) and require it to sit under an allowed root.

    Args:
        path: Caller-supplied path string.
        must_exist: Require the path to exist and match ``kind``.
        kind: ``"dir"``, ``"file"`` or ``"any"`` (only checked when it exists).
        allow_root: When False, a path equal to an allowed root is rejected
            (use before destructive operations such as ``rmtree``). This applies
            to every allowed root, including roots nested inside other roots.
        reject_link: When True, reject a path whose final component is a symlink
            or Windows junction (checked before resolving; use before ``rmtree``).

    Raises:
        HTTPException: 400 for empty, escaping, missing, reserved or wrong-kind paths.
    """

    raw = path.strip() if isinstance(path, str) else ""
    if not raw or "\x00" in raw:
        raise HTTPException(status_code=400, detail="A valid path is required")

    prefixes = [str(root) for root in allowed_roots()]
    if reject_link:
        _reject_link_before_resolution(raw, prefixes)

    canonical = _canonical(raw)
    if canonical is None:
        raise HTTPException(status_code=400, detail="Invalid path")
    if not allow_root and canonical in prefixes:
        raise HTTPException(status_code=400, detail=_REJECTED)

    for prefix in prefixes:
        if canonical.startswith(prefix.rstrip(os.sep) + os.sep):
            return _check_kind(Path(canonical), must_exist, kind)
        if canonical == prefix:
            return _check_kind(Path(prefix), must_exist, kind)
    raise HTTPException(status_code=400, detail=_REJECTED)