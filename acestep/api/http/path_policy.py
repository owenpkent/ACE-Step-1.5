"""Filesystem path confinement for API routes that accept caller-chosen paths.

Training, dataset and LoRA routes take directories and files from the request
body. Every such path must pass through :func:`resolve_allowed` before it is
read, written, scanned or deleted.
"""

from __future__ import annotations

import os
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


def _safe_resolve(raw: str) -> Optional[Path]:
    try:
        return Path(raw).expanduser().resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        return None


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


def resolve_allowed(
    path: str,
    *,
    must_exist: bool,
    kind: Literal["dir", "file", "any"],
    allow_root: bool = True,
) -> Path:
    """Resolve ``path`` (following symlinks) and require it to sit under an allowed root.

    Args:
        path: Caller-supplied path string.
        must_exist: Require the path to exist and match ``kind``.
        kind: ``"dir"``, ``"file"`` or ``"any"`` (only checked when it exists).
        allow_root: When False, a path equal to an allowed root is rejected
            (use before destructive operations such as ``rmtree``).

    Raises:
        HTTPException: 400 for empty, escaping, missing or wrong-kind paths.
    """

    raw = path.strip() if isinstance(path, str) else ""
    if not raw or "\x00" in raw:
        raise HTTPException(status_code=400, detail="A valid path is required")

    resolved = _safe_resolve(raw)
    if resolved is None:
        raise HTTPException(status_code=400, detail="Invalid path")

    for root in allowed_roots():
        try:
            rel = resolved.relative_to(root)
        except ValueError:
            continue
        if not allow_root and not rel.parts:
            continue
        break
    else:
        raise HTTPException(status_code=400, detail=_REJECTED)

    if must_exist and not resolved.exists():
        raise HTTPException(status_code=400, detail="Path not found")
    if resolved.exists():
        if kind == "dir" and not resolved.is_dir():
            raise HTTPException(status_code=400, detail="Path is not a directory")
        if kind == "file" and not resolved.is_file():
            raise HTTPException(status_code=400, detail="Path is not a file")
    return resolved
