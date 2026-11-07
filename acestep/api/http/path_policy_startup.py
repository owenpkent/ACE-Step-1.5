"""Startup reporting for the path confinement policy.

Logs the directories the training, dataset and LoRA routes may touch, and warns
when one of them is so broad that confinement gives little protection.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from loguru import logger

from acestep.api.http import path_policy
from acestep.api.http.path_policy import ALLOWED_PATHS_ENV, allowed_roots


def _home_directory() -> Path | None:
    """Return the resolved home directory, or None when it cannot be determined."""
    try:
        return Path.home().resolve()
    except (OSError, RuntimeError):
        return None


def is_broad_root(root: Path) -> bool:
    """Return True if ``root`` is a drive or filesystem root, or contains the home directory.

    Such a root lets any API caller read and write everything a user owns, so
    confinement to it is close to no confinement at all.
    """
    if root.parent == root:
        return True
    home = _home_directory()
    if home is None:
        return False
    return root == home or root in home.parents


def broad_roots(roots: List[Path]) -> List[Path]:
    """Return the subset of ``roots`` that :func:`is_broad_root` flags."""
    return [root for root in roots if is_broad_root(root)]


def root_origins(root: Path) -> List[str]:
    """Return which settings produced ``root``: startup directory, project root, env vars."""
    origins: List[str] = []
    if root == path_policy._STARTUP_CWD:
        origins.append("startup working directory")
    if root == path_policy._PROJECT_ROOT:
        origins.append("project directory")
    tmpdir = (os.getenv("ACESTEP_TMPDIR") or "").strip()
    if tmpdir and path_policy._safe_resolve(tmpdir) == root:
        origins.append("ACESTEP_TMPDIR")
    for entry in (os.getenv(ALLOWED_PATHS_ENV) or "").split(os.pathsep):
        if entry.strip() and path_policy._safe_resolve(entry.strip()) == root:
            origins.append(ALLOWED_PATHS_ENV)
            break
    return origins


_REMEDIES = {
    "startup working directory": (
        "Restart the server from the project directory or another narrow folder; the "
        "startup directory is always an allowed root."
    ),
    "project directory": "Install the project in its own folder rather than a home or drive root.",
    "ACESTEP_TMPDIR": "Point ACESTEP_TMPDIR at a dedicated folder.",
    ALLOWED_PATHS_ENV: f"List specific data folders in {ALLOWED_PATHS_ENV} instead of a broad one.",
}


_BUILT_IN_ORIGINS = ("startup working directory", "project directory")


def _remedy(origins: List[str]) -> str:
    """Return the remedy sentence(s) for the given origins.

    The caveat that ``ACESTEP_ALLOWED_PATHS`` cannot remove the root applies only to
    the built-in roots; a root supplied by an environment setting goes away when that
    setting is narrowed.
    """
    parts = [_REMEDIES[origin] for origin in origins if origin in _REMEDIES]
    if not parts:
        parts.append("Narrow the setting that produced it.")
    if any(origin in _BUILT_IN_ORIGINS for origin in origins):
        parts.append(
            f"{ALLOWED_PATHS_ENV} adds to these built-in roots and cannot remove this one."
        )
    return " ".join(parts)


def log_allowed_roots() -> List[Path]:
    """Log the allowed roots at startup and warn about broad ones.

    Returns:
        The roots that were logged, for callers that want to report them elsewhere.
    """
    roots = allowed_roots()
    logger.info(
        "[path policy] Training, dataset and LoRA routes may access: {}",
        "; ".join(str(root) for root in roots),
    )
    for root in broad_roots(roots):
        origins = root_origins(root)
        logger.warning(
            "[path policy] Allowed root {} ({}) is a drive root or a home directory, so API "
            "callers can read and write anything beneath it. {}",
            root,
            ", ".join(origins) or "unknown origin",
            _remedy(origins),
        )
    return roots