"""Startup reporting for the path confinement policy.

Logs the directories the training, dataset and LoRA routes may touch, and warns
when one of them is so broad that confinement gives little protection.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from loguru import logger

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
        logger.warning(
            "[path policy] Allowed root {} is a drive root or a home directory, so API callers "
            "can read and write anything beneath it. Start the server from the project "
            "directory, or set {} to the specific data folders instead.",
            root,
            ALLOWED_PATHS_ENV,
        )
    return roots