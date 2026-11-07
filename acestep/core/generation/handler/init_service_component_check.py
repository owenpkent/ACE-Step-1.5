"""Per-component presence check and download for the main model bundle."""

from pathlib import Path
from typing import Optional, Tuple

from loguru import logger

from acestep.model_downloader import (
    DEFAULT_LM_MODEL,
    MAIN_MODEL_COMPONENTS,
    _contains_model_weights,
    download_main_component,
)


def ensure_main_components(
    checkpoint_path: Path,
    prefer_source: Optional[str],
    require_lm: bool = True,
) -> Optional[Tuple[str, bool]]:
    """Download only the main-model components that are missing on disk.

    Args:
        checkpoint_path: Checkpoints directory.
        prefer_source: Preferred download source or None.
        require_lm: When False, the default LM component is neither checked nor fetched.

    Returns:
        None on success, or an ("ERROR: ...", False) tuple on the first failed download.
    """
    for component in MAIN_MODEL_COMPONENTS:
        if component == DEFAULT_LM_MODEL and not require_lm:
            continue
        if _contains_model_weights(Path(checkpoint_path) / component):
            continue
        logger.info(f"[initialize_service] '{component}' not found, starting auto-download...")
        success, msg = download_main_component(
            component, Path(checkpoint_path), prefer_source=prefer_source
        )
        if not success:
            return f"ERROR: Failed to download main model component '{component}': {msg}", False
        logger.info(f"[initialize_service] {msg}")
    return None
