"""Validation of the result returned by LyCORIS ``load_weights`` for LoKr adapters."""

from typing import Any

from loguru import logger

_MAX_EXAMPLES = 3


def _extract_keys(result: Any, dict_key: str, attr: str) -> list[str] | None:
    """Return key list from a LyCORIS dict result or torch ``_IncompatibleKeys``, else None."""
    if isinstance(result, dict):
        value = result.get(dict_key, [])
    else:
        value = getattr(result, attr, None)
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return None


def check_lokr_load_result(result: Any, lycoris_net: Any, weights_path: str) -> None:
    """Surface key mismatches reported by ``lycoris_net.load_weights``.

    Args:
        result: Return value of ``load_weights``. LyCORIS returns a dict with optional
            ``"missing keys"`` / ``"unexpected keys"`` entries; an object exposing
            ``missing_keys`` / ``unexpected_keys`` is also accepted. Anything else is ignored.
        lycoris_net: Network whose ``state_dict`` defines the expected keys.
        weights_path: Path of the loaded weights file, used in messages.

    Raises:
        ValueError: If the network expects keys but none of them were matched.
    """
    if result is None:
        return
    missing = _extract_keys(result, "missing keys", "missing_keys")
    unexpected = _extract_keys(result, "unexpected keys", "unexpected_keys")
    if missing is None or unexpected is None:
        return

    total = len(lycoris_net.state_dict())
    matched = total - len(missing)
    if total > 0 and matched == 0:
        raise ValueError(
            f"LoKr weights at {weights_path} have no keys matching the network "
            f"({len(missing)} missing, {len(unexpected)} unexpected); "
            f"the adapter would have no effect. "
            f"Example unexpected keys: {unexpected[:_MAX_EXAMPLES]}"
        )
    if missing or unexpected:
        logger.warning(
            f"LoKr weights at {weights_path} only partially match the network: "
            f"{len(missing)} missing keys (e.g. {missing[:_MAX_EXAMPLES]}), "
            f"{len(unexpected)} unexpected keys (e.g. {unexpected[:_MAX_EXAMPLES]})"
        )
        return
    logger.debug(f"LoKr weights at {weights_path}: all {total} keys matched")


def rollback_lokr_adapter(decoder: Any, lycoris_net: Any, previous_net: Any) -> None:
    """Detach a LyCORIS network whose weights failed to load, restoring the prior state.

    ``apply_to`` has already replaced the decoder's module forwards by the time
    ``load_weights`` runs, so a rejected adapter must be unwound or the decoder is left
    running an unloaded network that ``unload_lora`` cannot see.

    Args:
        decoder: Decoder the network was applied to.
        lycoris_net: The network to detach.
        previous_net: Value of ``decoder._lycoris_net`` before this load, restored as-is.
    """
    restore_fn = getattr(lycoris_net, "restore", None)
    if callable(restore_fn):
        restore_fn()
    else:
        logger.warning("Rejected LoKr network has no restore(); decoder forwards may remain wrapped")
    decoder._lycoris_net = previous_net
    logger.info("Rolled back rejected LoKr adapter; decoder restored to its previous state")

