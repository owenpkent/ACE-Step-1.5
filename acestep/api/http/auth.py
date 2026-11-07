"""API-key authentication helpers shared across HTTP route registration."""

from __future__ import annotations

import hmac
import os
from typing import Any, Optional

from fastapi import Header, HTTPException
from loguru import logger

_api_key: Optional[str] = None
_body_token_enabled_warned = False

BODY_TOKEN_ENV_VAR = "ACESTEP_ALLOW_BODY_API_TOKEN"
MISSING_HEADER_DETAIL = (
    "Missing Authorization header. Sending ai_token in the request body is disabled; "
    "set ACESTEP_ALLOW_BODY_API_TOKEN=true to allow it."
)


def body_api_token_allowed() -> bool:
    """Return True when ``ACESTEP_ALLOW_BODY_API_TOKEN`` enables body ``ai_token`` auth.

    The variable is read at call time. Truthy values are "1", "true", "yes", "on"
    (case-insensitive); anything else or unset means disabled.
    """

    return os.environ.get(BODY_TOKEN_ENV_VAR, "").strip().lower() in ("1", "true", "yes", "on")


def warn_body_token_enabled_once() -> None:
    """Log a once-per-process deprecation warning for the body ``ai_token`` path."""

    global _body_token_enabled_warned
    if not _body_token_enabled_warned:
        _body_token_enabled_warned = True
        logger.warning(
            "{} is enabled: accepting ai_token in request bodies is deprecated because "
            "bodies can leak into logs and proxies. Use the Authorization header.",
            BODY_TOKEN_ENV_VAR,
        )


def warn_body_token_ignored() -> None:
    """Log that a body ``ai_token`` was ignored (never logs the token value)."""

    logger.warning(
        "Ignored ai_token in request body; clients must use the Authorization header. "
        "Set {}=true to allow the deprecated body token.",
        BODY_TOKEN_ENV_VAR,
    )


def _keys_match(candidate: str, expected: str) -> bool:
    """Constant-time comparison of two API keys."""

    return hmac.compare_digest(candidate.encode("utf-8"), expected.encode("utf-8"))


def set_api_key(key: Optional[str]) -> None:
    """Set the process-level API key used by auth verification helpers.

    Args:
        key: API key string, or ``None`` to disable auth enforcement.
    """

    global _api_key
    _api_key = key


def verify_token_from_request(
    body: dict[str, Any], authorization: Optional[str] = None
) -> Optional[str]:
    """Validate request auth from the Authorization header (or body ``ai_token`` if enabled).

    Body ``ai_token`` is ignored unless ``ACESTEP_ALLOW_BODY_API_TOKEN`` is truthy.

    Args:
        body: Parsed request payload dictionary.
        authorization: Optional ``Authorization`` header value.

    Returns:
        Validated token value, or ``None`` when auth is disabled.

    Raises:
        HTTPException: If auth is required and token is missing/invalid.
    """

    if _api_key is None:
        return None

    ai_token = body.get("ai_token") if body else None
    if ai_token:
        if body_api_token_allowed():
            warn_body_token_enabled_once()
            if _keys_match(str(ai_token), _api_key):
                return ai_token
            raise HTTPException(status_code=401, detail="Invalid ai_token")
        warn_body_token_ignored()

    if authorization:
        token = authorization[7:] if authorization.startswith("Bearer ") else authorization
        if _keys_match(token, _api_key):
            return token
        raise HTTPException(status_code=401, detail="Invalid API key")

    raise HTTPException(status_code=401, detail=MISSING_HEADER_DETAIL)


async def verify_api_key(authorization: Optional[str] = Header(None)) -> None:
    """Validate Authorization header for endpoints that do not parse request body.

    Args:
        authorization: Optional ``Authorization`` header value.

    Raises:
        HTTPException: If auth is required and header token is missing/invalid.
    """

    if _api_key is None:
        return

    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    token = authorization[7:] if authorization.startswith("Bearer ") else authorization
    if not _keys_match(token, _api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
