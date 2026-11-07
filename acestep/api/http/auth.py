"""API-key authentication helpers shared across HTTP route registration."""

from __future__ import annotations

import hmac
from typing import Any, Optional

from fastapi import Header, HTTPException

_api_key: Optional[str] = None


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
    """Validate request auth from body ``ai_token`` or Authorization header.

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
        if _keys_match(str(ai_token), _api_key):
            return ai_token
        raise HTTPException(status_code=401, detail="Invalid ai_token")

    if authorization:
        token = authorization[7:] if authorization.startswith("Bearer ") else authorization
        if _keys_match(token, _api_key):
            return token
        raise HTTPException(status_code=401, detail="Invalid API key")

    raise HTTPException(status_code=401, detail="Missing ai_token or Authorization header")


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
