"""HTTP route for serving generated audio files by path."""

from __future__ import annotations

import os
from typing import Any, Callable

from fastapi import Depends, FastAPI, HTTPException, Request

_MEDIA_TYPES = {
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
}


def register_audio_route(
    app: FastAPI,
    verify_api_key: Callable[..., Any],
) -> None:
    """Register the ``GET /v1/audio`` route."""

    @app.get("/v1/audio")
    async def get_audio(path: str, request: Request, _: None = Depends(verify_api_key)):
        """Serve a generated audio file when path is within the allowed directory.

        The caller path is canonicalised with ``os.path.realpath`` and must pass a
        ``str.startswith`` check against the audio directory before any filesystem
        call, a shape static analysers recognise as a path-injection sanitizer.

        Raises:
            HTTPException: 403 when the path is outside the audio directory, 404 when
                it is not an existing file.
        """

        from fastapi.responses import FileResponse

        resolved_path = os.path.realpath(path)
        allowed_dir = os.path.realpath(request.app.state.temp_audio_dir)
        if resolved_path != allowed_dir and not resolved_path.startswith(allowed_dir + os.sep):
            raise HTTPException(status_code=403, detail="Access denied: path outside allowed directory")
        if not os.path.isfile(resolved_path):
            raise HTTPException(status_code=404, detail="Audio file not found")

        ext = os.path.splitext(resolved_path)[1].lower()
        return FileResponse(resolved_path, media_type=_MEDIA_TYPES.get(ext, "audio/mpeg"))
