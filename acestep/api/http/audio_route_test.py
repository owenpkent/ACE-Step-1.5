"""Unit tests for audio route registration."""

import asyncio
import os
import tempfile
import unittest
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.routing import APIRoute

from acestep.api.http.audio_route import register_audio_route


async def _verify_api_key(_authorization: str | None = None) -> None:
    """No-op auth dependency for unit-level route execution."""

    return None


def _get_endpoint(app: FastAPI, path: str, method: str):
    """Return endpoint callable matching route path and method."""

    for route in app.routes:
        if isinstance(route, APIRoute) and route.path == path and method.upper() in route.methods:
            return route.endpoint
    raise AssertionError(f"Missing route: {method} {path}")


class AudioRouteTests(unittest.TestCase):
    """Behavior tests for ``GET /v1/audio`` security and file checks."""

    def test_rejects_path_outside_allowed_dir(self):
        """Audio route should return HTTP 403 for paths outside allowed directory."""

        app = FastAPI()
        allowed_dir = Path.cwd() / "__allowed_audio_dir__"
        app.state.temp_audio_dir = str(allowed_dir)
        register_audio_route(app=app, verify_api_key=_verify_api_key)
        endpoint = _get_endpoint(app, "/v1/audio", "GET")
        request = type("Req", (), {"app": app})()
        outside_path = Path.cwd() / "__outside_audio__.mp3"

        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(endpoint(path=str(outside_path), request=request, _=None))
        self.assertEqual(403, ctx.exception.status_code)

    def test_returns_404_when_file_missing(self):
        """Audio route should return HTTP 404 for missing files inside allowed dir."""

        app = FastAPI()
        allowed_dir = Path.cwd() / "__allowed_audio_dir__"
        app.state.temp_audio_dir = str(allowed_dir)
        register_audio_route(app=app, verify_api_key=_verify_api_key)
        endpoint = _get_endpoint(app, "/v1/audio", "GET")
        request = type("Req", (), {"app": app})()

        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(endpoint(path=str(allowed_dir / "missing.mp3"), request=request, _=None))
        self.assertEqual(404, ctx.exception.status_code)

    def test_returns_404_for_directory_target(self):
        """Audio route should return HTTP 404 when target path is a directory."""

        app = FastAPI()
        app.state.temp_audio_dir = str(Path.cwd())
        register_audio_route(app=app, verify_api_key=_verify_api_key)
        endpoint = _get_endpoint(app, "/v1/audio", "GET")
        request = type("Req", (), {"app": app})()

        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(endpoint(path=str(Path.cwd()), request=request, _=None))
        self.assertEqual(404, ctx.exception.status_code)

    def _endpoint_for(self, allowed_dir: Path):
        """Return the audio endpoint and a fake request bound to ``allowed_dir``."""

        app = FastAPI()
        app.state.temp_audio_dir = str(allowed_dir)
        register_audio_route(app=app, verify_api_key=_verify_api_key)
        return _get_endpoint(app, "/v1/audio", "GET"), type("Req", (), {"app": app})()

    def test_rejects_dot_dot_traversal_out_of_allowed_dir(self):
        """A path that climbs out of the allowed dir with ``..`` should get HTTP 403."""

        with tempfile.TemporaryDirectory() as root:
            allowed = Path(root) / "audio"
            allowed.mkdir()
            (Path(root) / "secret.mp3").write_bytes(b"x")
            endpoint, request = self._endpoint_for(allowed)
            with self.assertRaises(HTTPException) as ctx:
                asyncio.run(endpoint(path=str(allowed / ".." / "secret.mp3"), request=request, _=None))
        self.assertEqual(403, ctx.exception.status_code)

    def test_rejects_sibling_dir_sharing_name_prefix(self):
        """``audio_evil/x.mp3`` must not pass as inside ``audio`` by string prefix."""

        with tempfile.TemporaryDirectory() as root:
            allowed = Path(root) / "audio"
            allowed.mkdir()
            sibling = Path(root) / "audio_evil"
            sibling.mkdir()
            (sibling / "x.mp3").write_bytes(b"x")
            endpoint, request = self._endpoint_for(allowed)
            with self.assertRaises(HTTPException) as ctx:
                asyncio.run(endpoint(path=str(sibling / "x.mp3"), request=request, _=None))
        self.assertEqual(403, ctx.exception.status_code)

    def test_serves_file_inside_allowed_dir_with_media_type(self):
        """A file inside the allowed dir should be served with its extension's media type."""

        with tempfile.TemporaryDirectory() as root:
            allowed = Path(root) / "audio"
            allowed.mkdir()
            target = allowed / "song.FLAC"
            target.write_bytes(b"x")
            endpoint, request = self._endpoint_for(allowed)
            response = asyncio.run(endpoint(path=str(target), request=request, _=None))
        self.assertIsInstance(response, FileResponse)
        self.assertEqual("audio/flac", response.media_type)
        self.assertEqual(os.path.realpath(target), response.path)


if __name__ == "__main__":
    unittest.main()
