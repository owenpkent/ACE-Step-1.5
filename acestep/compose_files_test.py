"""Regression checks for the Docker Compose files' network and auth settings."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parent.parent
_COMPOSE_FILES = ("docker-compose.yml", "docker-compose.jetson.yml")
_VAR = re.compile(r"\$\{(\w+):-([^}]*)\}")


def _interpolate(value: str, env: dict) -> str:
    """Resolve ``${NAME:-default}`` the way Compose does for set/unset variables."""
    return _VAR.sub(lambda m: env.get(m.group(1)) or m.group(2), value)


def _service(name: str) -> dict:
    """Return the single service defined in a compose file."""
    data = yaml.safe_load((_ROOT / name).read_text(encoding="utf-8"))
    (service,) = data["services"].values()
    return service


def _environment(service: dict, env: dict) -> dict:
    """Return the service environment after interpolation, as a dict."""
    pairs = (_interpolate(item, env).split("=", 1) for item in service["environment"])
    return {key: value for key, value in pairs}


class ComposeFilesTests(unittest.TestCase):
    """Both compose files keep ports local by default and forward the API key."""

    def test_ports_default_to_loopback(self):
        """Without PUBLISH_HOST, every published port binds 127.0.0.1."""
        for name in _COMPOSE_FILES:
            with self.subTest(name=name):
                for port in _service(name)["ports"]:
                    self.assertTrue(_interpolate(port, {}).startswith("127.0.0.1:"), port)

    def test_publish_host_opens_ports(self):
        """PUBLISH_HOST=0.0.0.0 publishes on every interface."""
        for name in _COMPOSE_FILES:
            with self.subTest(name=name):
                for port in _service(name)["ports"]:
                    resolved = _interpolate(port, {"PUBLISH_HOST": "0.0.0.0"})
                    self.assertTrue(resolved.startswith("0.0.0.0:"), port)

    def test_shell_api_key_reaches_container(self):
        """A key exported in the shell (or .env) lands in the container environment."""
        for name in _COMPOSE_FILES:
            with self.subTest(name=name):
                env = _environment(_service(name), {"ACESTEP_API_KEY": "k123"})
                self.assertEqual("k123", env["ACESTEP_API_KEY"])

    def test_unset_api_key_is_empty(self):
        """Non-target: with no key set, the container gets an empty value (auth off)."""
        for name in _COMPOSE_FILES:
            with self.subTest(name=name):
                self.assertEqual("", _environment(_service(name), {})["ACESTEP_API_KEY"])


if __name__ == "__main__":
    unittest.main()
