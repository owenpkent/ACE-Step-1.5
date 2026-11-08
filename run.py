#!/usr/bin/env python3
"""Double-click launcher: start ACE-Step and open it in the browser.

Finds uv, runs ``uv sync`` the first time, launches on loopback, opens the page once the
port answers, and keeps a double-clicked window open on failure. Stdlib only.

    py run.py                  # Gradio UI on http://127.0.0.1:7860
    py run.py --api            # REST API server on http://127.0.0.1:8001 instead
    py run.py -- --offload_to_cpu true   # anything after -- goes to the app unchanged

ROCm, Intel XPU and macOS keep their start_* launchers; this is the CUDA/CPU path.
"""

import argparse
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import webbrowser

ROOT = os.path.dirname(os.path.abspath(__file__))
HOST = "127.0.0.1"
UI_PORT = 7860
API_PORT = 8001
# (flag, .env key, fallback). The UI reads these keys only in API/service mode; a
# hardcoded model would download one nobody chose.
UI_DEFAULTS = [("--config_path", "ACESTEP_CONFIG_PATH", "acestep-v15-turbo"),
               ("--lm_model_path", "ACESTEP_LM_MODEL_PATH", "acestep-5Hz-lm-0.6B"),
               ("--init_service", None, "true")]
# Model download and init can take many minutes on a first run.
BROWSER_WAIT_S = 1800.0
# A human shell sharing this console means the output survives our exit: no need to hold.
_SHELLS = frozenset({"cmd.exe", "powershell.exe", "pwsh.exe", "bash.exe", "sh.exe",
                     "zsh.exe", "wt.exe", "windowsterminal.exe", "code.exe"})


def find_uv() -> str | None:
    """Return a path to uv: PATH first, then the installer's usual drop locations."""
    found = shutil.which("uv")
    if found:
        return found
    exe = "uv.exe" if os.name == "nt" else "uv"
    candidates = [os.path.join(os.path.expanduser("~"), ".local", "bin", exe),
                  os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "Microsoft", "WinGet",
                               "Links", exe)]
    return next((c for c in candidates if os.path.isfile(c)), None)


def read_env_file(path: str) -> dict[str, str]:
    """Parse simple ``KEY=value`` lines from a .env file; missing file gives {}."""
    values: dict[str, str] = {}
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                key, sep, value = line.strip().partition("=")
                if sep and key and not key.startswith("#"):
                    value = value.split(" #", 1)[0]  # inline comment, as python-dotenv
                    values[key.strip()] = value.strip().strip("\"'")
    except OSError:
        pass
    return values


def build_command(uv: str, api: bool, port: int | None, extra: list[str],
                  env: dict[str, str] | None = None) -> list[str]:
    """Build the ``uv run`` argv; ``env`` picks the models, ``extra`` is appended as is."""
    env = env or {}
    if api:
        app = ["acestep-api", "--host", HOST, "--port", str(port or API_PORT)]
    else:
        app = ["acestep", "--port", str(port or UI_PORT), "--server-name", HOST]
        given = {arg.split("=", 1)[0] for arg in extra}
        for flag, key, fallback in UI_DEFAULTS:
            if flag not in given:
                app += [flag, env.get(key) or fallback]
    return [uv, "run", "--no-sync", *app, *extra]


def ensure_venv(uv: str) -> bool:
    """Create the environment with ``uv sync`` (retrying offline) when .venv is missing."""
    if os.path.isdir(os.path.join(ROOT, ".venv")):
        return True
    print("First run: setting up the environment with uv sync (takes a few minutes)...")
    if subprocess.call([uv, "sync"], cwd=ROOT) == 0:
        return True
    print("Online sync failed, retrying offline...")
    return subprocess.call([uv, "sync", "--offline"], cwd=ROOT) == 0


def port_open(port: int) -> bool:
    """True when something accepts TCP connections on HOST:port."""
    try:
        with socket.create_connection((HOST, port), timeout=1.0):
            return True
    except OSError:
        return False


def open_when_ready(url: str, port: int, proc: subprocess.Popen) -> None:
    """Open ``url`` once the port answers; give up if the app exits or never comes up."""
    deadline = time.monotonic() + BROWSER_WAIT_S
    while proc.poll() is None and time.monotonic() < deadline:
        if port_open(port):
            webbrowser.open(url)
            return
        time.sleep(1.0)


def console_is_ours() -> bool:
    """True when the console exists only for us (double-clicked), so it closes on exit."""
    if os.name != "nt":
        return False
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        pids = (wintypes.DWORD * 64)()
        count = kernel32.GetConsoleProcessList(pids, len(pids))
        for pid in pids[:min(count, len(pids))]:
            handle = kernel32.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
            if not handle:  # a process we may not open is not a shell of ours
                continue
            try:
                name, size = ctypes.create_unicode_buffer(32768), wintypes.DWORD(32768)
                if kernel32.QueryFullProcessImageNameW(handle, 0, name, ctypes.byref(size)):
                    if os.path.basename(name.value).lower() in _SHELLS:
                        return False
            finally:
                kernel32.CloseHandle(handle)
        return count > 0
    except (OSError, AttributeError):
        return False


def hold_window_open(force: bool) -> None:
    """Pause a double-clicked console so the error above stays readable."""
    if not force and not console_is_ours():
        return
    print("\nRead the message above, then close this window (or press Enter).")
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        pass


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Split our own flags from the pass-through args that follow ``--``."""
    extra: list[str] = []
    if "--" in argv:
        cut = argv.index("--")
        argv, extra = argv[:cut], argv[cut + 1:]
    parser = argparse.ArgumentParser(description="Start ACE-Step (Gradio UI or REST API).")
    parser.add_argument("--api", action="store_true", help="start the REST API server")
    parser.add_argument("--port", type=int, help=f"port (default {UI_PORT}, API {API_PORT})")
    parser.add_argument("--no-browser", action="store_true", help="do not open the page")
    parser.add_argument("--hold", action="store_true", help="always pause before exiting")
    return parser.parse_args(argv), extra


def run(args: argparse.Namespace, extra: list[str]) -> int:
    """Launch the app and wait for it; returns the app's exit code."""
    uv = find_uv()
    if not uv:
        print("uv was not found. Install it with install_uv.bat, or:\n"
              "  winget install --id=astral-sh.uv -e")
        return 1
    if not ensure_venv(uv):
        print("Environment setup failed. Check the connection and disk space, then retry.")
        return 1
    env = {**read_env_file(os.path.join(ROOT, ".env")), **os.environ}
    cmd = build_command(uv, args.api, args.port, extra, env)
    port = args.port or (API_PORT if args.api else UI_PORT)
    url = f"http://{HOST}:{port}" + ("/docs" if args.api else "")
    print(f"Starting ACE-Step {'API server' if args.api else 'Gradio UI'} at {url}")
    print("(Ctrl+C or close this window to stop.)\n", flush=True)
    proc = subprocess.Popen(cmd, cwd=ROOT)
    if not args.no_browser:
        threading.Thread(target=open_when_ready, args=(url, port, proc), daemon=True).start()
    try:
        return proc.wait()
    except KeyboardInterrupt:
        proc.wait()
        return 0


def main() -> int:
    """Entry point: run the app, and hold a double-clicked window open on failure."""
    args, extra = parse_args(sys.argv[1:])
    code = run(args, extra)
    if code != 0 or args.hold:
        hold_window_open(args.hold)
    return code


if __name__ == "__main__":
    sys.exit(main())
