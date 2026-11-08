"""Console handling for the run.py launcher: keep a double-clicked window open on failure.

Stdlib only, because run.py imports it with the system Python before the venv exists.
"""

import os

# A human shell sharing this console means the output survives our exit: no need to hold.
_SHELLS = frozenset({"cmd.exe", "powershell.exe", "pwsh.exe", "bash.exe", "sh.exe",
                     "zsh.exe", "wt.exe", "windowsterminal.exe", "code.exe"})


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
