"""Background PyPI version check.

Runs once per 24h via a cached `~/.cache/bigsmall/version_check.json` file.
The actual PyPI call happens on a daemon thread so it never blocks the
caller's `import bigsmall`. If anything goes wrong (network failure,
DNS, timeout, parse error) the failure is silent -- the user's program
must never crash because of this check.

The user-facing output is a single one-line warning to `stderr` when an
update is available. We deliberately do not pretty-format anything or
hyperlink, and we never warn when the installed version is greater than
or equal to the latest.
"""
from __future__ import annotations

import json
import os
import sys
import time
import threading
from pathlib import Path
from typing import Optional

PYPI_URL = "https://pypi.org/pypi/bigsmall/json"
CACHE_TTL_SECONDS = 24 * 60 * 60   # 24h
NETWORK_TIMEOUT_SECONDS = 2
ENV_DISABLE = "BIGSMALL_DISABLE_VERSION_CHECK"


def _cache_path() -> Path:
    base = Path(os.environ.get("XDG_CACHE_HOME") or
                Path.home() / ".cache")
    return base / "bigsmall" / "version_check.json"


def _load_cache() -> Optional[dict]:
    p = _cache_path()
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return data


def _save_cache(latest: str) -> None:
    p = _cache_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps({"latest": latest, "checked_at": time.time()}),
            encoding="utf-8",
        )
    except OSError:
        # Read-only home, disk full, etc. -- silent.
        pass


def _fetch_latest_from_pypi(timeout: float = NETWORK_TIMEOUT_SECONDS) -> Optional[str]:
    """Return the latest version string from PyPI, or None on any failure."""
    import urllib.request
    try:
        with urllib.request.urlopen(PYPI_URL, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return payload.get("info", {}).get("version")
    except Exception:
        return None


def _version_tuple(s: str) -> tuple[int, ...]:
    """Parse 'a.b.c' (or with pre-release / dev tags) into a comparable tuple.

    We only need integer pieces; pre-release / post tags collapse to 0 so a
    `2.4.0` cache vs an installed `2.4.0.dev1` won't trigger a spurious
    "update available" warning.
    """
    out: list[int] = []
    for chunk in s.split("."):
        digits = ""
        for ch in chunk:
            if ch.isdigit():
                digits += ch
            else:
                break
        out.append(int(digits) if digits else 0)
    return tuple(out)


def _warn_if_outdated(installed: str, latest: str) -> None:
    if not latest:
        return
    try:
        cur = _version_tuple(installed)
        new = _version_tuple(latest)
    except Exception:
        return
    if new > cur:
        msg = (f"[bigsmall] Update available: {installed} -> {latest}. "
               "Run: pip install --upgrade bigsmall\n")
        try:
            sys.stderr.write(msg)
            sys.stderr.flush()
        except Exception:
            pass


def _check_version_sync(installed: str) -> None:
    """Synchronous body of the version check. Caller wraps in a thread."""
    if os.environ.get(ENV_DISABLE):
        return

    cache = _load_cache()
    if cache:
        checked_at = float(cache.get("checked_at") or 0)
        if (time.time() - checked_at) < CACHE_TTL_SECONDS:
            latest = cache.get("latest")
            if latest:
                _warn_if_outdated(installed, latest)
            return  # cache is fresh, do not hit the network

    latest = _fetch_latest_from_pypi()
    if not latest:
        return
    _save_cache(latest)
    _warn_if_outdated(installed, latest)


def check_version_async(installed: str) -> Optional[threading.Thread]:
    """Kick off the version check on a daemon thread. Returns the thread or
    None if checks are disabled via environment.

    Returning the thread lets tests `.join()` to wait deterministically.
    """
    if os.environ.get(ENV_DISABLE):
        return None
    t = threading.Thread(
        target=_check_version_sync, args=(installed,),
        name="bigsmall-version-check", daemon=True,
    )
    t.start()
    return t
