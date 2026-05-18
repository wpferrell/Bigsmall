"""Version-check tests.

Covers:
  1. Newer version on PyPI -> one-line warning to stderr.
  2. Same version (or newer locally) -> no warning.
  3. PyPI unreachable -> no crash, no warning.
  4. Fresh cache -> no network call.

We never let the check touch the real PyPI from the test suite -- everything
either calls `_check_version_sync(installed)` directly with a monkey-patched
fetcher or runs the daemon thread with the disable env set.
"""
import io
import json
import sys
import time
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_cache(monkeypatch, tmp_path):
    """Redirect the version-check cache to a tmpdir AND wipe any cache file
    that might exist there. The daemon thread fired at `import bigsmall` can
    write a cache file before the test runs; we delete it so each test
    starts from a known empty state."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    # Also redirect HOME on Windows (Path.home() uses USERPROFILE there).
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    # Make sure the disable env is NOT set (some tests set it explicitly).
    monkeypatch.delenv("BIGSMALL_DISABLE_VERSION_CHECK", raising=False)

    # Defensive: wipe the cache file in tmp_path AND in the real user cache.
    # The import-time daemon thread might have written to either depending on
    # when env vars took effect.
    from bigsmall import _version_check
    for p in [_version_check._cache_path(),
              Path.home() / ".cache" / "bigsmall" / "version_check.json"]:
        if p.exists():
            try:
                p.unlink()
            except OSError:
                pass
    yield


def test_warns_when_newer_version_available(monkeypatch, capsys):
    from bigsmall import _version_check

    monkeypatch.setattr(_version_check, "_fetch_latest_from_pypi",
                        lambda timeout=2: "99.0.0")
    _version_check._check_version_sync("2.4.0")

    err = capsys.readouterr().err
    assert "Update available: 2.4.0 -> 99.0.0" in err
    assert "pip install --upgrade bigsmall" in err


def test_no_warning_when_same_or_newer(monkeypatch, capsys):
    from bigsmall import _version_check

    # PyPI says same version
    monkeypatch.setattr(_version_check, "_fetch_latest_from_pypi",
                        lambda timeout=2: "2.4.0")
    _version_check._check_version_sync("2.4.0")
    assert capsys.readouterr().err == ""

    # PyPI says older than installed (e.g. pre-release locally)
    monkeypatch.setattr(_version_check, "_fetch_latest_from_pypi",
                        lambda timeout=2: "2.3.0")
    # Wipe cache so the second call re-checks.
    cache = _version_check._cache_path()
    if cache.exists():
        cache.unlink()
    _version_check._check_version_sync("2.4.0")
    assert capsys.readouterr().err == ""


def test_pypi_unreachable_silent(monkeypatch, capsys):
    from bigsmall import _version_check

    monkeypatch.setattr(_version_check, "_fetch_latest_from_pypi",
                        lambda timeout=2: None)
    # Must not raise.
    _version_check._check_version_sync("2.4.0")
    assert capsys.readouterr().err == ""


def test_fresh_cache_skips_network(monkeypatch, capsys):
    from bigsmall import _version_check

    # Pre-seed the cache with a recent timestamp.
    _version_check._save_cache("2.4.0")

    sentinel = {"called": False}

    def _should_not_be_called(timeout=2):
        sentinel["called"] = True
        return "99.0.0"

    monkeypatch.setattr(_version_check, "_fetch_latest_from_pypi",
                        _should_not_be_called)
    _version_check._check_version_sync("2.4.0")
    assert sentinel["called"] is False, "Network was called despite fresh cache"
    # Cache had 2.4.0 latest and we're on 2.4.0 -- no warning expected.
    assert capsys.readouterr().err == ""


def test_disable_env_short_circuits(monkeypatch, capsys):
    from bigsmall import _version_check
    monkeypatch.setenv("BIGSMALL_DISABLE_VERSION_CHECK", "1")

    sentinel = {"called": False}

    def _should_not_be_called(timeout=2):
        sentinel["called"] = True
        return "99.0.0"

    monkeypatch.setattr(_version_check, "_fetch_latest_from_pypi",
                        _should_not_be_called)
    thread = _version_check.check_version_async("2.4.0")
    assert thread is None
    assert sentinel["called"] is False
    assert capsys.readouterr().err == ""


def test_async_thread_is_daemon(monkeypatch):
    """The version check thread MUST be a daemon so it doesn't keep the
    process alive after the user's program exits."""
    from bigsmall import _version_check

    monkeypatch.setattr(_version_check, "_fetch_latest_from_pypi",
                        lambda timeout=2: "2.4.0")
    t = _version_check.check_version_async("2.4.0")
    assert t is not None
    assert t.daemon, "version-check thread must be a daemon"
    t.join(timeout=5)
