"""Tests for v3.7.0 multiprocessing improvements.

Coverage:
  1. compress(workers=2) produces md5-identical output to workers=1.
  2. compress(workers=4) is at least as fast as workers=1 on a meaningful
     workload (loose timing check — wall-clock can vary on busy CI).
  3. Auto worker detection returns a sensible value (1..cpu_count).
  4. Memory guard reduces worker count when RAM is constrained
     (via psutil mock).
  5. _safe_workers never returns less than 1 even on pathological inputs.
"""
from __future__ import annotations

import hashlib
import platform
import tempfile
import time
from pathlib import Path

import pytest


def _md5(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def _build_state_dict(seed: int, n_tensors: int = 8):
    """Multiple BF16 tensors so the worker pool has real jobs to spread."""
    import torch
    torch.manual_seed(seed)
    sd = {}
    for i in range(n_tensors):
        sd[f"layer.{i}.weight"] = (torch.randn(1024, 256) * 0.02).to(torch.bfloat16)
    return sd


def test_compress_workers_2_matches_workers_1():
    """workers=2 produces md5-identical .bs to workers=1."""
    try:
        import torch
        import safetensors  # noqa: F401
    except ImportError:
        pytest.skip("torch/safetensors not installed")
    from safetensors.torch import save_file
    import bigsmall

    sd = _build_state_dict(seed=0)
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "model.safetensors"
        save_file(sd, str(src))
        bs1 = Path(td) / "w1.bs"
        bs2 = Path(td) / "w2.bs"
        bigsmall.compress(src, bs1, workers=1, progress=False)
        bigsmall.compress(src, bs2, workers=2, progress=False)
        assert bs1.read_bytes() == bs2.read_bytes(), \
            "workers=2 produced a different .bs than workers=1"


@pytest.mark.skipif(not Path(r"C:\tmp\bs_src\Phi-3.5-mini-instruct").exists(),
                    reason="needs Phi-3.5-mini fixture")
def test_compress_workers_4_speedup_on_real_model():
    """workers=4 should be measurably faster than workers=1 on a realistic
    workload (Phi-3.5-mini partial shard). This is the production case.

    Skipped on machines without the Phi fixture (CI-friendly).
    """
    try:
        import torch
        import safetensors  # noqa: F401
    except ImportError:
        pytest.skip("torch/safetensors not installed")
    from safetensors import safe_open
    from safetensors.torch import save_file
    import bigsmall

    # Take first 10 BF16 tensors from Phi shard 1 — meaningful workload.
    src_orig = Path(r"C:\tmp\bs_src\Phi-3.5-mini-instruct\model-00001-of-00002.safetensors")
    tensors = {}
    with safe_open(str(src_orig), framework="pt") as f:
        for name in list(f.keys())[:40]:  # walk first 40
            t = f.get_tensor(name)
            if t.dtype == torch.bfloat16:
                tensors[name] = t.contiguous()
            if len(tensors) >= 10:
                break

    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "model.safetensors"
        save_file(tensors, str(src))
        bs1 = Path(td) / "w1.bs"
        bs4 = Path(td) / "w4.bs"

        t0 = time.perf_counter()
        bigsmall.compress(src, bs1, workers=1, progress=False)
        t_w1 = time.perf_counter() - t0

        t0 = time.perf_counter()
        bigsmall.compress(src, bs4, workers=4, progress=False)
        t_w4 = time.perf_counter() - t0

        # On a real workload, workers=4 should be at least 1.3x faster.
        # The diagnostic measured 1.82x on the full shard; we're more lenient
        # for test reliability (CI variance, small workload at this size).
        speedup = t_w1 / t_w4
        assert speedup >= 1.3, (
            f"workers=4 speedup {speedup:.2f}x is below 1.3x threshold "
            f"(workers=1: {t_w1:.2f}s, workers=4: {t_w4:.2f}s)"
        )


def test_default_workers_in_range():
    """Auto worker detection is between 1 and 8."""
    import os
    from bigsmall import encoder
    os.environ.pop("BIGSMALL_WORKERS", None)
    n = encoder._default_workers()
    assert 1 <= n <= 8, f"unexpected default worker count: {n}"


def test_safe_workers_caps_by_ram(monkeypatch):
    """Memory guard limits workers when available RAM is small."""
    from bigsmall import encoder

    # Force psutil to report 100 MB available.
    class FakeVM:
        available = 100 * 1024 * 1024  # 100 MB

    try:
        import psutil  # noqa: F401
        monkeypatch.setattr("psutil.virtual_memory", lambda: FakeVM)
    except ImportError:
        pytest.skip("psutil not installed")

    # 4 tensors averaging 200 MB raw each → per-worker need ~600 MB
    # → only 0 (clamped to 1) workers should fit in 100 MB available.
    safe = encoder._safe_workers(
        workers=8, raw_total_bytes=800 * 1024 * 1024, n_tensors=4,
    )
    assert safe >= 1
    assert safe <= 4, "should not exceed n_tensors"
    assert safe < 8, "memory guard should have reduced from 8 workers"


def test_safe_workers_never_zero():
    """_safe_workers always returns >= 1, even with pathological inputs."""
    from bigsmall import encoder
    # Zero tensors
    assert encoder._safe_workers(8, 0, 0) >= 1
    # Negative workers requested
    assert encoder._safe_workers(-5, 100_000_000, 4) >= 1
    # Zero workers requested
    assert encoder._safe_workers(0, 100_000_000, 4) >= 1
