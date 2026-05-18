"""Multi-worker round-trip and platform-default tests.

Covers two things:
  - `bigsmall.compress(..., workers=N)` produces a byte-identical round-trip
    for N=1 and N=2.
  - `_default_workers()` returns 1 on Windows and >=1 elsewhere unless
    BIGSMALL_WORKERS overrides.
"""
import hashlib
import os
import platform
import tempfile
from pathlib import Path

import pytest


def _md5_hex(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def _state_dict():
    import torch
    torch.manual_seed(0)
    return {
        "embed.weight": torch.randn(2048, 256, dtype=torch.bfloat16),
        "layer.0.weight": torch.randn(512, 512, dtype=torch.bfloat16),
        "layer.1.weight": torch.randn(512, 512, dtype=torch.bfloat16),
        "layer.2.weight": torch.randn(1024, 512, dtype=torch.bfloat16),
    }


@pytest.mark.parametrize("workers", [1, 2])
def test_roundtrip_with_workers(workers):
    try:
        import torch
        import safetensors  # noqa: F401
    except ImportError:
        pytest.skip("torch/safetensors not installed")
    import bigsmall
    from safetensors import safe_open
    from safetensors.torch import save_file

    tensors = _state_dict()
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "model.safetensors"
        save_file(tensors, str(src))
        bs = Path(td) / "model.bs"
        bigsmall.compress(src, bs, workers=workers, progress=False)
        out = bigsmall.decompress(bs, progress=False)

        with safe_open(str(src), framework="pt") as f:
            for name in f.keys():
                t = f.get_tensor(name)
                src_md5 = _md5_hex(t.contiguous().view(torch.uint8).cpu().numpy().tobytes())
                dec_md5 = _md5_hex(out[name].tobytes())
                assert src_md5 == dec_md5, f"{workers} workers: {name} differs"


def test_default_workers_uses_cpu_count():
    """Default worker count: min(cpu_count, 8), works on all platforms.

    v3.7.0 removed the Windows-only hard-coded workers=1 — diagnostics
    confirmed Windows spawn context works correctly and parallel encode
    delivers a 1.8x speedup at workers=4 on real Phi data.
    """
    import os
    from bigsmall import encoder
    os.environ.pop("BIGSMALL_WORKERS", None)
    n = encoder._default_workers()
    cpu = os.cpu_count() or 1
    assert n == min(cpu, 8), (
        f"_default_workers() should be min(cpu_count={cpu}, 8) = {min(cpu,8)}, got {n}"
    )


def test_bigsmall_workers_env_override(monkeypatch):
    """BIGSMALL_WORKERS env var wins on every platform."""
    from bigsmall import encoder
    monkeypatch.setenv("BIGSMALL_WORKERS", "3")
    assert encoder._default_workers() == 3
    monkeypatch.setenv("BIGSMALL_WORKERS", "garbage")
    n = encoder._default_workers()
    assert n >= 1
