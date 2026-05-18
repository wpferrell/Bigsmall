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


def test_default_workers_platform_aware(monkeypatch):
    """Default worker count: 1 on Windows, >=1 elsewhere."""
    from bigsmall import encoder
    monkeypatch.delenv("BIGSMALL_WORKERS", raising=False)
    n = encoder._default_workers()
    assert n >= 1
    if platform.system() == "Windows":
        assert n == 1, f"Windows default should be 1, got {n}"


def test_bigsmall_workers_env_override(monkeypatch):
    """BIGSMALL_WORKERS env var wins on every platform."""
    from bigsmall import encoder
    monkeypatch.setenv("BIGSMALL_WORKERS", "3")
    assert encoder._default_workers() == 3
    monkeypatch.setenv("BIGSMALL_WORKERS", "garbage")
    n = encoder._default_workers()
    assert n >= 1
