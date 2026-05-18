"""Tests for the GPU acceleration kernel (bigsmall.kernels.ac_triton).

Coverage:
  1. CPU encoder → Triton decoder: lossless md5 match.
  2. Triton output bit-identical to CPU decoder output.
  3. CPU fallback when BIGSMALL_FORCE_CPU=1.
  4. End-to-end: compress(gpu_optimised=True) → decompress() works on GPU host.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

import pytest


def _md5(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def _has_cuda_and_triton() -> bool:
    try:
        import torch
        if not torch.cuda.is_available():
            return False
    except Exception:
        return False
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(not _has_cuda_and_triton(), reason="needs CUDA + Triton")
def test_gpu_decode_lossless():
    """Triton decoder returns bit-identical output to CPU decoder."""
    import torch
    from bigsmall.codecs import bf16_parallel
    from bigsmall.kernels import ac_triton
    torch.manual_seed(0)
    n = 1 << 18  # 256K elements (fast test)
    raw = (torch.randn(n) * 0.02).to(torch.bfloat16).view(torch.uint8).numpy().tobytes()
    blob, extras = bf16_parallel.encode(raw, n_streams=64)
    cpu = bf16_parallel.decode(blob, extras, n)
    gpu = ac_triton.decode(blob, extras, n)
    assert _md5(raw) == _md5(cpu), "CPU decode broken"
    assert _md5(raw) == _md5(gpu), "GPU decode broken"
    assert cpu == gpu, "GPU output differs from CPU output"


@pytest.mark.skipif(not _has_cuda_and_triton(), reason="needs CUDA + Triton")
def test_kernels_module_picks_gpu_backend():
    """When CUDA+Triton available, kernels.backend_name() reports a GPU backend."""
    # Clear the cache so we re-probe
    from bigsmall import kernels
    kernels._GPU_BACKEND = None
    kernels._GPU_DECODE_FN = None
    # FORCE_CPU off
    os.environ.pop("BIGSMALL_FORCE_CPU", None)
    assert kernels.use_gpu() is True
    assert kernels.backend_name() in ("triton", "cuda_c")


def test_force_cpu_env_disables_gpu():
    """BIGSMALL_FORCE_CPU=1 disables GPU even when available."""
    from bigsmall import kernels
    # Clear probe cache
    kernels._GPU_BACKEND = None
    kernels._GPU_DECODE_FN = None
    os.environ["BIGSMALL_FORCE_CPU"] = "1"
    try:
        assert kernels.use_gpu() is False
        assert kernels.backend_name() == "cpu"
    finally:
        del os.environ["BIGSMALL_FORCE_CPU"]
        kernels._GPU_BACKEND = None
        kernels._GPU_DECODE_FN = None


def test_compress_gpu_optimised_end_to_end_with_kernel():
    """compress(gpu_optimised=True) → decompress() roundtrips losslessly
    via whatever backend (GPU if available, CPU otherwise)."""
    try:
        import torch
        import safetensors  # noqa: F401
    except ImportError:
        pytest.skip("torch/safetensors not installed")
    from safetensors import safe_open
    from safetensors.torch import save_file
    import bigsmall
    from bigsmall import container

    torch.manual_seed(0)
    sd = {"big.weight": (torch.randn(4096, 1024) * 0.02).to(torch.bfloat16)}
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "model.safetensors"
        save_file(sd, str(src))
        bs = Path(td) / "model.bs"
        bigsmall.compress(src, bs, gpu_optimised=True, workers=1, progress=False)

        header, _ = container.read_header(bs)
        codecs = {t["codec"] for t in header["tensors"]}
        assert "bf16_parallel_v1" in codecs

        out = bigsmall.decompress(bs)
        with safe_open(str(src), framework="pt") as f:
            t = f.get_tensor("big.weight")
            src_bytes = t.contiguous().view(torch.uint8).cpu().numpy().tobytes()
            assert _md5(src_bytes) == _md5(out["big.weight"].tobytes())
