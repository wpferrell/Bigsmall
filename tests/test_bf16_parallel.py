"""Tests for the `bf16_parallel_v1` codec (GPU-kernel infrastructure).

The codec emits N parallel AC streams that share one (sign,exp) and one
per-exp mantissa probability model. Coverage:

  1. Lossless round-trip on Gaussian-distributed BF16 (md5 match).
  2. Lossless on edge cases: NaN, +/-Inf, denormals, all-zero.
  3. Codec is in the registry under `bf16_parallel_v1`.
  4. Ratio cost on large synthetic tensors: < 0.5pp at N=256 vs `bf16_se_ac`.
  5. `compress(gpu_optimised=True)` selects the parallel codec on big BF16
     tensors and the resulting .bs file decompresses bit-identically.
  6. `compress()` default (gpu_optimised=False) does NOT select the codec.
"""
from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import pytest


def _md5(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def test_bf16_parallel_lossless_gaussian():
    import torch
    from bigsmall.codecs import bf16_parallel
    torch.manual_seed(0)
    n = 1 << 20
    raw = (torch.randn(n) * 0.02).to(torch.bfloat16).view(torch.uint8).numpy().tobytes()
    for ns in [1, 32, 128, 256]:
        blob, extras = bf16_parallel.encode(raw, n_streams=ns)
        out = bf16_parallel.decode(blob, extras, n)
        assert _md5(raw) == _md5(out), f"mismatch at N={ns}"


def test_bf16_parallel_lossless_edge_cases():
    import struct
    import torch
    from bigsmall.codecs import bf16_parallel
    # Build a tensor mixing NaN, +/-Inf, denormals, zeros, and normals
    special_words = [
        0x7FC0,  # NaN
        0x7F80,  # +Inf
        0xFF80,  # -Inf
        0x0000,  # +0
        0x8000,  # -0
        0x0001,  # smallest denormal
        0x0080,  # smallest normal
        0x3F80,  # 1.0
    ]
    torch.manual_seed(1)
    n_normal = 4096
    normals = (torch.randn(n_normal) * 0.02).to(torch.bfloat16).view(torch.uint8).numpy()
    raw = b"".join(struct.pack("<H", w) for w in special_words) + normals.tobytes()
    n = len(raw) // 2
    for ns in [1, 32, 128]:
        blob, extras = bf16_parallel.encode(raw, n_streams=ns)
        out = bf16_parallel.decode(blob, extras, n)
        assert raw == out, f"byte mismatch at N={ns}"


def test_bf16_parallel_codec_in_registry():
    from bigsmall import codec_registry
    pair = codec_registry.get_codec("bf16_parallel_v1")
    assert pair is not None, "bf16_parallel_v1 not registered"
    assert "bf16_parallel_v1" in codec_registry.registered_names()


def test_bf16_parallel_ratio_cost_within_spec():
    """On a large synthetic BF16 tensor, N=256 must cost < 0.5pp vs bf16_se_ac."""
    import torch
    from bigsmall.codecs import bf16, bf16_parallel
    torch.manual_seed(2)
    n = 1 << 23  # 8M elements
    raw = (torch.randn(n) * 0.02).to(torch.bfloat16).view(torch.uint8).numpy().tobytes()
    baseline_blob, _ = bf16.encode(raw)
    parallel_blob, _ = bf16_parallel.encode(raw, n_streams=256)
    delta_pp = (len(parallel_blob) - len(baseline_blob)) / len(raw) * 100
    assert delta_pp < 0.5, (
        f"N=256 ratio cost {delta_pp:.4f}pp exceeds 0.5pp spec gate"
    )


def test_compress_gpu_optimised_uses_parallel_codec():
    """End-to-end: compress(gpu_optimised=True) → bf16_parallel_v1 + lossless."""
    try:
        import torch
        import safetensors  # noqa: F401
    except ImportError:
        pytest.skip("torch/safetensors not installed")
    from safetensors import safe_open
    from safetensors.torch import save_file
    import bigsmall
    from bigsmall import container

    torch.manual_seed(3)
    # One big BF16 tensor so the +1% tolerance triggers in favour of parallel.
    sd = {
        "big.weight": (torch.randn(4096, 1024) * 0.02).to(torch.bfloat16),
    }
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "model.safetensors"
        save_file(sd, str(src))
        bs = Path(td) / "model.bs"
        bigsmall.compress(src, bs, gpu_optimised=True, workers=1, progress=False)
        # Header must show bf16_parallel_v1 picked for the big tensor.
        header, _ = container.read_header(bs)
        codecs_used = {t["codec"] for t in header["tensors"]}
        assert "bf16_parallel_v1" in codecs_used, (
            f"expected bf16_parallel_v1 in codecs_used={codecs_used}"
        )
        # Decompress and verify lossless.
        out = bigsmall.decompress(bs)
        with safe_open(str(src), framework="pt") as f:
            for name in f.keys():
                t = f.get_tensor(name)
                src_bytes = t.contiguous().view(torch.uint8).cpu().numpy().tobytes()
                assert _md5(src_bytes) == _md5(out[name].tobytes()), (
                    f"md5 mismatch on {name}"
                )


def test_compress_default_does_not_use_parallel():
    """Default compress() does NOT pick bf16_parallel — preserves prior behaviour."""
    try:
        import torch
        import safetensors  # noqa: F401
    except ImportError:
        pytest.skip("torch/safetensors not installed")
    from safetensors.torch import save_file
    import bigsmall
    from bigsmall import container

    torch.manual_seed(4)
    sd = {"big.weight": (torch.randn(4096, 1024) * 0.02).to(torch.bfloat16)}
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "model.safetensors"
        save_file(sd, str(src))
        bs = Path(td) / "model.bs"
        bigsmall.compress(src, bs, workers=1, progress=False)  # default
        header, _ = container.read_header(bs)
        codecs_used = {t["codec"] for t in header["tensors"]}
        assert "bf16_parallel_v1" not in codecs_used, (
            f"unexpected parallel codec in default mode: {codecs_used}"
        )
