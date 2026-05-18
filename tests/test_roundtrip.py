"""Round-trip md5 tests on synthetic safetensors fixtures.

The fixtures are tiny (~10 MB total) BF16 / FP32 / FP16 tensor dicts that we
build, save with safetensors, compress with bigsmall, decompress, and then
check that every tensor's md5 matches the original raw bytes. No real model
files are required.
"""
import hashlib
import tempfile
from pathlib import Path

import pytest


def _md5_hex(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def _make_synthetic(dtype, seed: int = 0) -> dict:
    """Build a small synthetic state-dict for round-trip tests.

    Sizes chosen to give ~10 MB total at bf16 / fp16, ~20 MB at fp32 -- big
    enough to exercise multi-tensor codepaths but trivial to compress.
    """
    import torch
    torch.manual_seed(seed)
    return {
        "embed.weight": torch.randn(4096, 256, dtype=dtype),
        "layer.0.attn.qkv.weight": torch.randn(512, 512, dtype=dtype),
        "layer.0.attn.out.weight": torch.randn(512, 512, dtype=dtype),
        "layer.0.mlp.up.weight": torch.randn(2048, 512, dtype=dtype),
        "layer.0.mlp.down.weight": torch.randn(512, 2048, dtype=dtype),
        "ln_f.weight": torch.randn(512, dtype=dtype),
        "ln_f.bias": torch.zeros(512, dtype=dtype),
    }


def _roundtrip(state_dict: dict):
    import bigsmall
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    with tempfile.TemporaryDirectory() as td:
        src_st = Path(td) / "model.safetensors"
        save_file(state_dict, str(src_st))
        bs_path = Path(td) / "model.bs"
        bigsmall.compress(src_st, bs_path)
        out = bigsmall.decompress(bs_path)

        with safe_open(str(src_st), framework="pt") as f:
            fail = []
            for name in f.keys():
                t = f.get_tensor(name)
                src_bytes = t.contiguous().view(torch.uint8).cpu().numpy().tobytes()
                if _md5_hex(src_bytes) != _md5_hex(out[name].tobytes()):
                    fail.append(name)
            assert not fail, f"md5 mismatch on: {fail[:3]}"


def test_roundtrip_bf16():
    try:
        import torch  # noqa: F401
        import safetensors  # noqa: F401
    except ImportError:
        pytest.skip("torch/safetensors not installed")
    import torch
    _roundtrip(_make_synthetic(torch.bfloat16))


def test_roundtrip_fp16():
    try:
        import torch  # noqa: F401
        import safetensors  # noqa: F401
    except ImportError:
        pytest.skip("torch/safetensors not installed")
    import torch
    _roundtrip(_make_synthetic(torch.float16))


def test_roundtrip_fp32():
    try:
        import torch  # noqa: F401
        import safetensors  # noqa: F401
    except ImportError:
        pytest.skip("torch/safetensors not installed")
    import torch
    _roundtrip(_make_synthetic(torch.float32))
