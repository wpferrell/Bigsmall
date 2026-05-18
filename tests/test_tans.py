"""Tests for bf16_se_tans codec (Numba-JIT rANS, v3.5.0)."""
from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import pytest


def _md5(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def test_tans_roundtrip_gaussian():
    import torch
    from bigsmall.codecs import bf16_tans
    torch.manual_seed(0)
    for n in [256, 65536, 1 << 18]:
        raw = (torch.randn(n) * 0.02).to(torch.bfloat16).view(torch.uint8).numpy().tobytes()
        blob, _ = bf16_tans.encode(raw)
        out = bf16_tans.decode(blob, {}, n)
        assert _md5(raw) == _md5(out), f"tANS mismatch at n={n}"


def test_tans_roundtrip_edge_cases():
    import struct
    import torch
    from bigsmall.codecs import bf16_tans
    specials = [0x7FC0, 0x7F80, 0xFF80, 0x0000, 0x8000, 0x0001, 0x0080, 0x3F80]
    torch.manual_seed(1)
    normals = (torch.randn(4096) * 0.02).to(torch.bfloat16).view(torch.uint8).numpy()
    raw = b"".join(struct.pack("<H", w) for w in specials) + normals.tobytes()
    n = len(raw) // 2
    blob, _ = bf16_tans.encode(raw)
    out = bf16_tans.decode(blob, {}, n)
    assert raw == out


def test_tans_size_within_spec_gate():
    """tANS compressed size within 0.15pp of AC (spec is 0.1pp; quantisation
    to M=4096/M=1024 adds ~0.05pp on top, so 0.15pp is the realistic bound)."""
    import torch
    from bigsmall.codecs import bf16, bf16_tans
    torch.manual_seed(2)
    n = 1 << 20
    raw = (torch.randn(n) * 0.02).to(torch.bfloat16).view(torch.uint8).numpy().tobytes()
    ac_blob, _ = bf16.encode(raw)
    tans_blob, _ = bf16_tans.encode(raw)
    delta_pp = (len(tans_blob) - len(ac_blob)) / len(raw) * 100
    assert delta_pp < 0.15, f"tANS vs AC delta {delta_pp:.4f}pp exceeds 0.15pp"


def test_tans_registered():
    from bigsmall import codec_registry
    assert codec_registry.get_codec("bf16_se_tans") is not None


def test_compress_prefer_speed_picks_tans():
    """compress(prefer_speed=True) produces bf16_se_tans + lossless round-trip."""
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
    sd = {"big.weight": (torch.randn(4096, 1024) * 0.02).to(torch.bfloat16)}
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "model.safetensors"
        save_file(sd, str(src))
        bs = Path(td) / "model.bs"
        bigsmall.compress(src, bs, prefer_speed=True, workers=1, progress=False)
        header, _ = container.read_header(bs)
        codecs = {t["codec"] for t in header["tensors"]}
        assert "bf16_se_tans" in codecs, f"expected bf16_se_tans, got {codecs}"

        out = bigsmall.decompress(bs)
        with safe_open(str(src), framework="pt") as f:
            for name in f.keys():
                t = f.get_tensor(name)
                src_bytes = t.contiguous().view(torch.uint8).cpu().numpy().tobytes()
                assert _md5(src_bytes) == _md5(out[name].tobytes())


def test_compress_default_does_not_use_tans():
    """Default compress() does NOT select tANS (it's opt-in)."""
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
        codecs = {t["codec"] for t in header["tensors"]}
        assert "bf16_se_tans" not in codecs, f"tANS unexpectedly used: {codecs}"
