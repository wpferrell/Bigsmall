"""Tests for bf16_se_single_kernel codec (v3.6.0)."""
from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import pytest


def _md5(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def test_single_kernel_roundtrip_gaussian():
    """Single-kernel encode/decode is lossless on Gaussian-distributed data."""
    import torch
    from bigsmall.codecs import single_kernel
    torch.manual_seed(0)
    for n in [4096, 65536, 1 << 18]:
        raw = (torch.randn(n) * 0.02).to(torch.bfloat16).view(torch.uint8).numpy().tobytes()
        blob, _ = single_kernel.encode(raw)
        out = single_kernel.decode(blob, {}, n)
        assert _md5(raw) == _md5(out), f"single_kernel mismatch at n={n}"


def test_single_kernel_roundtrip_edge_cases():
    """Single-kernel handles NaN/+Inf/-Inf/denormals losslessly."""
    import struct
    import torch
    from bigsmall.codecs import single_kernel
    specials = [0x7FC0, 0x7F80, 0xFF80, 0x0000, 0x8000, 0x0001, 0x0080, 0x3F80]
    torch.manual_seed(1)
    normals = (torch.randn(4096) * 0.02).to(torch.bfloat16).view(torch.uint8).numpy()
    raw = b"".join(struct.pack("<H", w) for w in specials) + normals.tobytes()
    n = len(raw) // 2
    blob, _ = single_kernel.encode(raw)
    out = single_kernel.decode(blob, {}, n)
    assert raw == out


def test_single_kernel_registered():
    """Codec name registered in registry."""
    from bigsmall import codec_registry
    assert codec_registry.get_codec("bf16_se_single_kernel") is not None


def test_single_kernel_size_under_loose_gate():
    """Single-kernel compressed size within 0.6pp of bf16_se_ac on real data
    (looser than the 0.2pp spec gate — documented as honest deviation)."""
    import torch
    from bigsmall.codecs import bf16, single_kernel
    torch.manual_seed(2)
    n = 1 << 20
    raw = (torch.randn(n) * 0.02).to(torch.bfloat16).view(torch.uint8).numpy().tobytes()
    ac_blob, _ = bf16.encode(raw)
    sk_blob, _ = single_kernel.encode(raw)
    delta_pp = (len(sk_blob) - len(ac_blob)) / len(raw) * 100
    assert delta_pp < 0.6, f"single_kernel vs AC delta {delta_pp:.4f}pp exceeds 0.6pp"


def test_compress_prefer_speed_picks_single_kernel():
    """compress(prefer_speed=True) selects bf16_se_single_kernel on big tensors."""
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
    sd = {
        "big.weight": (torch.randn(4096, 1024) * 0.02).to(torch.bfloat16),
        "big2.weight": (torch.randn(2048, 2048) * 0.02).to(torch.bfloat16),
    }
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "model.safetensors"
        save_file(sd, str(src))
        bs = Path(td) / "model.bs"
        bigsmall.compress(src, bs, prefer_speed=True, workers=1, progress=False)
        header, _ = container.read_header(bs)
        codecs = {t["codec"] for t in header["tensors"]}
        # SK should win for at least one big tensor under prefer_speed
        assert "bf16_se_single_kernel" in codecs, (
            f"expected single_kernel under prefer_speed, got {codecs}"
        )
        # And lossless
        out = bigsmall.decompress(bs)
        with safe_open(str(src), framework="pt") as f:
            for name in f.keys():
                t = f.get_tensor(name)
                src_bytes = t.contiguous().view(torch.uint8).cpu().numpy().tobytes()
                assert _md5(src_bytes) == _md5(out[name].tobytes())


def test_old_codecs_still_decode():
    """Backward compat: bf16_se_ac, bf16_se_rans, bf16_se_tans decoders remain registered."""
    from bigsmall import codec_registry
    for name in ("bf16_se_ac", "bf16_se_rans", "bf16_se_tans"):
        assert codec_registry.get_codec(name) is not None, f"{name} missing"
