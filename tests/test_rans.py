"""Tests for bf16_se_rans codec.

Covers:
  1. rANS encode → rANS decode is lossless (md5 round-trip).
  2. rANS compressed size within 0.1pp of bf16_se_ac (same data).
  3. Codec is registered and first in CODEC_CANDIDATES["bf16"].
  4. Old codec_name "bf16_se_ac" still decodes correctly (backward compat).
  5. compress() default produces bf16_se_rans (new files use rANS).
  6. End-to-end: compress() → decompress() lossless via rANS.
  7. rANS encode on NaN/Inf/denormal edge cases.
"""
from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import pytest


def _md5(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def test_bf16_rans_roundtrip_gaussian():
    """rANS roundtrip on Gaussian-distributed BF16."""
    import torch
    from bigsmall.codecs import bf16_rans
    torch.manual_seed(0)
    for n in [256, 65536, 1 << 20]:
        raw = (torch.randn(n) * 0.02).to(torch.bfloat16).view(torch.uint8).numpy().tobytes()
        blob, _ = bf16_rans.encode(raw)
        out = bf16_rans.decode(blob, {}, n)
        assert _md5(raw) == _md5(out), f"mismatch at n={n}"


def test_bf16_rans_roundtrip_edge_cases():
    """rANS roundtrip on NaN/+Inf/-Inf/denormals."""
    import struct
    import torch
    from bigsmall.codecs import bf16_rans
    specials = [0x7FC0, 0x7F80, 0xFF80, 0x0000, 0x8000, 0x0001, 0x0080, 0x3F80]
    torch.manual_seed(1)
    normals = (torch.randn(4096) * 0.02).to(torch.bfloat16).view(torch.uint8).numpy()
    raw = b"".join(struct.pack("<H", w) for w in specials) + normals.tobytes()
    n = len(raw) // 2
    blob, _ = bf16_rans.encode(raw)
    out = bf16_rans.decode(blob, {}, n)
    assert raw == out


def test_bf16_rans_size_matches_ac():
    """rANS compressed size is within 0.1pp of bf16_se_ac on the same data."""
    import torch
    from bigsmall.codecs import bf16, bf16_rans
    torch.manual_seed(2)
    n = 1 << 20
    raw = (torch.randn(n) * 0.02).to(torch.bfloat16).view(torch.uint8).numpy().tobytes()
    ac_blob, _ = bf16.encode(raw)
    rans_blob, _ = bf16_rans.encode(raw)
    delta_pp = abs(len(rans_blob) - len(ac_blob)) / len(raw) * 100
    assert delta_pp < 0.1, f"rANS vs AC size delta {delta_pp:.4f}pp exceeds 0.1pp"


def test_bf16_se_rans_registered_first():
    """rANS is the first BF16 candidate (wins ties with AC for speed)."""
    from bigsmall import codec_registry
    candidates = codec_registry.CODEC_CANDIDATES["bf16"]
    assert candidates[0] == "bf16_se_rans"
    # AC stays as a fallback so safety-net invariant holds on rare tensors
    # where rANS framing happens to be a few bytes larger.
    assert "bf16_se_ac" in candidates
    assert codec_registry.get_codec("bf16_se_rans") is not None


def test_bf16_se_ac_backward_compat():
    """Files with codec_name 'bf16_se_ac' still decode (backward compat)."""
    try:
        import torch
        import safetensors  # noqa: F401
    except ImportError:
        pytest.skip("torch/safetensors not installed")
    from safetensors import safe_open
    from safetensors.torch import save_file
    from bigsmall import container
    from bigsmall.decoder import _decode_blob
    from bigsmall.codecs import bf16

    torch.manual_seed(3)
    raw = (torch.randn(1024) * 0.02).to(torch.bfloat16).view(torch.uint8).numpy().tobytes()
    # Hand-build a tensor meta dict that uses the AC codec name
    blob, extras = bf16.encode(raw)
    t = {
        "codec": "bf16_se_ac",
        "shape": [1024],
        "extra": extras,
    }
    out = _decode_blob(t, blob)
    assert _md5(raw) == _md5(out), "backward-compat AC decode broken"


def test_compress_default_prefers_rans():
    """compress() default uses bf16_se_rans OR bf16_se_ac (tie-break dependent),
    and the round-trip is lossless either way."""
    try:
        import torch
        import safetensors  # noqa: F401
    except ImportError:
        pytest.skip("torch/safetensors not installed")
    from safetensors.torch import save_file
    import bigsmall
    from bigsmall import container

    torch.manual_seed(4)
    # Multiple tensors of different sizes — at least one should pick rANS.
    sd = {
        "a.weight": (torch.randn(4096, 256) * 0.02).to(torch.bfloat16),
        "b.weight": (torch.randn(2048, 512) * 0.02).to(torch.bfloat16),
        "c.weight": (torch.randn(1024, 1024) * 0.02).to(torch.bfloat16),
        "d.weight": (torch.randn(8192, 128) * 0.02).to(torch.bfloat16),
    }
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "model.safetensors"
        save_file(sd, str(src))
        bs = Path(td) / "model.bs"
        bigsmall.compress(src, bs, workers=1, progress=False)
        header, _ = container.read_header(bs)
        codecs_used = {t["codec"] for t in header["tensors"]}
        # At least one of rANS / AC must be used (no surprise codec picks),
        # and rANS should be selected at least once across these four tensors.
        assert codecs_used <= {"bf16_se_rans", "bf16_se_ac"}, (
            f"unexpected codec(s) in output: {codecs_used}"
        )
        assert "bf16_se_rans" in codecs_used, (
            f"expected at least one tensor to use bf16_se_rans, got {codecs_used}"
        )
        # And lossless on all
        out = bigsmall.decompress(bs)
        from safetensors import safe_open
        with safe_open(str(src), framework="pt") as f:
            for name in f.keys():
                t = f.get_tensor(name)
                src_bytes = t.contiguous().view(torch.uint8).cpu().numpy().tobytes()
                assert _md5(src_bytes) == _md5(out[name].tobytes())
