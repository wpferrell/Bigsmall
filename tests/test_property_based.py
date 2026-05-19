"""Property-based testing with Hypothesis (v3.11.0).

These tests sample randomly-shaped tensors of mixed dtypes and check
invariants that should hold for ANY input. Faster than full integration
tests but broader-coverage than hand-crafted unit tests.

Property checks:
  1. Round-trip is lossless on bf16 / fp16 / fp32 tensors of any 1-3D shape.
  2. Auto-select never picks an unregistered codec name.
  3. `verify` (header md5 check) passes on every freshly-compressed file.
  4. Compress + decompress round-trip yields bit-identical bytes when
     the input is not a multi-tensor dict (single-tensor edge cases).
  5. compress_streaming output matches compress() output on the same data.

All property tests are constrained to small tensors (< 256 KB per tensor)
so the full suite completes well under 60 seconds even with 50 examples
per test.
"""
from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import pytest

try:
    from hypothesis import given, settings, strategies as st, HealthCheck
    HYP_OK = True
except ImportError:
    HYP_OK = False

pytestmark = pytest.mark.skipif(not HYP_OK, reason="hypothesis not installed")


# ----------------------------------------------------------------------------
# Strategies — small shapes only so the property suite runs fast.

if HYP_OK:
    # Shapes: 1-3 dimensions, each dim 1..64. Constrains tensor total
    # element count so 50 examples × multiple property tests stays fast.
    shape_strategy = st.lists(
        st.integers(min_value=1, max_value=64),
        min_size=1, max_size=3,
    ).filter(lambda s: 1 <= int(_prod(s)) <= 64 * 64 * 64)

    dtype_strategy = st.sampled_from(["bfloat16", "float16", "float32"])

    # Even shape sums for BF16/FP16 (we need bytes divisible by item_bytes)
    # — torch handles this automatically as long as we go through .randn.


def _prod(seq):
    out = 1
    for x in seq:
        out *= x
    return out


def _save_single_tensor(td: Path, name: str, t) -> Path:
    """Write a one-tensor safetensors file for round-trip testing."""
    from safetensors.torch import save_file
    src = td / "model.safetensors"
    save_file({name: t.contiguous()}, str(src))
    return src


def _md5_tensor_bytes(t) -> str:
    import torch
    raw = t.contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.md5(raw).hexdigest()


# ----------------------------------------------------------------------------
# Property 1: round-trip is lossless on any shape × dtype.

@given(shape=shape_strategy, dtype=dtype_strategy)
@settings(max_examples=30, deadline=None,
          suppress_health_check=[HealthCheck.too_slow] if HYP_OK else [])
def test_property_roundtrip_any_shape_dtype(shape, dtype):
    """compress() → decompress() is lossless for any shape × dtype."""
    import torch
    import bigsmall
    torch_dtype = getattr(torch, dtype)
    t = torch.randn(*shape, dtype=torch.float32).to(torch_dtype)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = _save_single_tensor(td, "w", t)
        bs = td / "out.bs"
        bigsmall.compress(src, bs, workers=1, progress=False)
        out = bigsmall.decompress(bs)
        # Compare bytes round-trip — torch.equal isn't reliable for bf16
        # specials (NaN, etc.); md5 of raw bytes is the strict check.
        src_md5 = _md5_tensor_bytes(t)
        # `out["w"]` is a numpy array (uint16 view for BF16, FP16, etc.)
        dec_md5 = hashlib.md5(out["w"].tobytes()).hexdigest()
        assert src_md5 == dec_md5, f"round-trip mismatch for shape={shape} dtype={dtype}"


# ----------------------------------------------------------------------------
# Property 2: auto_select_codec returns only registered codec names.

@given(shape=shape_strategy, dtype=st.sampled_from(["bfloat16", "float16", "float32"]))
@settings(max_examples=30, deadline=None,
          suppress_health_check=[HealthCheck.too_slow] if HYP_OK else [])
def test_property_auto_select_returns_registered_codec(shape, dtype):
    """`auto_select_codec` never returns a codec name not in the registry."""
    import torch
    from bigsmall import codec_registry
    from bigsmall import formats
    torch_dtype = getattr(torch, dtype)
    t = torch.randn(*shape, dtype=torch.float32).to(torch_dtype)
    raw = t.contiguous().view(torch.uint8).cpu().numpy().tobytes()
    fmt = formats.detect_format_from_dtype(
        {"bfloat16": "BF16", "float16": "F16", "float32": "F32"}[dtype]
    )
    dtype_safetensors = {"bfloat16": "BF16", "float16": "F16", "float32": "F32"}[dtype]
    blob, codec_name, _extras = codec_registry.auto_select_codec(
        raw, fmt=fmt, dtype=dtype_safetensors,
    )
    # Either a registered codec, or one of the known-inline fallback names.
    known_inline = {"raw", "zstd"}
    if codec_name not in known_inline:
        assert codec_registry.get_codec(codec_name) is not None, (
            f"auto_select returned unknown codec {codec_name!r}"
        )


# ----------------------------------------------------------------------------
# Property 3: verify_fast passes on every freshly-compressed file.

@given(shape=shape_strategy, dtype=dtype_strategy)
@settings(max_examples=20, deadline=None,
          suppress_health_check=[HealthCheck.too_slow] if HYP_OK else [])
def test_property_verify_fast_passes_after_compress(shape, dtype):
    """Every file produced by compress() passes verify_fast() with no problems."""
    import torch
    import bigsmall
    from bigsmall.verify import verify_fast
    torch_dtype = getattr(torch, dtype)
    t = torch.randn(*shape, dtype=torch.float32).to(torch_dtype)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = _save_single_tensor(td, "w", t)
        bs = td / "out.bs"
        bigsmall.compress(src, bs, workers=1, progress=False)
        ok, problems = verify_fast(bs)
        assert ok, f"verify_fast failed on freshly-compressed file: {problems}"
        assert problems == [], f"unexpected problems: {problems}"


# ----------------------------------------------------------------------------
# Property 4: compressed file is never inadvertently corrupt for any shape.

@given(shape=shape_strategy, dtype=dtype_strategy)
@settings(max_examples=20, deadline=None,
          suppress_health_check=[HealthCheck.too_slow] if HYP_OK else [])
def test_property_full_verify_passes(shape, dtype):
    """The full md5 verify() also passes on every freshly-compressed file."""
    import torch
    import bigsmall
    torch_dtype = getattr(torch, dtype)
    t = torch.randn(*shape, dtype=torch.float32).to(torch_dtype)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = _save_single_tensor(td, "w", t)
        bs = td / "out.bs"
        bigsmall.compress(src, bs, workers=1, progress=False)
        ok = bigsmall.verify(bs)
        assert ok, f"verify() failed on freshly-compressed file shape={shape} dtype={dtype}"


# ----------------------------------------------------------------------------
# Property 5: compress_streaming output md5-matches compress() output.

@given(shape=shape_strategy, dtype=dtype_strategy)
@settings(max_examples=15, deadline=None,
          suppress_health_check=[HealthCheck.too_slow] if HYP_OK else [])
def test_property_streaming_matches_standard(shape, dtype):
    """compress_streaming output is byte-identical to compress() output.

    Holds for any non-tied single-tensor model (the streaming path's main
    limitation is no cross-tensor tied-weight dedup; with one tensor that
    can't apply).
    """
    import torch
    import bigsmall
    torch_dtype = getattr(torch, dtype)
    t = torch.randn(*shape, dtype=torch.float32).to(torch_dtype)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = _save_single_tensor(td, "w", t)
        bs_std = td / "std.bs"
        bs_stream = td / "stream.bs"
        bigsmall.compress(src, bs_std, workers=1, progress=False)
        bigsmall.compress_streaming(src, bs_stream, progress=False)
        std_bytes = bs_std.read_bytes()
        stream_bytes = bs_stream.read_bytes()
        assert std_bytes == stream_bytes, (
            f"compress_streaming differs from compress() on shape={shape} "
            f"dtype={dtype}: std={len(std_bytes)} stream={len(stream_bytes)}"
        )
