"""Cross-layer XOR delta codec tests (V4 B2).

What this file proves:

1. `encode_group` -> `decode_group` round-trips exactly on a chain of
   arbitrary-content byte buffers, regardless of whether the bytes are
   numerically meaningful.

2. The pair API (`encode_pair` / `decode_pair`) round-trips and the
   recorded `delta_from` extras key is wired through.

3. XOR-delta length-mismatch + edge cases (NaN, Inf, denormals, large
   magnitudes) all survive intact -- the codec is purely byte-level XOR
   so the input distribution is irrelevant for correctness.

4. The single-tensor delta blob CAN be larger than the standalone blob
   (Session B finding: lossless XOR doesn't reduce entropy across
   transformer layers). The test asserts that the codec returns *valid*
   blobs in both directions, not that they're smaller -- the encoder's
   safety net is responsible for picking the smaller blob.
"""
import hashlib

import pytest


def _md5_hex(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def _bf16_raw_bytes(t):
    import torch
    return t.contiguous().view(torch.uint8).cpu().numpy().tobytes()


def _normal_bf16(n_elements: int, seed: int = 0, scale: float = 0.05):
    import torch
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n_elements, generator=g) * scale).to(torch.bfloat16)


def test_cross_layer_delta_group_roundtrip():
    """A 5-element chain encodes and decodes byte-identically."""
    try:
        import torch  # noqa: F401
    except ImportError:
        pytest.skip("torch not installed")
    from bigsmall.codecs import cross_layer_delta

    raws = [_bf16_raw_bytes(_normal_bf16(100_000, seed=i)) for i in range(5)]
    md5s = [_md5_hex(r) for r in raws]
    blobs = cross_layer_delta.encode_group(raws)
    decoded = cross_layer_delta.decode_group(blobs, n_weights_each=100_000)
    assert len(decoded) == 5
    for i, (orig_md5, dec_bytes) in enumerate(zip(md5s, decoded)):
        assert _md5_hex(dec_bytes) == orig_md5, f"layer {i} mismatch"


def test_cross_layer_delta_edge_cases():
    """NaN, Inf, denormals, large magnitudes survive XOR round-trip."""
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed")
    from bigsmall.codecs import cross_layer_delta

    bulk = _normal_bf16(100_000, seed=42)
    specials = torch.tensor(
        [0.0, -0.0, float("inf"), float("-inf"), float("nan"),
         1.0, -1.0, 1e-38, -1e-38, 65000.0, -65000.0,
         float("nan"), float("inf"), 0.5, -0.5, 0.25, -0.25, 0.125],
        dtype=torch.bfloat16,
    )
    # Pad to multiple of bulk length so the test is small / fast.
    t0 = torch.cat([bulk, specials], dim=0)
    t1 = torch.cat([bulk * 2.0, specials], dim=0).to(torch.bfloat16)
    raws = [_bf16_raw_bytes(t0), _bf16_raw_bytes(t1)]
    md5s = [_md5_hex(r) for r in raws]
    blobs = cross_layer_delta.encode_group(raws)
    decoded = cross_layer_delta.decode_group(blobs, n_weights_each=t0.numel())
    for i, (orig_md5, dec_bytes) in enumerate(zip(md5s, decoded)):
        assert _md5_hex(dec_bytes) == orig_md5, f"edge case layer {i} mismatch"


def test_cross_layer_delta_pair_api():
    """Pair API roundtrips and records the predecessor name in extras."""
    try:
        import torch  # noqa: F401
    except ImportError:
        pytest.skip("torch not installed")
    from bigsmall.codecs import cross_layer_delta

    prev = _bf16_raw_bytes(_normal_bf16(80_000, seed=0))
    curr = _bf16_raw_bytes(_normal_bf16(80_000, seed=1))
    blob, extras = cross_layer_delta.encode_pair(curr, prev, "model.layers.0.attn.weight")
    assert extras == {"delta_from": "model.layers.0.attn.weight"}
    decoded = cross_layer_delta.decode_pair(blob, prev, n_weights=80_000 // 1)
    assert _md5_hex(decoded) == _md5_hex(curr)


def test_cross_layer_delta_xor_length_mismatch_raises():
    """xor_bytes raises on mismatched-length inputs."""
    from bigsmall.codecs import cross_layer_delta
    with pytest.raises(ValueError):
        cross_layer_delta.xor_bytes(b"\x01" * 8, b"\x02" * 7)


def test_cross_layer_delta_safety_net_documented():
    """Empirical Session B finding: XOR delta of trained-transformer
    consecutive layers can be LARGER than the standalone blob. This test
    just documents the expected behaviour: the codec produces a valid blob
    even when it's not a size win, and the encoder's auto-select is the
    safety net.

    The test compares the delta blob size to a plain blob and reports the
    relationship without asserting on it (so the test passes on either
    sign of the comparison). The behaviour ASSERTED is: both blobs decode
    correctly.
    """
    try:
        import torch  # noqa: F401
    except ImportError:
        pytest.skip("torch not installed")
    from bigsmall.codecs import bf16, cross_layer_delta

    prev = _bf16_raw_bytes(_normal_bf16(120_000, seed=10))
    curr = _bf16_raw_bytes(_normal_bf16(120_000, seed=11))

    plain_blob, _ = bf16.encode(curr)
    delta_blob, _ = cross_layer_delta.encode_pair(
        curr, prev, "prev_tensor")

    # Both must decode correctly regardless of which is smaller.
    decoded_plain = bf16.decode(plain_blob, {}, n_weights=120_000)
    decoded_delta = cross_layer_delta.decode_pair(
        delta_blob, prev, n_weights=120_000)
    assert _md5_hex(decoded_plain) == _md5_hex(curr)
    assert _md5_hex(decoded_delta) == _md5_hex(curr)
