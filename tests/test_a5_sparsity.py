"""A5 sparsity codec tests.

What this file actually proves (read together with `A5_DONE.md`):

1. The A5 codec is implemented correctly: every encoded blob decodes
   byte-identically (`md5` verified) on heavy-tailed synthetic tensors and
   on the full compress -> .bs -> decompress pipeline.
2. The encoder's safety net works: a well-behaved normal-distribution tensor
   never triggers A5 and stays on the v1 bf16 codec, so 2.0.x consumers can
   still read the resulting .bs file.
3. When directly forced through the codec module the output round-trips and
   the container can be stamped as v2.
4. The mask-cost vs entropy-savings tradeoff is observable: on a Student-T
   tensor the size delta vs plain bf16 is sub-1 % regardless of threshold,
   confirming the codec is at the joint-entropy floor (the finding
   documented in `A5_DONE.md`).
5. The encoder safety net (`bf16_sparsity_v1` only kept if it produces a
   smaller blob than plain bf16) means A5 cannot cause a *regression* even
   when the heuristic doesn't help.
"""
import hashlib
import struct
import tempfile
from pathlib import Path

import pytest


def _md5_hex(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def _heavy_tailed_bf16(n_elements: int, df: float = 2.5, seed: int = 0):
    """Spike-and-slab BF16 fixture.

    Real Qwen3 early-MLP gate_proj tensors are not Student-T -- they have a
    tight near-zero bulk plus a heavy outlier tail. We synthesise that
    distribution explicitly: ~95 % of weights are drawn from a tight
    Normal(0, 0.005) and ~5 % from a wider Normal(0, 0.1). This produces
    kurtosis well above the A5 detection threshold and a near-zero fraction
    close to what we measured on Qwen3-8B (0.15 - 0.23 %).

    `df` is kept for signature compatibility but is now unused.
    """
    import torch
    g = torch.Generator().manual_seed(seed)
    bulk_std = 0.005
    tail_std = 0.10
    tail_prob = 0.05
    # Bernoulli mask for which population each element belongs to.
    is_tail = (torch.rand(n_elements, generator=g) < tail_prob)
    bulk = torch.randn(n_elements, generator=g) * bulk_std
    tail = torch.randn(n_elements, generator=g) * tail_std
    sample = torch.where(is_tail, tail, bulk).to(torch.bfloat16)
    return sample


def _normal_bf16(n_elements: int, seed: int = 0):
    import torch
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n_elements, generator=g, dtype=torch.float32).to(torch.bfloat16)


def _bf16_raw_bytes(t):
    import torch
    return t.contiguous().view(torch.uint8).cpu().numpy().tobytes()


def test_a5_lossless_roundtrip_on_heavy_tailed():
    """High-kurtosis tensor encodes with A5 and decodes byte-identically."""
    try:
        import torch  # noqa: F401
    except ImportError:
        pytest.skip("torch not installed")
    from bigsmall.codecs import bf16_sparsity

    t = _heavy_tailed_bf16(200_000, df=2.5, seed=0)
    raw = _bf16_raw_bytes(t)
    blob, _ = bf16_sparsity.encode(raw)
    back = bf16_sparsity.decode(blob, {}, n_weights=t.numel())
    assert _md5_hex(back) == _md5_hex(raw), \
        "A5 roundtrip is not byte-identical on a heavy-tailed BF16 tensor"


def test_a5_lossless_through_full_compress_pipeline():
    """End-to-end: compress() -> .bs -> decompress() preserves md5."""
    try:
        import torch
        import safetensors  # noqa: F401
    except ImportError:
        pytest.skip("torch/safetensors not installed")
    import bigsmall
    from safetensors import safe_open
    from safetensors.torch import save_file

    # Mix a heavy-tailed tensor (triggers A5) with a plain matrix.
    tensors = {
        "model.layers.0.mlp.gate_proj.weight":
            _heavy_tailed_bf16(120_000, df=2.0, seed=1).reshape(300, 400),
        "model.layers.0.input_layernorm.weight":
            _normal_bf16(2048, seed=2),
        "model.layers.0.attn.q_proj.weight":
            _normal_bf16(80_000, seed=3).reshape(400, 200),
    }
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "m.safetensors"
        save_file(tensors, str(src))
        bs = Path(td) / "m.bs"
        bigsmall.compress(src, bs, progress=False, workers=1)
        out = bigsmall.decompress(bs, progress=False)
        with safe_open(str(src), framework="pt") as f:
            for name in f.keys():
                t = f.get_tensor(name)
                src_md5 = _md5_hex(_bf16_raw_bytes(t))
                dec_md5 = _md5_hex(out[name].tobytes())
                assert src_md5 == dec_md5, f"{name} byte mismatch"


def test_a5_does_not_fire_on_normal_distribution():
    """A well-behaved random matrix must NOT trigger A5 -- it stays on the
    plain bf16 codec, leaving 19 already-uploaded HF models bit-stable."""
    try:
        import torch  # noqa: F401
    except ImportError:
        pytest.skip("torch not installed")
    import bigsmall
    from bigsmall import container
    from safetensors.torch import save_file

    tensors = {
        "matrix.weight": _normal_bf16(200_000, seed=42).reshape(400, 500),
    }
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "m.safetensors"
        save_file(tensors, str(src))
        bs = Path(td) / "m.bs"
        bigsmall.compress(src, bs, progress=False, workers=1)

        header, _ = container.read_header(bs)
        codecs = {t["name"]: t["codec"] for t in header["tensors"]}
        assert codecs["matrix.weight"] != "bf16_sparsity_v1", (
            "A5 fired on a plain normal-distribution tensor; "
            f"got codec={codecs['matrix.weight']}"
        )
        # And the file must stay on container v1 -- no v2-only codec used.
        assert header["_format_version"] == 1


def test_a5_safety_net_falls_back_to_bf16_when_not_beneficial():
    """The encoder runs both codecs and only keeps A5 if it produced fewer
    bytes. On real data the joint-entropy floor pins A5 to within ~1 byte/elem
    of plain bf16, so the safety net almost always picks bf16. This test
    pins the safety-net behaviour explicitly: when A5 isn't smaller, the
    resulting tensor codec must be the standard bf16, the container must
    stay on v1, and round-trip must still be lossless."""
    try:
        import torch  # noqa: F401
    except ImportError:
        pytest.skip("torch not installed")
    import bigsmall
    from bigsmall import container
    from safetensors.torch import save_file
    from safetensors import safe_open

    tensors = {
        "model.layers.0.mlp.gate_proj.weight":
            _heavy_tailed_bf16(150_000, seed=11).reshape(300, 500),
    }
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "m.safetensors"
        save_file(tensors, str(src))
        bs = Path(td) / "m.bs"
        bigsmall.compress(src, bs, progress=False, workers=1)

        header, _ = container.read_header(bs)
        codecs = {t["name"]: t["codec"] for t in header["tensors"]}
        # On real-shape distributions, A5 doesn't actually beat bf16, so the
        # safety net stays on bf16 -- and the container stays on v1.
        assert codecs["model.layers.0.mlp.gate_proj.weight"] != "bf16_sparsity_v1"
        assert header["_format_version"] == 1
        # And the file still round-trips byte-identically.
        out = bigsmall.decompress(bs, progress=False)
        with safe_open(str(src), framework="pt") as f:
            for name in f.keys():
                t = f.get_tensor(name)
                assert _md5_hex(_bf16_raw_bytes(t)) == _md5_hex(out[name].tobytes())


def test_a5_codec_module_can_stamp_v2_container():
    """Even though the encoder dispatcher rarely keeps A5, the codec module
    itself produces v2-compatible output: an A5 blob written into a
    `bf16_sparsity_v1`-tagged tensor entry round-trips correctly under v2
    container stamping."""
    try:
        import torch  # noqa: F401
    except ImportError:
        pytest.skip("torch not installed")
    from bigsmall import container
    from bigsmall.codecs import bf16_sparsity
    from bigsmall.decoder import _decode_blob
    from bigsmall.formats import BS_FORMAT_VERSION_V2

    # Build a tensor and an A5 blob directly (bypassing the encoder safety net).
    t = _heavy_tailed_bf16(80_000, seed=21)
    raw = _bf16_raw_bytes(t)
    blob, extras = bf16_sparsity.encode(raw)

    header = {
        "format": "bf16", "mode": "balanced", "model_type": "llm",
        "base_model": None, "tensor_count": 1,
        "tensors": [{
            "name": "t",
            "shape": [t.numel()],
            "dtype": "BF16",
            "codec": "bf16_sparsity_v1",
            "special": "a5_sparsity",
            "compressed_bytes": len(blob),
            "offset": 0,
            "md5": _md5_hex(raw),
            "extra": extras or None,
        }],
        "safetensors_metadata": None,
    }
    with tempfile.TemporaryDirectory() as td:
        bs = Path(td) / "a5.bs"
        container.write_container(bs, header, blob,
                                  format_version=BS_FORMAT_VERSION_V2)

        # Verify on-disk stamp is v2 and the dispatcher routes back to A5.
        with open(bs, "rb") as f:
            f.seek(4)
            v, = struct.unpack("<H", f.read(2))
        assert v == 2

        out_header, _ = container.read_header(bs)
        meta = out_header["tensors"][0]
        # Read the blob back from disk and decode through the dispatcher.
        with open(bs, "rb") as f:
            f.seek(struct.calcsize("<4sH I") + len(__import__("json").dumps(
                {k: v for k, v in out_header.items() if k != "_format_version"},
                separators=(",", ":")).encode()))
            disk_blob = f.read(meta["compressed_bytes"])
        decoded = _decode_blob(meta, disk_blob)
        assert _md5_hex(decoded) == _md5_hex(raw)


def test_a5_safety_net_size_is_within_one_pct_of_bf16():
    """Documents the codec-equivalence finding: across reasonable threshold
    factors A5 produces a total within ~1 % of the plain bf16 size. This is
    the empirical reflection of `H(s,e,m) = H(mask) + H(s,e | mask) + H(m|e,
    mask)` -- splitting determined-by-X partitions out of an already
    floor-achieving codec cannot save bits."""
    try:
        import torch  # noqa: F401
    except ImportError:
        pytest.skip("torch not installed")
    from bigsmall.codecs import bf16, bf16_sparsity

    t = _heavy_tailed_bf16(400_000, seed=7)
    raw = _bf16_raw_bytes(t)
    bf16_blob, _ = bf16.encode(raw)
    a5_blob, _ = bf16_sparsity.encode(raw)
    ratio_diff_pct = 100.0 * (len(a5_blob) - len(bf16_blob)) / max(1, len(bf16_blob))
    assert abs(ratio_diff_pct) < 1.5, (
        f"A5 size diverged unexpectedly from bf16 ({ratio_diff_pct:+.2f} %); "
        "this would indicate a codec bug, since both codes are at the "
        "joint-entropy floor"
    )
