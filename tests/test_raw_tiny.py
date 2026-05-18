"""Tiny-tensor short-circuit tests.

Tensors below `encoder.RAW_TINY_THRESHOLD` bytes should be stored with
`codec="raw"` -- no entropy coder, no special-tensor wrapping. The round-trip
must still be byte-identical.
"""
import hashlib
import tempfile
from pathlib import Path

import pytest


def _md5_hex(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def _build():
    """A mix of tensors: tiny bias-shape ones (~256 B) and a normal-size matrix."""
    import torch
    torch.manual_seed(0)
    return {
        "tiny.bias": torch.randn(128, dtype=torch.bfloat16),               # 256 B
        "very_tiny.scale": torch.tensor([1.5, -0.25], dtype=torch.float32),  # 8 B
        "norm.weight": torch.randn(2048, dtype=torch.bfloat16),            # 4096 B
        "matrix.weight": torch.randn(512, 512, dtype=torch.bfloat16),      # 524288 B
    }


def test_tiny_tensor_uses_raw_codec():
    try:
        import torch  # noqa: F401
        import safetensors  # noqa: F401
    except ImportError:
        pytest.skip("torch/safetensors not installed")
    import bigsmall
    from bigsmall import container, encoder
    from safetensors.torch import save_file

    tensors = _build()
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "model.safetensors"
        save_file(tensors, str(src))
        bs = Path(td) / "model.bs"
        bigsmall.compress(src, bs, progress=False, workers=1)

        header, _ = container.read_header(bs)
        codec_by_name = {t["name"]: t["codec"] for t in header["tensors"]}
        raw_sizes = {t["name"]: int(t["compressed_bytes"]) for t in header["tensors"]}

        # Tiny tensors should use the raw codec...
        assert codec_by_name["tiny.bias"] == "raw"
        assert codec_by_name["very_tiny.scale"] == "raw"
        # ...and stored as exactly the raw byte count.
        assert raw_sizes["tiny.bias"] == 256
        assert raw_sizes["very_tiny.scale"] == 8
        # Above-threshold tensors should NOT be raw.
        assert codec_by_name["norm.weight"] != "raw"
        assert codec_by_name["matrix.weight"] != "raw"


def test_tiny_tensor_roundtrip_md5():
    try:
        import torch
        import safetensors  # noqa: F401
    except ImportError:
        pytest.skip("torch/safetensors not installed")
    import bigsmall
    from safetensors import safe_open
    from safetensors.torch import save_file

    tensors = _build()
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "model.safetensors"
        save_file(tensors, str(src))
        bs = Path(td) / "model.bs"
        bigsmall.compress(src, bs, progress=False, workers=1)
        out = bigsmall.decompress(bs, progress=False)

        with safe_open(str(src), framework="pt") as f:
            for name in f.keys():
                t = f.get_tensor(name)
                src_md5 = _md5_hex(t.contiguous().view(torch.uint8).cpu().numpy().tobytes())
                dec_md5 = _md5_hex(out[name].tobytes())
                assert src_md5 == dec_md5, f"{name} differs after raw-tiny round-trip"


def test_threshold_boundary():
    """A tensor exactly at the threshold uses the codec path, not raw."""
    import torch
    from safetensors.torch import save_file
    import bigsmall
    from bigsmall import container, encoder

    threshold = encoder.RAW_TINY_THRESHOLD
    # `threshold` bytes at bf16 == threshold/2 elements
    just_at_threshold = torch.zeros(threshold // 2, dtype=torch.bfloat16)
    just_below = torch.zeros((threshold // 2) - 1, dtype=torch.bfloat16)
    tensors = {
        "at_threshold.weight": just_at_threshold,
        "below_threshold.weight": just_below,
    }
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "model.safetensors"
        save_file(tensors, str(src))
        bs = Path(td) / "model.bs"
        bigsmall.compress(src, bs, progress=False, workers=1)
        header, _ = container.read_header(bs)
        codec_by_name = {t["name"]: t["codec"] for t in header["tensors"]}

        assert codec_by_name["below_threshold.weight"] == "raw"
        assert codec_by_name["at_threshold.weight"] != "raw"
