"""End-to-end lossless roundtrip tests for FP8 tensors (E4M3 + E5M2).

Locks the baseline that `fp8_cat_ac` is at the joint-entropy floor of the
FP8 byte stream. The codec is a 256-symbol Categorical AC; H(byte) on
trained-weight-like distributions is ~67-71% of raw, which already beats
the published ECF8 ratio (73.1%) by 2-6pp. These tests are pure
correctness checks plus an upper-bound ratio assertion so a future codec
change cannot silently regress.
"""
import hashlib
import tempfile
from pathlib import Path

import pytest


def _md5(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def _roundtrip_fp8(dtype, ratio_upper_bound: float) -> None:
    import torch
    import bigsmall
    from safetensors import safe_open
    from safetensors.torch import save_file

    torch.manual_seed(0)
    # Mix of attention/mlp/embedding/norm-shaped tensors, scaled like
    # trained LLM weights (~0.02 std).
    sd = {
        "embed.weight":              (torch.randn(2048, 256) * 0.02).to(dtype),
        "layer.0.attn.qkv.weight":   (torch.randn(512, 512) * 0.02).to(dtype),
        "layer.0.attn.out.weight":   (torch.randn(512, 512) * 0.02).to(dtype),
        "layer.0.mlp.down.weight":   (torch.randn(512, 1024) * 0.02).to(dtype),
        "ln_f.weight":               (torch.randn(512) * 0.02).to(dtype),
    }
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "model.safetensors"
        save_file(sd, str(src))
        bs = Path(td) / "model.bs"
        bigsmall.compress(src, bs)
        out = bigsmall.decompress(bs)

        src_size = src.stat().st_size
        bs_size = bs.stat().st_size
        ratio = bs_size / src_size
        assert ratio < ratio_upper_bound, (
            f"{dtype} ratio {ratio:.4f} regressed past upper bound "
            f"{ratio_upper_bound:.4f} (ECF8 baseline 0.731)"
        )

        with safe_open(str(src), framework="pt") as f:
            fail = []
            for name in f.keys():
                t = f.get_tensor(name)
                src_bytes = t.contiguous().view(torch.uint8).cpu().numpy().tobytes()
                if _md5(src_bytes) != _md5(out[name].tobytes()):
                    fail.append(name)
            assert not fail, f"md5 mismatch on: {fail[:3]}"


def test_roundtrip_fp8_e4m3():
    try:
        import torch  # noqa: F401
        import safetensors  # noqa: F401
    except ImportError:
        pytest.skip("torch/safetensors not installed")
    import torch
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch build lacks float8_e4m3fn")
    _roundtrip_fp8(torch.float8_e4m3fn, ratio_upper_bound=0.731)


def test_roundtrip_fp8_e5m2():
    try:
        import torch  # noqa: F401
        import safetensors  # noqa: F401
    except ImportError:
        pytest.skip("torch/safetensors not installed")
    import torch
    if not hasattr(torch, "float8_e5m2"):
        pytest.skip("torch build lacks float8_e5m2")
    _roundtrip_fp8(torch.float8_e5m2, ratio_upper_bound=0.731)


def test_fp8_codec_at_shannon_floor():
    """Direct unit test that fp8_cat_ac achieves H(byte) on synthetic data.

    The Categorical AC codec is mathematically at the joint entropy floor of
    the FP8 byte stream. Overhead vs Shannon should be <= 0.1pp of raw size
    on large tensors.
    """
    try:
        import torch
        import numpy as np
    except ImportError:
        pytest.skip("torch/numpy not installed")
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch build lacks float8_e4m3fn")
    from bigsmall.codecs import fp8

    torch.manual_seed(0)
    n = 1 << 20  # 1M samples for a stable measurement
    fp32 = torch.distributions.StudentT(4.0).sample((n,)) * 0.02
    raw = fp32.to(torch.float8_e4m3fn).view(torch.uint8).numpy().tobytes()

    blob, _ = fp8.encode(raw)
    ratio = len(blob) / len(raw)

    u8 = np.frombuffer(raw, dtype=np.uint8)
    freq = np.bincount(u8, minlength=256).astype(np.float64)
    p = freq[freq > 0] / freq.sum()
    shannon_bits = -(p * np.log2(p)).sum()
    shannon_ratio = shannon_bits / 8.0

    # Codec must be within 0.1pp of Shannon and lossless
    assert ratio - shannon_ratio < 0.001, (
        f"cat_ac overhead {(ratio - shannon_ratio)*100:.4f}pp exceeds 0.1pp"
    )
    assert fp8.decode(blob, {}, n) == raw, "lossless roundtrip failed"
