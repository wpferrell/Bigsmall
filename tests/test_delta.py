"""Delta compression round-trip tests."""
import hashlib
import tempfile
from pathlib import Path

import pytest


def _md5_hex(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def test_delta_roundtrip_synthetic():
    try:
        import torch
        from safetensors.torch import save_file
        from safetensors import safe_open
    except ImportError:
        pytest.skip("torch/safetensors not installed")
    import bigsmall

    torch.manual_seed(0)
    base = {
        "embed.weight": torch.randn(2000, 256, dtype=torch.bfloat16),
        "layer.0.weight": torch.randn(256, 512, dtype=torch.bfloat16),
        "layer.1.weight": torch.randn(512, 256, dtype=torch.bfloat16),
    }
    ft = {}
    for k, v in base.items():
        perturb = (torch.rand_like(v.float()) < 0.01).to(torch.bfloat16)
        ft[k] = v + perturb * 0.01

    with tempfile.TemporaryDirectory() as td:
        base_st = Path(td) / "base.safetensors"
        ft_st = Path(td) / "ft.safetensors"
        save_file(base, str(base_st))
        save_file(ft, str(ft_st))

        # Standalone
        ft_bs = Path(td) / "ft.bs"
        bigsmall.compress(ft_st, ft_bs)
        # Delta
        delta_bs = Path(td) / "delta.bs"
        bigsmall.compress_delta(ft_st, base_st, delta_bs)
        # Delta should be smaller than standalone (since most weights identical)
        assert delta_bs.stat().st_size < ft_bs.stat().st_size

        # Roundtrip via delta decompression
        out = bigsmall.decompress_delta(delta_bs, base_st)
        with safe_open(str(ft_st), framework="pt") as f:
            for name in f.keys():
                t = f.get_tensor(name)
                src_md5 = _md5_hex(t.contiguous().view(torch.uint8).cpu().numpy().tobytes())
                dec_md5 = _md5_hex(out[name].tobytes())
                assert src_md5 == dec_md5, f"md5 mismatch on {name}"
