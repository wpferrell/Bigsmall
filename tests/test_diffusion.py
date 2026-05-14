"""Diffusion model 4D conv tensor handling test."""
import hashlib
import tempfile
from pathlib import Path

import pytest


def _md5_hex(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def test_diffusion_4d_conv_roundtrip():
    try:
        import torch
        from safetensors.torch import save_file
        from safetensors import safe_open
    except ImportError:
        pytest.skip("torch/safetensors not installed")
    from bigsmall.integrations import diffusion as diff

    torch.manual_seed(42)
    tensors = {
        "unet.conv_in.weight": torch.randn(64, 4, 3, 3, dtype=torch.bfloat16),
        "unet.conv_in.bias": torch.zeros(64, dtype=torch.bfloat16),
        "unet.down_blocks.0.resnets.0.conv1.weight": torch.randn(64, 64, 3, 3, dtype=torch.bfloat16),
        "unet.transformer_blocks.0.attn.to_q.weight": torch.randn(512, 512, dtype=torch.bfloat16),
        "unet.transformer_blocks.0.attn.to_k.weight": torch.randn(512, 512, dtype=torch.bfloat16),
        "vae.encoder.conv_in.weight": torch.randn(128, 3, 3, 3, dtype=torch.bfloat16),
        "time_embed.linear_1.weight": torch.randn(1280, 320, dtype=torch.bfloat16),
    }
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "diffusion.safetensors"
        save_file(tensors, str(src))
        assert diff.is_diffusion_model(src)

        bs = Path(td) / "diffusion.bs"
        diff.compress_diffusion(src, bs)
        out = diff.decompress_diffusion(bs)

        with safe_open(str(src), framework="pt") as f:
            for name in f.keys():
                t = f.get_tensor(name)
                src_md5 = _md5_hex(t.contiguous().view(torch.uint8).cpu().numpy().tobytes())
                dec_md5 = _md5_hex(out[name].tobytes())
                assert src_md5 == dec_md5, f"md5 mismatch {name}"
                assert list(t.shape) == list(out[name].shape), f"shape mismatch {name}"
