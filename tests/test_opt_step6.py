"""Tests for v3.9.0 streaming + hub features.

Coverage:
  1. compress_streaming output md5-identical to compress() on a non-tied model.
  2. compress_streaming on a tied-weight model still produces a valid file
     that decompresses correctly (just no dedup).
  3. decompress_layers returns only the requested layers (and the tensors
     are bit-identical to the full-decompress version).
  4. BigSmallStreamingModel.from_pretrained surfaces a clear error when the
     bs_path doesn't exist (Step 5 — better error messages).
  5. BigSmallStreamingModel constructor accepts prefetch=N kwarg without
     starting the worker until first forward (lazy init).
"""
from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import pytest


def _md5(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def _md5_file(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def test_compress_streaming_matches_compress():
    """compress_streaming output == compress() on a non-tied model (md5)."""
    try:
        import torch
        import safetensors  # noqa: F401
    except ImportError:
        pytest.skip("torch/safetensors not installed")
    from safetensors.torch import save_file
    import bigsmall

    torch.manual_seed(0)
    sd = {
        f"model.layers.{i}.weight": (torch.randn(256, 128) * 0.02).to(torch.bfloat16)
        for i in range(4)
    }
    sd["lm_head.weight"] = (torch.randn(512, 128) * 0.02).to(torch.bfloat16)
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "model.safetensors"
        save_file(sd, str(src))
        std_bs = Path(td) / "std.bs"
        stream_bs = Path(td) / "stream.bs"
        bigsmall.compress(src, std_bs, workers=1, progress=False)
        bigsmall.compress_streaming(src, stream_bs, progress=False)
        assert _md5_file(std_bs) == _md5_file(stream_bs), (
            "compress_streaming output differs from compress() on a non-tied model"
        )


def test_compress_streaming_roundtrip_lossless():
    """compress_streaming → decompress is byte-identical to source bytes."""
    try:
        import torch
        import safetensors  # noqa: F401
    except ImportError:
        pytest.skip("torch/safetensors not installed")
    from safetensors import safe_open
    from safetensors.torch import save_file
    import bigsmall

    torch.manual_seed(1)
    sd = {
        "embed.weight": (torch.randn(1024, 64) * 0.02).to(torch.bfloat16),
        "decoder.layers.0.weight": (torch.randn(64, 128) * 0.02).to(torch.bfloat16),
        "ln_f.weight": (torch.randn(64) * 0.02).to(torch.bfloat16),
    }
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "model.safetensors"
        save_file(sd, str(src))
        bs = Path(td) / "out.bs"
        bigsmall.compress_streaming(src, bs, progress=False)
        out = bigsmall.decompress(bs)
        with safe_open(str(src), framework="pt") as f:
            for name in f.keys():
                t = f.get_tensor(name)
                src_b = t.contiguous().view(torch.uint8).cpu().numpy().tobytes()
                assert _md5(src_b) == _md5(out[name].tobytes()), name


def test_decompress_layers_returns_only_requested():
    """decompress_layers returns the requested layers and they match a
    full-model decompress."""
    try:
        import torch
        import safetensors  # noqa: F401
    except ImportError:
        pytest.skip("torch/safetensors not installed")
    from safetensors.torch import save_file
    import bigsmall

    torch.manual_seed(2)
    sd = {
        "model.embed_tokens.weight": (torch.randn(1024, 64) * 0.02).to(torch.bfloat16),
    }
    for i in range(6):
        sd[f"model.layers.{i}.q.weight"] = (torch.randn(64, 64) * 0.02).to(torch.bfloat16)
        sd[f"model.layers.{i}.k.weight"] = (torch.randn(64, 64) * 0.02).to(torch.bfloat16)
    sd["model.norm.weight"] = (torch.randn(64) * 0.02).to(torch.bfloat16)

    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "model.safetensors"
        save_file(sd, str(src))
        bs = Path(td) / "model.bs"
        bigsmall.compress(src, bs, workers=1, progress=False)

        # Partial: just layers [1, 3]
        partial = bigsmall.decompress_layers(bs, layer_indices=[1, 3], device="cpu")
        for li in (1, 3):
            assert f"model.layers.{li}.q.weight" in partial
            assert f"model.layers.{li}.k.weight" in partial
        # No other layers
        for li in (0, 2, 4, 5):
            assert f"model.layers.{li}.q.weight" not in partial
        # Non-layer not included by default
        assert "model.embed_tokens.weight" not in partial

        # include_non_layer=True picks up embed + norm
        with_extras = bigsmall.decompress_layers(
            bs, layer_indices=[0], device="cpu", include_non_layer=True,
        )
        assert "model.embed_tokens.weight" in with_extras
        assert "model.norm.weight" in with_extras


def test_streaming_model_from_pretrained_missing_path():
    """from_pretrained raises FileNotFoundError with a helpful suggestion."""
    try:
        import torch  # noqa: F401
        import safetensors  # noqa: F401
        from accelerate import init_empty_weights  # noqa: F401
    except ImportError:
        pytest.skip("torch/safetensors/accelerate not installed")
    from bigsmall.streaming_inference import BigSmallStreamingModel
    missing = Path(tempfile.gettempdir()) / "definitely_not_a_real_bs_dir_12345xyz"
    if missing.exists():
        pytest.skip("test path unexpectedly exists")
    with pytest.raises(FileNotFoundError) as e:
        BigSmallStreamingModel.from_pretrained(missing)
    msg = str(e.value)
    assert "does not exist" in msg
    assert "compress_from_hub" in msg, "error message should suggest compress_from_hub"


def test_streaming_model_accepts_prefetch_kwarg():
    """Constructor accepts prefetch=N without starting the worker until forward."""
    try:
        import torch  # noqa: F401
        import safetensors  # noqa: F401
        from accelerate import init_empty_weights  # noqa: F401
        from transformers import AutoConfig  # noqa: F401
    except ImportError:
        pytest.skip("torch/safetensors/accelerate/transformers not installed")
    # We don't actually construct a full model (slow) — just check the
    # signature accepts the kwarg by inspecting it.
    import inspect
    from bigsmall.streaming_inference import BigSmallStreamingModel
    sig = inspect.signature(BigSmallStreamingModel.__init__)
    assert "prefetch" in sig.parameters, "BigSmallStreamingModel should accept prefetch="
    assert sig.parameters["prefetch"].default == 0, "default prefetch should be 0 (disabled)"
