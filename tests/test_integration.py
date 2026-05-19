"""Real-model integration tests (v3.11.0).

These tests download a small real model (GPT-2, ~500 MB) and exercise the
full compress → decompress → md5 verify pipeline against actual trained
weights. They are marked `integration` so the default `pytest tests/` run
skips them; opt in with `pytest -m integration tests/`.

Run manually to validate before release. Network access required.
"""
from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import pytest


def _md5(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


@pytest.mark.integration
def test_compress_from_hub_gpt2_roundtrip():
    """Real model: gpt2 from HF Hub → .bs → decompress → md5-identical.

    Validates the full chain that ships in the user-facing API:
      - `compress_from_hub` downloads via huggingface_hub
      - `compress_streaming` runs over the downloaded shard
      - `decompress` reads back what was written

    Compares each tensor's md5 against the source.
    """
    try:
        import torch  # noqa: F401
        import safetensors  # noqa: F401
        from safetensors import safe_open
        import huggingface_hub  # noqa: F401
    except ImportError as e:
        pytest.skip(f"required dep missing: {e}")

    import bigsmall

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        out_dir = td / "gpt2_bs"
        # Compress via the public API: streams from HF cache, one shard.
        bigsmall.compress_from_hub(
            "gpt2",
            output_path=out_dir,
            progress=False,
        )

        # Find the produced .bs shard(s)
        shards = sorted(out_dir.glob("*.bs"))
        assert shards, f"compress_from_hub produced no .bs files in {out_dir}"

        # Decompress and md5-check each tensor against the cached
        # safetensors source (huggingface_hub puts it in the default cache).
        from huggingface_hub import hf_hub_download
        src_path = Path(hf_hub_download("gpt2", "model.safetensors"))

        decoded = bigsmall.decompress(shards[0], progress=False)
        with safe_open(str(src_path), framework="pt") as f:
            failures = []
            for name in f.keys():
                t = f.get_tensor(name)
                import torch
                src_bytes = t.contiguous().view(torch.uint8).cpu().numpy().tobytes()
                dec_arr = decoded.get(name)
                if dec_arr is None:
                    failures.append(f"missing tensor {name}")
                    continue
                if _md5(src_bytes) != _md5(dec_arr.tobytes()):
                    failures.append(f"md5 mismatch {name}")
            assert not failures, "\n".join(failures[:5])


@pytest.mark.integration
def test_verify_full_on_real_model():
    """End-to-end: compress a real model, then verify() (md5 round-trip)."""
    try:
        import torch  # noqa: F401
        import safetensors  # noqa: F401
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        pytest.skip(f"required dep missing: {e}")
    import bigsmall

    src = Path(hf_hub_download("gpt2", "model.safetensors"))
    with tempfile.TemporaryDirectory() as td:
        bs = Path(td) / "gpt2.bs"
        bigsmall.compress(src, bs, workers=1, progress=False)
        ok = bigsmall.verify(bs)
        assert ok, "verify() failed on real GPT-2 compressed file"
