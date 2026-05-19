"""End-to-end inference test: GPT-2 decompressed produces identical generations."""
import os
import shutil
import tempfile
from pathlib import Path

import pytest


def _gpt2_dir() -> Path | None:
    cache = Path(os.path.expandvars(r"%USERPROFILE%/.cache/huggingface/hub")) / "models--gpt2" / "snapshots"
    if not cache.exists():
        return None
    for d in cache.iterdir():
        if (d / "model.safetensors").exists():
            return d
    return None


@pytest.mark.integration
def test_gpt2_inference_identical():
    """Decompressed GPT-2 produces same generations as original.

    Marked `integration` (v3.11.0): requires GPT-2 to be cached locally
    (~500 MB). Skipped from the default `pytest tests/` run; opt in
    with `pytest -m integration tests/`.
    """
    gpt2 = _gpt2_dir()
    if gpt2 is None:
        pytest.skip("GPT-2 not in HF cache")
    try:
        import torch
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
    except ImportError:
        pytest.skip("transformers not installed")
    import bigsmall

    with tempfile.TemporaryDirectory() as td:
        bs = Path(td) / "model.bs"
        bigsmall.compress(gpt2 / "model.safetensors", bs)
        dec_st = Path(td) / "dec.safetensors"
        bigsmall.decompress(bs, dec_st)
        # Build temp HF dir with decompressed weights
        hf_dir = Path(td) / "hf"
        hf_dir.mkdir()
        for f in gpt2.iterdir():
            if f.is_file() and f.name != "model.safetensors":
                shutil.copy2(f, hf_dir)
        shutil.copy2(dec_st, hf_dir / "model.safetensors")

        tok = GPT2Tokenizer.from_pretrained(str(gpt2))
        m_orig = GPT2LMHeadModel.from_pretrained(str(gpt2)).eval()
        m_dec = GPT2LMHeadModel.from_pretrained(str(hf_dir)).eval()
        ids = tok.encode("Hello, world.", return_tensors="pt")
        with torch.no_grad():
            o = m_orig.generate(ids, max_new_tokens=10, do_sample=False)
            d = m_dec.generate(ids, max_new_tokens=10, do_sample=False)
        assert tok.decode(o[0]) == tok.decode(d[0])
