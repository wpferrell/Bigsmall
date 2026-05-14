"""Round-trip md5 tests on real safetensors models.

Each test compresses a model to .bs, decompresses back to a tensor dict,
and verifies that every tensor's md5 matches the original raw bytes.
"""
import hashlib
import os
import tempfile
from pathlib import Path

import pytest

# Locate test models in the HF cache
HF_CACHE = Path(os.environ.get("HF_HOME") or
                os.path.expandvars(r"%USERPROFILE%\.cache\huggingface")) / "hub"


def _find_first_safetensors(model_repo: str) -> Path | None:
    """Find a model.safetensors file in the HF cache snapshot."""
    p = HF_CACHE / f"models--{model_repo.replace('/', '--')}" / "snapshots"
    if not p.exists():
        return None
    for snap in p.iterdir():
        st = snap / "model.safetensors"
        if st.exists():
            return st
        # Multi-shard
        idx = snap / "model.safetensors.index.json"
        if idx.exists():
            shards = sorted(snap.glob("model-*.safetensors"))
            if shards:
                return shards[0]  # test first shard only
    return None


def _md5_hex(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def _roundtrip(safetensors_path: Path, fmt: str = "bf16"):
    import bigsmall
    with tempfile.TemporaryDirectory() as td:
        bs_path = Path(td) / "model.bs"
        bigsmall.compress(safetensors_path, bs_path)
        out = bigsmall.decompress(bs_path)

    # Compare md5s with originals
    from safetensors import safe_open
    import torch
    fail = []
    with safe_open(str(safetensors_path), framework="pt") as f:
        for name in f.keys():
            t = f.get_tensor(name)
            src_bytes = t.contiguous().view(torch.uint8).cpu().numpy().tobytes()
            src_md5 = _md5_hex(src_bytes)
            dec_md5 = _md5_hex(out[name].tobytes())
            if src_md5 != dec_md5:
                fail.append((name, src_md5, dec_md5))
    assert not fail, f"md5 mismatch on {len(fail)} tensors: {fail[:3]}"


def test_gpt2_bf16():
    p = _find_first_safetensors("gpt2")
    if p is None:
        pytest.skip("GPT-2 not in HF cache")
    _roundtrip(p, "bf16")
