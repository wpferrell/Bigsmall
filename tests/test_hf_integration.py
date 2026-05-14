"""HuggingFace Hub integration tests (Phase 4).

Two tests:
 1. test_compress_for_hub_writes_valid_index   - compress_for_hub('gpt2')
    produces a populated bigsmall.index.json with byte-correct totals.
 2. test_from_pretrained_roundtrip_gpt2        - from_pretrained() returns a
    state_dict whose tensors are byte-identical to the source safetensors.
"""
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest


HF_CACHE = Path(os.environ.get("HF_HOME") or
                os.path.expandvars(r"%USERPROFILE%\.cache\huggingface")) / "hub"


def _gpt2_cached() -> Path | None:
    """Find a cached gpt2 model.safetensors in HF_HOME."""
    p = HF_CACHE / "models--gpt2" / "snapshots"
    if not p.exists():
        return None
    for snap in p.iterdir():
        st = snap / "model.safetensors"
        if st.exists():
            return st
    return None


@pytest.fixture(scope="module")
def gpt2_bs_dir():
    """Compress cached GPT-2 once for both tests, clean up at end."""
    if _gpt2_cached() is None:
        pytest.skip("GPT-2 not in HF cache")
    import bigsmall
    out = Path(tempfile.mkdtemp(prefix="bigsmall_hf_test_"))
    try:
        bigsmall.compress_for_hub("gpt2", output_dir=out, overwrite=True)
        yield out
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_compress_for_hub_writes_valid_index(gpt2_bs_dir):
    out = gpt2_bs_dir
    idx_path = out / "bigsmall.index.json"
    assert idx_path.exists(), "bigsmall.index.json was not written"

    with open(idx_path, "r", encoding="utf-8") as f:
        idx = json.load(f)

    meta = idx["metadata"]
    assert "bigsmall_version" in meta
    assert meta["container_version"] == 1
    assert meta["shard_count"] >= 1
    assert meta["tensor_count"] > 0
    assert meta["format"] in ("fp32", "fp16", "bf16", "fp8", "fp4", "mixed")
    assert meta["mode"] in ("storage", "balanced", "inference", "mixed")
    assert isinstance(meta["shards"], list) and len(meta["shards"]) == meta["shard_count"]

    # Every listed shard exists on disk and is non-empty.
    total_bytes = 0
    for shard_name in meta["shards"]:
        sp = out / shard_name
        assert sp.exists(), f"shard {shard_name} not on disk"
        total_bytes += sp.stat().st_size
    assert total_bytes == meta["total_size"], (
        f"recorded total_size {meta['total_size']} != on-disk {total_bytes}"
    )

    # weight_map covers all tensors and every value points at a listed shard.
    assert len(idx["weight_map"]) == meta["tensor_count"]
    for name, shard in idx["weight_map"].items():
        assert shard in meta["shards"], f"weight {name} -> unknown shard {shard}"

    # ratio is sensible (10-100%).
    assert 10.0 <= meta["ratio_pct"] <= 100.0


def test_from_pretrained_roundtrip_gpt2(gpt2_bs_dir):
    """state_dict returned by from_pretrained == source safetensors, byte-for-byte."""
    import bigsmall
    import torch
    from safetensors.torch import load_file

    src = _gpt2_cached()
    assert src is not None

    sd = bigsmall.from_pretrained(str(gpt2_bs_dir), device="cpu", show_progress=False)
    orig = load_file(str(src))

    assert set(sd.keys()) == set(orig.keys()), (
        f"key sets differ: only-sd={list(set(sd) - set(orig))[:3]} "
        f"only-orig={list(set(orig) - set(sd))[:3]}"
    )

    bad = []
    for k in sd:
        a = sd[k]; b = orig[k]
        if a.shape != b.shape or a.dtype != b.dtype:
            bad.append((k, "shape/dtype"))
            continue
        if not torch.equal(a, b):
            bad.append((k, "values"))
    assert not bad, f"{len(bad)} tensors differ from source: {bad[:3]}"
