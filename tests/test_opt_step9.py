"""Tests for v3.12.0 storage features.

Covers:
  1. Binary index roundtrip — write_binary_index then read_binary_index
     reproduces the same tensor → shard mapping as the JSON index.
  2. Binary index is written by compress_for_hub when tensor count
     crosses BINARY_INDEX_MIN_TENSORS.
  3. Binary index is NOT written for small models (below threshold).
  4. maybe_read_binary_index returns None when .bin missing.
  5. read_binary_index rejects a corrupted file (bad magic).
  6. Tensor deduplication (already in v2.2.0): synthetic safetensors
     with two md5-identical tensors compresses with `tied_ref` for
     the duplicate, decompress recovers both tensors bit-identically.
"""
from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import pytest


def _md5(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def _build_many_tensor_dict(n: int, seed: int = 0):
    """A dict of `n` distinct BF16 tensors, all small."""
    import torch
    torch.manual_seed(seed)
    return {
        f"model.layers.{i}.q.weight":
            (torch.randn(32, 32) * 0.02).to(torch.bfloat16)
        for i in range(n)
    }


def _make_synthetic_bs_dir(td: Path, n_tensors: int, seed: int = 0) -> Path:
    """Compress a synthetic model into td. Returns the directory."""
    from safetensors.torch import save_file
    import bigsmall
    src = td / "model.safetensors"
    save_file(_build_many_tensor_dict(n_tensors, seed), str(src))
    out = td / "bs_out"
    out.mkdir(exist_ok=True)
    # Use the public compress_for_hub which wires the binary-index write.
    bigsmall.compress_for_hub(td, output_dir=out, overwrite=True, mode="balanced")
    return out


def test_binary_index_roundtrip():
    """write_binary_index → read_binary_index returns identical weight_map."""
    try:
        import torch  # noqa: F401
        import safetensors  # noqa: F401
    except ImportError:
        pytest.skip("torch/safetensors not installed")
    from bigsmall import hub_index
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        out = _make_synthetic_bs_dir(td, n_tensors=12)
        # Force-write the binary index even though the model is small —
        # call write_binary_index directly with the shard list from the JSON.
        idx_json = hub_index.read_index(out)
        shards = hub_index.shard_paths_from_index(out, index=idx_json)
        bin_path = hub_index.write_binary_index(out, shards)
        assert bin_path.exists()

        # Now read it back
        bin_idx = hub_index.read_binary_index(bin_path)
        assert bin_idx["weight_map"] == idx_json["weight_map"], \
            "binary index weight_map differs from JSON"
        assert bin_idx["metadata"]["tensor_count"] == \
            idx_json["metadata"]["stored_tensor_count"]


def test_binary_index_written_for_large_models():
    """compress_for_hub writes .bin when tensor count crosses threshold."""
    try:
        import torch  # noqa: F401
        import safetensors  # noqa: F401
    except ImportError:
        pytest.skip("torch/safetensors not installed")
    from bigsmall import hub_index
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # Threshold is 100. Build 120 tensors.
        out = _make_synthetic_bs_dir(td, n_tensors=120)
        assert (out / hub_index.BINARY_INDEX_FILENAME).exists(), \
            "binary index NOT written for a 120-tensor model"


def test_binary_index_skipped_for_small_models():
    """compress_for_hub does NOT write .bin for tiny models."""
    try:
        import torch  # noqa: F401
        import safetensors  # noqa: F401
    except ImportError:
        pytest.skip("torch/safetensors not installed")
    from bigsmall import hub_index
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        out = _make_synthetic_bs_dir(td, n_tensors=10)
        assert not (out / hub_index.BINARY_INDEX_FILENAME).exists(), \
            "binary index unexpectedly written for a 10-tensor model"


def test_maybe_read_binary_index_returns_none_when_missing():
    """maybe_read_binary_index returns None when the .bin doesn't exist."""
    from bigsmall import hub_index
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        assert hub_index.maybe_read_binary_index(td) is None


def test_read_binary_index_rejects_bad_magic():
    """read_binary_index raises ValueError on a non-binary-index file."""
    from bigsmall import hub_index
    with tempfile.TemporaryDirectory() as td:
        bad = Path(td) / hub_index.BINARY_INDEX_FILENAME
        bad.write_bytes(b"NOTABS\x00\x00\x00\x00\x00\x00\x00\x00")
        with pytest.raises(ValueError, match="Not a BigSmall binary index"):
            hub_index.read_binary_index(bad)


def test_tied_tensor_deduplication():
    """Dedup (v2.2.0 feature, re-verified): two md5-identical tensors
    are stored once + tied_ref, decompress recovers both bit-identically.

    This protects against future refactors silently breaking the dedup
    code path. The Step 0 measurement showed no local model has tied
    tensors, but the feature itself must keep working for the rare
    models that do (older GPT-2, some early T5 variants).
    """
    try:
        import torch
        import safetensors  # noqa: F401
    except ImportError:
        pytest.skip("torch/safetensors not installed")
    from safetensors import safe_open
    from safetensors.torch import save_file
    import bigsmall
    from bigsmall import container

    torch.manual_seed(7)
    shared = (torch.randn(64, 64) * 0.02).to(torch.bfloat16)
    sd = {
        "model.embed_tokens.weight": shared,
        "lm_head.weight": shared.clone(),  # md5-identical bytes
        "model.layers.0.q.weight": (torch.randn(32, 32) * 0.02).to(torch.bfloat16),
    }
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = td / "model.safetensors"
        save_file(sd, str(src))
        bs = td / "model.bs"
        bigsmall.compress(src, bs, workers=1, progress=False)

        # The header should mark one of the duplicates as tied_ref.
        header, _ = container.read_header(bs)
        codecs = {t["name"]: t["codec"] for t in header["tensors"]}
        tied = [n for n, c in codecs.items() if c == "tied_ref"]
        assert len(tied) >= 1, (
            f"expected at least one tied_ref tensor; got codecs: {codecs}"
        )

        # Decompress and verify both copies recover bit-identically.
        out = bigsmall.decompress(bs)
        with safe_open(str(src), framework="pt") as f:
            for name in ("model.embed_tokens.weight", "lm_head.weight"):
                t = f.get_tensor(name)
                src_b = t.contiguous().view(torch.uint8).cpu().numpy().tobytes()
                assert _md5(src_b) == _md5(out[name].tobytes()), \
                    f"dedup decompress failed for {name}"
