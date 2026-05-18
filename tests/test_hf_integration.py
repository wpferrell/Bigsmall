"""HuggingFace Hub round-trip tests on a synthetic model directory.

We build a tiny safetensors file in a temp dir, run `compress_for_hub` on it,
check the resulting index, and verify that `from_pretrained` returns a
state_dict that is byte-identical to the source.
"""
import json
import tempfile
from pathlib import Path

import pytest


def _make_synthetic_hf_dir(tmp_dir: Path, dtype) -> Path:
    """Write a synthetic safetensors file + minimal config.json into tmp_dir."""
    import torch
    from safetensors.torch import save_file

    torch.manual_seed(0)
    tensors = {
        "embed.weight": torch.randn(2048, 256, dtype=dtype),
        "layer.0.attn.weight": torch.randn(512, 512, dtype=dtype),
        "layer.0.mlp.weight": torch.randn(1024, 512, dtype=dtype),
        "ln_f.weight": torch.randn(512, dtype=dtype),
    }
    save_file(tensors, str(tmp_dir / "model.safetensors"))
    (tmp_dir / "config.json").write_text(
        json.dumps({"model_type": "synthetic", "hidden_size": 512}),
        encoding="utf-8",
    )
    return tmp_dir


@pytest.fixture(scope="module")
def synthetic_bs_dir():
    """Compress a synthetic 1-shard model once for both tests."""
    try:
        import torch  # noqa: F401
        import safetensors  # noqa: F401
    except ImportError:
        pytest.skip("torch/safetensors not installed")
    import bigsmall
    import torch

    work = Path(tempfile.mkdtemp(prefix="bigsmall_hf_test_"))
    src_dir = work / "src"
    src_dir.mkdir()
    out_dir = work / "out"
    _make_synthetic_hf_dir(src_dir, torch.bfloat16)
    try:
        bigsmall.compress_for_hub(str(src_dir), output_dir=out_dir, overwrite=True)
        yield {"src": src_dir, "out": out_dir}
    finally:
        import shutil
        shutil.rmtree(work, ignore_errors=True)


def test_compress_for_hub_writes_valid_index(synthetic_bs_dir):
    out = synthetic_bs_dir["out"]
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

    total_bytes = 0
    for shard_name in meta["shards"]:
        sp = out / shard_name
        assert sp.exists(), f"shard {shard_name} not on disk"
        total_bytes += sp.stat().st_size
    assert total_bytes == meta["total_size"], (
        f"recorded total_size {meta['total_size']} != on-disk {total_bytes}"
    )

    assert len(idx["weight_map"]) == meta["tensor_count"]
    for name, shard in idx["weight_map"].items():
        assert shard in meta["shards"], f"weight {name} -> unknown shard {shard}"

    assert 10.0 <= meta["ratio_pct"] <= 100.0


def test_from_pretrained_roundtrip_synthetic(synthetic_bs_dir):
    """state_dict from from_pretrained == source safetensors, byte-for-byte."""
    import bigsmall
    import torch
    from safetensors.torch import load_file

    out_dir = synthetic_bs_dir["out"]
    src_st = synthetic_bs_dir["src"] / "model.safetensors"

    sd = bigsmall.from_pretrained(str(out_dir), device="cpu", show_progress=False)
    orig = load_file(str(src_st))

    assert set(sd.keys()) == set(orig.keys()), (
        f"key sets differ: only-sd={list(set(sd) - set(orig))[:3]} "
        f"only-orig={list(set(orig) - set(sd))[:3]}"
    )

    bad = []
    for k in sd:
        a = sd[k]
        b = orig[k]
        if a.shape != b.shape or a.dtype != b.dtype:
            bad.append((k, "shape/dtype"))
            continue
        if not torch.equal(a, b):
            bad.append((k, "values"))
    assert not bad, f"{len(bad)} tensors differ from source: {bad[:3]}"
