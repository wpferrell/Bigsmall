"""Cross-shard tied-weight deduplication tests.

We build a synthetic 2-shard HF model where `lm_head.weight` (shard 2) is
byte-identical to `embed.weight` (shard 1) and an unrelated tied pair
`copy_a.weight` / `copy_b.weight` is split across shards. `compress_for_hub`
must detect both, store only the master copies, and record the duplicates in
`bigsmall.index.json`. `from_pretrained` must reconstruct the duplicates;
`StreamingLoader.load_tensor` must transparently return master data when
queried for a duplicate name.
"""
import json
import tempfile
from pathlib import Path

import pytest


HIDDEN = 256
VOCAB = 1024


def _make_two_shard_model(work: Path):
    """Two safetensors files: shard1 has embed + copy_a + ln; shard2 has lm_head
    (== embed bytes) + copy_b (== copy_a bytes) + matrix.
    """
    import torch
    from safetensors.torch import save_file

    torch.manual_seed(0)
    embed = torch.randn(VOCAB, HIDDEN, dtype=torch.bfloat16)
    copy_a = torch.randn(HIDDEN, HIDDEN, dtype=torch.bfloat16)
    matrix = torch.randn(HIDDEN, HIDDEN, dtype=torch.bfloat16)
    ln = torch.randn(HIDDEN, dtype=torch.bfloat16)

    shard1 = {
        "model.embed.weight": embed,
        "model.copy_a.weight": copy_a,
        "model.ln.weight": ln,
    }
    shard2 = {
        "model.lm_head.weight": embed.clone(),     # tied to embed
        "model.copy_b.weight": copy_a.clone(),     # tied to copy_a
        "model.layers.0.mlp.weight": matrix,
    }

    src = work / "src"
    src.mkdir()
    save_file(shard1, str(src / "model-00001-of-00002.safetensors"))
    save_file(shard2, str(src / "model-00002-of-00002.safetensors"))

    # Minimal safetensors index so compress_for_hub walks the shards in order.
    weight_map = {n: "model-00001-of-00002.safetensors" for n in shard1}
    weight_map.update({n: "model-00002-of-00002.safetensors" for n in shard2})
    (src / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}), encoding="utf-8"
    )
    (src / "config.json").write_text(json.dumps({"model_type": "synthetic"}), encoding="utf-8")
    return src


def _shard_tensor_count(out_dir: Path, shard_name: str) -> int:
    """Count tensors in a single .bs shard via its header."""
    from bigsmall import container
    header, _ = container.read_header(out_dir / shard_name)
    return header["tensor_count"]


@pytest.fixture(scope="module")
def deduped_model():
    try:
        import torch  # noqa: F401
        import safetensors  # noqa: F401
    except ImportError:
        pytest.skip("torch/safetensors not installed")
    import bigsmall

    work = Path(tempfile.mkdtemp(prefix="bigsmall_dedup_"))
    try:
        src = _make_two_shard_model(work)
        out = work / "bs"
        bigsmall.compress_for_hub(str(src), output_dir=out, overwrite=True,
                                  workers=1)
        yield {"src": src, "out": out}
    finally:
        import shutil
        shutil.rmtree(work, ignore_errors=True)


def test_index_records_duplicate_map(deduped_model):
    out = deduped_model["out"]
    idx = json.loads((out / "bigsmall.index.json").read_text(encoding="utf-8"))
    dup = idx["metadata"].get("duplicate_map") or {}
    assert set(dup.keys()) == {"model.lm_head.weight", "model.copy_b.weight"}, dup
    assert dup["model.lm_head.weight"]["master"] == "model.embed.weight"
    assert dup["model.copy_b.weight"]["master"] == "model.copy_a.weight"

    # The duplicates participate in weight_map (so consumers can locate them)
    # but stored_tensor_count is exactly the master count.
    meta = idx["metadata"]
    assert idx["weight_map"]["model.lm_head.weight"] == \
        idx["weight_map"]["model.embed.weight"]
    assert meta["stored_tensor_count"] + len(dup) == meta["tensor_count"]


def test_duplicates_are_excluded_from_shard(deduped_model):
    out = deduped_model["out"]
    # Shard 1 has 3 tensors (embed, copy_a, ln); shard 2 originally had 3
    # but two are deduped so only `matrix` remains.
    assert _shard_tensor_count(out, "model-00001-of-00002.bs") == 3
    assert _shard_tensor_count(out, "model-00002-of-00002.bs") == 1


def test_from_pretrained_reconstructs_duplicates(deduped_model):
    import bigsmall
    sd = bigsmall.from_pretrained(str(deduped_model["out"]), device="cpu",
                                  show_progress=False)
    assert "model.embed.weight" in sd
    assert "model.lm_head.weight" in sd
    # Both must be byte-identical (we alias, not copy).
    import torch
    assert torch.equal(sd["model.embed.weight"], sd["model.lm_head.weight"])
    assert torch.equal(sd["model.copy_a.weight"], sd["model.copy_b.weight"])


def test_streaming_loader_resolves_duplicates(deduped_model):
    import torch
    from bigsmall.streaming import StreamingLoader

    with StreamingLoader(str(deduped_model["out"]), device="cpu") as L:
        names = set(L.tensor_names())
        # Both the master AND the duplicate name should be queryable.
        assert "model.embed.weight" in names
        assert "model.lm_head.weight" in names
        master_t = L.load_tensor("model.embed.weight")
        dup_t = L.load_tensor("model.lm_head.weight")
        assert torch.equal(master_t, dup_t)


def test_total_bytes_smaller_than_naive(deduped_model):
    """Cross-shard dedup must actually save bytes on disk."""
    out = deduped_model["out"]
    shard1 = (out / "model-00001-of-00002.bs").stat().st_size
    shard2 = (out / "model-00002-of-00002.bs").stat().st_size
    total = shard1 + shard2
    # The deduped shard2 should be very small (only the `matrix` payload plus
    # container overhead) -- well under what a non-deduped shard would be.
    # Naive worst case is ~equal to shard1, so just sanity-check the obvious.
    assert shard2 < shard1, (shard1, shard2)
