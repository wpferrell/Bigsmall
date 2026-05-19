"""StreamingLoader tests on a synthetic transformer-shaped model.

We build a tiny multi-layer state-dict with the `model.layers.N.<...>` naming
convention the StreamingLoader recognises, compress it via `compress_for_hub`,
then iterate the resulting .bs through `StreamingLoader` and check that every
decompressed tensor is byte-identical to the source.

The original GPT-2 specific test (`StreamingGPT2.generate_greedy` vs.
`GPT2LMHeadModel.generate`) is retained but skipped when GPT-2 is not present
in the local HF cache.
"""
import hashlib
import os
import shutil
import tempfile
from pathlib import Path

import pytest


HF_CACHE = Path(os.environ.get("HF_HOME") or
                os.path.expandvars(r"%USERPROFILE%\.cache\huggingface")) / "hub"


def _gpt2_cached() -> Path | None:
    p = HF_CACHE / "models--gpt2" / "snapshots"
    if not p.exists():
        return None
    for snap in p.iterdir():
        st = snap / "model.safetensors"
        if st.exists():
            return st
    return None


N_LAYERS = 4
HIDDEN = 256


def _make_synthetic_transformer(tmp_dir: Path):
    """Write a synthetic transformer-shaped safetensors file into tmp_dir."""
    import torch
    from safetensors.torch import save_file

    torch.manual_seed(0)
    tensors: dict = {
        "embed.weight": torch.randn(1024, HIDDEN, dtype=torch.bfloat16),
        "ln_f.weight": torch.randn(HIDDEN, dtype=torch.bfloat16),
        "ln_f.bias": torch.zeros(HIDDEN, dtype=torch.bfloat16),
    }
    for i in range(N_LAYERS):
        prefix = f"model.layers.{i}"
        tensors[f"{prefix}.self_attn.q_proj.weight"] = torch.randn(HIDDEN, HIDDEN, dtype=torch.bfloat16)
        tensors[f"{prefix}.self_attn.k_proj.weight"] = torch.randn(HIDDEN, HIDDEN, dtype=torch.bfloat16)
        tensors[f"{prefix}.mlp.up_proj.weight"] = torch.randn(HIDDEN * 2, HIDDEN, dtype=torch.bfloat16)
        tensors[f"{prefix}.mlp.down_proj.weight"] = torch.randn(HIDDEN, HIDDEN * 2, dtype=torch.bfloat16)
        tensors[f"{prefix}.input_layernorm.weight"] = torch.randn(HIDDEN, dtype=torch.bfloat16)
    save_file(tensors, str(tmp_dir / "model.safetensors"))
    return tensors


@pytest.fixture(scope="module")
def synthetic_bs_dir():
    try:
        import torch  # noqa: F401
        import safetensors  # noqa: F401
    except ImportError:
        pytest.skip("torch/safetensors not installed")
    import bigsmall

    work = Path(tempfile.mkdtemp(prefix="bigsmall_streaming_test_"))
    src_dir = work / "src"
    src_dir.mkdir()
    out_dir = work / "out"
    src_tensors = _make_synthetic_transformer(src_dir)
    try:
        bigsmall.compress_for_hub(str(src_dir), output_dir=out_dir, overwrite=True)
        yield {"out": out_dir, "tensors": src_tensors}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_streaming_layer_count(synthetic_bs_dir):
    from bigsmall.streaming import StreamingLoader

    with StreamingLoader(str(synthetic_bs_dir["out"]), device="cpu") as L:
        assert L.layer_count() == N_LAYERS
        non_layer = L.non_layer_tensor_names()
        assert sorted(non_layer) == ["embed.weight", "ln_f.bias", "ln_f.weight"]
        total = sum(len(L.layer_tensor_names(i)) for i in range(N_LAYERS)) + len(non_layer)
        assert total == len(synthetic_bs_dir["tensors"])


def test_streaming_md5_roundtrip_all_layers(synthetic_bs_dir):
    """Every decompressed tensor must md5-match its source."""
    import torch
    from bigsmall.streaming import StreamingLoader, layer_index

    orig = synthetic_bs_dir["tensors"]
    name_to_md5 = {
        k: hashlib.md5(v.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
        for k, v in orig.items()
    }

    bad: list[tuple[str, str]] = []
    seen: set[str] = set()
    with StreamingLoader(str(synthetic_bs_dir["out"]), device="cpu") as L:
        for name, t in L.load_non_layer_tensors().items():
            seen.add(name)
            md5 = hashlib.md5(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
            if md5 != name_to_md5[name]:
                bad.append((name, "md5 mismatch"))
        for layer_idx, layer in L.iter_layers():
            for name, t in layer.items():
                seen.add(name)
                assert layer_index(name) == layer_idx
                md5 = hashlib.md5(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
                if md5 != name_to_md5[name]:
                    bad.append((name, "md5 mismatch"))

    missing = set(name_to_md5) - seen
    extra = seen - set(name_to_md5)
    assert not bad, f"md5 mismatch: {bad[:3]}"
    assert not missing, f"missing tensors: {sorted(missing)[:3]}"
    assert not extra, f"extra tensors: {sorted(extra)[:3]}"


@pytest.mark.integration
def test_streaming_inference_identical_to_full():
    """StreamingGPT2 greedy generation == GPT2LMHeadModel.

    Marked `integration` (v3.11.0): requires a real GPT-2 in the HF cache
    (~500 MB). Skipped from the default `pytest tests/` run; opt in
    with `pytest -m integration tests/`.
    """
    src = _gpt2_cached()
    if src is None:
        pytest.skip("GPT-2 not in HF cache")
    try:
        from transformers import GPT2Tokenizer, GPT2LMHeadModel  # noqa: F401
    except ImportError:
        pytest.skip("transformers not installed")

    import bigsmall
    import torch
    from bigsmall.streaming import StreamingLoader
    from bigsmall.streaming_model import StreamingGPT2

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "gpt2_bs"
        bigsmall.compress_for_hub("gpt2", output_dir=out, overwrite=True)

        tok = GPT2Tokenizer.from_pretrained("gpt2")
        prompt = "The future of artificial intelligence is"
        inp = tok(prompt, return_tensors="pt").input_ids

        with StreamingLoader(str(out), device="cpu") as L:
            sm = StreamingGPT2(L, device="cpu")
            out_streaming = sm.generate_greedy(inp, max_new_tokens=5)

        orig_model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
        out_full = orig_model.generate(
            input_ids=inp, max_new_tokens=5, do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
        assert torch.equal(out_streaming.cpu(), out_full.cpu()), (
            f"streaming={out_streaming.tolist()} full={out_full.tolist()}"
        )
