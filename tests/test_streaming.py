"""Streaming loader tests (Phase 4 cont.).

Three tests on GPT-2:
 1. test_streaming_layer_count                 - layer_count() == 12 for GPT-2.
 2. test_streaming_md5_roundtrip_all_layers    - iterate every layer, md5 every
    tensor vs the source safetensors.
 3. test_streaming_inference_identical_to_full - StreamingGPT2.generate_greedy
    output equals GPT2LMHeadModel.generate output (same ids, greedy).
"""
# transformers must be imported before torch on this Windows env to avoid a
# native crash. The streaming model module imports torch directly, so users
# of streaming on Windows generally need to import transformers first.
from transformers import GPT2Tokenizer, GPT2LMHeadModel  # noqa: F401

import hashlib
import os
import shutil
import tempfile
from pathlib import Path

import pytest
import torch

import bigsmall
from bigsmall.streaming import StreamingLoader, layer_index
from bigsmall.streaming_model import StreamingGPT2


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


@pytest.fixture(scope="module")
def gpt2_bs_dir():
    """Compress GPT-2 once for all streaming tests."""
    if _gpt2_cached() is None:
        pytest.skip("GPT-2 not in HF cache")
    out = Path(tempfile.mkdtemp(prefix="bigsmall_streaming_test_"))
    try:
        bigsmall.compress_for_hub("gpt2", output_dir=out, overwrite=True)
        yield out
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_streaming_layer_count(gpt2_bs_dir):
    """GPT-2 has 12 transformer layers and 4 non-layer tensors."""
    with StreamingLoader(str(gpt2_bs_dir), device="cpu") as L:
        assert L.layer_count() == 12
        non_layer = L.non_layer_tensor_names()
        # GPT-2 safetensors: wte, wpe, ln_f.weight, ln_f.bias (lm_head is tied)
        assert sorted(non_layer) == ["ln_f.bias", "ln_f.weight",
                                     "wpe.weight", "wte.weight"]
        # 12 layers x 13 tensors each + 4 non-layer = 160
        total = sum(len(L.layer_tensor_names(i)) for i in range(12)) + len(non_layer)
        assert total == 160


def test_streaming_md5_roundtrip_all_layers(gpt2_bs_dir):
    """Iterate every layer + non-layer through the streaming loader; every
    decompressed tensor must md5-match the source safetensors bytes."""
    from safetensors.torch import load_file
    src = _gpt2_cached()
    orig = load_file(str(src))
    name_to_md5 = {
        k: hashlib.md5(v.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
        for k, v in orig.items()
    }

    bad: list[tuple[str, str]] = []
    seen: set[str] = set()
    with StreamingLoader(str(gpt2_bs_dir), device="cpu") as L:
        for name, t in L.load_non_layer_tensors().items():
            seen.add(name)
            md5 = hashlib.md5(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
            if md5 != name_to_md5[name]:
                bad.append((name, "md5 mismatch"))
        for layer_idx, layer in L.iter_layers():
            for name, t in layer.items():
                seen.add(name)
                # Verify the layer index matches the regex
                assert layer_index(name) == layer_idx
                md5 = hashlib.md5(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
                if md5 != name_to_md5[name]:
                    bad.append((name, "md5 mismatch"))

    missing = set(name_to_md5) - seen
    extra = seen - set(name_to_md5)
    assert not bad, f"md5 mismatch: {bad[:3]}"
    assert not missing, f"missing tensors: {sorted(missing)[:3]}"
    assert not extra, f"extra tensors: {sorted(extra)[:3]}"


def test_streaming_inference_identical_to_full(gpt2_bs_dir):
    """StreamingGPT2 greedy generation produces the exact same token ids as
    a full-load GPT2LMHeadModel."""
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    prompt = "The future of artificial intelligence is"
    inp = tok(prompt, return_tensors="pt").input_ids

    with StreamingLoader(str(gpt2_bs_dir), device="cpu") as L:
        sm = StreamingGPT2(L, device="cpu")
        # 5 new tokens keeps the test fast; correctness signal is the same.
        out_streaming = sm.generate_greedy(inp, max_new_tokens=5)

    orig = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    out_full = orig.generate(
        input_ids=inp, max_new_tokens=5, do_sample=False,
        pad_token_id=tok.eos_token_id,
    )
    assert torch.equal(out_streaming.cpu(), out_full.cpu()), (
        f"streaming={out_streaming.tolist()} full={out_full.tolist()}"
    )
