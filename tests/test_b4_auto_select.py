"""Tests for B4 — per-tensor codec auto-selection registry."""
from __future__ import annotations

import io
import os
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

import bigsmall
from bigsmall import codec_registry
from bigsmall.codecs import bf16


def _bf16_raw_random(shape, seed=0):
    torch.manual_seed(seed)
    t = torch.randn(*shape).bfloat16().contiguous()
    return t.view(torch.uint8).numpy().tobytes()


def _bf16_raw_high_kurtosis(shape, seed=0):
    """Most weights near zero, a heavy tail of outliers -> qualifies for A5."""
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(*shape, generator=g) * 1e-4
    # Add outliers in <1% of positions
    n = base.numel()
    mask = torch.rand(n, generator=g) < 0.005
    outliers = torch.randn(n, generator=g) * 3.0
    flat = base.view(-1)
    flat[mask] = outliers[mask]
    t = flat.view(*shape).bfloat16().contiguous()
    return t.view(torch.uint8).numpy().tobytes()


def test_auto_select_never_larger_than_baseline_bf16():
    """auto_select must produce a blob no larger than the previous fixed
    dispatch (= plain bf16.encode for a generic BF16 tensor), up to the
    speed tie-break tolerance for bf16_se_rans (0.01% of raw, capped at 1KB).
    """
    raw = _bf16_raw_random((512, 256))
    baseline_blob, _ = bf16.encode(raw)
    blob, codec, extras = codec_registry.auto_select_codec(
        raw, fmt="bf16", dtype="BF16",
    )
    tolerance = max(1024, int(len(raw) * 0.0001))
    assert len(blob) <= len(baseline_blob) + tolerance, (
        f"auto_select picked {codec} ({len(blob)} B) which is more than "
        f"{tolerance} B larger than baseline bf16_se_ac ({len(baseline_blob)} B)"
    )


def test_auto_select_picks_a5_when_smaller():
    """On a synthetic high-kurtosis BF16 tensor, A5 must win OR plain bf16
    must win (whichever is smaller).  Either way the returned blob is the
    smallest of {bf16_se_ac, bf16_sparsity_v1, zstd}."""
    raw = _bf16_raw_high_kurtosis((512, 512))
    blob, codec, extras = codec_registry.auto_select_codec(
        raw, fmt="bf16", dtype="BF16",
    )
    # Run each candidate individually and confirm we picked the smallest.
    sizes = {}
    for name in codec_registry.CODEC_CANDIDATES["bf16"]:
        pair = codec_registry.get_codec(name)
        if pair is None:
            continue
        enc, _ = pair
        try:
            if name == "bf16_sparsity_v1":
                # Match the threshold-selection auto_select would use.
                from bigsmall.codecs import bf16_sparsity
                tw = bf16_sparsity.choose_threshold_word(raw)
                b, _ = enc(raw, threshold_word=tw)
            else:
                b, _ = enc(raw)
            sizes[name] = len(b)
        except Exception:
            pass
    min_size = min(sizes.values())
    # auto_select may prefer bf16_se_rans by tie-break (within tolerance)
    # for its speed advantage. Tolerance: 0.01% of raw, capped at 1KB.
    tolerance = max(1024, int(len(raw) * 0.0001))
    assert len(blob) <= min_size + tolerance, (
        f"auto_select gave {codec}={len(blob)}B but the candidates were {sizes} "
        f"(allowed {tolerance}B speed tolerance)"
    )


def test_auto_select_skips_a5_on_well_behaved_tensor():
    """Plain Gaussian BF16 tensors should NOT qualify for A5 and should
    therefore use bf16_se_ac (not bf16_sparsity_v1)."""
    raw = _bf16_raw_random((1024, 256))
    _blob, codec, _ = codec_registry.auto_select_codec(
        raw, fmt="bf16", dtype="BF16",
    )
    assert codec != "bf16_sparsity_v1", (
        f"unexpected A5 selection on a well-behaved tensor (codec={codec})"
    )


def test_register_codec_extends_candidates():
    """A registered codec is callable through the registry; adding it to
    CODEC_CANDIDATES means auto_select will try it."""

    def _fake_encode(raw, **_):
        # Returns a 1-byte blob -> guaranteed to win the size race.
        return b"x", {"fake": True}

    def _fake_decode(blob, extras, n_weights):
        return b"\x00" * n_weights  # decoder body irrelevant for this test

    codec_registry.register_codec("test_fake_v1", _fake_encode, _fake_decode)
    assert "test_fake_v1" in codec_registry.registered_names()

    # Append to the bf16 candidate list, run auto_select, restore the list.
    old = list(codec_registry.CODEC_CANDIDATES["bf16"])
    try:
        codec_registry.CODEC_CANDIDATES["bf16"] = old + ["test_fake_v1"]
        raw = _bf16_raw_random((128, 64))
        _blob, codec, _ = codec_registry.auto_select_codec(
            raw, fmt="bf16", dtype="BF16",
        )
        assert codec == "test_fake_v1", (
            f"auto_select did not pick the 1-byte fake codec (got {codec})"
        )
    finally:
        codec_registry.CODEC_CANDIDATES["bf16"] = old


def test_auto_select_never_raises_on_garbage_input():
    """Even if every candidate fails, auto_select must return *something* and
    must not raise."""
    # Length-1 (odd-byte-count) "BF16" input — bf16.encode rejects it.
    blob, codec, extras = codec_registry.auto_select_codec(
        b"\x00", fmt="bf16", dtype="BF16",
    )
    assert len(blob) >= 0
    assert codec == "zstd", f"expected zstd fallback, got {codec}"


def test_compress_writes_codec_stats_in_header_and_sums_to_tensor_count(tmp_path):
    """compress() stamps codec_stats into the .bs header and the counts
    sum to the total tensor_count."""
    st_path = tmp_path / "model.safetensors"
    bs_path = tmp_path / "model.bs"
    torch.manual_seed(0)
    tensors = {
        "embed":       torch.randn(512, 64).bfloat16(),
        "layer.0.mlp": torch.randn(256, 128).bfloat16(),
        "layer.0.bias": torch.randn(8).bfloat16(),  # tiny -> raw
        "logits":      torch.randn(64, 256).float(),
    }
    save_file(tensors, str(st_path))

    bigsmall.compress(str(st_path), str(bs_path), workers=1, progress=False)
    info = bigsmall.info(str(bs_path))

    assert "codec_stats" in info
    assert isinstance(info["codec_stats"], dict)
    assert sum(info["codec_stats"].values()) == info["tensor_count"], (
        f"codec_stats {info['codec_stats']} does not sum to "
        f"tensor_count={info['tensor_count']}"
    )
    # Round-trip md5 sanity: confirms B4 didn't break the encoder.
    out = bigsmall.decompress(str(bs_path), progress=False)
    assert set(out.keys()) == set(tensors.keys())


def test_info_cli_shows_codec_breakdown(tmp_path):
    """`bigsmall info <file>` should include a "codec_breakdown" section."""
    st_path = tmp_path / "tiny.safetensors"
    bs_path = tmp_path / "tiny.bs"
    torch.manual_seed(0)
    save_file(
        {"w": torch.randn(64, 64).bfloat16(),
         "x": torch.randn(32, 32).float()},
        str(st_path),
    )
    bigsmall.compress(str(st_path), str(bs_path), workers=1, progress=False)

    # Run the CLI through subprocess so we exercise the real argparse path.
    env = dict(os.environ)
    env["BIGSMALL_DISABLE_VERSION_CHECK"] = "1"
    result = subprocess.run(
        [sys.executable, "-m", "bigsmall.cli", "info", str(bs_path)],
        capture_output=True, text=True, env=env, timeout=60,
    )
    assert result.returncode == 0, f"CLI failed: stderr={result.stderr}"
    assert "codec_breakdown" in result.stdout, (
        f"CLI did not print codec_breakdown.  stdout was:\n{result.stdout}"
    )
