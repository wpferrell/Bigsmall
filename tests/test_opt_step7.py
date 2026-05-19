"""Tests for v3.10.0 CLI improvements.

Covers:
  1. `bigsmall verify --fast` on a valid .bs file → exit 0
  2. `bigsmall verify --fast` on a corrupted .bs file (truncated header offset)
     → exit 1 with diagnostic problems printed
  3. `bigsmall stat` outputs expected tensor count + summary
  4. `bigsmall diff` correctly identifies added / removed / changed tensors
  5. `bigsmall benchmark` runs without error and prints per-layer-type breakdown

All tests invoke the CLI module directly (subprocess) so they cover the
real arg-parsing + handler-dispatch path.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


def _run_cli(*args: str, check_zero: bool = False) -> subprocess.CompletedProcess:
    """Run `python -m bigsmall.cli <args>` and return the process."""
    proc = subprocess.run(
        [sys.executable, "-m", "bigsmall.cli", *args],
        capture_output=True, text=True, timeout=300,
    )
    if check_zero:
        assert proc.returncode == 0, (
            f"CLI exit {proc.returncode}\nstdout: {proc.stdout}\nstderr: {proc.stderr}"
        )
    return proc


def _build_sample_bs(tmp: Path, n_layers: int = 4, seed: int = 0) -> Path:
    """Compress a small synthetic safetensors and return the .bs path."""
    import torch
    from safetensors.torch import save_file
    import bigsmall
    torch.manual_seed(seed)
    sd = {}
    for i in range(n_layers):
        sd[f"model.layers.{i}.self_attn.qkv_proj.weight"] = (
            torch.randn(128, 128) * 0.02
        ).to(torch.bfloat16)
    sd["lm_head.weight"] = (torch.randn(256, 128) * 0.02).to(torch.bfloat16)
    src = tmp / f"src_seed{seed}.safetensors"
    save_file(sd, str(src))
    dst = tmp / f"out_seed{seed}.bs"
    bigsmall.compress(src, dst, workers=1, progress=False)
    return dst


def test_verify_fast_valid_file():
    """verify --fast on a freshly-compressed .bs exits 0."""
    try:
        import torch  # noqa: F401
        import safetensors  # noqa: F401
    except ImportError:
        pytest.skip("torch/safetensors not installed")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        bs = _build_sample_bs(td)
        proc = _run_cli("verify", "--fast", str(bs))
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "OK" in proc.stdout


def test_verify_fast_corrupted_file():
    """verify --fast on a corrupted .bs returns exit 1 with diagnostics."""
    try:
        import torch  # noqa: F401
        import safetensors  # noqa: F401
    except ImportError:
        pytest.skip("torch/safetensors not installed")
    from bigsmall import container
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        bs = _build_sample_bs(td)
        # Read the header, corrupt one tensor's offset, write back.
        header, data_offset = container.read_header(bs)
        header["tensors"][1]["offset"] = 999_999_999  # past end of data
        # Hand-rewrite the .bs with the broken header.
        import struct
        header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
        # Read the data section from the original
        with open(bs, "rb") as f:
            f.seek(data_offset)
            data_bytes = f.read()
        with open(bs, "wb") as f:
            f.write(container.MAGIC)
            f.write(struct.pack("<H", 2))  # v2 to be safe
            f.write(struct.pack("<I", len(header_json)))
            f.write(header_json)
            f.write(data_bytes)
        proc = _run_cli("verify", "--fast", str(bs))
        assert proc.returncode == 1
        assert "FAIL" in proc.stdout
        assert "past data section end" in proc.stdout or "offset" in proc.stdout


def test_stat_outputs_tensor_count_and_summary():
    """bigsmall stat prints the expected tensor count and summary lines."""
    try:
        import torch  # noqa: F401
        import safetensors  # noqa: F401
    except ImportError:
        pytest.skip("torch/safetensors not installed")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        bs = _build_sample_bs(td, n_layers=3)
        proc = _run_cli("stat", str(bs), check_zero=True)
        # 3 layers + lm_head = 4 tensors
        assert "tensors:    4" in proc.stdout
        assert "overall:" in proc.stdout
        assert "codecs:" in proc.stdout
        # Each tensor should appear by name
        for i in range(3):
            assert f"model.layers.{i}.self_attn.qkv_proj.weight" in proc.stdout
        assert "lm_head.weight" in proc.stdout


def test_stat_tensor_filter():
    """--tensor <substring> filters the output."""
    try:
        import torch  # noqa: F401
        import safetensors  # noqa: F401
    except ImportError:
        pytest.skip("torch/safetensors not installed")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        bs = _build_sample_bs(td, n_layers=3)
        proc = _run_cli("stat", str(bs), "--tensor", "lm_head", check_zero=True)
        assert "lm_head.weight" in proc.stdout
        assert "model.layers.0" not in proc.stdout
        # Exactly 1 tensor matched
        assert "tensors:    1" in proc.stdout


def test_diff_detects_changes():
    """bigsmall diff lists added/removed/changed correctly + exits 1 on differences."""
    try:
        import torch
        import safetensors  # noqa: F401
    except ImportError:
        pytest.skip("torch/safetensors not installed")
    from safetensors.torch import save_file
    import bigsmall

    torch.manual_seed(0)
    common_tensor = (torch.randn(128, 128) * 0.02).to(torch.bfloat16)
    sd_a = {
        "kept.weight": common_tensor.clone(),
        "removed.weight": (torch.randn(128, 64) * 0.02).to(torch.bfloat16),
        "changed.weight": (torch.randn(128, 128) * 0.02).to(torch.bfloat16),
    }
    sd_b = {
        "kept.weight": common_tensor.clone(),
        "changed.weight": (torch.randn(128, 128) * 0.02).to(torch.bfloat16),  # different
        "added.weight": (torch.randn(64, 64) * 0.02).to(torch.bfloat16),
    }
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src_a = td / "a.safetensors"
        src_b = td / "b.safetensors"
        save_file(sd_a, str(src_a))
        save_file(sd_b, str(src_b))
        bs_a = td / "a.bs"
        bs_b = td / "b.bs"
        bigsmall.compress(src_a, bs_a, workers=1, progress=False)
        bigsmall.compress(src_b, bs_b, workers=1, progress=False)

        proc = _run_cli("diff", str(bs_a), str(bs_b))
        # exit 1 because there ARE differences
        assert proc.returncode == 1, proc.stdout
        assert "identical:  1" in proc.stdout
        assert "changed:    1" in proc.stdout
        assert "only in A:  1" in proc.stdout
        assert "only in B:  1" in proc.stdout
        assert "removed.weight" in proc.stdout
        assert "added.weight" in proc.stdout
        assert "changed.weight" in proc.stdout


def test_diff_identical_files_exit_zero():
    """bigsmall diff <same> <same> exits 0."""
    try:
        import torch  # noqa: F401
        import safetensors  # noqa: F401
    except ImportError:
        pytest.skip("torch/safetensors not installed")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        bs = _build_sample_bs(td)
        proc = _run_cli("diff", str(bs), str(bs))
        assert proc.returncode == 0
        assert "identical:" in proc.stdout


def test_benchmark_runs_and_emits_breakdown():
    """bigsmall benchmark runs end-to-end and includes the per-layer breakdown."""
    try:
        import torch
        import safetensors  # noqa: F401
    except ImportError:
        pytest.skip("torch/safetensors not installed")
    from safetensors.torch import save_file
    torch.manual_seed(0)
    sd = {
        "model.layers.0.self_attn.qkv_proj.weight": (torch.randn(256, 256) * 0.02).to(torch.bfloat16),
        "model.layers.0.input_layernorm.weight": (torch.randn(256) * 0.02).to(torch.bfloat16),
        "model.layers.0.mlp.gate_up_proj.weight": (torch.randn(512, 256) * 0.02).to(torch.bfloat16),
        "lm_head.weight": (torch.randn(256, 256) * 0.02).to(torch.bfloat16),
    }
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = td / "model.safetensors"
        save_file(sd, str(src))
        proc = _run_cli("benchmark", str(src), "--no-progress",
                        "-o", str(td / "model.bs"), check_zero=True)
        # Output should include ratio + encode/decode rates + per-layer table
        out = proc.stdout
        assert "encode:" in out and "decode:" in out
        assert "ratio:" in out
        assert "per-layer-type breakdown:" in out
        # Recognised buckets we expect from these tensor names:
        assert "attn_qkv" in out
        assert "mlp_gate_up" in out
        assert "lm_head" in out
        assert "norm" in out
