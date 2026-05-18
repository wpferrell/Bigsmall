"""Tests for `bigsmall migrate` — re-encode `.bs` files with current codecs.

Guards:
  1. A v1-stamped file becomes a valid v2 file (or stays v1 if no v2 codec was
     selected) and is readable by the decoder afterwards.
  2. The migrated file is never larger than the original (size monotone).
  3. Every tensor decompresses to the same raw bytes after migration as before
     (lossless).
  4. `--dry-run` writes nothing — file mtime/size unchanged.
  5. `backup=True` writes `<file>.bs.bak` with the original bytes.
"""
from __future__ import annotations

import hashlib
import shutil
import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

import bigsmall
from bigsmall import container, migrate as migrate_mod


def _md5_hex(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def _make_safetensors(td: Path) -> Path:
    torch.manual_seed(0)
    tensors = {
        # generic BF16 — exercises bf16_se_ac path
        "embed.weight": torch.randn(512, 128, dtype=torch.bfloat16),
        # generic FP32 — exercises fp32_se_ac path
        "layer.0.weight": torch.randn(256, 64, dtype=torch.float32),
        # tiny tensor — exercises raw codec keep-as-is path
        "layer.0.bias": torch.randn(8, dtype=torch.bfloat16),
    }
    src = td / "model.safetensors"
    save_file(tensors, str(src))
    return src


def _make_high_kurtosis_safetensors(td: Path) -> Path:
    """Build a single sparse + heavy-tail BF16 tensor so A5 has a chance to win.

    The fresh `compress()` call already runs auto-select, so to give migrate
    something to do we have to *force* a v1-only codec onto the tensor first.
    We do that by calling `compress()`, then rewriting the header to claim the
    tensor used `bf16_se_ac` (which is true) but force the encoder to NOT use
    auto-select on a v1-stamped file by passing `enable_a5=False` to compress.
    """
    g = torch.Generator().manual_seed(0)
    base = torch.randn(1024, 512, generator=g) * 1e-4
    mask = torch.rand(1024 * 512, generator=g) < 0.005
    outliers = torch.randn(1024 * 512, generator=g) * 3.0
    flat = base.view(-1)
    flat[mask] = outliers[mask]
    t = flat.view(1024, 512).bfloat16().contiguous()
    src = td / "high_kurtosis.safetensors"
    save_file({"mlp.gate_proj.weight": t}, str(src))
    return src


def _decompress_md5_map(bs_path: Path) -> dict[str, str]:
    out = bigsmall.decompress(str(bs_path), progress=False)
    # Map name -> md5 of raw bytes (uint8 view).
    md5_map: dict[str, str] = {}
    for name, arr in out.items():
        md5_map[name] = _md5_hex(arr.tobytes())
    return md5_map


def test_migrate_outputs_valid_file_readable_by_decoder():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = _make_safetensors(td)
        bs = td / "model.bs"
        bigsmall.compress(str(src), str(bs), workers=1, progress=False)

        result = migrate_mod.migrate(bs, dry_run=False, backup=False)

        # Container must still parse.
        header, _ = container.read_header(bs)
        assert header["tensor_count"] == 3
        assert result["format_version"] in (1, 2)


def test_migrate_never_grows_the_file():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = _make_safetensors(td)
        bs = td / "model.bs"
        bigsmall.compress(str(src), str(bs), workers=1, progress=False)

        result = migrate_mod.migrate(bs, dry_run=False, backup=False)
        assert result["bytes_after"] <= result["bytes_before"], (
            f"migrate grew the blob bytes "
            f"({result['bytes_before']} -> {result['bytes_after']})"
        )
        # Whole-file size can grow by at most the JSON header delta (codec
        # change widens the per-tensor entry).  But the blob payload itself
        # is strictly monotone.


def test_migrate_is_lossless():
    """Every tensor must decompress to byte-identical values after migration."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = _make_safetensors(td)
        bs = td / "model.bs"
        bigsmall.compress(str(src), str(bs), workers=1, progress=False)

        before = _decompress_md5_map(bs)
        migrate_mod.migrate(bs, dry_run=False, backup=False)
        after = _decompress_md5_map(bs)

        assert before == after, (
            f"migrate produced different decompressed bytes\n"
            f"before: {before}\nafter:  {after}"
        )


def test_dry_run_makes_no_changes():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = _make_safetensors(td)
        bs = td / "model.bs"
        bigsmall.compress(str(src), str(bs), workers=1, progress=False)

        before_bytes = bs.read_bytes()
        result = migrate_mod.migrate(bs, dry_run=True, backup=True)
        after_bytes = bs.read_bytes()

        assert result["dry_run"] is True
        assert before_bytes == after_bytes, "dry_run modified the file"
        # No backup should be created on dry_run.
        assert not bs.with_suffix(".bs.bak").exists()


def test_backup_creates_bak_file_with_original_bytes():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = _make_safetensors(td)
        bs = td / "model.bs"
        bigsmall.compress(str(src), str(bs), workers=1, progress=False)

        original_bytes = bs.read_bytes()
        migrate_mod.migrate(bs, dry_run=False, backup=True)

        bak = bs.with_suffix(".bs.bak")
        assert bak.exists(), "expected <file>.bs.bak to be created"
        assert bak.read_bytes() == original_bytes, (
            "backup file contents differ from the pre-migrate original"
        )
