"""Format-version handling and v1/v2 backward compatibility tests.

What this guards:
  - Default writer still emits v1 (so 2.0.x consumers can read 2.3.0 files).
  - `write_container(format_version=2)` produces a file that the 2.3.0 reader
    accepts, and the file's stamped version is 2.
  - The 2.3.0 reader rejects any unsupported version with a clear error.
  - The full `compress / decompress` round-trip is byte-identical when the
    container is forced to v2.
  - `bigsmall.index.json` records the format version it bundled.
"""
import hashlib
import json
import struct
import tempfile
from pathlib import Path

import pytest


def _md5_hex(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def _state_dict():
    import torch
    torch.manual_seed(0)
    return {
        "embed.weight": torch.randn(512, 128, dtype=torch.bfloat16),
        "layer.0.weight": torch.randn(256, 128, dtype=torch.bfloat16),
    }


def test_default_writer_emits_v1():
    """Default `compress()` must produce a v1 file -- backward compat for the
    19 already-uploaded HF models and any 2.0.x consumer still pinned."""
    import torch
    from safetensors.torch import save_file
    import bigsmall
    from bigsmall import container

    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "model.safetensors"
        save_file(_state_dict(), str(src))
        bs = Path(td) / "model.bs"
        bigsmall.compress(src, bs, progress=False, workers=1)

        with open(bs, "rb") as f:
            magic = f.read(4)
            assert magic == container.MAGIC
            version, = struct.unpack("<H", f.read(2))
        assert version == 1, f"default writer must emit v1; got {version}"


def test_v2_writer_emits_v2_and_reads_back():
    """Forcing write_container(format_version=2) stamps v2 and the reader
    round-trips the header."""
    from bigsmall import container
    from bigsmall.formats import BS_FORMAT_VERSION_V2

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "synthetic.bs"
        header = {
            "format": "bf16", "mode": "balanced", "model_type": "llm",
            "base_model": None, "tensor_count": 0, "tensors": [],
            "safetensors_metadata": None,
        }
        container.write_container(path, header, data=b"",
                                  format_version=BS_FORMAT_VERSION_V2)

        # Raw-byte version stamp must be 2.
        with open(path, "rb") as f:
            f.seek(4)
            version, = struct.unpack("<H", f.read(2))
        assert version == BS_FORMAT_VERSION_V2

        # read_header surfaces it via the synthetic `_format_version` key.
        h, _ = container.read_header(path)
        assert h["_format_version"] == BS_FORMAT_VERSION_V2


def test_unsupported_version_rejected_clearly():
    """A future v3 file must fail with a helpful error, not silent corruption."""
    from bigsmall import container

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "future.bs"
        header_bytes = b"{}"
        with open(path, "wb") as f:
            f.write(container.MAGIC)
            f.write(struct.pack("<H", 99))           # impossibly future version
            f.write(struct.pack("<I", len(header_bytes)))
            f.write(header_bytes)
        with pytest.raises(ValueError, match="Unsupported BigSmall version"):
            container.read_header(path)


def test_v2_writer_roundtrips_byte_identical():
    """End-to-end: force v2 stamp, compress, decompress, every tensor's
    raw bytes match the source."""
    try:
        import torch  # noqa: F401
        import safetensors  # noqa: F401
    except ImportError:
        pytest.skip("torch/safetensors not installed")

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    import bigsmall
    from bigsmall import container
    from bigsmall.formats import BS_FORMAT_VERSION_V2

    tensors = _state_dict()
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "model.safetensors"
        save_file(tensors, str(src))
        v1_bs = Path(td) / "v1.bs"
        bigsmall.compress(src, v1_bs, progress=False, workers=1)

        # Re-stamp as v2 by re-reading the v1 file and writing it back.
        header, data_offset = container.read_header(v1_bs)
        with open(v1_bs, "rb") as f:
            f.seek(data_offset)
            data = f.read()
        v2_bs = Path(td) / "v2.bs"
        # Strip the synthetic _format_version key before re-emit so the on-disk
        # JSON stays clean.
        header.pop("_format_version", None)
        container.write_container(v2_bs, header, data,
                                  format_version=BS_FORMAT_VERSION_V2)

        out_v2 = bigsmall.decompress(v2_bs, progress=False)
        out_v1 = bigsmall.decompress(v1_bs, progress=False)

        with safe_open(str(src), framework="pt") as f:
            for name in f.keys():
                t = f.get_tensor(name)
                src_md5 = _md5_hex(t.contiguous().view(torch.uint8).cpu().numpy().tobytes())
                assert _md5_hex(out_v1[name].tobytes()) == src_md5, f"v1 mismatch on {name}"
                assert _md5_hex(out_v2[name].tobytes()) == src_md5, f"v2 mismatch on {name}"


def test_compress_for_hub_records_format_version():
    """`bigsmall.index.json:metadata.format_version` should be present and
    equal to the version stamped into the shard files."""
    try:
        import torch  # noqa: F401
        import safetensors  # noqa: F401
    except ImportError:
        pytest.skip("torch/safetensors not installed")
    import bigsmall

    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "src"
        src.mkdir()
        from safetensors.torch import save_file
        save_file({"w": _state_dict()["layer.0.weight"]},
                  str(src / "model.safetensors"))
        (src / "config.json").write_text("{}", encoding="utf-8")
        out = Path(td) / "out"
        bigsmall.compress_for_hub(str(src), output_dir=out, overwrite=True, workers=1)

        idx = json.loads((out / "bigsmall.index.json").read_text(encoding="utf-8"))
        meta = idx["metadata"]
        assert "format_version" in meta
        # Default writer still emits v1 -> the index mirrors that.
        assert meta["format_version"] == 1
        # The legacy `container_version` key kept its meaning too.
        assert meta["container_version"] == 1
