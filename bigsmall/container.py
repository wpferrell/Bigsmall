"""BigSmall container format (.bs file).

Layout:
    [4 bytes]  magic = b"BGSM"
    [2 bytes]  version (uint16, little-endian, current = 1)
    [4 bytes]  header JSON length (uint32 LE)
    [N bytes]  header JSON (utf-8)
    [...]      data section: concatenated compressed blobs

Header JSON:
    {
        "format":        "bf16" | "fp32" | "fp16" | "fp8" | "fp4",
        "mode":          "storage" | "balanced" | "inference",
        "model_type":    "llm" | "diffusion" | "base" | "delta",
        "base_model":    null | "<path-or-hash>",
        "tensor_count":  N,
        "tensors": [
            {
                "name":             str,
                "shape":            [int, ...],
                "dtype":            str,            # numpy dtype string
                "codec":            str,            # codec name
                "special":          null | str,     # attn_bias | wpe_delta | tied | delta
                "compressed_bytes": int,
                "offset":           int,            # byte offset into data section
                "md5":              str,            # md5 of original raw bytes (hex)
                "extra":            dict | null     # codec-specific extras
            },
            ...
        ]
    }
"""
import json
import struct
from pathlib import Path
from typing import Any

MAGIC = b"BGSM"
VERSION = 1


def write_container(path, header: dict, data: bytes) -> None:
    """Write a complete .bs container to disk."""
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<H", VERSION))
        f.write(struct.pack("<I", len(header_bytes)))
        f.write(header_bytes)
        f.write(data)


def read_header(path) -> tuple[dict, int]:
    """Return (header_dict, data_offset)."""
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != MAGIC:
            raise ValueError(f"Not a BigSmall .bs file (magic={magic!r})")
        version, = struct.unpack("<H", f.read(2))
        if version != VERSION:
            raise ValueError(f"Unsupported BigSmall version: {version}")
        header_len, = struct.unpack("<I", f.read(4))
        header_bytes = f.read(header_len)
        data_offset = f.tell()
    header = json.loads(header_bytes.decode("utf-8"))
    return header, data_offset


def read_blob(path, header: dict, data_offset: int, tensor_idx: int) -> bytes:
    """Read a single tensor blob from the data section."""
    t = header["tensors"][tensor_idx]
    with open(path, "rb") as f:
        f.seek(data_offset + t["offset"])
        return f.read(t["compressed_bytes"])


def info(path) -> dict[str, Any]:
    """Return summary info for a .bs file (no decompression)."""
    p = Path(path)
    header, data_offset = read_header(p)
    file_size = p.stat().st_size
    data_size = file_size - data_offset
    total_raw = 0
    dtype_to_bytes = {
        "fp32": 4, "bf16": 2, "fp16": 2, "fp8": 1, "fp4": 1,  # fp4 stored unpacked
    }
    fmt = header["format"]
    bytes_per_w = dtype_to_bytes.get(fmt, 1)
    if fmt == "fp4":
        bytes_per_w = 0.5
    for t in header["tensors"]:
        n = 1
        for d in t["shape"]:
            n *= d
        total_raw += int(n * bytes_per_w)
    ratio = (file_size / total_raw * 100) if total_raw > 0 else 0.0
    return {
        "path": str(p),
        "file_size": file_size,
        "data_offset": data_offset,
        "data_size": data_size,
        "format": fmt,
        "mode": header.get("mode", "balanced"),
        "model_type": header.get("model_type", "llm"),
        "base_model": header.get("base_model"),
        "tensor_count": header["tensor_count"],
        "estimated_raw_bytes": total_raw,
        "ratio_pct": ratio,
        "version": VERSION,
    }
