"""BigSmall container format (.bs file).

Layout:
    [4 bytes]  magic = b"BGSM"
    [2 bytes]  version (uint16 LE; 1 = legacy, 2 = future-codec-extensions)
    [4 bytes]  header JSON length (uint32 LE)
    [N bytes]  header JSON (utf-8)
    [...]      data section: concatenated compressed blobs

Version policy: a v2 reader MUST accept v1 files transparently (we have 19
already-uploaded HF models on v1). A v1 reader rejects v2 files explicitly
rather than misparse the header. New per-tensor codec extensions (planned for
later sessions: shared probability tables, row-delta, sparsity, QKV dedup)
are signalled by additive keys inside the per-tensor header dict, never by
moving bytes around.

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

from .exceptions import BigSmallVersionError
from .formats import (
    BS_FORMAT_VERSION,
    BS_FORMAT_VERSION_V1,
    BS_FORMAT_VERSION_V2,
    BS_SUPPORTED_FORMAT_VERSIONS,
)

MAGIC = b"BGSM"
# `VERSION` kept for callers that imported it from older releases. Writes go
# through `write_container` which now accepts an explicit `format_version`.
VERSION = BS_FORMAT_VERSION


def write_container(path, header: dict, data: bytes,
                    format_version: int = BS_FORMAT_VERSION) -> None:
    """Write a complete .bs container to disk.

    Args:
        path: destination file path.
        header: header dict (will be json-serialised).
        data: concatenated tensor data blobs.
        format_version: container version to stamp into the file. Defaults to
            `BS_FORMAT_VERSION` (currently v2). Pass `BS_FORMAT_VERSION_V1`
            explicitly when re-encoding for v1-only consumers.
    """
    if format_version not in BS_SUPPORTED_FORMAT_VERSIONS:
        raise ValueError(
            f"Unsupported BigSmall format version {format_version!r}; "
            f"supported: {sorted(BS_SUPPORTED_FORMAT_VERSIONS)}"
        )
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<H", format_version))
        f.write(struct.pack("<I", len(header_bytes)))
        f.write(header_bytes)
        f.write(data)


def read_header(path) -> tuple[dict, int]:
    """Return (header_dict, data_offset).

    Accepts any supported format version. The returned header dict gains a
    `_format_version` key when reading from disk so downstream code can
    distinguish v1 / v2 without re-reading the file.
    """
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != MAGIC:
            raise ValueError(f"Not a BigSmall .bs file (magic={magic!r})")
        version, = struct.unpack("<H", f.read(2))
        if version not in BS_SUPPORTED_FORMAT_VERSIONS:
            # Tell the user exactly what to do. We hard-code the next-version
            # string here because the version that introduced the v2 format
            # is 2.4.0; any future on-disk format will raise the same error
            # but with its own required-version annotation in the detail.
            from . import __version__ as _bs_version
            raise BigSmallVersionError(
                required="2.4.0",
                installed=_bs_version,
                detail=(
                    f"file uses container format v{version}; "
                    f"this build supports {sorted(BS_SUPPORTED_FORMAT_VERSIONS)}"
                ),
            )
        header_len, = struct.unpack("<I", f.read(4))
        header_bytes = f.read(header_len)
        data_offset = f.tell()
    header = json.loads(header_bytes.decode("utf-8"))
    header["_format_version"] = version
    return header, data_offset


def read_blob(path, header: dict, data_offset: int, tensor_idx: int) -> bytes:
    """Read a single tensor blob from the data section."""
    t = header["tensors"][tensor_idx]
    with open(path, "rb") as f:
        f.seek(data_offset + t["offset"])
        return f.read(t["compressed_bytes"])


_DTYPE_BYTES = {
    "F32": 4, "float32": 4,
    "F16": 2, "float16": 2,
    "BF16": 2, "bfloat16": 2,
    "F8_E4M3": 1, "float8_e4m3fn": 1,
    "F8_E5M2": 1, "float8_e5m2": 1,
    "I64": 8, "int64": 8,
    "I32": 4, "int32": 4,
    "I16": 2, "int16": 2,
    "I8": 1, "int8": 1,
    "U8": 1, "uint8": 1,
    "BOOL": 1, "bool": 1,
}


def _dtype_to_format(dtype_str: str) -> str:
    """Map a tensor dtype string to a friendly format label."""
    s = dtype_str
    if s in ("F32", "float32"):
        return "fp32"
    if s in ("F16", "float16"):
        return "fp16"
    if s in ("BF16", "bfloat16"):
        return "bf16"
    if s in ("F8_E4M3", "float8_e4m3fn", "F8_E5M2", "float8_e5m2"):
        return "fp8"
    return "other"


def _tensor_raw_bytes(t: dict) -> int:
    """Raw byte count for one tensor entry."""
    n = 1
    for d in t["shape"]:
        n *= d
    return int(n * _DTYPE_BYTES.get(t["dtype"], 1))


def _layer_index_from_name(name: str) -> Any:
    """Lazy import to avoid circular dependency at module load."""
    from .streaming import layer_index
    return layer_index(name)


def info(path) -> dict[str, Any]:
    """Return summary info for a .bs file (no decompression).

    Includes per-tensor compression ratios, top/worst-5 tensor lists,
    format breakdown, special-tensor counts, and a streaming peak RAM
    estimate (embeddings/non-layer tensors plus the largest single layer).
    """
    p = Path(path)
    header, data_offset = read_header(p)
    file_size = p.stat().st_size
    data_size = file_size - data_offset

    fmt = header["format"]
    per_tensor: list[dict] = []
    total_raw = 0
    format_breakdown: dict[str, int] = {}
    special_counts: dict[str, int] = {}

    for t in header["tensors"]:
        raw_bytes = _tensor_raw_bytes(t)
        total_raw += raw_bytes
        comp = int(t["compressed_bytes"])
        ratio = (comp / raw_bytes * 100.0) if raw_bytes > 0 else 0.0
        tf = _dtype_to_format(t["dtype"])
        format_breakdown[tf] = format_breakdown.get(tf, 0) + 1
        special = t.get("special")
        if special:
            special_counts[special] = special_counts.get(special, 0) + 1
        per_tensor.append({
            "name": t["name"],
            "shape": t["shape"],
            "dtype": t["dtype"],
            "codec": t["codec"],
            "special": special,
            "raw_bytes": raw_bytes,
            "compressed_bytes": comp,
            "ratio_pct": ratio,
        })

    # Streaming peak RAM estimate:
    #   peak = sum(non-layer tensor raw bytes) + max(per-layer total raw bytes)
    non_layer_raw = 0
    layer_raw_totals: dict[int, int] = {}
    for pt in per_tensor:
        li = _layer_index_from_name(pt["name"])
        if li is None:
            non_layer_raw += pt["raw_bytes"]
        else:
            layer_raw_totals[li] = layer_raw_totals.get(li, 0) + pt["raw_bytes"]
    largest_layer = max(layer_raw_totals.values()) if layer_raw_totals else 0
    streaming_peak_ram_bytes = non_layer_raw + largest_layer

    # Filter ratios for "best/worst" lists - skip tied entries (compressed_bytes=0)
    ranked = [pt for pt in per_tensor if pt["compressed_bytes"] > 0 and pt["raw_bytes"] > 0]
    ranked.sort(key=lambda x: x["ratio_pct"])
    best5 = [
        {"name": x["name"], "ratio_pct": x["ratio_pct"],
         "raw_bytes": x["raw_bytes"], "compressed_bytes": x["compressed_bytes"]}
        for x in ranked[:5]
    ]
    worst5 = [
        {"name": x["name"], "ratio_pct": x["ratio_pct"],
         "raw_bytes": x["raw_bytes"], "compressed_bytes": x["compressed_bytes"]}
        for x in ranked[-5:][::-1]
    ]

    ratio = (file_size / total_raw * 100) if total_raw > 0 else 0.0
    # Codec breakdown.  Prefer the header-stamped value (written since v2.5.0)
    # so old readers and pre-v2.5.0 files still get something useful by
    # re-tallying the per-tensor `codec` field.
    codec_stats: dict[str, int] = dict(header.get("codec_stats") or {})
    if not codec_stats:
        for pt in per_tensor:
            cn = pt.get("codec", "?")
            codec_stats[cn] = codec_stats.get(cn, 0) + 1
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
        "format_breakdown": format_breakdown,
        "special_counts": special_counts,
        "codec_stats": codec_stats,
        "per_tensor": per_tensor,
        "top5_best": best5,
        "top5_worst": worst5,
        "layer_count": len(layer_raw_totals),
        "non_layer_raw_bytes": non_layer_raw,
        "largest_layer_raw_bytes": largest_layer,
        "streaming_peak_ram_bytes": streaming_peak_ram_bytes,
    }
