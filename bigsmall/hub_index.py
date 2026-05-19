"""BigSmall multi-shard index (bigsmall.index.json).

Parallel to HuggingFace's model.safetensors.index.json. Makes a multi-shard
.bs model loadable as a single unit.

Layout:
    {
        "metadata": {
            "bigsmall_version":      str,   # package version that wrote the index
            "container_version":     int,   # .bs container format version
            "format":                str,   # "bf16" | "fp32" | "fp16" | "fp8" | "fp4" | "mixed"
            "mode":                  str,   # "storage" | "balanced" | "inference"
            "model_type":            str,   # "llm" | "diffusion" | "base" | "delta"
            "total_size":            int,   # sum of compressed shard sizes
            "total_raw_size":        int,   # sum of estimated raw bytes
            "ratio_pct":             float, # total_size / total_raw_size * 100
            "shard_count":           int,
            "tensor_count":          int,
            "shards":                [str, ...]   # ordered shard filenames
        },
        "weight_map": {
            "<tensor_name>": "<shard_filename>",
            ...
        }
    }
"""
from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

from . import container

INDEX_FILENAME = "bigsmall.index.json"
BINARY_INDEX_FILENAME = "bigsmall.index.bin"

# Threshold below which a binary index is pure overhead vs the JSON.
# At 100+ tensors the binary form starts saving meaningful parse time
# (and bytes if the names share long prefixes via the string table).
BINARY_INDEX_MIN_TENSORS = 100

# Binary-index file layout (little-endian throughout):
#
#   [4]   magic = b"BSIX"
#   [1]   version = 1
#   [3]   reserved (zero)
#   [4]   n_shards
#   [4]   n_tensors
#   [4]   shard_names_section_size
#   For each shard, in order:
#     [2]  shard_name_length
#     [N]  shard_name (UTF-8)
#   [4]   string_table_size
#   [N]   string table (utf-8 bytes, names referenced by offset)
#   For each tensor record (24 bytes each):
#     [4]  name_offset (into string table)
#     [2]  name_length
#     [2]  shard_index
#     [4]  reserved (zero)
#     [4]  blob_offset_lo / [4] blob_offset_hi  — uint64 split as 2x uint32
#         (no — collapse into one uint64 below for clarity)
#
# Concretely the per-tensor record is:
#     <I H H Q Q B 5x  =  4+2+2+8+8+1+5 = 30 bytes
#   = (name_offset, name_length, shard_index, blob_offset, compressed_bytes,
#      codec_index, padding5)
#
# `codec_index` is into a separate codec-name string table written before
# the per-tensor records.

BINARY_INDEX_MAGIC = b"BSIX"
BINARY_INDEX_VERSION = 1
_TENSOR_RECORD = struct.Struct("<IHHQQB5x")
assert _TENSOR_RECORD.size == 30


def build_index(shard_paths: list[str | Path],
                model_type: str | None = None,
                duplicate_map: dict[str, dict] | None = None) -> dict[str, Any]:
    """Walk a list of .bs shard files and produce the index dict.

    Args:
        shard_paths: list of .bs files in the order they should appear.
        model_type: optional override; otherwise read from the first shard.
        duplicate_map: optional `{dup_name: {"master": str}}` recording
                       cross-shard tied-weight aliases. The duplicate tensors
                       are NOT present in any shard's header; the decoder
                       materialises them by aliasing the master tensor.

    Returns:
        Index dict ready to json.dump.
    """
    shard_paths = [Path(p) for p in shard_paths]
    if not shard_paths:
        raise ValueError("build_index needs at least one shard")

    weight_map: dict[str, str] = {}
    total_size = 0
    total_raw_size = 0
    tensor_count = 0
    formats_seen: set[str] = set()
    modes_seen: set[str] = set()
    types_seen: set[str] = set()
    container_versions: set[int] = set()

    for shard in shard_paths:
        info = container.info(shard)
        shard_name = shard.name
        total_size += int(info["file_size"])
        total_raw_size += int(info["estimated_raw_bytes"])
        tensor_count += int(info["tensor_count"])
        formats_seen.add(info["format"])
        modes_seen.add(info["mode"])
        types_seen.add(info["model_type"])
        container_versions.add(int(info["version"]))

        header, _ = container.read_header(shard)
        for t in header["tensors"]:
            name = t["name"]
            if name in weight_map and weight_map[name] != shard_name:
                raise ValueError(
                    f"Tensor {name!r} appears in two shards: "
                    f"{weight_map[name]} and {shard_name}"
                )
            weight_map[name] = shard_name

    if len(container_versions) != 1:
        raise ValueError(f"Mixed container versions across shards: {container_versions}")

    fmt = next(iter(formats_seen)) if len(formats_seen) == 1 else "mixed"
    mode = next(iter(modes_seen)) if len(modes_seen) == 1 else "mixed"
    mtype = model_type or (next(iter(types_seen)) if len(types_seen) == 1 else "llm")

    ratio = (total_size / total_raw_size * 100.0) if total_raw_size > 0 else 0.0

    try:
        from . import __version__ as bs_version
    except Exception:
        bs_version = "unknown"

    dup_map = duplicate_map or {}
    # Duplicate aliases participate in the public tensor count and the weight map
    # so consumers can locate them without needing to know about the dedup.
    for dup_name, info in dup_map.items():
        master = info.get("master")
        if master in weight_map and dup_name not in weight_map:
            weight_map[dup_name] = weight_map[master]

    metadata = {
        "bigsmall_version": str(bs_version),
        "container_version": next(iter(container_versions)),
        "format_version": next(iter(container_versions)),
        "format": fmt,
        "mode": mode,
        "model_type": mtype,
        "total_size": total_size,
        "total_raw_size": total_raw_size,
        "ratio_pct": ratio,
        "shard_count": len(shard_paths),
        "tensor_count": tensor_count + len(dup_map),
        "stored_tensor_count": tensor_count,
        "shards": [p.name for p in shard_paths],
    }
    if dup_map:
        metadata["duplicate_map"] = dup_map
    return {
        "metadata": metadata,
        "weight_map": weight_map,
    }


def write_index(directory: str | Path, shard_paths: list[str | Path],
                model_type: str | None = None,
                duplicate_map: dict[str, dict] | None = None) -> Path:
    """Build and write bigsmall.index.json into directory. Returns the path."""
    index = build_index(shard_paths, model_type=model_type,
                        duplicate_map=duplicate_map)
    out = Path(directory) / INDEX_FILENAME
    with open(out, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, sort_keys=False)
    return out


def read_index(directory_or_path: str | Path) -> dict[str, Any]:
    """Read bigsmall.index.json from a directory or full path."""
    p = Path(directory_or_path)
    if p.is_dir():
        p = p / INDEX_FILENAME
    if not p.exists():
        raise FileNotFoundError(f"No bigsmall.index.json at {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def shard_paths_from_index(directory: str | Path, index: dict | None = None) -> list[Path]:
    """Return ordered shard file paths from the index in `directory`."""
    directory = Path(directory)
    if index is None:
        index = read_index(directory)
    return [directory / name for name in index["metadata"]["shards"]]


# ----------------------------------------------------------------------------
# Binary index (v3.12.0)
#
# `bigsmall.index.bin` is an optional compact form of `bigsmall.index.json`.
# Both files are written together; the JSON remains the source of truth for
# human inspection. Readers that want the speedup (millisecond seek lookup)
# can use `read_binary_index()`.


def _build_tensor_lookup(shard_paths: list[Path]) -> list[dict]:
    """Walk each shard header and collect per-tensor records for the binary index."""
    records: list[dict] = []
    for shard_idx, shard in enumerate(shard_paths):
        header, _data_offset = container.read_header(shard)
        for t in header["tensors"]:
            records.append({
                "name": t["name"],
                "shard_index": shard_idx,
                "blob_offset": int(t.get("offset", 0)),
                "compressed_bytes": int(t.get("compressed_bytes", 0)),
                "codec": t.get("codec") or "?",
            })
    return records


def write_binary_index(directory: str | Path,
                       shard_paths: list[str | Path]) -> Path:
    """Write `bigsmall.index.bin` alongside the JSON for fast lookup.

    Returns the written path. Reads each shard's header to build the
    tensor → (shard_index, blob_offset, compressed_bytes, codec) map.
    """
    directory = Path(directory)
    shard_paths = [Path(p) for p in shard_paths]
    records = _build_tensor_lookup(shard_paths)

    # Build the name and codec string tables. Deduplicate names so repeated
    # prefixes share bytes via shared offsets.
    name_table = bytearray()
    name_offsets: dict[str, tuple[int, int]] = {}  # name -> (offset, len)
    for r in records:
        if r["name"] not in name_offsets:
            encoded = r["name"].encode("utf-8")
            name_offsets[r["name"]] = (len(name_table), len(encoded))
            name_table += encoded

    codec_names: list[str] = []
    codec_index: dict[str, int] = {}
    for r in records:
        if r["codec"] not in codec_index:
            codec_index[r["codec"]] = len(codec_names)
            codec_names.append(r["codec"])
    # Codec name table: [n_codecs (uint8)] [for each: name_length (uint8) || name bytes]
    if len(codec_names) > 255:
        raise ValueError(
            f"binary index: more than 255 distinct codec names "
            f"({len(codec_names)}); needs format extension"
        )

    out = Path(directory) / BINARY_INDEX_FILENAME
    with open(out, "wb") as f:
        # File header
        f.write(BINARY_INDEX_MAGIC)
        f.write(struct.pack("<B", BINARY_INDEX_VERSION))
        f.write(b"\x00\x00\x00")  # reserved padding
        f.write(struct.pack("<I", len(shard_paths)))
        f.write(struct.pack("<I", len(records)))

        # Shard name section
        shard_names_bytes = bytearray()
        for shard in shard_paths:
            name_b = shard.name.encode("utf-8")
            if len(name_b) > 65535:
                raise ValueError(
                    f"shard name too long for binary index: {shard.name!r}"
                )
            shard_names_bytes += struct.pack("<H", len(name_b))
            shard_names_bytes += name_b
        f.write(struct.pack("<I", len(shard_names_bytes)))
        f.write(bytes(shard_names_bytes))

        # Codec name section
        f.write(struct.pack("<B", len(codec_names)))
        for c in codec_names:
            cb = c.encode("utf-8")
            if len(cb) > 255:
                raise ValueError(f"codec name too long: {c!r}")
            f.write(struct.pack("<B", len(cb)))
            f.write(cb)

        # String table (tensor names) — write its size, then the bytes.
        f.write(struct.pack("<I", len(name_table)))
        f.write(bytes(name_table))

        # Tensor records
        for r in records:
            n_off, n_len = name_offsets[r["name"]]
            f.write(_TENSOR_RECORD.pack(
                n_off, n_len, r["shard_index"],
                r["blob_offset"], r["compressed_bytes"],
                codec_index[r["codec"]],
            ))
    return out


def read_binary_index(path: str | Path) -> dict[str, Any]:
    """Parse a binary index file. Returns the same shape as `read_index()`.

    The returned dict has:
      - "metadata": {"shards": [...], "tensor_count": int, "shard_count": int}
      - "weight_map": {tensor_name: shard_filename}
      - "binary": full per-tensor list with offsets + codec names (extra)

    Raises FileNotFoundError if path doesn't exist, or ValueError if the
    file isn't a binary index (bad magic or version).
    """
    path = Path(path)
    if path.is_dir():
        path = path / BINARY_INDEX_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"No binary index at {path}")
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != BINARY_INDEX_MAGIC:
            raise ValueError(f"Not a BigSmall binary index: magic {magic!r}")
        version = struct.unpack("<B", f.read(1))[0]
        f.read(3)  # reserved
        if version != BINARY_INDEX_VERSION:
            raise ValueError(
                f"Unsupported binary index version {version} "
                f"(expected {BINARY_INDEX_VERSION})"
            )
        n_shards = struct.unpack("<I", f.read(4))[0]
        n_tensors = struct.unpack("<I", f.read(4))[0]
        shard_names_size = struct.unpack("<I", f.read(4))[0]
        shard_names_bytes = f.read(shard_names_size)
        # Parse shard names
        shard_names: list[str] = []
        pos = 0
        for _ in range(n_shards):
            ln = struct.unpack("<H", shard_names_bytes[pos:pos + 2])[0]
            pos += 2
            shard_names.append(shard_names_bytes[pos:pos + ln].decode("utf-8"))
            pos += ln

        # Codec table
        n_codecs = struct.unpack("<B", f.read(1))[0]
        codecs: list[str] = []
        for _ in range(n_codecs):
            cln = struct.unpack("<B", f.read(1))[0]
            codecs.append(f.read(cln).decode("utf-8"))

        # String table (tensor names)
        name_table_size = struct.unpack("<I", f.read(4))[0]
        name_table = f.read(name_table_size)

        # Per-tensor records
        records = []
        weight_map: dict[str, str] = {}
        for _ in range(n_tensors):
            (name_off, name_len, shard_idx,
             blob_off, cmp_bytes, codec_idx) = _TENSOR_RECORD.unpack(
                 f.read(_TENSOR_RECORD.size)
             )
            name = name_table[name_off:name_off + name_len].decode("utf-8")
            shard_name = shard_names[shard_idx]
            records.append({
                "name": name,
                "shard_index": shard_idx,
                "shard": shard_name,
                "blob_offset": blob_off,
                "compressed_bytes": cmp_bytes,
                "codec": codecs[codec_idx],
            })
            weight_map[name] = shard_name

    return {
        "metadata": {
            "shards": shard_names,
            "shard_count": n_shards,
            "tensor_count": n_tensors,
            "binary_index_version": version,
        },
        "weight_map": weight_map,
        "binary": records,
    }


def maybe_read_binary_index(directory: str | Path) -> dict[str, Any] | None:
    """Try `read_binary_index(directory)`; return None if no .bin file."""
    directory = Path(directory)
    bin_path = directory / BINARY_INDEX_FILENAME
    if not bin_path.exists():
        return None
    try:
        return read_binary_index(bin_path)
    except (ValueError, FileNotFoundError):
        return None
