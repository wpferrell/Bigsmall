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
from pathlib import Path
from typing import Any

from . import container

INDEX_FILENAME = "bigsmall.index.json"


def build_index(shard_paths: list[str | Path],
                model_type: str | None = None) -> dict[str, Any]:
    """Walk a list of .bs shard files and produce the index dict.

    Args:
        shard_paths: list of .bs files in the order they should appear.
        model_type: optional override; otherwise read from the first shard.

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

    return {
        "metadata": {
            "bigsmall_version": str(bs_version),
            "container_version": next(iter(container_versions)),
            "format": fmt,
            "mode": mode,
            "model_type": mtype,
            "total_size": total_size,
            "total_raw_size": total_raw_size,
            "ratio_pct": ratio,
            "shard_count": len(shard_paths),
            "tensor_count": tensor_count,
            "shards": [p.name for p in shard_paths],
        },
        "weight_map": weight_map,
    }


def write_index(directory: str | Path, shard_paths: list[str | Path],
                model_type: str | None = None) -> Path:
    """Build and write bigsmall.index.json into directory. Returns the path."""
    index = build_index(shard_paths, model_type=model_type)
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
