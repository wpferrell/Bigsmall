"""BigSmall streaming loader: decompress one transformer layer at a time.

Lets you run a model whose decompressed size is larger than your VRAM/RAM by
materialising only one layer's worth of weights at any moment.

Usage:
    import bigsmall
    with bigsmall.StreamingLoader("model.bs", device="cuda") as loader:
        non_layer = loader.load_non_layer_tensors()
        for i, layer_tensors in loader.iter_layers():
            ...
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from . import container, hub_index
from .decoder import _decode_blob, _raw_to_numpy, _numpy_to_torch


# Match the layer-index segment in tensor names. Examples that match:
#   transformer.h.0.attn.c_attn.weight       -> 0
#   model.layers.5.self_attn.q_proj.weight   -> 5
#   h.11.mlp.c_fc.bias                       -> 11
#   gpt_neox.layers.3.attention.dense.bias   -> 3
_LAYER_RE = re.compile(r"(?:^|\.)(?:layers|h)\.(\d+)\.")


def layer_index(name: str) -> Optional[int]:
    """Return the transformer layer index encoded in `name`, or None if non-layer."""
    m = _LAYER_RE.search(name)
    if not m:
        return None
    return int(m.group(1))


class _TensorEntry:
    """Per-tensor index entry pointing into a shard file."""
    __slots__ = ("name", "meta", "shard_path", "data_offset")

    def __init__(self, name: str, meta: dict, shard_path: Path, data_offset: int):
        self.name = name
        self.meta = meta
        self.shard_path = shard_path
        self.data_offset = data_offset


class StreamingLoader:
    """Decompress a .bs model one transformer layer at a time.

    Args:
        path: path to a single .bs file OR a directory containing
            bigsmall.index.json + .bs shards.
        device: device for the returned torch tensors ("cuda" | "cpu" | ...).
        dtype: optional torch dtype to cast to (None = keep native dtype).

    Use as a context manager so file handles are cleaned up.
    """

    def __init__(self,
                 path: str | Path,
                 device: str = "cuda",
                 dtype=None):
        self.device = device
        self.dtype = dtype
        self.path = Path(path)
        self._open_files: dict[Path, "object"] = {}
        self._index: dict[str, _TensorEntry] = {}      # tensor name -> entry
        self._layers: dict[int, list[str]] = {}        # layer idx -> [names]
        self._non_layer: list[str] = []                # non-layer tensor names
        self._build_index()

    # ---------------------- context manager ---------------------------------

    def __enter__(self) -> "StreamingLoader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        for f in self._open_files.values():
            try:
                f.close()
            except Exception:
                pass
        self._open_files.clear()

    # ---------------------- index construction ------------------------------

    def _build_index(self) -> None:
        """Walk shard(s), read headers, build name->entry and layer maps."""
        shard_paths: list[Path]
        if self.path.is_dir():
            index_file = self.path / hub_index.INDEX_FILENAME
            if index_file.exists():
                idx = hub_index.read_index(self.path)
                shard_paths = hub_index.shard_paths_from_index(self.path, index=idx)
            else:
                shard_paths = sorted(self.path.glob("*.bs"))
                if not shard_paths:
                    raise FileNotFoundError(
                        f"No .bs files and no bigsmall.index.json in {self.path}"
                    )
        elif self.path.is_file():
            shard_paths = [self.path]
        else:
            raise FileNotFoundError(f"Path not found: {self.path}")

        for shard in shard_paths:
            header, data_offset = container.read_header(shard)
            for t in header["tensors"]:
                name = t["name"]
                if name in self._index:
                    raise ValueError(
                        f"Tensor {name!r} appears in multiple shards "
                        f"({self._index[name].shard_path.name}, {shard.name})"
                    )
                self._index[name] = _TensorEntry(name, t, shard, data_offset)
                li = layer_index(name)
                if li is None:
                    self._non_layer.append(name)
                else:
                    self._layers.setdefault(li, []).append(name)

    # ---------------------- file-handle pool --------------------------------

    def _file(self, shard_path: Path):
        f = self._open_files.get(shard_path)
        if f is None:
            f = open(shard_path, "rb")
            self._open_files[shard_path] = f
        return f

    # ---------------------- low-level decode --------------------------------

    def _read_blob(self, entry: _TensorEntry) -> bytes:
        f = self._file(entry.shard_path)
        f.seek(entry.data_offset + entry.meta["offset"])
        return f.read(entry.meta["compressed_bytes"])

    def _decode_raw(self, entry: _TensorEntry, _seen: set | None = None) -> bytes:
        """Return raw decompressed bytes for one tensor. Handles tied_ref by
        looking up the master from the global index on demand."""
        meta = entry.meta
        if meta["codec"] == "tied_ref":
            extras = meta.get("extra") or {}
            master_name = extras["tied_to"]
            seen = _seen or set()
            if master_name in seen:
                raise ValueError(f"Tied-ref cycle: {seen | {master_name}}")
            seen.add(entry.name)
            master_entry = self._index.get(master_name)
            if master_entry is None:
                raise KeyError(
                    f"tied_ref master {master_name!r} not found in index"
                )
            return self._decode_raw(master_entry, _seen=seen)
        blob = self._read_blob(entry)
        return _decode_blob(meta, blob)

    def _to_torch(self, entry: _TensorEntry, raw: bytes):
        meta = entry.meta
        arr = _raw_to_numpy(raw, meta["dtype"], meta["shape"])
        t = _numpy_to_torch(arr, meta["dtype"])
        if self.dtype is not None:
            t = t.to(self.dtype)
        if self.device != "cpu":
            t = t.to(self.device)
        return t

    # ---------------------- public API --------------------------------------

    def tensor_names(self) -> list[str]:
        """All tensor names in the model (layer + non-layer)."""
        return list(self._index.keys())

    def layer_count(self) -> int:
        if not self._layers:
            return 0
        return max(self._layers.keys()) + 1

    def layer_tensor_names(self, layer_idx: int) -> list[str]:
        return list(self._layers.get(layer_idx, []))

    def non_layer_tensor_names(self) -> list[str]:
        return list(self._non_layer)

    def load_tensor(self, name: str) -> "torch.Tensor":
        """Decompress a single named tensor and return it on `device`."""
        entry = self._index.get(name)
        if entry is None:
            raise KeyError(f"Tensor {name!r} not in this BigSmall model")
        raw = self._decode_raw(entry)
        return self._to_torch(entry, raw)

    def load_layer(self, layer_idx: int) -> dict[str, "torch.Tensor"]:
        """Decompress all tensors belonging to one transformer layer."""
        names = self._layers.get(layer_idx)
        if names is None:
            raise IndexError(f"Layer {layer_idx} not in this model "
                             f"(have {self.layer_count()} layers)")
        out: dict[str, "torch.Tensor"] = {}
        for n in names:
            out[n] = self.load_tensor(n)
        return out

    def load_non_layer_tensors(self) -> dict[str, "torch.Tensor"]:
        """Decompress every non-layer tensor (embeddings, norms, lm_head, ...)."""
        out: dict[str, "torch.Tensor"] = {}
        for n in self._non_layer:
            out[n] = self.load_tensor(n)
        return out

    def iter_layers(self) -> Iterator[tuple[int, dict[str, "torch.Tensor"]]]:
        """Yield (layer_idx, tensors_dict) one layer at a time.

        Each yielded dict is the only reference to that layer's tensors held
        by the loader; once the caller drops the reference the memory is
        eligible for reclamation. On CUDA devices we also empty the cache
        between iterations to actively release VRAM.
        """
        import torch
        for i in sorted(self._layers.keys()):
            layer = self.load_layer(i)
            try:
                yield i, layer
            finally:
                layer.clear()
                del layer
                if isinstance(self.device, str) and self.device.startswith("cuda") \
                        and torch.cuda.is_available():
                    torch.cuda.empty_cache()
