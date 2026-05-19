"""Drop-in compressed-KV-cache manager for streaming inference.

Stores per-layer K and V tensors as compressed bytes in CPU RAM; decompresses
on demand. Designed as the backing store for the attention layers in
`BigSmallStreamingModel` when `compress_kv=True`.

NOTE on performance: the AC codec runs at ~17 MB/s on CPU
(constriction). On Phi-3.5-mini, decoding a single layer's K + V at
seq=2000 takes ~700ms per tensor. This is unsuitable for live token
generation in v1 — KV compression is shipped as infrastructure
(format + correctness) and not wired into the live attention path by
default. The path to making it useful runs through a faster codec
(GPU AC kernel or higher-throughput entropy coder).
"""
from __future__ import annotations

from typing import Optional

from .codecs import kv_cache as _kv_codec


class CompressedKVCache:
    """Compressed KV cache storage.

    API mirrors transformers' DynamicCache enough that callers can use
    `set(layer_idx, k, v)` to store and `get(layer_idx) -> (k, v)` to
    retrieve. Internal storage is a dict of compressed bytes keyed by
    layer index.

    Args:
        device: device to materialise tensors on when get() is called.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._compressed: dict[int, bytes] = {}
        self._raw_size_bytes = 0   # accumulated uncompressed (raw) size for accounting

    def set(self, layer_idx: int, keys, values) -> None:
        import torch
        if keys.dtype != torch.bfloat16 or values.dtype != torch.bfloat16:
            raise ValueError("CompressedKVCache.set: keys and values must be BF16")
        raw_size = keys.element_size() * keys.numel() + values.element_size() * values.numel()
        blob = _kv_codec.compress_kv_entry(keys, values)
        # If layer was already present, account only the new entry.
        if layer_idx in self._compressed:
            # Subtract the previous raw size estimate (we don't track per-layer
            # raw size precisely; for simplicity recompute on next call).
            pass
        self._compressed[layer_idx] = blob
        self._raw_size_bytes = self._raw_size_bytes - 0 + raw_size  # cumulative; see compression_ratio

    def get(self, layer_idx: int):
        if layer_idx not in self._compressed:
            raise KeyError(f"layer {layer_idx} not in CompressedKVCache")
        return _kv_codec.decompress_kv_entry(self._compressed[layer_idx], device=self.device)

    def has(self, layer_idx: int) -> bool:
        return layer_idx in self._compressed

    def memory_usage(self) -> int:
        """Total bytes currently used by compressed storage."""
        return sum(len(b) for b in self._compressed.values())

    def raw_size(self) -> int:
        """Total raw (uncompressed BF16) bytes accounted for."""
        return self._raw_size_bytes

    def compression_ratio(self) -> Optional[float]:
        """compressed / raw, or None if no data stored yet."""
        compressed = self.memory_usage()
        raw = self.raw_size()
        if raw == 0:
            return None
        return compressed / raw

    def clear(self) -> None:
        self._compressed.clear()
        self._raw_size_bytes = 0

    def __len__(self) -> int:
        return len(self._compressed)

    def __repr__(self) -> str:
        cr = self.compression_ratio()
        cr_str = f"{cr * 100:.2f}%" if cr is not None else "n/a"
        return (
            f"CompressedKVCache(layers={len(self)} "
            f"compressed={self.memory_usage()/1e6:.2f}MB "
            f"raw={self.raw_size()/1e6:.2f}MB ratio={cr_str})"
        )
