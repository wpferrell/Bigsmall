"""KV cache compression codec.

Wraps the existing `bf16` (sign-exp joint AC + per-exp conditional
mantissa) codec used for model weights. The KV cache on Phi-3.5-mini was
measured to compress to ~68% of raw BF16, matching the weight compression
regime. K and V are encoded independently
in v1; future work can attempt joint coding if measurable correlation is
present.

Functions are tensor-level (work on torch.Tensor for ergonomic use during
inference) but internally reuse the byte-level codec.
"""
from __future__ import annotations

import struct
from typing import Optional

import torch

from . import bf16 as _bf16
from . import bf16_rans as _bf16_rans


def _tensor_to_bytes(t: torch.Tensor) -> tuple[bytes, list[int]]:
    """Return contiguous raw uint8 bytes of a BF16 tensor and its shape."""
    if t.dtype != torch.bfloat16:
        raise ValueError(f"compress_kv_entry expects BF16 tensors, got {t.dtype}")
    raw = t.contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return raw, list(t.shape)


def _bytes_to_tensor(raw: bytes, shape: list[int], device: str) -> torch.Tensor:
    import numpy as np
    arr = np.frombuffer(raw, dtype=np.uint16).copy().reshape(shape)
    t = torch.from_numpy(arr).view(torch.bfloat16)
    if device != "cpu":
        t = t.to(device)
    return t


def compress_kv_entry(keys: torch.Tensor, values: torch.Tensor) -> bytes:
    """Compress one layer's K and V tensors.

    Args:
        keys, values: BF16 tensors with arbitrary shape (typically
                      [B, n_kv_heads, seq, head_dim]).

    Returns:
        Compressed bytes encoding both K and V plus shape metadata.

    Layout:
        [1B] version=1
        [1B] ndim_k
        [ndim_k * 4B] shape_k (int32 each)
        [4B] k_blob_len
        [k_blob_len] k bf16-codec blob
        [1B] ndim_v
        [ndim_v * 4B] shape_v
        [4B] v_blob_len
        [v_blob_len] v bf16-codec blob
    """
    if keys.dtype != torch.bfloat16 or values.dtype != torch.bfloat16:
        raise ValueError("compress_kv_entry requires BF16 keys and values")
    k_raw, k_shape = _tensor_to_bytes(keys)
    v_raw, v_shape = _tensor_to_bytes(values)
    # v2 format: use bf16_se_rans (1.1-1.3x faster than bf16_se_ac at the
    # same compression ratio). v1 readers (v3.3.0) cannot decode this.
    k_blob, _ = _bf16_rans.encode(k_raw)
    v_blob, _ = _bf16_rans.encode(v_raw)

    out = bytearray()
    out += struct.pack("<B", 2)
    out += struct.pack("<B", len(k_shape))
    for s in k_shape:
        out += struct.pack("<i", int(s))
    out += struct.pack("<I", len(k_blob))
    out += k_blob
    out += struct.pack("<B", len(v_shape))
    for s in v_shape:
        out += struct.pack("<i", int(s))
    out += struct.pack("<I", len(v_blob))
    out += v_blob
    return bytes(out)


def decompress_kv_entry(data: bytes,
                        device: str = "cuda") -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse of compress_kv_entry. Returns (keys, values) on `device`."""
    pos = 0
    version = data[pos]
    pos += 1
    if version not in (1, 2):
        raise ValueError(f"Unknown KV cache codec version {version}")
    # v1 = bf16_se_ac (3.3.0), v2 = bf16_se_rans (3.4.0+, 1.18x faster decode).
    decoder = _bf16.decode if version == 1 else _bf16_rans.decode

    def _read_tensor(pos: int) -> tuple[torch.Tensor, int]:
        ndim = data[pos]
        pos += 1
        shape = []
        for _ in range(ndim):
            s = struct.unpack("<i", data[pos:pos + 4])[0]
            pos += 4
            shape.append(s)
        blob_len = struct.unpack("<I", data[pos:pos + 4])[0]
        pos += 4
        blob = data[pos:pos + blob_len]
        pos += blob_len
        n_elements = 1
        for s in shape:
            n_elements *= max(1, s)
        raw = decoder(blob, {}, n_elements)
        return _bytes_to_tensor(raw, shape, device), pos

    keys, pos = _read_tensor(pos)
    values, pos = _read_tensor(pos)
    return keys, values
