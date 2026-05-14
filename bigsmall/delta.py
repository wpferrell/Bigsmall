"""Delta compression utilities (XOR base XOR finetune -> sparse stream).

The encoder API lives in encoder.compress_delta and decoder.decompress_delta.
This module exposes lower-level helpers for advanced users.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np


def xor_bytes(a: bytes, b: bytes) -> bytes:
    """XOR two equal-length byte streams."""
    if len(a) != len(b):
        raise ValueError(f"length mismatch {len(a)} vs {len(b)}")
    aa = np.frombuffer(a, dtype=np.uint8)
    bb = np.frombuffer(b, dtype=np.uint8)
    return (aa ^ bb).tobytes()


def hash_safetensors(path: str | Path) -> str:
    """Return md5 over the entire safetensors file (used as base hash)."""
    p = Path(path)
    h = hashlib.md5()
    with open(p, "rb") as f:
        while True:
            chunk = f.read(4 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()
