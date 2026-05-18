"""Cross-layer XOR delta codec (V4 B2).

Encodes a group of same-shape same-dtype tensors as:
    [ layer_0_full_blob,
      xor(layer_1, layer_0) -> compress,
      xor(layer_2, layer_1) -> compress,
      ... ]

Decoding walks the chain: layer_N = xor(layer_{N-1}, delta_N).

Why u16 XOR is the lossless choice
----------------------------------
The Session A B3c entropy measurement used FP32 subtraction followed by
BF16 rounding -- that bound is **not realizable losslessly** because BF16
rounding throws away information that has to be stored back as a correction
stream (see `tests/test_fp2_residual.py` for the same finding on B1).

The strictly-lossless cross-layer transform is bitwise XOR on the u16 word
representation: XOR is its own inverse, so every bit pattern (NaN, +/-Inf,
denormals included) round-trips exactly. The realised compression gain
depends on whether consecutive transformer layers share enough bit-level
structure for the XOR'd stream to have lower joint (sign, exp) entropy
than either layer alone.

Empirical finding from Phi-3.5-mini (see `research/v4_session_b/`):
the XOR'd stream of consecutive `self_attn.qkv_proj.weight` tensors has
HIGHER entropy than each tensor individually -- because the high mantissa
bits of trained weights are effectively random across layers. The safety
net path in the encoder must therefore reject the cross-layer delta on
every measured tensor.

The codec is still useful as an infrastructure hook for future work
(lossy / quantize-then-delta strategies), and the tests below verify that
the round-trip is exact for any byte sequence.
"""
from __future__ import annotations

import io
import struct

import numpy as np

from . import bf16


def xor_bytes(a: bytes, b: bytes) -> bytes:
    """Bitwise XOR of two equal-length byte strings, as numpy u8 arrays."""
    if len(a) != len(b):
        raise ValueError(
            f"cross_layer_delta XOR: length mismatch ({len(a)} vs {len(b)})"
        )
    if len(a) == 0:
        return b""
    aa = np.frombuffer(a, dtype=np.uint8)
    bb = np.frombuffer(b, dtype=np.uint8)
    return (aa ^ bb).tobytes()


def encode_group(raws: list[bytes]) -> list[bytes]:
    """Encode a chain of same-shape same-dtype tensors as XOR-deltas.

    Args:
        raws: list of raw byte buffers, all same length and same encoding.

    Returns:
        list of compressed blobs:
          - blobs[0] is plain bf16.encode(raws[0])
          - blobs[i>0] is bf16.encode(xor(raws[i], raws[i-1]))

    No safety net is applied at this level: every tensor after the first is
    delta-encoded. Callers that need a per-tensor smallest-wins rule should
    encode both ways and compare.
    """
    if not raws:
        return []
    out: list[bytes] = []
    head_blob, _ = bf16.encode(raws[0])
    out.append(head_blob)
    for i in range(1, len(raws)):
        delta = xor_bytes(raws[i], raws[i - 1])
        delta_blob, _ = bf16.encode(delta)
        out.append(delta_blob)
    return out


def decode_group(blobs: list[bytes], n_weights_each: int) -> list[bytes]:
    """Decode a chain of XOR-deltas back to the original raws.

    Args:
        blobs:           list of compressed blobs from `encode_group`.
        n_weights_each:  element count per tensor (must be the same).

    Returns:
        list of raw byte buffers, exactly recovering the originals.
    """
    if not blobs:
        return []
    raws: list[bytes] = []
    head_raw = bf16.decode(blobs[0], {}, n_weights_each)
    raws.append(head_raw)
    for i in range(1, len(blobs)):
        delta = bf16.decode(blobs[i], {}, n_weights_each)
        raws.append(xor_bytes(delta, raws[-1]))
    return raws


# --- Single-tensor convenience ------------------------------------------------
#
# When wired into the encoder as a per-tensor candidate (each tensor knows its
# predecessor by name), it's useful to compute the delta blob with a single
# call that also returns a header dict pointing at the predecessor. The
# encoder is then responsible for the safety-net comparison vs a plain
# `auto_select_codec` blob.


def encode_pair(curr_raw: bytes, prev_raw: bytes,
                prev_name: str) -> tuple[bytes, dict]:
    """Encode `curr_raw` as a delta from `prev_raw`.

    Args:
        curr_raw:  raw bytes of the current tensor.
        prev_raw:  raw bytes of the predecessor tensor (same length).
        prev_name: name of the predecessor tensor (recorded in extras so the
                   decoder can resolve the reference at load time).

    Returns:
        (blob, extras). `extras = {"delta_from": prev_name}`.
    """
    delta = xor_bytes(curr_raw, prev_raw)
    blob, _ = bf16.encode(delta)
    return blob, {"delta_from": prev_name}


def decode_pair(blob: bytes, prev_raw: bytes,
                n_weights: int) -> bytes:
    """Inverse of `encode_pair`."""
    delta = bf16.decode(blob, {}, n_weights)
    return xor_bytes(delta, prev_raw)
