"""BF16 + FP2 quantize + lossless residual codec (V4 B1).

For a BF16 tensor with well-clustered, near-symmetric value distribution
(typical of attention and MLP weight matrices in transformers), we can:

  1. Quantize each weight to one of 4 levels {-s, -s/3, +s/3, +s} where s is
     the per-tensor absmax.
  2. Compute a BF16-rounded residual:  e_bf16 = bf16_round(w - dequant).
     The residual stream has tight (sign, exponent) distribution because the
     subtracted weights and dequant values share the same sign and exponent
     bucket — so H(s,e) collapses onto a couple of high-frequency entries
     and the joint codec exploits it.
  3. Capture the rounding error:  corr = w_u16 XOR bf16_round(dequant + e_bf16).
     This XOR is exactly zero on the elements that round-trip cleanly through
     BF16 arithmetic (10-25 % of elements depending on the tensor) and is a
     handful of small bit-patterns elsewhere. Its u16 distribution is very
     skewed so it compresses to a fraction of a bit per element.
  4. Compress both the BF16 residual and the correction with the existing
     BF16 codec; pack the FP2 indices as a 2-bit stream.

Why this can beat plain BF16 AC
-------------------------------
V4 Session A measured the FP32-subtraction lower bound on Phi-3.5-mini:
  - attention: 8.13 bits/el vs 10.50 BF16 baseline (-23.4 %)
  - mlp:       8.51 bits/el vs 10.48 BF16 baseline (-19.6 %)

The lossless codec realises a fraction of that (because correction adds
overhead) but still beats plain BF16 AC on real transformer attention and
MLP tensors. The encoder-side safety net in `codec_registry.auto_select_codec`
prevents regressions on tensors where the bound doesn't beat baseline.

Lossless guarantee
------------------
The encode path is deterministic numpy arithmetic (no torch dependency at
encode time, only at the top of the encoder). The decode path uses the
identical numpy operations to reproduce `w_check`, then XORs the correction
to restore the original u16. Every BF16 word — including NaN, +/-Inf, and
denormals — is preserved exactly.

Container layout
----------------
    u32  n_elements
    u16  absmax_bf16_word          (the BF16 representation of |w|_max;
                                    decoder rebuilds the 4 levels from this)
    u32  fp2_byte_count            (= ceil(n_elements / 4))
    [fp2_byte_count bytes]         packed 2-bit FP2 indices (LSB-first)
    u32  residual_blob_length
    [residual_blob_length bytes]   bf16.encode(e_bf16_u16)
    u32  correction_blob_length
    [correction_blob_length bytes] bf16.encode(correction_u16)
"""
from __future__ import annotations

import io
import struct

import numpy as np

from . import bf16


# Minimum elements required for the codec to be considered. Below this the
# FP2 + residual header overhead dominates and plain bf16 wins.
MIN_ELEMENTS = 65_536


def _bf16_word_to_float32(word: int) -> float:
    """Convert a BF16 uint16 word to its IEEE-754 fp32 value."""
    arr = np.array([word], dtype=np.uint16)
    return float((arr.astype(np.uint32) << 16).view(np.float32)[0])


def _float32_to_bf16_word(x: float) -> int:
    """Round-to-nearest-even fp32 -> bf16 (matches torch.bfloat16 semantics)."""
    f32 = np.float32(x)
    if isinstance(f32, np.floating):
        u32 = f32.view(np.uint32)
    else:
        u32 = np.array([f32], dtype=np.float32).view(np.uint32)[0]
    rounding_bias = 0x7FFF + ((u32 >> 16) & 1)
    u32_rounded = u32 + rounding_bias
    return int((u32_rounded >> 16) & 0xFFFF)


def _bf16_array_from_fp32(f: np.ndarray) -> np.ndarray:
    """Vectorised round-to-nearest-even fp32 -> bf16 u16 word array.

    Mirrors the scalar `_float32_to_bf16_word` and matches torch.bfloat16
    rounding so any tensor produced by torch round-trips bit-exactly through
    this conversion.
    """
    u32 = f.astype(np.float32).view(np.uint32)
    rounding_bias = 0x7FFF + ((u32 >> 16) & 1)
    return ((u32 + rounding_bias) >> 16).astype(np.uint16)


def _bf16_op(a_u16: np.ndarray, b_u16: np.ndarray,
             op: str) -> np.ndarray:
    """Apply bf16-rounded arithmetic to two u16 streams."""
    a_f = (a_u16.astype(np.uint32) << 16).view(np.float32)
    b_f = (b_u16.astype(np.uint32) << 16).view(np.float32)
    if op == "sub":
        out_f = a_f - b_f
    elif op == "add":
        out_f = a_f + b_f
    else:
        raise ValueError(f"unsupported op: {op}")
    return _bf16_array_from_fp32(out_f)


def _compute_levels_bf16(absmax_word: int) -> np.ndarray:
    """Reconstruct the four FP2 dequant level words deterministically.

    Levels are derived from a single stored absmax BF16 word so encoder and
    decoder agree to the bit. Each level is rounded to BF16 independently
    after the FP32 arithmetic so the dequant u16 values match exactly.
    """
    absmax_f = _bf16_word_to_float32(int(absmax_word))
    level_fp32 = np.array([-absmax_f, -absmax_f / 3.0,
                           absmax_f / 3.0, absmax_f], dtype=np.float32)
    return _bf16_array_from_fp32(level_fp32)


def _pack_2bit_indices(idx: np.ndarray) -> bytes:
    """Pack uint8 indices in [0,3] into 2-bit LSB-first nibbles per byte."""
    n = idx.size
    pad = (-n) % 4
    if pad:
        padded = np.zeros(n + pad, dtype=np.uint8)
        padded[:n] = idx
    else:
        padded = idx.astype(np.uint8)
    grouped = padded.reshape(-1, 4)
    packed = (
        (grouped[:, 0] & 0x3)
        | ((grouped[:, 1] & 0x3) << 2)
        | ((grouped[:, 2] & 0x3) << 4)
        | ((grouped[:, 3] & 0x3) << 6)
    ).astype(np.uint8)
    return packed.tobytes()


def _unpack_2bit_indices(packed: bytes, n: int) -> np.ndarray:
    """Inverse of `_pack_2bit_indices`."""
    arr = np.frombuffer(packed, dtype=np.uint8)
    out = np.empty(arr.size * 4, dtype=np.uint8)
    out[0::4] = arr & 0x3
    out[1::4] = (arr >> 2) & 0x3
    out[2::4] = (arr >> 4) & 0x3
    out[3::4] = (arr >> 6) & 0x3
    return out[:n]


def quantize_indices(u16: np.ndarray, absmax_word: int) -> tuple[np.ndarray, np.ndarray]:
    """Quantize a BF16 u16 stream to FP2 indices [0..3] and the matching dequant words.

    Boundaries between adjacent FP2 levels are the FP32 midpoints
    {-2/3*absmax, 0, +2/3*absmax}. NaN values map deterministically into a
    single bucket; round-trip is preserved by the XOR correction step anyway.
    """
    absmax_f = _bf16_word_to_float32(int(absmax_word))
    f = (u16.astype(np.uint32) << 16).view(np.float32)
    b1 = -2.0 * absmax_f / 3.0
    b3 = 2.0 * absmax_f / 3.0

    idx = np.full(f.size, 2, dtype=np.uint8)
    idx[f < b1] = 0
    idx[(f >= b1) & (f < 0.0)] = 1
    idx[(f >= 0.0) & (f < b3)] = 2
    idx[f >= b3] = 3

    levels = _compute_levels_bf16(absmax_word)
    dequant_u16 = levels[idx]
    return idx, dequant_u16


def encode(raw: bytes, **_) -> tuple[bytes, dict]:
    """Encode a BF16 tensor as FP2 indices + bf16 residual + bf16 correction.

    Args:
        raw: little-endian BF16 bytes; len must be even.

    Returns:
        (blob, extras). Extras is empty; the codec is self-describing.

    Raises:
        ValueError: if raw length is not a multiple of 2.
    """
    if len(raw) % 2 != 0:
        raise ValueError(
            f"FP2 residual encode: byte length must be even, got {len(raw)}"
        )
    u16 = np.frombuffer(raw, dtype=np.uint16)
    n = int(u16.size)
    if n == 0:
        return b"", {}

    f = (u16.astype(np.uint32) << 16).view(np.float32)
    finite = f[np.isfinite(f)]
    if finite.size == 0 or float(np.abs(finite).max()) == 0.0:
        absmax_word = 0
    else:
        absmax_word = _float32_to_bf16_word(float(np.abs(finite).max()))

    idx, dequant_u16 = quantize_indices(u16, absmax_word)

    # BF16-rounded residual: small in magnitude, very low H(s,e) because
    # most residuals fall in a tiny exponent range.
    e_bf16 = _bf16_op(u16, dequant_u16, op="sub")

    # Round-trip check in BF16 arithmetic, then capture the rounding error
    # as an XOR correction. Most elements have corr == 0; the non-zero values
    # are 1-2 bit patterns that compress easily.
    w_check = _bf16_op(dequant_u16, e_bf16, op="add")
    corr = (u16 ^ w_check).astype(np.uint16)

    residual_blob, _ = bf16.encode(e_bf16.tobytes())
    correction_blob, _ = bf16.encode(corr.tobytes())
    idx_packed = _pack_2bit_indices(idx)

    buf = io.BytesIO()
    buf.write(struct.pack("<I", n))
    buf.write(struct.pack("<H", int(absmax_word) & 0xFFFF))
    buf.write(struct.pack("<I", len(idx_packed)))
    buf.write(idx_packed)
    buf.write(struct.pack("<I", len(residual_blob)))
    buf.write(residual_blob)
    buf.write(struct.pack("<I", len(correction_blob)))
    buf.write(correction_blob)
    return buf.getvalue(), {}


def decode(blob: bytes, extras: dict, n_weights: int) -> bytes:
    """Decode an FP2 residual blob back to exact BF16 raw bytes."""
    if n_weights == 0 or len(blob) == 0:
        return b""
    inp = io.BytesIO(blob)
    n, = struct.unpack("<I", inp.read(4))
    if n != n_weights:
        raise ValueError(
            f"FP2 residual decode: weight count mismatch ({n} vs {n_weights})"
        )
    absmax_word, = struct.unpack("<H", inp.read(2))
    idx_byte_count, = struct.unpack("<I", inp.read(4))
    idx_packed = inp.read(idx_byte_count)
    idx = _unpack_2bit_indices(idx_packed, n)

    levels = _compute_levels_bf16(absmax_word)
    dequant_u16 = levels[idx]

    residual_blob_len, = struct.unpack("<I", inp.read(4))
    residual_blob = inp.read(residual_blob_len)
    e_bf16 = np.frombuffer(bf16.decode(residual_blob, {}, n), dtype=np.uint16)

    correction_blob_len, = struct.unpack("<I", inp.read(4))
    correction_blob = inp.read(correction_blob_len)
    corr = np.frombuffer(bf16.decode(correction_blob, {}, n), dtype=np.uint16)

    w_check = _bf16_op(dequant_u16, e_bf16, op="add")
    w_u16 = (w_check ^ corr).astype(np.uint16)
    return w_u16.tobytes()


def expected_to_beat_bf16(raw: bytes) -> bool:
    """Cheap pre-check: skip FP2 residual when it clearly won't help."""
    if len(raw) < 2 * MIN_ELEMENTS:
        return False
    u16 = np.frombuffer(raw, dtype=np.uint16)
    f = (u16.astype(np.uint32) << 16).view(np.float32)
    finite = f[np.isfinite(f)]
    if finite.size == 0:
        return False
    if float(np.abs(finite).max()) == 0.0:
        return False
    return True
