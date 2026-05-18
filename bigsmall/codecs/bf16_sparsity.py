"""BF16 sparsity-aware codec (A5).

Splits a BF16 tensor into two populations by a threshold T on |w|, codes the
two populations separately with the existing per-tensor joint AC coder, and
codes the per-element 0/1 selector mask with the same AC machinery on a
two-symbol alphabet.

Why this can beat the plain BF16 codec
--------------------------------------
The plain bf16 codec pays H(sign,exp) + H(mant|exp) per element across the
whole tensor. On a high-kurtosis tensor (e.g. Qwen3 early MLP gate_proj) the
exponent histogram is wide because of the outlier tail, even though the bulk
of weights are near zero. Splitting the tensor by |w| against T yields:

  - near-zero population: small dynamic range -> tight H(exp) -> cheap.
  - outlier population: also smaller H(exp) than the mixed distribution
                        because the tiny-weight exponent bins drop out.

Plus a one-shot mask whose entropy is small (~0.5 bit per element when the
populations are close to 50/50, much less when one dominates).

Container layout
----------------
The codec emits a self-describing byte blob:

    u32  n_elements
    u16  threshold_bf16_word         (the BF16 representation of T; lets the
                                       decoder reconstruct T without metadata)
    u32  near_zero_count
    u32  outlier_count

    -- mask block (2-symbol AC) --
    u16  mask_n_nonzero_symbols      (1 or 2)
    [mask_n_nonzero_symbols * u16]   nonzero symbol values (0 or 1)
    [mask_n_nonzero_symbols * u32]   nonzero symbol frequencies
    u32  mask_codeword_length
    [mask_codeword_length bytes]     mask AC codeword

    -- near-zero sub-population (encoded with bf16.encode) --
    u32  nz_blob_length
    [nz_blob_length bytes]           bf16 codec output

    -- outlier sub-population (encoded with bf16.encode) --
    u32  outlier_blob_length
    [outlier_blob_length bytes]      bf16 codec output

Lossless guarantee
------------------
Encoding reorders elements into two ordered sub-sequences keyed by the mask.
Decoding re-interleaves them by walking the mask in order, so the output byte
sequence is byte-identical to the input. Round-trips are md5-verified in
`tests/test_a5_sparsity.py`.
"""
from __future__ import annotations

import io
import struct

import numpy as np

from . import bf16


def _bf16_word_to_float32(word: int) -> float:
    """Convert a BF16 uint16 word to its IEEE-754 fp32 value."""
    arr = np.array([word], dtype=np.uint16)
    return float((arr.astype(np.uint32) << 16).view(np.float32)[0])


def _float32_to_bf16_word(x: float) -> int:
    """Round-to-nearest-even fp32 -> bf16, return the resulting uint16 word.

    Uses numpy's bfloat16 conversion if available; falls back to a manual
    round-truncate that matches `torch.bfloat16` semantics on values within the
    bf16 representable range.
    """
    # bfloat16 isn't a numpy dtype, so we drop the bottom 16 bits of an fp32
    # word with round-to-nearest-even, exactly like torch does.
    f32 = np.float32(x)
    u32 = f32.view(np.uint32) if isinstance(f32, np.floating) else \
        np.array([f32], dtype=np.float32).view(np.uint32)[0]
    # Round to nearest, ties to even on the bottom 16 bits.
    rounding_bias = 0x7FFF + ((u32 >> 16) & 1)
    u32_rounded = u32 + rounding_bias
    return int((u32_rounded >> 16) & 0xFFFF)


def choose_threshold_word(raw: bytes, factor: float = 0.25) -> int:
    """Pick the per-tensor sparsity threshold T as a BF16 word.

    Heuristic from the A5 spec: T = mean(|w|) * factor (default factor=0.25).
    Returns the BF16 representation of T so it can be stored in 2 bytes.
    """
    u16 = np.frombuffer(raw, dtype=np.uint16)
    if u16.size == 0:
        return 0
    u32 = u16.astype(np.uint32) << 16
    f = u32.view(np.float32)
    finite = f[np.isfinite(f)]
    if finite.size == 0:
        return 0
    mean_abs = float(np.abs(finite).mean())
    return _float32_to_bf16_word(mean_abs * factor)


def encode(raw: bytes, threshold_word: int | None = None) -> tuple[bytes, dict]:
    """Encode a BF16 tensor with the A5 sparsity-aware codec.

    Args:
        raw: bf16 little-endian bytes; len must be even.
        threshold_word: optional pre-computed BF16 threshold word. If None,
            picked automatically via `choose_threshold_word(raw)`.

    Returns:
        (blob, extras). Extras is empty -- the codec is self-describing.

    Raises:
        ValueError: if `raw` length is not a multiple of 2.
    """
    if len(raw) % 2 != 0:
        raise ValueError(f"BF16 sparsity encode: byte length must be even, got {len(raw)}")
    u16 = np.frombuffer(raw, dtype=np.uint16)
    n = int(u16.size)
    if n == 0:
        return b"", {}

    if threshold_word is None:
        threshold_word = choose_threshold_word(raw)
    threshold_f = _bf16_word_to_float32(threshold_word)

    # |w| comparison -- convert bf16 to fp32 only for the comparison.
    f = (u16.astype(np.uint32) << 16).view(np.float32)
    # `np.abs` on fp32 yields fp32; the comparison handles NaN safely
    # (NaN >= T is False, which puts NaN into the near-zero bucket. That's
    # fine for losslessness because the values themselves are preserved.)
    mask = (np.abs(f) >= threshold_f).astype(np.uint8)

    nz_idx = (mask == 0)
    out_idx = (mask == 1)
    nz_values = u16[nz_idx]
    out_values = u16[out_idx]
    nz_count = int(nz_values.size)
    outlier_count = int(out_values.size)

    # Encode the 0/1 mask with the existing AC coder on alphabet=2. The mask
    # tends to be very skewed (mostly 0 on sparse tensors) so this packs
    # tightly. We reuse `bf16._encode_cat` for symmetry with the rest of the
    # codec; it accepts an int32 array and returns (cw, nz_idx, freqs).
    mask_cw, mask_nz_syms, mask_freqs = bf16._encode_cat(
        mask.astype(np.int32), alphabet=2,
    )

    # Encode each sub-population with the existing per-tensor bf16 codec.
    # Both populations have a tighter H(exp) than the mixed distribution.
    nz_blob = b""
    out_blob = b""
    if nz_count > 0:
        nz_blob, _ = bf16.encode(nz_values.tobytes())
    if outlier_count > 0:
        out_blob, _ = bf16.encode(out_values.tobytes())

    buf = io.BytesIO()
    buf.write(struct.pack("<I", n))
    buf.write(struct.pack("<H", int(threshold_word) & 0xFFFF))
    buf.write(struct.pack("<I", nz_count))
    buf.write(struct.pack("<I", outlier_count))

    # Mask block
    buf.write(struct.pack("<H", int(mask_nz_syms.size)))
    buf.write(mask_nz_syms.astype(np.uint16).tobytes())
    buf.write(mask_freqs.astype(np.uint32).tobytes())
    buf.write(struct.pack("<I", len(mask_cw)))
    buf.write(mask_cw)

    # Sub-population blobs
    buf.write(struct.pack("<I", len(nz_blob)))
    buf.write(nz_blob)
    buf.write(struct.pack("<I", len(out_blob)))
    buf.write(out_blob)

    return buf.getvalue(), {}


def decode(blob: bytes, extras: dict, n_weights: int) -> bytes:
    """Decode a BF16 sparsity blob back to raw bf16 bytes."""
    if n_weights == 0 or len(blob) == 0:
        return b""
    inp = io.BytesIO(blob)
    n, = struct.unpack("<I", inp.read(4))
    if n != n_weights:
        raise ValueError(
            f"BF16 sparsity decode: weight count mismatch ({n} vs {n_weights})"
        )
    threshold_word, = struct.unpack("<H", inp.read(2))
    _threshold_f = _bf16_word_to_float32(threshold_word)
    nz_count, = struct.unpack("<I", inp.read(4))
    outlier_count, = struct.unpack("<I", inp.read(4))

    # Mask block
    mask_n_syms, = struct.unpack("<H", inp.read(2))
    mask_nz_syms = np.frombuffer(inp.read(mask_n_syms * 2),
                                 dtype=np.uint16).astype(np.int32)
    mask_freqs = np.frombuffer(inp.read(mask_n_syms * 4), dtype=np.uint32)
    mask_cw_len, = struct.unpack("<I", inp.read(4))
    mask_cw = inp.read(mask_cw_len)
    mask = bf16._decode_cat(mask_cw, mask_nz_syms, mask_freqs,
                            alphabet=2, n=n).astype(np.uint8)

    # Defensive: counts in header must match what the mask says.
    expected_outliers = int(mask.sum())
    if expected_outliers != outlier_count or (n - expected_outliers) != nz_count:
        raise ValueError(
            f"BF16 sparsity decode: mask population mismatch "
            f"(header nz={nz_count} outlier={outlier_count}, mask says "
            f"nz={n - expected_outliers} outlier={expected_outliers})"
        )

    # Sub-populations
    nz_blob_len, = struct.unpack("<I", inp.read(4))
    nz_blob = inp.read(nz_blob_len)
    nz_values: np.ndarray
    if nz_count > 0:
        nz_raw = bf16.decode(nz_blob, {}, nz_count)
        nz_values = np.frombuffer(nz_raw, dtype=np.uint16)
    else:
        nz_values = np.empty(0, dtype=np.uint16)

    out_blob_len, = struct.unpack("<I", inp.read(4))
    out_blob = inp.read(out_blob_len)
    out_values: np.ndarray
    if outlier_count > 0:
        out_raw = bf16.decode(out_blob, {}, outlier_count)
        out_values = np.frombuffer(out_raw, dtype=np.uint16)
    else:
        out_values = np.empty(0, dtype=np.uint16)

    # Re-interleave by walking the mask in order.
    u16 = np.empty(n, dtype=np.uint16)
    nz_mask = (mask == 0)
    u16[nz_mask] = nz_values
    u16[~nz_mask] = out_values
    return u16.tobytes()
