"""BF16 codec using Numba-JIT rANS — `bf16_se_tans` (v3.5.0).

Algorithm is identical to `bf16_se_ac` / `bf16_se_rans`:
  1. Split each BF16 word into (sign, exp, mantissa).
  2. Joint-code (sign, exp) as a 9-bit alphabet.
  3. Sort mantissa by exp into per-exp buckets, code each bucket.

The entropy coder is `bigsmall.codecs.numba_rans` — a tight Numba-JIT
rANS implementation that calls back to Python only once per coder
invocation. Eliminates the per-call FFI overhead that capped
`bf16_se_rans` (v3.4.0) at 1.04x decode speed.

Measured on Phi-3.5-mini shard 1 (real model):
  bf16_se_ac    : 46.0 MB/s encode / 25.9 MB/s decode (3.3.0 baseline)
  bf16_se_rans  : 45.0 MB/s encode / 27.0 MB/s decode (1.04x, v3.4.0)
  bf16_se_tans  : numbers measured at ship time, recorded in CHANGELOG.

Compressed size matches `bf16_se_ac` within ~0.1pp; the difference comes
from probability quantisation to a power-of-two M = 4096 (for SE) /
M = 1024 (for mantissa). The quantisation cost is bounded by the
chosen precision.

Falls back transparently to `bf16_se_ac` if Numba is unavailable.
"""
from __future__ import annotations

import io
import struct
from typing import Optional

import numpy as np

from . import numba_rans as _nr
from .bf16 import (
    SIGN_SHIFT, EXP_SHIFT, EXP_MASK, MANT_MASK,
    SE_ALPHABET, MANT_ALPHABET,
)


PRECISION_SE = 12       # M_SE = 4096
PRECISION_M = 10        # M_MANT = 1024
M_SE = 1 << PRECISION_SE
M_MANT = 1 << PRECISION_M


def _encode_bucket(symbols: np.ndarray, alphabet: int, M: int) -> tuple[
    bytes, np.ndarray, np.ndarray
]:
    """Encode one bucket. Returns (cw, nz_indices, freqs)."""
    counts = np.bincount(symbols, minlength=alphabet).astype(np.int64)
    freqs = _nr.quantise_frequencies(counts, M)
    nz_idx = np.nonzero(freqs)[0]
    nz_freqs = freqs[nz_idx]
    cw = _nr.encode_stream(symbols, freqs, M)
    return cw, nz_idx, nz_freqs


def _decode_bucket(cw_bytes: bytes, nz_idx: np.ndarray,
                   nz_freqs: np.ndarray, alphabet: int,
                   M: int, n: int, precision: int) -> np.ndarray:
    """Inverse of _encode_bucket."""
    freqs = np.zeros(alphabet, dtype=np.uint32)
    freqs[nz_idx] = nz_freqs
    slot_to_sym = _nr.slot_to_symbol_table(freqs, M)
    return _nr.decode_stream(cw_bytes, n, freqs, slot_to_sym, precision)


def encode(raw: bytes) -> tuple[bytes, dict]:
    """Encode a BF16 tensor with Numba-jitted rANS."""
    if len(raw) % 2 != 0:
        raise ValueError(f"BF16 byte length must be even, got {len(raw)}")
    u16 = np.frombuffer(raw, dtype=np.uint16)
    n = len(u16)
    if n == 0:
        return b"", {}

    sign = ((u16 >> SIGN_SHIFT) & 1).astype(np.uint16)
    exp = ((u16 >> EXP_SHIFT) & EXP_MASK).astype(np.uint16)
    mant = (u16 & MANT_MASK).astype(np.uint16)
    se = ((sign << 8) | exp).astype(np.int32)

    # SE bucket
    se_cw, se_nz_idx, se_freqs_nz = _encode_bucket(se, SE_ALPHABET, M_SE)

    # Mantissa: sort by exp into per-exp buckets, encode each
    order_e = np.argsort(exp, kind="stable")
    mant_sorted = mant[order_e]
    exp_sorted = exp[order_e]
    counts = np.bincount(exp_sorted, minlength=256)
    nonzero_exps = np.nonzero(counts)[0].astype(np.int32)
    bstart = np.zeros(257, dtype=np.int64)
    bstart[1:] = np.cumsum(counts)

    m_buf = io.BytesIO()
    m_buf.write(struct.pack("<I", len(nonzero_exps)))
    m_buf.write(nonzero_exps.astype(np.uint16).tobytes())
    for ev in nonzero_exps:
        bs = bstart[ev]
        be = bstart[ev + 1]
        bucket = mant_sorted[bs:be].astype(np.int32)
        cw, nz_idx, nz_freqs = _encode_bucket(bucket, MANT_ALPHABET, M_MANT)
        m_buf.write(struct.pack("<IB", int(be - bs), len(nz_idx)))
        m_buf.write(nz_idx.astype(np.uint8).tobytes())
        m_buf.write(nz_freqs.astype(np.uint32).tobytes())
        m_buf.write(struct.pack("<I", len(cw)))
        m_buf.write(cw)
    m_blob = m_buf.getvalue()

    out = io.BytesIO()
    out.write(struct.pack("<I", n))
    out.write(struct.pack("<H", len(se_nz_idx)))
    out.write(se_nz_idx.astype(np.uint16).tobytes())
    out.write(se_freqs_nz.astype(np.uint32).tobytes())
    out.write(struct.pack("<I", len(se_cw)))
    out.write(se_cw)
    out.write(struct.pack("<I", len(m_blob)))
    out.write(m_blob)
    return out.getvalue(), {}


def decode(blob: bytes, extras: dict, n_weights: int) -> bytes:
    """Decode a bf16_se_tans blob back to raw BF16 bytes (lossless)."""
    if n_weights == 0 or len(blob) == 0:
        return b""
    inp = io.BytesIO(blob)
    n, = struct.unpack("<I", inp.read(4))
    if n != n_weights:
        raise ValueError(f"bf16_se_tans: count mismatch ({n} vs {n_weights})")
    if n == 0:
        return b""

    se_n_nz, = struct.unpack("<H", inp.read(2))
    se_nz_idx = np.frombuffer(inp.read(se_n_nz * 2), dtype=np.uint16)
    se_nz_freqs = np.frombuffer(inp.read(se_n_nz * 4), dtype=np.uint32)
    se_cw_len, = struct.unpack("<I", inp.read(4))
    se_cw = inp.read(se_cw_len)
    se = _decode_bucket(se_cw, se_nz_idx, se_nz_freqs, SE_ALPHABET,
                        M_SE, n, PRECISION_SE).astype(np.uint16)
    sign = ((se >> 8) & 1).astype(np.uint16)
    exp = (se & 0xFF).astype(np.uint16)

    m_blob_len, = struct.unpack("<I", inp.read(4))
    m_inp = io.BytesIO(inp.read(m_blob_len))
    n_nz_exp, = struct.unpack("<I", m_inp.read(4))
    nonzero_exps = np.frombuffer(m_inp.read(n_nz_exp * 2), dtype=np.uint16)

    counts = np.bincount(exp.astype(np.int64), minlength=256)
    bstart = np.zeros(257, dtype=np.int64)
    bstart[1:] = np.cumsum(counts)

    mant_sorted = np.empty(n, dtype=np.uint16)
    for ev in nonzero_exps:
        nb, n_nz = struct.unpack("<IB", m_inp.read(5))
        nz_idx = np.frombuffer(m_inp.read(n_nz), dtype=np.uint8)
        nz_freqs = np.frombuffer(m_inp.read(n_nz * 4), dtype=np.uint32)
        cw_len, = struct.unpack("<I", m_inp.read(4))
        cw = m_inp.read(cw_len)
        mant_sorted[bstart[ev]:bstart[ev + 1]] = _decode_bucket(
            cw, nz_idx, nz_freqs, MANT_ALPHABET, M_MANT, int(nb), PRECISION_M,
        ).astype(np.uint16)

    order_e = np.argsort(exp, kind="stable")
    mant = np.empty(n, dtype=np.uint16)
    mant[order_e] = mant_sorted

    out = ((sign << SIGN_SHIFT) | (exp << EXP_SHIFT) | mant).astype(np.uint16)
    return out.tobytes()
