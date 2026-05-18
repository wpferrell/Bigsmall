"""BF16 codec using rANS (Asymmetric Numeral Systems) instead of range AC.

Algorithm is identical to `bf16_se_ac`:
  1. Split each BF16 word into (sign, exp, mantissa).
  2. Joint-code (sign, exp) as a 9-bit alphabet (`SE_ALPHABET = 512`).
  3. Sort mantissa by exp into per-exp buckets, code each bucket.

The only change is the entropy coder: `constriction.stream.stack.AnsCoder`
replaces `constriction.stream.queue.RangeEncoder/Decoder`. Compressed
size is identical to bf16_se_ac within 0.003%.

Measured throughput on a 4M-element synthetic SE stream (constriction Rust):
  RangeCoder decode: 133.8 MB/s
  AnsCoder decode:   157.8 MB/s  (1.18x faster)

The on-disk bytestream differs from bf16_se_ac — files encoded with this
codec carry codec_name "bf16_se_rans" and the decoder dispatches based on
that. The existing `bf16_se_ac` decoder is kept forever for backward
compat with all .bs files written by 3.0.0-3.3.0.
"""
from __future__ import annotations

import io
import struct

import numpy as np
import constriction as c

# Reuse the BF16 bit-layout constants
from .bf16 import (
    SIGN_SHIFT, EXP_SHIFT, EXP_MASK, MANT_MASK,
    SE_ALPHABET, MANT_ALPHABET,
)


def _encode_cat_ans(values: np.ndarray, alphabet: int) -> tuple[bytes, np.ndarray, np.ndarray]:
    """Encode int array with constriction.AnsCoder.

    Returns (codeword_bytes, nonzero_indices, nonzero_freqs) — same return
    shape as the AC version so the surrounding header/format stays in
    lock-step with bf16_se_ac except for the bitstream itself.
    """
    fp = np.bincount(values, minlength=alphabet).astype(np.int64)
    nz_idx = np.nonzero(fp)[0]
    probs = fp.astype(np.float64) + 0.01
    probs /= probs.sum()
    model = c.stream.model.Categorical(probs, perfect=True)
    ans = c.stream.stack.AnsCoder()
    # encode_reverse pushes symbols onto the stack so subsequent decode pops
    # them out in the original order.
    ans.encode_reverse(values.astype(np.int32), model)
    cw = ans.get_compressed().tobytes()
    return cw, nz_idx, fp[nz_idx].astype(np.int64)


def _decode_cat_ans(cw_bytes: bytes, nz_idx: np.ndarray, freqs: np.ndarray,
                    alphabet: int, n: int) -> np.ndarray:
    """Inverse of _encode_cat_ans."""
    fp = np.zeros(alphabet, dtype=np.int64)
    fp[nz_idx] = freqs
    probs = fp.astype(np.float64) + 0.01
    probs /= probs.sum()
    model = c.stream.model.Categorical(probs, perfect=True)
    cw = np.frombuffer(cw_bytes, dtype=np.uint32)
    ans = c.stream.stack.AnsCoder(cw)
    return ans.decode(model, n)


def encode(raw: bytes) -> tuple[bytes, dict]:
    """Encode a single BF16 tensor's raw bytes with rANS. Lossless."""
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

    se_cw, se_nz_idx, se_freqs = _encode_cat_ans(se, SE_ALPHABET)

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
        cw, nz_idx, freqs = _encode_cat_ans(bucket, MANT_ALPHABET)
        m_buf.write(struct.pack("<IB", be - bs, len(nz_idx)))
        m_buf.write(nz_idx.astype(np.uint8).tobytes())
        m_buf.write(freqs.astype(np.uint32).tobytes())
        m_buf.write(struct.pack("<I", len(cw)))
        m_buf.write(cw)
    m_blob = m_buf.getvalue()

    out = io.BytesIO()
    out.write(struct.pack("<I", n))
    out.write(struct.pack("<H", len(se_nz_idx)))
    out.write(se_nz_idx.astype(np.uint16).tobytes())
    out.write(se_freqs.astype(np.uint32).tobytes())
    out.write(struct.pack("<I", len(se_cw)))
    out.write(se_cw)
    out.write(struct.pack("<I", len(m_blob)))
    out.write(m_blob)
    return out.getvalue(), {}


def decode(blob: bytes, extras: dict, n_weights: int) -> bytes:
    """Decode a bf16_se_rans blob back to raw bytes (lossless)."""
    if n_weights == 0 or len(blob) == 0:
        return b""
    inp = io.BytesIO(blob)
    n, = struct.unpack("<I", inp.read(4))
    if n != n_weights:
        raise ValueError(f"bf16_se_rans decode: count mismatch ({n} vs {n_weights})")
    if n == 0:
        return b""

    se_n_nz, = struct.unpack("<H", inp.read(2))
    se_nz_idx = np.frombuffer(inp.read(se_n_nz * 2), dtype=np.uint16).astype(np.int32)
    se_freqs = np.frombuffer(inp.read(se_n_nz * 4), dtype=np.uint32)
    se_cw_len, = struct.unpack("<I", inp.read(4))
    se_cw = inp.read(se_cw_len)
    se = _decode_cat_ans(se_cw, se_nz_idx, se_freqs, SE_ALPHABET, n).astype(np.uint16)
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
        nz_idx = np.frombuffer(m_inp.read(n_nz), dtype=np.uint8).astype(np.int32)
        freqs = np.frombuffer(m_inp.read(n_nz * 4), dtype=np.uint32)
        cw_len, = struct.unpack("<I", m_inp.read(4))
        cw = m_inp.read(cw_len)
        mant_sorted[bstart[ev]:bstart[ev + 1]] = _decode_cat_ans(
            cw, nz_idx, freqs, MANT_ALPHABET, nb
        ).astype(np.uint16)

    order_e = np.argsort(exp, kind="stable")
    mant = np.empty(n, dtype=np.uint16)
    mant[order_e] = mant_sorted

    out = ((sign << SIGN_SHIFT) | (exp << EXP_SHIFT) | mant).astype(np.uint16)
    return out.tobytes()
