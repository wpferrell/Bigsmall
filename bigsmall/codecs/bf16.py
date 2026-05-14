"""BF16 codec: per-tensor (sign,exp) joint AC + per-tensor (mantissa | exp) AC.

BF16 layout: 1 sign bit | 8 exp bits | 7 mantissa bits = 16 bits total.

Encoding strategy (mirrors cc10_v2.py but operates per-tensor so it works on
ANY safetensors model - no GPT-2-specific handling here):

  1. Split each weight into (sign, exp, mantissa).
  2. Encode (sign, exp) jointly as a 9-bit alphabet per tensor with Categorical AC.
  3. Encode mantissa per-tensor, sorted by exp, with one Cat AC bucket per exp value.

This per-tensor approach generalises cleanly. The cc10_v2 script encoded the
mantissa stream globally across all non-special tensors; that gives a tiny
extra ratio improvement on GPT-2 but requires a fixed 'special tensor set' to
be known across the encode/decode path. Per-tensor encoding is simpler,
generalises, and the per-tensor overhead is negligible relative to the AC
codeword cost.
"""
import struct
import io
import numpy as np
import constriction as c

# Bit layout for BF16
SIGN_SHIFT = 15
EXP_SHIFT = 7
EXP_MASK = 0xFF
MANT_MASK = 0x7F
SE_ALPHABET = 512    # 2 sign * 256 exp
MANT_ALPHABET = 128  # 7 mantissa bits


def _encode_cat(values: np.ndarray, alphabet: int) -> tuple[bytes, np.ndarray, np.ndarray]:
    """Encode an int array with per-symbol Categorical AC.

    Returns (codeword_bytes, nonzero_indices, nonzero_freqs).
    """
    fp = np.bincount(values, minlength=alphabet).astype(np.int64)
    nz_idx = np.nonzero(fp)[0]
    probs = fp.astype(np.float64) + 0.01
    probs /= probs.sum()
    m = c.stream.model.Categorical(probs, perfect=True)
    enc = c.stream.queue.RangeEncoder()
    enc.encode(values.astype(np.int32), m)
    cw = enc.get_compressed().tobytes()
    return cw, nz_idx, fp[nz_idx].astype(np.int64)


def _decode_cat(cw_bytes: bytes, nz_idx: np.ndarray, freqs: np.ndarray, alphabet: int, n: int) -> np.ndarray:
    fp = np.zeros(alphabet, dtype=np.int64)
    fp[nz_idx] = freqs
    probs = fp.astype(np.float64) + 0.01
    probs /= probs.sum()
    m = c.stream.model.Categorical(probs, perfect=True)
    cw = np.frombuffer(cw_bytes, dtype=np.uint32)
    dec = c.stream.queue.RangeDecoder(cw)
    return dec.decode(m, n)


def encode(raw: bytes) -> tuple[bytes, dict]:
    """Encode a single BF16 tensor's raw bytes.

    Args:
        raw: little-endian bytes of the tensor (length must be even)

    Returns:
        (compressed_blob, extras_dict)

    The extras dict contains nothing - the codec is self-describing inside the blob.
    """
    if len(raw) % 2 != 0:
        raise ValueError(f"BF16 tensor byte length must be even, got {len(raw)}")
    u16 = np.frombuffer(raw, dtype=np.uint16)
    n = len(u16)
    if n == 0:
        return b"", {}

    sign = ((u16 >> SIGN_SHIFT) & 1).astype(np.uint16)
    exp = ((u16 >> EXP_SHIFT) & EXP_MASK).astype(np.uint16)
    mant = (u16 & MANT_MASK).astype(np.uint16)
    se = ((sign << 8) | exp).astype(np.int32)

    # SE block
    se_cw, se_nz_idx, se_freqs = _encode_cat(se, SE_ALPHABET)

    # Sort mantissa by exp -> per-exp buckets
    order_e = np.argsort(exp, kind="stable")
    mant_sorted = mant[order_e]
    exp_sorted = exp[order_e]
    counts = np.bincount(exp_sorted, minlength=256)
    nonzero_exps = np.nonzero(counts)[0].astype(np.int32)
    bstart = np.zeros(257, dtype=np.int64)
    bstart[1:] = np.cumsum(counts)

    # Encode mantissa per nonzero-exp bucket
    m_buf = io.BytesIO()
    m_buf.write(struct.pack("<I", len(nonzero_exps)))
    m_buf.write(nonzero_exps.astype(np.uint16).tobytes())
    for ev in nonzero_exps:
        bs = bstart[ev]
        be = bstart[ev + 1]
        bucket = mant_sorted[bs:be].astype(np.int32)
        cw, nz_idx, freqs = _encode_cat(bucket, MANT_ALPHABET)
        m_buf.write(struct.pack("<IB", be - bs, len(nz_idx)))
        m_buf.write(nz_idx.astype(np.uint8).tobytes())
        m_buf.write(freqs.astype(np.uint32).tobytes())
        m_buf.write(struct.pack("<I", len(cw)))
        m_buf.write(cw)
    m_blob = m_buf.getvalue()

    # Container blob
    out = io.BytesIO()
    out.write(struct.pack("<I", n))                         # tensor weight count
    out.write(struct.pack("<H", len(se_nz_idx)))            # SE nonzero count
    out.write(se_nz_idx.astype(np.uint16).tobytes())        # SE nonzero indices
    out.write(se_freqs.astype(np.uint32).tobytes())         # SE freqs
    out.write(struct.pack("<I", len(se_cw)))                # SE codeword length
    out.write(se_cw)
    out.write(struct.pack("<I", len(m_blob)))               # M block length
    out.write(m_blob)
    return out.getvalue(), {}


def decode(blob: bytes, extras: dict, n_weights: int) -> bytes:
    """Decode a BF16 blob back to raw bytes."""
    if n_weights == 0 or len(blob) == 0:
        return b""
    inp = io.BytesIO(blob)
    n, = struct.unpack("<I", inp.read(4))
    if n != n_weights:
        raise ValueError(f"BF16 decode: weight count mismatch ({n} vs {n_weights})")
    if n == 0:
        return b""

    se_n_nz, = struct.unpack("<H", inp.read(2))
    se_nz_idx = np.frombuffer(inp.read(se_n_nz * 2), dtype=np.uint16).astype(np.int32)
    se_freqs = np.frombuffer(inp.read(se_n_nz * 4), dtype=np.uint32)
    se_cw_len, = struct.unpack("<I", inp.read(4))
    se_cw = inp.read(se_cw_len)
    se = _decode_cat(se_cw, se_nz_idx, se_freqs, SE_ALPHABET, n).astype(np.uint16)
    sign = ((se >> 8) & 1).astype(np.uint16)
    exp = (se & 0xFF).astype(np.uint16)

    m_blob_len, = struct.unpack("<I", inp.read(4))
    m_inp = io.BytesIO(inp.read(m_blob_len))
    n_nz_exp, = struct.unpack("<I", m_inp.read(4))
    nonzero_exps = np.frombuffer(m_inp.read(n_nz_exp * 2), dtype=np.uint16)

    # Reconstruct sort order from decoded exp
    order_e = np.argsort(exp, kind="stable")
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
        mant_sorted[bstart[ev]:bstart[ev + 1]] = _decode_cat(
            cw, nz_idx, freqs, MANT_ALPHABET, nb
        ).astype(np.uint16)

    mant = np.empty(n, dtype=np.uint16)
    mant[order_e] = mant_sorted

    out = ((sign << SIGN_SHIFT) | (exp << EXP_SHIFT) | mant).astype(np.uint16)
    return out.tobytes()
