"""FP16 codec: per-tensor (sign,exp) joint AC + per-tensor (mantissa | exp) AC.

FP16 layout: 1 sign bit | 5 exp bits | 10 mantissa bits = 16 bits total.

Same structure as bf16.py with different alphabet sizes.
"""
import struct
import io
import numpy as np
import constriction as c

SIGN_SHIFT = 15
EXP_SHIFT = 10
EXP_MASK = 0x1F        # 5 bits
MANT_MASK = 0x3FF      # 10 bits
SE_ALPHABET = 64       # 2 sign * 32 exp
MANT_ALPHABET = 1024   # 10 bits


def _encode_cat(values: np.ndarray, alphabet: int):
    fp = np.bincount(values, minlength=alphabet).astype(np.int64)
    nz_idx = np.nonzero(fp)[0]
    probs = fp.astype(np.float64) + 0.01
    probs /= probs.sum()
    m = c.stream.model.Categorical(probs, perfect=True)
    enc = c.stream.queue.RangeEncoder()
    enc.encode(values.astype(np.int32), m)
    cw = enc.get_compressed().tobytes()
    return cw, nz_idx, fp[nz_idx].astype(np.int64)


def _decode_cat(cw_bytes, nz_idx, freqs, alphabet, n):
    fp = np.zeros(alphabet, dtype=np.int64)
    fp[nz_idx] = freqs
    probs = fp.astype(np.float64) + 0.01
    probs /= probs.sum()
    m = c.stream.model.Categorical(probs, perfect=True)
    cw = np.frombuffer(cw_bytes, dtype=np.uint32)
    dec = c.stream.queue.RangeDecoder(cw)
    return dec.decode(m, n)


def encode(raw: bytes) -> tuple[bytes, dict]:
    if len(raw) % 2 != 0:
        raise ValueError(f"FP16 tensor byte length must be even, got {len(raw)}")
    u16 = np.frombuffer(raw, dtype=np.uint16)
    n = len(u16)
    if n == 0:
        return b"", {}

    sign = ((u16 >> SIGN_SHIFT) & 1).astype(np.uint16)
    exp = ((u16 >> EXP_SHIFT) & EXP_MASK).astype(np.uint16)
    mant = (u16 & MANT_MASK).astype(np.uint16)
    se = ((sign << 5) | exp).astype(np.int32)

    se_cw, se_nz_idx, se_freqs = _encode_cat(se, SE_ALPHABET)

    order_e = np.argsort(exp, kind="stable")
    mant_sorted = mant[order_e]
    exp_sorted = exp[order_e]
    counts = np.bincount(exp_sorted, minlength=32)
    nonzero_exps = np.nonzero(counts)[0].astype(np.int32)
    bstart = np.zeros(33, dtype=np.int64)
    bstart[1:] = np.cumsum(counts)

    m_buf = io.BytesIO()
    m_buf.write(struct.pack("<I", len(nonzero_exps)))
    m_buf.write(nonzero_exps.astype(np.uint8).tobytes())
    for ev in nonzero_exps:
        bs = bstart[ev]; be = bstart[ev + 1]
        bucket = mant_sorted[bs:be].astype(np.int32)
        cw, nz_idx, freqs = _encode_cat(bucket, MANT_ALPHABET)
        m_buf.write(struct.pack("<IH", be - bs, len(nz_idx)))
        m_buf.write(nz_idx.astype(np.uint16).tobytes())
        m_buf.write(freqs.astype(np.uint32).tobytes())
        m_buf.write(struct.pack("<I", len(cw)))
        m_buf.write(cw)
    m_blob = m_buf.getvalue()

    out = io.BytesIO()
    out.write(struct.pack("<I", n))
    out.write(struct.pack("<B", len(se_nz_idx)))
    out.write(se_nz_idx.astype(np.uint8).tobytes())
    out.write(se_freqs.astype(np.uint32).tobytes())
    out.write(struct.pack("<I", len(se_cw)))
    out.write(se_cw)
    out.write(struct.pack("<I", len(m_blob)))
    out.write(m_blob)
    return out.getvalue(), {}


def decode(blob: bytes, extras: dict, n_weights: int) -> bytes:
    if n_weights == 0 or len(blob) == 0:
        return b""
    inp = io.BytesIO(blob)
    n, = struct.unpack("<I", inp.read(4))
    if n != n_weights:
        raise ValueError(f"FP16 decode: weight count mismatch ({n} vs {n_weights})")
    if n == 0:
        return b""

    se_n_nz, = struct.unpack("<B", inp.read(1))
    se_nz_idx = np.frombuffer(inp.read(se_n_nz), dtype=np.uint8).astype(np.int32)
    se_freqs = np.frombuffer(inp.read(se_n_nz * 4), dtype=np.uint32)
    se_cw_len, = struct.unpack("<I", inp.read(4))
    se_cw = inp.read(se_cw_len)
    se = _decode_cat(se_cw, se_nz_idx, se_freqs, SE_ALPHABET, n).astype(np.uint16)
    sign = ((se >> 5) & 1).astype(np.uint16)
    exp = (se & 0x1F).astype(np.uint16)

    m_blob_len, = struct.unpack("<I", inp.read(4))
    m_inp = io.BytesIO(inp.read(m_blob_len))
    n_nz_exp, = struct.unpack("<I", m_inp.read(4))
    nonzero_exps = np.frombuffer(m_inp.read(n_nz_exp), dtype=np.uint8)

    order_e = np.argsort(exp, kind="stable")
    counts = np.bincount(exp.astype(np.int64), minlength=32)
    bstart = np.zeros(33, dtype=np.int64)
    bstart[1:] = np.cumsum(counts)

    mant_sorted = np.empty(n, dtype=np.uint16)
    for ev in nonzero_exps:
        nb, n_nz = struct.unpack("<IH", m_inp.read(6))
        nz_idx = np.frombuffer(m_inp.read(n_nz * 2), dtype=np.uint16).astype(np.int32)
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
