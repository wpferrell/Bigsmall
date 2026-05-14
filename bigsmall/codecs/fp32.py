"""FP32 codec: per-tensor (sign,exp) joint AC + per-tensor (mantissa | exp) AC.

FP32 layout: 1 sign | 8 exp | 23 mantissa = 32 bits total.

The 23-bit mantissa cannot be efficiently AC-encoded as a single 8M-symbol
alphabet, so we encode mantissa per-(exp) bucket using 3 byte-streams
(low, mid, high) compressed with zstd. This mirrors cc10_v2 byte-transpose
intuition while staying simple.

Strategy per tensor:
  1. Split into sign(1), exp(8), mant_lo8 + mant_mid8 + mant_hi7.
  2. Encode (sign, exp) as 9-bit alphabet (512) per tensor with Cat AC.
  3. Compress mant_lo / mant_mid / mant_hi as three byte streams with zstd L9
     (the high 7 bits of mantissa have lower entropy than the low 16).
"""
import struct
import io
import numpy as np
import constriction as c
import zstandard as zstd

SE_ALPHABET = 512  # 2 sign * 256 exp


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
    if len(raw) % 4 != 0:
        raise ValueError(f"FP32 tensor byte length must be % 4 == 0, got {len(raw)}")
    u32 = np.frombuffer(raw, dtype=np.uint32)
    n = len(u32)
    if n == 0:
        return b"", {}

    sign = ((u32 >> 31) & 1).astype(np.uint16)
    exp = ((u32 >> 23) & 0xFF).astype(np.uint16)
    mant = (u32 & 0x7FFFFF).astype(np.uint32)
    se = ((sign << 8) | exp).astype(np.int32)

    se_cw, se_nz_idx, se_freqs = _encode_cat(se, SE_ALPHABET)

    # Split mantissa into low 8, mid 8, high 7 bits and zstd-compress each
    # (these 3 byte-streams approximate per-byte entropy of the mantissa)
    mant_lo = (mant & 0xFF).astype(np.uint8)
    mant_mid = ((mant >> 8) & 0xFF).astype(np.uint8)
    mant_hi = ((mant >> 16) & 0x7F).astype(np.uint8)

    cctx = zstd.ZstdCompressor(level=9)
    blob_lo = cctx.compress(mant_lo.tobytes())
    blob_mid = cctx.compress(mant_mid.tobytes())
    blob_hi = cctx.compress(mant_hi.tobytes())

    out = io.BytesIO()
    out.write(struct.pack("<I", n))
    out.write(struct.pack("<H", len(se_nz_idx)))
    out.write(se_nz_idx.astype(np.uint16).tobytes())
    out.write(se_freqs.astype(np.uint32).tobytes())
    out.write(struct.pack("<I", len(se_cw)))
    out.write(se_cw)
    out.write(struct.pack("<I", len(blob_lo)))
    out.write(blob_lo)
    out.write(struct.pack("<I", len(blob_mid)))
    out.write(blob_mid)
    out.write(struct.pack("<I", len(blob_hi)))
    out.write(blob_hi)
    return out.getvalue(), {}


def decode(blob: bytes, extras: dict, n_weights: int) -> bytes:
    if n_weights == 0 or len(blob) == 0:
        return b""
    inp = io.BytesIO(blob)
    n, = struct.unpack("<I", inp.read(4))
    if n != n_weights:
        raise ValueError(f"FP32 decode: weight count mismatch ({n} vs {n_weights})")
    if n == 0:
        return b""

    se_n_nz, = struct.unpack("<H", inp.read(2))
    se_nz_idx = np.frombuffer(inp.read(se_n_nz * 2), dtype=np.uint16).astype(np.int32)
    se_freqs = np.frombuffer(inp.read(se_n_nz * 4), dtype=np.uint32)
    se_cw_len, = struct.unpack("<I", inp.read(4))
    se_cw = inp.read(se_cw_len)
    se = _decode_cat(se_cw, se_nz_idx, se_freqs, SE_ALPHABET, n).astype(np.uint32)
    sign = ((se >> 8) & 1).astype(np.uint32)
    exp = (se & 0xFF).astype(np.uint32)

    blob_lo_len, = struct.unpack("<I", inp.read(4)); blob_lo = inp.read(blob_lo_len)
    blob_mid_len, = struct.unpack("<I", inp.read(4)); blob_mid = inp.read(blob_mid_len)
    blob_hi_len, = struct.unpack("<I", inp.read(4)); blob_hi = inp.read(blob_hi_len)
    dctx = zstd.ZstdDecompressor()
    mant_lo = np.frombuffer(dctx.decompress(blob_lo), dtype=np.uint8).astype(np.uint32)
    mant_mid = np.frombuffer(dctx.decompress(blob_mid), dtype=np.uint8).astype(np.uint32)
    mant_hi = np.frombuffer(dctx.decompress(blob_hi), dtype=np.uint8).astype(np.uint32)
    mant = mant_lo | (mant_mid << 8) | (mant_hi << 16)

    out = ((sign << 31) | (exp << 23) | mant).astype(np.uint32)
    return out.tobytes()
