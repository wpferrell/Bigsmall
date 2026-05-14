"""FP4 codec: per-tensor Categorical AC on 4-bit indices (alphabet 16).

The codec operates on UNPACKED FP4 (one byte per 4-bit value). The encoder
takes raw bytes representing already-unpacked 4-bit indices in the low nibble.
"""
import struct
import io
import numpy as np
import constriction as c


def encode(raw: bytes) -> tuple[bytes, dict]:
    """Encode FP4 unpacked stream (low nibble of each byte)."""
    if len(raw) == 0:
        return b"", {}
    u8 = np.frombuffer(raw, dtype=np.uint8)
    # Defensive: ensure values are in [0, 15]
    if u8.max() > 15:
        # Caller passed packed FP4 - unpack
        n_total = len(u8) * 2
        unp = np.empty(n_total, dtype=np.uint8)
        unp[0::2] = u8 & 0x0F
        unp[1::2] = (u8 >> 4) & 0x0F
        u8 = unp
        was_packed = True
    else:
        was_packed = False
    n = len(u8)
    vals = u8.astype(np.int32)
    fp = np.bincount(vals, minlength=16).astype(np.int64)
    nz_idx = np.nonzero(fp)[0]
    probs = fp.astype(np.float64) + 0.01
    probs /= probs.sum()
    m = c.stream.model.Categorical(probs, perfect=True)
    enc = c.stream.queue.RangeEncoder()
    enc.encode(vals, m)
    cw = enc.get_compressed().tobytes()

    out = io.BytesIO()
    out.write(struct.pack("<IBB", n, len(nz_idx), 1 if was_packed else 0))
    out.write(nz_idx.astype(np.uint8).tobytes())
    out.write(fp[nz_idx].astype(np.uint32).tobytes())
    out.write(struct.pack("<I", len(cw)))
    out.write(cw)
    return out.getvalue(), {"was_packed": was_packed}


def decode(blob: bytes, extras: dict, n_weights: int) -> bytes:
    """Decode FP4 to unpacked stream (low nibble of each byte).

    Note: n_weights is interpreted as the unpacked count.
    If extras['was_packed'], the original was packed; we re-pack on output.
    """
    if n_weights == 0 or len(blob) == 0:
        return b""
    inp = io.BytesIO(blob)
    n, n_nz, was_packed_flag = struct.unpack("<IBB", inp.read(6))
    nz_idx = np.frombuffer(inp.read(n_nz), dtype=np.uint8)
    freqs = np.frombuffer(inp.read(n_nz * 4), dtype=np.uint32)
    cw_len, = struct.unpack("<I", inp.read(4))
    cw_bytes = inp.read(cw_len)
    fp = np.zeros(16, dtype=np.int64)
    fp[nz_idx] = freqs
    probs = fp.astype(np.float64) + 0.01
    probs /= probs.sum()
    m = c.stream.model.Categorical(probs, perfect=True)
    cw = np.frombuffer(cw_bytes, dtype=np.uint32)
    dec = c.stream.queue.RangeDecoder(cw)
    out = dec.decode(m, n).astype(np.uint8)

    if was_packed_flag:
        if n % 2 != 0:
            raise ValueError("FP4 packed decode requires even unpacked count")
        packed = (out[0::2] | (out[1::2] << 4)).astype(np.uint8)
        return packed.tobytes()
    return out.tobytes()
