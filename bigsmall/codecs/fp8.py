"""FP8 codec: per-tensor Categorical AC on the byte stream.

Since FP8 is 1 byte per weight, per-tensor AC on the full byte is the natural
operation. Alphabet 256.
"""
import struct
import io
import numpy as np
import constriction as c


def encode(raw: bytes) -> tuple[bytes, dict]:
    if len(raw) == 0:
        return b"", {}
    u8 = np.frombuffer(raw, dtype=np.uint8)
    n = len(u8)
    vals = u8.astype(np.int32)
    fp = np.bincount(vals, minlength=256).astype(np.int64)
    nz_idx = np.nonzero(fp)[0]
    probs = fp.astype(np.float64) + 0.01
    probs /= probs.sum()
    m = c.stream.model.Categorical(probs, perfect=True)
    enc = c.stream.queue.RangeEncoder()
    enc.encode(vals, m)
    cw = enc.get_compressed().tobytes()
    freqs = fp[nz_idx].astype(np.uint32).tobytes()

    out = io.BytesIO()
    out.write(struct.pack("<IH", n, len(nz_idx)))
    out.write(nz_idx.astype(np.uint8).tobytes())
    out.write(freqs)
    out.write(struct.pack("<I", len(cw)))
    out.write(cw)
    return out.getvalue(), {}


def decode(blob: bytes, extras: dict, n_weights: int) -> bytes:
    if n_weights == 0 or len(blob) == 0:
        return b""
    inp = io.BytesIO(blob)
    n, n_nz = struct.unpack("<IH", inp.read(6))
    if n != n_weights:
        raise ValueError(f"FP8 decode: weight count mismatch ({n} vs {n_weights})")
    nz_idx = np.frombuffer(inp.read(n_nz), dtype=np.uint8)
    freqs = np.frombuffer(inp.read(n_nz * 4), dtype=np.uint32)
    cw_len, = struct.unpack("<I", inp.read(4))
    cw_bytes = inp.read(cw_len)
    fp = np.zeros(256, dtype=np.int64)
    fp[nz_idx] = freqs
    probs = fp.astype(np.float64) + 0.01
    probs /= probs.sum()
    m = c.stream.model.Categorical(probs, perfect=True)
    cw = np.frombuffer(cw_bytes, dtype=np.uint32)
    dec = c.stream.queue.RangeDecoder(cw)
    out = dec.decode(m, n).astype(np.uint8)
    return out.tobytes()
