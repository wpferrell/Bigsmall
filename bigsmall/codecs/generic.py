"""Generic fallback codecs: zstd and blosc2 SHUFFLE+ZSTD."""
import struct
import io
import numpy as np
import zstandard as zstd
import blosc2


def encode_zstd(raw: bytes, level: int = 9) -> tuple[bytes, dict]:
    cctx = zstd.ZstdCompressor(level=level)
    out = cctx.compress(raw)
    return out, {"level": level, "n_bytes": len(raw)}


def decode_zstd(blob: bytes, extras: dict, n_bytes: int = None) -> bytes:
    dctx = zstd.ZstdDecompressor()
    out = dctx.decompress(blob)
    return out


def encode_blosc2_shuffle(raw: bytes, typesize: int, level: int = 9, blocksize: int = 256 * 1024) -> tuple[bytes, dict]:
    arr = np.frombuffer(raw, dtype=np.uint8)
    cp = blosc2.CParams(
        codec=blosc2.Codec.ZSTD, clevel=level,
        filters=[blosc2.Filter.SHUFFLE], nthreads=1,
        typesize=typesize, blocksize=blocksize,
        splitmode=blosc2.SplitMode.ALWAYS_SPLIT,
    )
    out = blosc2.compress2(arr, cparams=cp)
    return bytes(out), {"typesize": typesize, "level": level, "blocksize": blocksize, "n_bytes": len(raw)}


def decode_blosc2_shuffle(blob: bytes, extras: dict, n_bytes: int = None) -> bytes:
    out = blosc2.decompress2(blob, dparams=blosc2.DParams(nthreads=1))
    return bytes(out)


# Wrapper API for codec registry compatibility
def encode(raw: bytes, **kwargs) -> tuple[bytes, dict]:
    return encode_zstd(raw)


def decode(blob: bytes, extras: dict, n_weights: int = None) -> bytes:
    return decode_zstd(blob, extras)
