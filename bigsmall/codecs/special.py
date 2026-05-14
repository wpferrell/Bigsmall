"""Special tensor codecs: low-cardinality masks, row-delta embeddings, tied refs.

Each function returns (compressed_blob, extras_dict). Decode takes (blob, extras, n_bytes).

Special types:
  - "lowcard"   : tensor with very few unique values (e.g. attn.bias mask).
                  Stored as: list of unique uint8 values + per-position index stream
                  compressed with zstd. For binary masks this collapses to <100 bytes.
  - "wpe_delta" : 2D float tensor where consecutive rows are highly correlated
                  (position embeddings). Compressed via per-row delta of raw bytes
                  + blosc2 SHUFFLE+ZSTD.
  - "tied_ref"  : Empty blob, header marks 'special.tied_ref_to' = name of master tensor.
"""
import struct
import io
import numpy as np
import zstandard as zstd
import blosc2


# --------------------- Low-cardinality (mask-like) ----------------------------

def encode_lowcard(raw: bytes, item_bytes: int) -> tuple[bytes, dict]:
    """Encode a tensor with very few unique values.

    item_bytes: size of one element in the tensor (1 for fp8/fp4, 2 for bf16/fp16, 4 for fp32).
    """
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(-1, item_bytes)
    # Build unique map by treating each element as raw bytes
    # Use a view by computing a hash key
    if item_bytes == 1:
        keys = arr[:, 0].astype(np.uint64)
    elif item_bytes == 2:
        keys = arr[:, 0].astype(np.uint64) | (arr[:, 1].astype(np.uint64) << 8)
    elif item_bytes == 4:
        keys = (arr[:, 0].astype(np.uint64) |
                (arr[:, 1].astype(np.uint64) << 8) |
                (arr[:, 2].astype(np.uint64) << 16) |
                (arr[:, 3].astype(np.uint64) << 24))
    else:
        raise ValueError(f"Unsupported item_bytes={item_bytes}")

    unique_keys, inverse = np.unique(keys, return_inverse=True)
    n_unique = len(unique_keys)
    if n_unique > 256:
        raise ValueError(f"lowcard codec requires <= 256 unique values, got {n_unique}")

    # Pack unique values back to raw bytes
    unique_bytes = np.zeros((n_unique, item_bytes), dtype=np.uint8)
    for i, k in enumerate(unique_keys):
        for b in range(item_bytes):
            unique_bytes[i, b] = (int(k) >> (8 * b)) & 0xFF

    # Indices as uint8
    idx_u8 = inverse.astype(np.uint8)
    cctx = zstd.ZstdCompressor(level=19)
    idx_blob = cctx.compress(idx_u8.tobytes())

    out = io.BytesIO()
    out.write(struct.pack("<BB", item_bytes, n_unique))
    out.write(unique_bytes.tobytes())
    out.write(struct.pack("<I", len(idx_blob)))
    out.write(idx_blob)
    out.write(struct.pack("<I", len(arr)))  # n elements
    return out.getvalue(), {"item_bytes": item_bytes, "n_unique": n_unique}


def decode_lowcard(blob: bytes, extras: dict, n_bytes: int) -> bytes:
    inp = io.BytesIO(blob)
    item_bytes, n_unique = struct.unpack("<BB", inp.read(2))
    unique_bytes = np.frombuffer(inp.read(n_unique * item_bytes), dtype=np.uint8).reshape(n_unique, item_bytes)
    idx_blob_len, = struct.unpack("<I", inp.read(4))
    idx_blob = inp.read(idx_blob_len)
    n_elem, = struct.unpack("<I", inp.read(4))

    dctx = zstd.ZstdDecompressor()
    idx = np.frombuffer(dctx.decompress(idx_blob), dtype=np.uint8)
    if len(idx) != n_elem:
        raise ValueError(f"lowcard decode: index count mismatch {len(idx)} vs {n_elem}")

    out = unique_bytes[idx].reshape(-1)
    if out.nbytes != n_bytes:
        raise ValueError(f"lowcard decode: byte count mismatch {out.nbytes} vs {n_bytes}")
    return out.tobytes()


# --------------------- WPE delta (row-correlated 2D embeddings) --------------

def encode_wpe_delta(raw: bytes, item_bytes: int, shape: list[int]) -> tuple[bytes, dict]:
    """Encode 2D embedding tensor via per-row delta of raw byte view + blosc2.

    shape: tensor shape, must be 2D (rows, cols).
    """
    if len(shape) != 2:
        raise ValueError(f"wpe_delta requires 2D shape, got {shape}")
    rows, cols = shape
    np_dtype = {1: np.uint8, 2: np.uint16, 4: np.uint32}[item_bytes]
    arr = np.frombuffer(raw, dtype=np_dtype).reshape(rows, cols)
    delta = arr.copy()
    delta[1:] = arr[1:] - arr[:-1]
    cp = blosc2.CParams(
        codec=blosc2.Codec.ZSTD, clevel=9,
        filters=[blosc2.Filter.SHUFFLE], nthreads=1,
        typesize=item_bytes, blocksize=256 * 1024,
        splitmode=blosc2.SplitMode.ALWAYS_SPLIT,
    )
    delta_blob = bytes(blosc2.compress2(delta.ravel(), cparams=cp))
    raw_blob = bytes(blosc2.compress2(arr.ravel(), cparams=cp))
    if len(delta_blob) < len(raw_blob):
        chosen, use_delta = delta_blob, 1
    else:
        chosen, use_delta = raw_blob, 0

    out = io.BytesIO()
    out.write(struct.pack("<BBII", item_bytes, use_delta, rows, cols))
    out.write(struct.pack("<I", len(chosen)))
    out.write(chosen)
    return out.getvalue(), {"item_bytes": item_bytes, "use_delta": bool(use_delta)}


def decode_wpe_delta(blob: bytes, extras: dict, n_bytes: int) -> bytes:
    inp = io.BytesIO(blob)
    item_bytes, use_delta, rows, cols = struct.unpack("<BBII", inp.read(10))
    np_dtype = {1: np.uint8, 2: np.uint16, 4: np.uint32}[item_bytes]
    blob_len, = struct.unpack("<I", inp.read(4))
    payload = inp.read(blob_len)
    raw = np.frombuffer(blosc2.decompress2(payload, dparams=blosc2.DParams(nthreads=1)), dtype=np_dtype).reshape(rows, cols)
    if use_delta:
        out = np.empty_like(raw)
        out[0] = raw[0]
        for i in range(1, rows):
            out[i] = out[i - 1] + raw[i]
    else:
        out = raw
    return out.ravel().tobytes()


# --------------------- Codec registry interface ------------------------------

def encode(raw: bytes, **kwargs) -> tuple[bytes, dict]:
    """Dispatch by 'special' kwarg."""
    sp = kwargs.get("special")
    if sp == "lowcard":
        return encode_lowcard(raw, kwargs["item_bytes"])
    if sp == "wpe_delta":
        return encode_wpe_delta(raw, kwargs["item_bytes"], kwargs["shape"])
    raise ValueError(f"Unknown special codec: {sp}")


def decode(blob: bytes, extras: dict, n_bytes: int) -> bytes:
    sp = extras.get("special_kind")
    if sp == "lowcard":
        return decode_lowcard(blob, extras, n_bytes)
    if sp == "wpe_delta":
        return decode_wpe_delta(blob, extras, n_bytes)
    raise ValueError(f"Unknown special codec: {sp}")
