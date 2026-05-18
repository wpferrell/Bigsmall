"""BF16 parallel-stream codec (`bf16_parallel_v1`, GPU-decodable format).

N interleaved slices of the tensor are each rANS-encoded using SHARED
probability tables:
  - One (sign,exp) joint frequency table built from the full tensor.
  - One per-exp mantissa frequency table built from the full tensor.

The frequency tables are quantised to power-of-two totals (M_SE and M_MANT)
so the rANS state update can use bit-shift instead of integer division on
GPU. Each slice gets two independent rANS bitstreams (one for SE, one for
mantissa). The decoder pops SE first to recover exps, then pops the
mantissa stream using the per-exp model dictated by the previously
decoded exp values.

Format (little-endian):

  HEADER:
    [4] magic = b"PAR1"
    [4] n_weights (uint32)
    [2] n_streams (uint16)
    [2] precision_se (uint16)   -- log2(M_SE)
    [2] precision_m  (uint16)   -- log2(M_MANT)
    [2] n_se_nz (uint16)
    [n_se_nz * 2] SE nonzero indices (uint16)
    [n_se_nz * 4] SE quantised frequencies (uint32, sum to M_SE)
    [4] n_nonzero_exps (uint32)
    [n_nonzero_exps * 2] nonzero exp values (uint16)
    for each exp in nonzero_exps:
      [1] mant_n_nz (uint8)
      [mant_n_nz * 1] mant nonzero indices (uint8)
      [mant_n_nz * 4] mant quantised frequencies (uint32, sum to M_MANT)

  PAYLOAD:
    [n_streams * 4] per-stream byte offset into payload (relative to payload start)
    for each stream k:
      [4] se_blob_len (uint32)
      [se_blob_len] SE rANS bytestream
      [4] n_nz_exps_in_stream_k (uint32)
      [n_nz_exps_in_stream_k * 2] exps with at least one element in this stream
      for each such exp:
        [4] count_in_stream (uint32)
        [4] mant_blob_len (uint32)
        [mant_blob_len] mantissa rANS bytestream for this (stream, exp) pair

The per-exp mantissa stream is split per-exp within each slice (not
globally) so the decoder doesn't need to know the exp of each element
before decoding mantissa — it decodes one bucket at a time. The encoder
sorts the slice's mantissas by exp inside this loop.

Per-tensor ratio cost vs single-stream bf16_se_ac on Phi-3.5-mini big
tensors (measured): +0.07-0.34 pp at N=128.
"""
from __future__ import annotations

import io
import struct
from typing import Optional

import numpy as np

from . import rans
from .bf16 import (
    SIGN_SHIFT, EXP_SHIFT, EXP_MASK, MANT_MASK,
    SE_ALPHABET, MANT_ALPHABET,
)


DEFAULT_N_STREAMS = 128
PRECISION_SE = 12       # M_SE = 4096
PRECISION_M = 10        # M_MANT = 1024
M_SE = 1 << PRECISION_SE
M_MANT = 1 << PRECISION_M

MAGIC = b"PAR1"


def _build_shared_freqs(
    se: np.ndarray, exp: np.ndarray, mant: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    """Build quantised shared frequency tables from the full tensor.

    Returns:
        se_freqs: uint32[SE_ALPHABET], sums to M_SE
        nonzero_exps: uint16 sorted array of exp values present in the tensor
        mant_freqs_by_exp: {exp_value: uint32[MANT_ALPHABET] summing to M_MANT}
    """
    se_counts = np.bincount(se, minlength=SE_ALPHABET).astype(np.int64)
    se_freqs = rans.quantise_frequencies(se_counts, M_SE)

    counts_by_exp = np.bincount(exp, minlength=256)
    nonzero_exps = np.nonzero(counts_by_exp)[0].astype(np.uint16)

    mant_freqs_by_exp: dict[int, np.ndarray] = {}
    for ev in nonzero_exps:
        mask = exp == ev
        mant_counts = np.bincount(
            mant[mask].astype(np.int64),
            minlength=MANT_ALPHABET,
        ).astype(np.int64)
        # quantise_frequencies requires non_nz <= M. M=1024 >= 128 alphabet OK.
        mant_freqs_by_exp[int(ev)] = rans.quantise_frequencies(mant_counts, M_MANT)

    return se_freqs, nonzero_exps, mant_freqs_by_exp


def _pack_header(
    n_weights: int, n_streams: int,
    se_freqs: np.ndarray, nonzero_exps: np.ndarray,
    mant_freqs_by_exp: dict[int, np.ndarray],
) -> bytes:
    """Pack the header bytes."""
    buf = io.BytesIO()
    buf.write(MAGIC)
    buf.write(struct.pack("<I", n_weights))
    buf.write(struct.pack("<H", n_streams))
    buf.write(struct.pack("<H", PRECISION_SE))
    buf.write(struct.pack("<H", PRECISION_M))

    # SE nonzero indices + freqs
    se_nz_idx = np.nonzero(se_freqs)[0].astype(np.uint16)
    buf.write(struct.pack("<H", len(se_nz_idx)))
    buf.write(se_nz_idx.tobytes())
    buf.write(se_freqs[se_nz_idx].astype(np.uint32).tobytes())

    # Per-exp mantissa tables
    buf.write(struct.pack("<I", len(nonzero_exps)))
    buf.write(nonzero_exps.tobytes())
    for ev in nonzero_exps:
        mf = mant_freqs_by_exp[int(ev)]
        nz = np.nonzero(mf)[0].astype(np.uint8)
        buf.write(struct.pack("<B", len(nz)))
        buf.write(nz.tobytes())
        buf.write(mf[nz].astype(np.uint32).tobytes())

    return buf.getvalue()


def _unpack_header(blob: bytes) -> tuple[
    int, int, int,
    np.ndarray, np.ndarray, dict[int, np.ndarray],
    int,  # cursor position after header
]:
    """Parse the header, returning the inverse of _pack_header.

    Returns:
        (n_weights, n_streams, payload_start,
         se_freqs (full SE_ALPHABET), nonzero_exps,
         mant_freqs_by_exp (full MANT_ALPHABET each),
         header_end_offset)
    """
    inp = io.BytesIO(blob)
    magic = inp.read(4)
    if magic != MAGIC:
        raise ValueError(f"bf16_parallel: bad magic {magic!r}")
    n_weights, = struct.unpack("<I", inp.read(4))
    n_streams, = struct.unpack("<H", inp.read(2))
    p_se, = struct.unpack("<H", inp.read(2))
    p_m, = struct.unpack("<H", inp.read(2))
    if p_se != PRECISION_SE or p_m != PRECISION_M:
        raise ValueError(
            f"bf16_parallel: precision mismatch (header SE={p_se} M={p_m} "
            f"vs codec SE={PRECISION_SE} M={PRECISION_M})"
        )

    n_se_nz, = struct.unpack("<H", inp.read(2))
    se_nz_idx = np.frombuffer(inp.read(n_se_nz * 2), dtype=np.uint16)
    se_nz_freqs = np.frombuffer(inp.read(n_se_nz * 4), dtype=np.uint32)
    se_freqs = np.zeros(SE_ALPHABET, dtype=np.uint32)
    se_freqs[se_nz_idx] = se_nz_freqs

    n_nz_exps, = struct.unpack("<I", inp.read(4))
    nonzero_exps = np.frombuffer(inp.read(n_nz_exps * 2), dtype=np.uint16).copy()

    mant_freqs_by_exp: dict[int, np.ndarray] = {}
    for ev in nonzero_exps:
        nz_count, = struct.unpack("<B", inp.read(1))
        nz = np.frombuffer(inp.read(nz_count), dtype=np.uint8)
        nz_freqs = np.frombuffer(inp.read(nz_count * 4), dtype=np.uint32)
        mf = np.zeros(MANT_ALPHABET, dtype=np.uint32)
        mf[nz] = nz_freqs
        mant_freqs_by_exp[int(ev)] = mf

    return n_weights, n_streams, inp.tell(), se_freqs, nonzero_exps, mant_freqs_by_exp, inp.tell()


def encode(raw: bytes, n_streams: int = DEFAULT_N_STREAMS) -> tuple[bytes, dict]:
    """Encode a BF16 tensor as N rANS-coded parallel streams sharing probability tables."""
    if len(raw) % 2 != 0:
        raise ValueError(f"BF16 byte length must be even, got {len(raw)}")
    u16 = np.frombuffer(raw, dtype=np.uint16)
    n = len(u16)
    if n == 0:
        return b"", {"n_streams": n_streams}

    n_streams = max(1, min(n_streams, n))

    sign = ((u16 >> SIGN_SHIFT) & 1).astype(np.uint16)
    exp = ((u16 >> EXP_SHIFT) & EXP_MASK).astype(np.uint16)
    mant = (u16 & MANT_MASK).astype(np.uint16)
    se = ((sign << 8) | exp).astype(np.int32)

    se_freqs, nonzero_exps, mant_freqs_by_exp = _build_shared_freqs(se, exp, mant)

    header_bytes = _pack_header(n, n_streams, se_freqs, nonzero_exps, mant_freqs_by_exp)

    # Encode each slice
    stream_blobs: list[bytes] = []
    for k in range(n_streams):
        se_slice = se[k::n_streams].astype(np.int32)
        exp_slice = exp[k::n_streams]
        mant_slice = mant[k::n_streams]

        sbuf = io.BytesIO()
        if len(se_slice) == 0:
            sbuf.write(struct.pack("<I", 0))    # se_blob_len = 0
            sbuf.write(struct.pack("<I", 0))    # n_nz_exps_in_stream = 0
        else:
            # SE rANS stream
            se_blob = rans.encode_stream(se_slice, se_freqs, M_SE)
            sbuf.write(struct.pack("<I", len(se_blob)))
            sbuf.write(se_blob)

            # Mantissa: sort by exp within slice, encode each bucket independently
            order = np.argsort(exp_slice, kind="stable")
            exp_sorted = exp_slice[order]
            mant_sorted = mant_slice[order]
            counts = np.bincount(exp_sorted, minlength=256)
            stream_nz_exps = np.nonzero(counts)[0].astype(np.uint16)
            bstart = np.zeros(257, dtype=np.int64)
            bstart[1:] = np.cumsum(counts)

            sbuf.write(struct.pack("<I", len(stream_nz_exps)))
            sbuf.write(stream_nz_exps.tobytes())
            for ev in stream_nz_exps:
                bs = bstart[ev]
                be = bstart[ev + 1]
                bucket = mant_sorted[bs:be].astype(np.int32)
                mant_blob = rans.encode_stream(
                    bucket, mant_freqs_by_exp[int(ev)], M_MANT,
                )
                sbuf.write(struct.pack("<II", int(be - bs), len(mant_blob)))
                sbuf.write(mant_blob)

        stream_blobs.append(sbuf.getvalue())

    # Per-stream offsets
    offsets: list[int] = []
    cur = 0
    for sb in stream_blobs:
        offsets.append(cur)
        cur += len(sb)
    offsets.append(cur)  # sentinel total

    out = io.BytesIO()
    out.write(header_bytes)
    out.write(struct.pack(f"<{n_streams + 1}I", *offsets))
    for sb in stream_blobs:
        out.write(sb)
    return out.getvalue(), {"n_streams": n_streams}


def decode(blob: bytes, extras: dict, n_weights: int) -> bytes:
    """Decode a bf16_parallel_v1 blob back to raw BF16 bytes (lossless, CPU path).

    The GPU path (bigsmall.kernels.decode_bf16_parallel) is bit-equivalent
    to this function on its output bytes.
    """
    if n_weights == 0 or len(blob) == 0:
        return b""

    (n, n_streams, _payload_start,
     se_freqs, nonzero_exps, mant_freqs_by_exp,
     header_end) = _unpack_header(blob)
    if n != n_weights:
        raise ValueError(f"bf16_parallel: weight count mismatch ({n} vs {n_weights})")

    # Slot-to-symbol tables (uint16 for SE since alphabet=512; uint8 for mantissa)
    se_slot_to_sym = rans.build_slot_to_symbol(se_freqs, M_SE)
    mant_slot_to_sym_by_exp: dict[int, np.ndarray] = {}
    for ev in nonzero_exps:
        mant_slot_to_sym_by_exp[int(ev)] = rans.build_slot_to_symbol(
            mant_freqs_by_exp[int(ev)], M_MANT,
        ).astype(np.uint8)

    # Per-stream offsets
    n_off = n_streams + 1
    offset_table = struct.unpack(
        f"<{n_off}I", blob[header_end:header_end + n_off * 4],
    )
    payload_start = header_end + n_off * 4

    # Per-stream element counts (every Nth element starting at offset k)
    stream_n_elements = [
        (n - 1 - k) // n_streams + 1 if k < n else 0
        for k in range(n_streams)
    ]
    assert sum(stream_n_elements) == n

    full_se = np.empty(n, dtype=np.uint16)
    full_mant = np.empty(n, dtype=np.uint16)

    for k in range(n_streams):
        n_in_stream = stream_n_elements[k]
        if n_in_stream == 0:
            continue
        stream_bytes = blob[
            payload_start + offset_table[k]:
            payload_start + offset_table[k + 1]
        ]
        sub = io.BytesIO(stream_bytes)

        se_blob_len, = struct.unpack("<I", sub.read(4))
        se_blob = sub.read(se_blob_len)
        se_slice = rans.decode_stream(
            se_blob, n_in_stream, se_freqs, se_slot_to_sym, M_SE,
        ).astype(np.uint16)

        exp_slice = (se_slice & 0xFF).astype(np.uint16)
        counts = np.bincount(exp_slice.astype(np.int64), minlength=256)
        bstart = np.zeros(257, dtype=np.int64)
        bstart[1:] = np.cumsum(counts)

        n_nz_exps_in_stream, = struct.unpack("<I", sub.read(4))
        stream_nz_exps = np.frombuffer(
            sub.read(n_nz_exps_in_stream * 2), dtype=np.uint16,
        )

        # Buckets are decoded in the same order they were encoded, then
        # un-sorted back to slice-order via the same argsort.
        order = np.argsort(exp_slice, kind="stable")
        mant_sorted = np.empty(n_in_stream, dtype=np.uint16)
        for ev in stream_nz_exps:
            count, blen = struct.unpack("<II", sub.read(8))
            mblob = sub.read(blen)
            bucket = rans.decode_stream(
                mblob, int(count),
                mant_freqs_by_exp[int(ev)],
                mant_slot_to_sym_by_exp[int(ev)],
                M_MANT,
            ).astype(np.uint16)
            mant_sorted[bstart[ev]:bstart[ev + 1]] = bucket

        mant_slice = np.empty(n_in_stream, dtype=np.uint16)
        mant_slice[order] = mant_sorted

        full_se[k::n_streams] = se_slice
        full_mant[k::n_streams] = mant_slice

    sign = ((full_se >> 8) & 1).astype(np.uint16)
    exp_full = (full_se & 0xFF).astype(np.uint16)
    out = ((sign << SIGN_SHIFT) | (exp_full << EXP_SHIFT) | full_mant).astype(np.uint16)
    return out.tobytes()
