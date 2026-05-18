"""Single-kernel BF16 codec — `bf16_se_single_kernel` (v3.6.0).

One Numba `@njit` function handles the entire per-tensor encode, including:
  - sign/exp/mantissa extraction from u16
  - SE frequency table build + quantisation
  - SE rANS encode
  - O(n) counting-sort into per-exp mantissa buckets (replaces numpy argsort)
  - Per-bucket mantissa frequency tables + quantisation
  - Per-bucket mantissa rANS encode
  - Output blob assembly

Zero Python orchestration between phases. The whole tensor goes in, the
final compressed blob comes out.

Speedup vs bf16_se_tans (v3.5.0): the argsort cost (~32% of v3.5.0 encode
time) is eliminated via counting sort, and per-bucket FFI calls are
eliminated by inlining. Lossless md5-verified.
"""
from __future__ import annotations

import struct
from typing import Tuple

import numpy as np

try:
    from numba import njit
    _NUMBA_OK = True
except ImportError:
    _NUMBA_OK = False
    def njit(*args, **kwargs):  # type: ignore
        def wrap(fn):
            return fn
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return wrap

from .bf16 import SIGN_SHIFT, EXP_SHIFT, EXP_MASK, MANT_MASK


SE_ALPHABET = 512
MANT_ALPHABET = 128
PRECISION_SE = 12       # M_SE = 4096
PRECISION_M = 10        # M_MANT = 1024
M_SE = 1 << PRECISION_SE
M_MANT = 1 << PRECISION_M
L_BOUND = 1 << 16


# ----------------------------- Numba kernels -----------------------------


@njit(cache=True, boundscheck=False)
def _quantise_inplace(counts: np.ndarray, M: np.int64, out: np.ndarray) -> None:
    """Quantise raw counts to integer frequencies summing exactly to M.

    Writes into `out` (same shape as `counts`, must be uint32).
    Handles the n_nz==0 edge case by putting all mass on symbol 0.
    """
    alphabet = counts.shape[0]
    n_nz = 0
    total = np.int64(0)
    for i in range(alphabet):
        if counts[i] > 0:
            n_nz += 1
            total += counts[i]
    if n_nz == 0:
        for i in range(alphabet):
            out[i] = 0
        out[0] = np.uint32(M)
        return
    # First pass: floor-scale
    s = np.int64(0)
    for i in range(alphabet):
        if counts[i] > 0:
            v = (counts[i] * M) // total
            if v < 1:
                v = 1
            out[i] = np.uint32(v)
            s += v
        else:
            out[i] = np.uint32(0)
    # Fix the sum to exactly M by adjusting the largest-freq entries
    diff = np.int64(M) - s
    if diff > 0:
        # Add diff to the symbol(s) with the highest freq
        while diff > 0:
            best = 0
            best_v = np.uint32(0)
            for i in range(alphabet):
                if out[i] > best_v:
                    best_v = out[i]
                    best = i
            out[best] += np.uint32(1)
            diff -= 1
    elif diff < 0:
        guard = 0
        while diff < 0 and guard < 50 * alphabet:
            best = 0
            best_v = np.uint32(0)
            for i in range(alphabet):
                if out[i] > best_v:
                    best_v = out[i]
                    best = i
            if out[best] > 1:
                out[best] -= np.uint32(1)
                diff += 1
            else:
                break
            guard += 1


@njit(cache=True, boundscheck=False)
def _cumulative(freqs: np.ndarray, out: np.ndarray) -> None:
    """Exclusive prefix sum into `out`."""
    out[0] = np.uint32(0)
    for i in range(1, freqs.shape[0]):
        out[i] = out[i - 1] + freqs[i - 1]


@njit(cache=True, boundscheck=False)
def _rans_encode_one(symbols: np.ndarray,
                     freqs: np.ndarray,
                     cumfreqs: np.ndarray,
                     M_int: np.int64,
                     out_buf: np.ndarray,
                     out_start: np.int64) -> Tuple[np.int64, np.uint32]:
    """Encode `symbols` into `out_buf[out_start:]` as a sequence of u16 chunks.

    rANS is LIFO: walk symbols in reverse, push onto state, emit u16 when
    state too large. The emitted u16s land in `out_buf` in REVERSE of their
    eventual decode order — caller reverses the slice afterwards.

    Returns (count_of_u16s_written, final_state).
    The final 4-byte state goes in `out_buf` AFTER the reversed u16s.
    """
    n = symbols.shape[0]
    if n == 0:
        return np.int64(0), np.uint32(L_BOUND)
    state_max = np.uint64(1) << np.uint64(32)
    M = np.uint64(M_int)
    L_b_mul = state_max // M
    state = np.uint64(L_BOUND)
    count = np.int64(0)

    for i in range(n - 1, -1, -1):
        s = symbols[i]
        f = np.uint64(freqs[s])
        cf = np.uint64(cumfreqs[s])
        upper = L_b_mul * f
        while state >= upper:
            out_buf[out_start + count] = np.uint16(state & np.uint64(0xFFFF))
            count += 1
            state >>= np.uint64(16)
        state = (state // f) * M + cf + (state % f)
    return count, np.uint32(state)


@njit(cache=True, boundscheck=False)
def _rans_decode_one(data_u16: np.ndarray,
                     in_start: np.int64,
                     in_count: np.int64,
                     n: np.int64,
                     freqs: np.ndarray,
                     cumfreqs: np.ndarray,
                     slot_to_sym: np.ndarray,
                     precision: np.int64,
                     out: np.ndarray,
                     out_start: np.int64) -> None:
    """Inverse of _rans_encode_one. Writes decoded symbols into out[out_start:]."""
    if n == 0:
        return
    # First 2 u16s = state, then renorm u16s follow (in forward order).
    state = (np.uint32(data_u16[in_start])
             | (np.uint32(data_u16[in_start + 1]) << np.uint32(16)))
    pos = in_start + np.int64(2)
    end = in_start + in_count
    M_mask = np.uint32((np.int64(1) << precision) - np.int64(1))
    prec = np.uint32(precision)
    L = np.uint32(L_BOUND)
    for i in range(n):
        slot = state & M_mask
        s = slot_to_sym[slot]
        out[out_start + i] = s
        f = freqs[s]
        cf = cumfreqs[s]
        state = f * (state >> prec) + slot - cf
        while state < L and pos < end:
            state = (state << np.uint32(16)) | np.uint32(data_u16[pos])
            pos += np.int64(1)


@njit(cache=True, boundscheck=False)
def _encode_full_kernel(u16: np.ndarray,
                        M_se: np.int64, M_mant: np.int64,
                        out_buf: np.ndarray) -> np.int64:
    """Encode entire BF16 tensor into `out_buf`. Returns bytes written.

    `out_buf` must be a uint8 buffer pre-sized to worst-case (input + ~16 KB).
    """
    n = u16.shape[0]
    if n == 0:
        return np.int64(0)

    # Step 1: split sign / exp / mantissa, build SE values and exp counts.
    se = np.empty(n, dtype=np.int32)
    exp = np.empty(n, dtype=np.uint8)
    mant = np.empty(n, dtype=np.uint8)
    se_counts = np.zeros(SE_ALPHABET, dtype=np.int64)
    exp_counts = np.zeros(256, dtype=np.int64)
    for i in range(n):
        w = u16[i]
        sgn = (w >> 15) & 1
        e = (w >> 7) & 0xFF
        m = w & 0x7F
        se_v = (sgn << 8) | e
        se[i] = se_v
        exp[i] = e
        mant[i] = m
        se_counts[se_v] += 1
        exp_counts[e] += 1

    # Step 2: SE quantised freqs + cumfreqs.
    se_freqs = np.zeros(SE_ALPHABET, dtype=np.uint32)
    _quantise_inplace(se_counts, M_se, se_freqs)
    se_cumfreqs = np.zeros(SE_ALPHABET, dtype=np.uint32)
    _cumulative(se_freqs, se_cumfreqs)

    # Step 3: O(n) counting-sort mantissas by exp.
    bstart = np.zeros(257, dtype=np.int64)
    bstart[0] = 0
    for i in range(256):
        bstart[i + 1] = bstart[i] + exp_counts[i]
    mant_sorted = np.empty(n, dtype=np.uint8)
    cursors = np.empty(256, dtype=np.int64)
    for i in range(256):
        cursors[i] = bstart[i]
    for i in range(n):
        e = exp[i]
        mant_sorted[cursors[e]] = mant[i]
        cursors[e] += 1

    # Step 4: per-bucket mantissa freqs (only for nonzero exps).
    mant_freqs_all = np.zeros((256, MANT_ALPHABET), dtype=np.uint32)
    mant_cumfreqs_all = np.zeros((256, MANT_ALPHABET), dtype=np.uint32)
    n_nonzero_exps = 0
    nonzero_exps = np.empty(256, dtype=np.int32)
    for e in range(256):
        if exp_counts[e] > 0:
            nonzero_exps[n_nonzero_exps] = e
            n_nonzero_exps += 1
            # Build mantissa counts for this bucket
            bc = np.zeros(MANT_ALPHABET, dtype=np.int64)
            bs_ = bstart[e]
            be_ = bstart[e + 1]
            for i in range(bs_, be_):
                bc[mant_sorted[i]] += 1
            _quantise_inplace(bc, M_mant, mant_freqs_all[e])
            _cumulative(mant_freqs_all[e], mant_cumfreqs_all[e])

    # Step 5: encode SE rANS stream into a scratch u16 buffer.
    scratch_u16 = np.empty(n * 4 + 64, dtype=np.uint16)
    se_int32 = se  # already int32
    n_se_chunks, se_state = _rans_encode_one(
        se_int32, se_freqs, se_cumfreqs, M_se, scratch_u16, np.int64(0),
    )
    # Reverse scratch_u16[:n_se_chunks] in place
    for i in range(n_se_chunks // 2):
        a = scratch_u16[i]
        b = scratch_u16[n_se_chunks - 1 - i]
        scratch_u16[i] = b
        scratch_u16[n_se_chunks - 1 - i] = a

    # Encode mantissa streams per bucket into scratch_u16 contiguously
    scratch_pos = n_se_chunks
    bucket_n_chunks = np.zeros(256, dtype=np.int64)
    bucket_state = np.zeros(256, dtype=np.uint32)
    bucket_chunk_start = np.zeros(256, dtype=np.int64)
    for b in range(n_nonzero_exps):
        e = nonzero_exps[b]
        bs_ = bstart[e]
        be_ = bstart[e + 1]
        bucket_size = be_ - bs_
        # Convert mant_sorted[bs_:be_] to int32 for the rANS encoder
        sym_int32 = np.empty(bucket_size, dtype=np.int32)
        for i in range(bucket_size):
            sym_int32[i] = mant_sorted[bs_ + i]
        chunks, st = _rans_encode_one(
            sym_int32, mant_freqs_all[e], mant_cumfreqs_all[e],
            M_mant, scratch_u16, scratch_pos,
        )
        # Reverse this bucket's chunks
        for i in range(chunks // 2):
            a = scratch_u16[scratch_pos + i]
            bb = scratch_u16[scratch_pos + chunks - 1 - i]
            scratch_u16[scratch_pos + i] = bb
            scratch_u16[scratch_pos + chunks - 1 - i] = a
        bucket_n_chunks[e] = chunks
        bucket_state[e] = st
        bucket_chunk_start[e] = scratch_pos
        scratch_pos += chunks

    # Step 6: write final blob layout into out_buf (uint8).
    # Layout:
    #   [4] n
    #   [2] n_se_nz
    #   [n_se_nz * 2] se_nz_idx (uint16)
    #   [n_se_nz * 4] se_freqs_nz (uint32)
    #   [4] se_state (uint32)
    #   [4] se_chunks_count
    #   [n_se_chunks * 2] se renorm u16s
    #   [4] n_nz_exps
    #   For each nonzero exp:
    #     [2] exp value (uint16)
    #     [1] mant_n_nz (uint8)
    #     [mant_n_nz] nz indices (uint8)
    #     [mant_n_nz * 4] nz freqs (uint32)
    #     [4] mant state (uint32)
    #     [4] bucket_size (uint32)
    #     [4] mant_n_chunks
    #     [mant_n_chunks * 2] mant renorm u16s
    pos = np.int64(0)
    # n
    out_buf[pos] = n & 0xFF
    out_buf[pos + 1] = (n >> 8) & 0xFF
    out_buf[pos + 2] = (n >> 16) & 0xFF
    out_buf[pos + 3] = (n >> 24) & 0xFF
    pos += 4

    # SE freqs nonzero indices + freqs
    n_se_nz = 0
    for i in range(SE_ALPHABET):
        if se_freqs[i] > 0:
            n_se_nz += 1
    out_buf[pos] = n_se_nz & 0xFF
    out_buf[pos + 1] = (n_se_nz >> 8) & 0xFF
    pos += 2
    nz_pos = pos
    for i in range(SE_ALPHABET):
        if se_freqs[i] > 0:
            out_buf[nz_pos] = i & 0xFF
            out_buf[nz_pos + 1] = (i >> 8) & 0xFF
            nz_pos += 2
    pos = nz_pos
    for i in range(SE_ALPHABET):
        if se_freqs[i] > 0:
            v = se_freqs[i]
            out_buf[pos] = np.uint8(v & 0xFF)
            out_buf[pos + 1] = np.uint8((v >> 8) & 0xFF)
            out_buf[pos + 2] = np.uint8((v >> 16) & 0xFF)
            out_buf[pos + 3] = np.uint8((v >> 24) & 0xFF)
            pos += 4

    # SE state + chunk count + chunks
    st = se_state
    out_buf[pos] = np.uint8(st & 0xFF)
    out_buf[pos + 1] = np.uint8((st >> 8) & 0xFF)
    out_buf[pos + 2] = np.uint8((st >> 16) & 0xFF)
    out_buf[pos + 3] = np.uint8((st >> 24) & 0xFF)
    pos += 4
    cc = n_se_chunks
    out_buf[pos] = np.uint8(cc & 0xFF)
    out_buf[pos + 1] = np.uint8((cc >> 8) & 0xFF)
    out_buf[pos + 2] = np.uint8((cc >> 16) & 0xFF)
    out_buf[pos + 3] = np.uint8((cc >> 24) & 0xFF)
    pos += 4
    for i in range(n_se_chunks):
        v = scratch_u16[i]
        out_buf[pos] = np.uint8(v & 0xFF)
        out_buf[pos + 1] = np.uint8((v >> 8) & 0xFF)
        pos += 2

    # Mantissa: n_nz_exps + per-exp records
    out_buf[pos] = np.uint8(n_nonzero_exps & 0xFF)
    out_buf[pos + 1] = np.uint8((n_nonzero_exps >> 8) & 0xFF)
    out_buf[pos + 2] = np.uint8((n_nonzero_exps >> 16) & 0xFF)
    out_buf[pos + 3] = np.uint8((n_nonzero_exps >> 24) & 0xFF)
    pos += 4
    for b in range(n_nonzero_exps):
        e = nonzero_exps[b]
        out_buf[pos] = np.uint8(e & 0xFF)
        out_buf[pos + 1] = np.uint8((e >> 8) & 0xFF)
        pos += 2
        # mant_n_nz + nz indices + freqs
        mnz = 0
        for i in range(MANT_ALPHABET):
            if mant_freqs_all[e, i] > 0:
                mnz += 1
        out_buf[pos] = np.uint8(mnz)
        pos += 1
        for i in range(MANT_ALPHABET):
            if mant_freqs_all[e, i] > 0:
                out_buf[pos] = np.uint8(i)
                pos += 1
        for i in range(MANT_ALPHABET):
            if mant_freqs_all[e, i] > 0:
                v = mant_freqs_all[e, i]
                out_buf[pos] = np.uint8(v & 0xFF)
                out_buf[pos + 1] = np.uint8((v >> 8) & 0xFF)
                out_buf[pos + 2] = np.uint8((v >> 16) & 0xFF)
                out_buf[pos + 3] = np.uint8((v >> 24) & 0xFF)
                pos += 4
        # state + bucket_size + n_chunks + chunks
        st = bucket_state[e]
        out_buf[pos] = np.uint8(st & 0xFF)
        out_buf[pos + 1] = np.uint8((st >> 8) & 0xFF)
        out_buf[pos + 2] = np.uint8((st >> 16) & 0xFF)
        out_buf[pos + 3] = np.uint8((st >> 24) & 0xFF)
        pos += 4
        bsize = bstart[e + 1] - bstart[e]
        out_buf[pos] = np.uint8(bsize & 0xFF)
        out_buf[pos + 1] = np.uint8((bsize >> 8) & 0xFF)
        out_buf[pos + 2] = np.uint8((bsize >> 16) & 0xFF)
        out_buf[pos + 3] = np.uint8((bsize >> 24) & 0xFF)
        pos += 4
        nc = bucket_n_chunks[e]
        out_buf[pos] = np.uint8(nc & 0xFF)
        out_buf[pos + 1] = np.uint8((nc >> 8) & 0xFF)
        out_buf[pos + 2] = np.uint8((nc >> 16) & 0xFF)
        out_buf[pos + 3] = np.uint8((nc >> 24) & 0xFF)
        pos += 4
        bcs = bucket_chunk_start[e]
        for i in range(nc):
            v = scratch_u16[bcs + i]
            out_buf[pos] = np.uint8(v & 0xFF)
            out_buf[pos + 1] = np.uint8((v >> 8) & 0xFF)
            pos += 2
    return pos


@njit(cache=True, boundscheck=False)
def _decode_full_kernel(in_buf: np.ndarray,
                        n_weights: np.int64,
                        M_se: np.int64, M_mant: np.int64,
                        out_u16: np.ndarray) -> None:
    """Inverse of _encode_full_kernel.

    Reads the blob from `in_buf` (uint8 array), writes BF16 u16 values to
    `out_u16` (uint16 array of length n_weights).
    """
    pos = np.int64(0)
    n = (in_buf[pos] | (in_buf[pos + 1] << 8)
         | (in_buf[pos + 2] << 16) | (in_buf[pos + 3] << 24))
    pos += 4
    if n != n_weights:
        return  # caller should validate

    # SE freqs
    n_se_nz = in_buf[pos] | (in_buf[pos + 1] << 8)
    pos += 2
    se_freqs = np.zeros(SE_ALPHABET, dtype=np.uint32)
    nz_idx = np.empty(n_se_nz, dtype=np.int32)
    for i in range(n_se_nz):
        v = in_buf[pos] | (in_buf[pos + 1] << 8)
        nz_idx[i] = v
        pos += 2
    for i in range(n_se_nz):
        v = (np.uint32(in_buf[pos])
             | (np.uint32(in_buf[pos + 1]) << 8)
             | (np.uint32(in_buf[pos + 2]) << 16)
             | (np.uint32(in_buf[pos + 3]) << 24))
        se_freqs[nz_idx[i]] = v
        pos += 4
    se_cumfreqs = np.zeros(SE_ALPHABET, dtype=np.uint32)
    _cumulative(se_freqs, se_cumfreqs)
    # Slot table for SE
    se_slot_to_sym = np.empty(M_se, dtype=np.uint32)
    for s in range(SE_ALPHABET):
        f = se_freqs[s]
        if f > 0:
            cf = se_cumfreqs[s]
            for j in range(f):
                se_slot_to_sym[cf + j] = s

    # SE state + chunks
    se_state_low = (np.uint32(in_buf[pos]) | (np.uint32(in_buf[pos + 1]) << 8)
                    | (np.uint32(in_buf[pos + 2]) << 16)
                    | (np.uint32(in_buf[pos + 3]) << 24))
    pos += 4
    se_chunks_count = (in_buf[pos] | (in_buf[pos + 1] << 8)
                       | (in_buf[pos + 2] << 16) | (in_buf[pos + 3] << 24))
    pos += 4
    # SE u16 array: state (2 u16s) + chunks
    se_data = np.empty(se_chunks_count + 2, dtype=np.uint16)
    se_data[0] = np.uint16(se_state_low & 0xFFFF)
    se_data[1] = np.uint16((se_state_low >> 16) & 0xFFFF)
    for i in range(se_chunks_count):
        se_data[2 + i] = np.uint16(in_buf[pos] | (in_buf[pos + 1] << 8))
        pos += 2

    # Decode SE rANS stream
    se_decoded = np.empty(n, dtype=np.int32)
    _rans_decode_one(se_data, np.int64(0),
                     np.int64(se_chunks_count + 2),
                     n, se_freqs, se_cumfreqs, se_slot_to_sym,
                     np.int64(PRECISION_SE), se_decoded, np.int64(0))

    # Extract sign + exp from decoded SE; compute bucket boundaries
    exp_arr = np.empty(n, dtype=np.uint8)
    sign_arr = np.empty(n, dtype=np.uint8)
    exp_counts = np.zeros(256, dtype=np.int64)
    for i in range(n):
        se_v = se_decoded[i]
        sign_arr[i] = (se_v >> 8) & 1
        exp_arr[i] = se_v & 0xFF
        exp_counts[exp_arr[i]] += 1
    bstart = np.zeros(257, dtype=np.int64)
    for i in range(256):
        bstart[i + 1] = bstart[i] + exp_counts[i]

    # Read n_nz_exps
    n_nz_exps = (in_buf[pos] | (in_buf[pos + 1] << 8)
                 | (in_buf[pos + 2] << 16) | (in_buf[pos + 3] << 24))
    pos += 4

    mant_sorted = np.empty(n, dtype=np.uint8)
    for b in range(n_nz_exps):
        e = in_buf[pos] | (in_buf[pos + 1] << 8)
        pos += 2
        mnz = in_buf[pos]
        pos += 1
        nz = np.empty(mnz, dtype=np.int32)
        for i in range(mnz):
            nz[i] = in_buf[pos]
            pos += 1
        mfreqs = np.zeros(MANT_ALPHABET, dtype=np.uint32)
        for i in range(mnz):
            v = (np.uint32(in_buf[pos])
                 | (np.uint32(in_buf[pos + 1]) << 8)
                 | (np.uint32(in_buf[pos + 2]) << 16)
                 | (np.uint32(in_buf[pos + 3]) << 24))
            mfreqs[nz[i]] = v
            pos += 4
        mcumfreqs = np.zeros(MANT_ALPHABET, dtype=np.uint32)
        _cumulative(mfreqs, mcumfreqs)
        slot_to_sym = np.empty(M_mant, dtype=np.uint32)
        for s in range(MANT_ALPHABET):
            f = mfreqs[s]
            if f > 0:
                cf = mcumfreqs[s]
                for j in range(f):
                    slot_to_sym[cf + j] = s
        # mant state + bucket_size + n_chunks
        mstate = (np.uint32(in_buf[pos])
                  | (np.uint32(in_buf[pos + 1]) << 8)
                  | (np.uint32(in_buf[pos + 2]) << 16)
                  | (np.uint32(in_buf[pos + 3]) << 24))
        pos += 4
        bsize = (in_buf[pos] | (in_buf[pos + 1] << 8)
                 | (in_buf[pos + 2] << 16) | (in_buf[pos + 3] << 24))
        pos += 4
        nc = (in_buf[pos] | (in_buf[pos + 1] << 8)
              | (in_buf[pos + 2] << 16) | (in_buf[pos + 3] << 24))
        pos += 4
        mdata = np.empty(nc + 2, dtype=np.uint16)
        mdata[0] = np.uint16(mstate & 0xFFFF)
        mdata[1] = np.uint16((mstate >> 16) & 0xFFFF)
        for i in range(nc):
            mdata[2 + i] = np.uint16(in_buf[pos] | (in_buf[pos + 1] << 8))
            pos += 2
        # Decode into mant_sorted[bstart[e]:bstart[e]+bsize]
        bucket_out_i32 = np.empty(bsize, dtype=np.int32)
        _rans_decode_one(mdata, np.int64(0), np.int64(nc + 2),
                         np.int64(bsize), mfreqs, mcumfreqs, slot_to_sym,
                         np.int64(PRECISION_M), bucket_out_i32, np.int64(0))
        bs_ = bstart[e]
        for i in range(bsize):
            mant_sorted[bs_ + i] = np.uint8(bucket_out_i32[i])

    # Un-sort mantissas back to original order via cursor walk
    cursors = np.empty(256, dtype=np.int64)
    for i in range(256):
        cursors[i] = bstart[i]
    for i in range(n):
        e = exp_arr[i]
        m = mant_sorted[cursors[e]]
        cursors[e] += 1
        out_u16[i] = np.uint16((np.uint16(sign_arr[i]) << 15)
                                | (np.uint16(e) << 7)
                                | np.uint16(m))


# ----------------------------- Public API --------------------------------


def encode(raw: bytes) -> Tuple[bytes, dict]:
    """Encode a BF16 tensor via the single-kernel codec. Lossless."""
    if len(raw) % 2 != 0:
        raise ValueError(f"BF16 byte length must be even, got {len(raw)}")
    u16 = np.frombuffer(raw, dtype=np.uint16)
    n = len(u16)
    if n == 0:
        return b"", {}
    # Worst-case buffer: input size + headroom for headers and rANS framing.
    cap = n * 4 + 64 * 1024
    out_buf = np.empty(cap, dtype=np.uint8)
    n_bytes = _encode_full_kernel(np.ascontiguousarray(u16),
                                  np.int64(M_SE), np.int64(M_MANT),
                                  out_buf)
    return bytes(out_buf[:int(n_bytes)]), {}


def decode(blob: bytes, extras: dict, n_weights: int) -> bytes:
    """Decode a single-kernel BF16 blob. Lossless."""
    if n_weights == 0 or len(blob) == 0:
        return b""
    in_buf = np.frombuffer(blob, dtype=np.uint8)
    out_u16 = np.empty(n_weights, dtype=np.uint16)
    _decode_full_kernel(np.ascontiguousarray(in_buf),
                        np.int64(n_weights),
                        np.int64(M_SE), np.int64(M_MANT),
                        out_u16)
    return out_u16.tobytes()


def numba_available() -> bool:
    return _NUMBA_OK
