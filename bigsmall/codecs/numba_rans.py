"""Numba-JIT-compiled rANS encoder and decoder.

Replaces constriction's `AnsCoder` for hot-path encoding/decoding. The
Python↔Rust FFI boundary was the dominant cost on bf16's per-exp-bucket
coding (v3.4.0 measured 1.04x decode vs AC). Numba JIT eliminates that
boundary — the per-symbol loop compiles to native code that calls back
to Python only once per coder invocation.

This module is the substrate for `bigsmall.codecs.bf16_tans` (v3.5.0).
The bitstream is NOT compatible with constriction's `AnsCoder` output;
this is a self-contained format.

State machine constants:
  - 32-bit state, lower bound L = 1 << 16
  - 16-bit renorm chunks
  - Precision: caller picks log2(M), typically 12 (M=4096) for SE and
    10 (M=1024) for the mantissa alphabet.
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
    # Decorator stub so the module still imports without numba
    def njit(*args, **kwargs):  # type: ignore
        def wrap(fn):
            return fn
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return wrap


L_BOUND = 1 << 16
RENORM_BITS = 16
STATE_BITS = 32


def quantise_frequencies(counts: np.ndarray, M: int) -> np.ndarray:
    """Map raw symbol counts to integer frequencies summing to exactly M.

    Each originally-nonzero symbol gets at least frequency 1.
    """
    counts = counts.astype(np.int64)
    n_nz = int((counts > 0).sum())
    if n_nz == 0:
        freqs = np.zeros_like(counts, dtype=np.uint32)
        freqs[0] = M
        return freqs
    if n_nz > M:
        raise ValueError(
            f"quantise_frequencies: {n_nz} nonzero symbols exceed M={M}"
        )
    total = int(counts.sum())
    scaled = (counts * M) // total
    scaled = np.where((counts > 0) & (scaled < 1), 1, scaled).astype(np.int64)
    scaled = np.where(counts == 0, 0, scaled)
    diff = M - int(scaled.sum())
    if diff > 0:
        order = np.argsort(-scaled, kind="stable")
        i = 0
        while diff > 0:
            scaled[order[i % len(order)]] += 1
            diff -= 1
            i += 1
    elif diff < 0:
        order = np.argsort(-scaled, kind="stable")
        i = 0
        guard = 0
        while diff < 0:
            j = order[i % len(order)]
            if scaled[j] > 1:
                scaled[j] -= 1
                diff += 1
            i += 1
            guard += 1
            if guard > 100 * len(order):
                raise RuntimeError("quantise_frequencies failed to converge")
    return scaled.astype(np.uint32)


def cumulative(freqs: np.ndarray) -> np.ndarray:
    """Exclusive prefix sum, full-alphabet."""
    cum = np.zeros(len(freqs), dtype=np.uint32)
    cum[1:] = np.cumsum(freqs.astype(np.uint32))[:-1]
    return cum


def slot_to_symbol_table(freqs: np.ndarray, M: int) -> np.ndarray:
    """uint32 table of shape (M,) mapping slot -> symbol index."""
    cum = cumulative(freqs)
    tbl = np.zeros(M, dtype=np.uint32)
    for sym, (cf, f) in enumerate(zip(cum, freqs)):
        if f > 0:
            tbl[cf:cf + f] = sym
    return tbl


# --- Numba kernels --------------------------------------------------------


@njit(cache=True, boundscheck=False)
def _encode_kernel(symbols: np.ndarray,
                   freqs_u32: np.ndarray,
                   cumfreqs_u32: np.ndarray,
                   M_int: np.int64) -> Tuple[np.ndarray, np.uint32, np.int64]:
    """Encode a symbol sequence with rANS. Returns (renorm_u16s_reversed, final_state, count_emitted).

    Pre-allocates a scratch buffer of size 4 * len(symbols) + 8 uint16s,
    which is more than enough for normal distributions (renorm emits ≤ 1
    uint16 per encoded symbol in the streaming variant).
    """
    n = symbols.shape[0]
    state_max = np.uint64(1) << np.uint64(32)
    M = np.uint64(M_int)
    L_b_mul = state_max // M
    state = np.uint64(L_BOUND)

    cap = n * 4 + 64
    buf = np.empty(cap, dtype=np.uint16)
    out_count = np.int64(0)

    for i in range(n - 1, -1, -1):
        s = symbols[i]
        f = np.uint64(freqs_u32[s])
        cf = np.uint64(cumfreqs_u32[s])
        # state must stay strictly below state_max after encode.
        # Renorm condition: while state >= (state_max / M) * f, emit u16.
        upper = L_b_mul * f
        while state >= upper:
            buf[out_count] = np.uint16(state & np.uint64(0xFFFF))
            out_count += 1
            state >>= np.uint64(16)
        # Encode: state = (state / f) * M + cumfreq + (state % f)
        state = (state // f) * M + cf + (state % f)

    return buf[:out_count], np.uint32(state), out_count


def encode_stream(symbols: np.ndarray, freqs: np.ndarray, M: int) -> bytes:
    """Encode a sequence of symbols. Returns bytes ready for decode."""
    n = len(symbols)
    if n == 0:
        return struct.pack("<I", L_BOUND)
    freqs_u32 = freqs.astype(np.uint32)
    cumfreqs_u32 = cumulative(freqs_u32)
    sym_arr = np.ascontiguousarray(symbols, dtype=np.int32)
    buf_used, state, count = _encode_kernel(sym_arr, freqs_u32, cumfreqs_u32, np.int64(M))
    # Final state goes first (decoder reads it first), then renorm u16s in
    # REVERSE order (encoder emitted them latest-first because rANS is LIFO).
    out = bytearray()
    out += struct.pack("<I", int(state))
    if count > 0:
        # buf_used[0] is the LAST emitted u16 (most recent), so we need to
        # output it LAST. Reverse the buf_used array first.
        rev = buf_used[::-1]
        out += rev.tobytes()
    return bytes(out)


@njit(cache=True, boundscheck=False)
def _decode_kernel(data_u16: np.ndarray,
                   n: np.int64,
                   freqs_u32: np.ndarray,
                   cumfreqs_u32: np.ndarray,
                   slot_to_sym_u32: np.ndarray,
                   precision: np.int64) -> np.ndarray:
    """Decode n symbols from a u16-encoded bytestream. Returns int32 array."""
    state = np.uint32(data_u16[0]) | (np.uint32(data_u16[1]) << np.uint32(16))
    pos = np.int64(2)
    n_data = data_u16.shape[0]
    M_mask = np.uint32((np.int64(1) << precision) - np.int64(1))
    prec = np.uint32(precision)
    L = np.uint32(L_BOUND)
    out = np.empty(n, dtype=np.int32)
    for i in range(n):
        slot = state & M_mask
        s = slot_to_sym_u32[slot]
        out[i] = s
        f = freqs_u32[s]
        cf = cumfreqs_u32[s]
        state = f * (state >> prec) + slot - cf
        while state < L and pos < n_data:
            state = (state << np.uint32(16)) | np.uint32(data_u16[pos])
            pos += np.int64(1)
    return out


def decode_stream(data: bytes,
                  n_symbols: int,
                  freqs: np.ndarray,
                  slot_to_sym: np.ndarray,
                  precision: int) -> np.ndarray:
    """Decode `n_symbols` symbols from bytes. Returns int32 array."""
    if n_symbols == 0:
        return np.zeros(0, dtype=np.int32)
    if len(data) < 4:
        raise ValueError("decode_stream: blob too short for final state")
    # Reinterpret bytes as u16 array — we need at least the 4-byte state
    # padded to an even length. The 4-byte final state is read as 2 u16s.
    # If the renorm payload has an odd byte count, pad. The encoder
    # guarantees even byte counts because it emits in u16 chunks.
    if len(data) % 2 != 0:
        data = data + b"\x00"
    data_u16 = np.frombuffer(data, dtype=np.uint16)
    freqs_u32 = freqs.astype(np.uint32)
    cumfreqs_u32 = cumulative(freqs_u32)
    sts = slot_to_sym.astype(np.uint32)
    out = _decode_kernel(data_u16, np.int64(n_symbols), freqs_u32,
                         cumfreqs_u32, sts, np.int64(precision))
    return out


def numba_available() -> bool:
    return _NUMBA_OK
