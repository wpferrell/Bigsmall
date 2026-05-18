"""Minimal rANS (range Asymmetric Numeral Systems) coder.

Chosen because:
  1. State update is a single multiplicative step — GPU-friendly.
  2. Renormalisation is byte/u16-aligned — easy to port to CUDA / Triton.
  3. Decode is sequential within a stream (one block per stream on GPU),
     so parallelism comes from N independent streams, exactly matching
     bigsmall.codecs.bf16_parallel's design.

Parameters used throughout this module:
  - State is 32-bit unsigned, lower bound `L = 1 << 16`.
  - Renormalisation emits / consumes 16-bit chunks.
  - Frequency table total `M` is a power of two passed in by the caller.

The CPU encoder/decoder here is the reference. A CUDA C extension at
bigsmall/kernels/ac_cuda decodes the same bytestream on GPU.
"""
from __future__ import annotations

import io
import struct

import numpy as np


# rANS streaming constants — bytewise renorm of u16 chunks.
L = 1 << 16              # State lower bound after renorm
RENORM_BITS = 16
RENORM_MASK = (1 << RENORM_BITS) - 1
STATE_BITS = 32
STATE_MAX = 1 << STATE_BITS


def quantise_frequencies(counts: np.ndarray, M: int) -> np.ndarray:
    """Map raw symbol counts to integer frequencies summing to exactly M.

    Each nonzero count gets at least frequency 1 (so the symbol can be
    encoded). Frequencies are then proportionally scaled to sum to M.

    Args:
        counts: nonnegative int array of shape (alphabet,).
        M: target total (must be a power of two; M > number of nonzero counts).

    Returns:
        freqs: uint32 array of shape (alphabet,) summing to exactly M.
    """
    counts = counts.astype(np.int64)
    n_nz = int((counts > 0).sum())
    if n_nz == 0:
        # Pathological: tensor empty or all-one-symbol. Force a flat-on-zero
        # distribution so the codec doesn't divide by zero.
        freqs = np.zeros_like(counts, dtype=np.uint32)
        freqs[0] = M
        return freqs
    if n_nz > M:
        raise ValueError(
            f"quantise_frequencies: {n_nz} nonzero symbols exceed total M={M}; "
            "need a larger precision parameter."
        )

    total = int(counts.sum())
    # First pass: scale proportionally, floor.
    scaled = (counts * M) // total
    # Force each originally-nonzero count to have at least 1.
    scaled = np.where((counts > 0) & (scaled < 1), 1, scaled).astype(np.int64)
    # Force originally-zero counts to stay zero.
    scaled = np.where(counts == 0, 0, scaled)
    # Adjust to sum to exactly M.
    diff = M - int(scaled.sum())
    if diff > 0:
        # Add to the symbol with the largest current frequency (most stable).
        order = np.argsort(-scaled, kind="stable")
        i = 0
        while diff > 0:
            scaled[order[i % len(order)]] += 1
            diff -= 1
            i += 1
    elif diff < 0:
        # Remove from largest-freq symbols, never going below 1 for nonzero.
        order = np.argsort(-scaled, kind="stable")
        i = 0
        while diff < 0:
            j = order[i % len(order)]
            if scaled[j] > 1:
                scaled[j] -= 1
                diff += 1
            i += 1
            if i > 100 * len(order):
                # Defensive — should never happen with sane counts.
                raise RuntimeError("quantise_frequencies failed to converge")
    assert int(scaled.sum()) == M
    return scaled.astype(np.uint32)


def cumulative_from_freqs(freqs: np.ndarray) -> np.ndarray:
    """Exclusive prefix sum: cumfreqs[s] = sum(freqs[:s])."""
    cum = np.zeros(len(freqs) + 1, dtype=np.uint32)
    cum[1:] = np.cumsum(freqs.astype(np.uint32))
    return cum[:-1].copy()


def build_slot_to_symbol(freqs: np.ndarray, M: int) -> np.ndarray:
    """Build a uint16 table of shape (M,) mapping slot -> symbol index.

    slot_to_symbol[s] for s in [cumfreq, cumfreq+freq) returns symbol idx.
    """
    cum = cumulative_from_freqs(freqs)
    tbl = np.zeros(M, dtype=np.uint16)
    for sym, (cf, f) in enumerate(zip(cum, freqs)):
        if f > 0:
            tbl[cf:cf + f] = sym
    return tbl


def encode_stream(symbols: np.ndarray, freqs: np.ndarray, M: int) -> bytes:
    """Encode an int32 array of symbols into rANS bitstream bytes.

    Returns: bytes containing [4-byte final state][u16 ... u16] in order
    the decoder consumes them (state first, then renorm u16s).
    """
    if len(symbols) == 0:
        # No data: emit only an initial state so decoder can still parse.
        return struct.pack("<I", L)

    M_int = int(M)
    cumfreqs = cumulative_from_freqs(freqs).astype(np.int64)
    freqs64 = freqs.astype(np.int64)

    state = L
    renorm_buf: list[int] = []   # u16s, append-order = reverse decode order
    # Need to encode in reverse: rANS decode pops the FIRST encoded symbol last
    syms_int = symbols.astype(np.int64)
    # The condition "state >= (state_max / M) * f_s" is the renorm gate.
    # state_max = 2**32 ; with M power of two, state_max/M = 2**(32 - log2(M)).
    # For 16-bit renorm and M = 2^precision: renorm bound = freq << (32 - precision - 16) = freq << (16 - precision).
    # Equivalent: threshold[s] = (freqs[s] << (32 - precision)) >> 16  -- but easier just to compare directly.
    state_max = STATE_MAX
    for i in range(len(syms_int) - 1, -1, -1):
        s = int(syms_int[i])
        f = int(freqs64[s])
        cf = int(cumfreqs[s])
        # Renorm: emit lower 16 bits while state would overflow on next encode.
        # Bound: state < (state_max // M) * f, i.e. (state * M) < state_max * f
        # Rearrange: state < (state_max / M) * f
        upper = (state_max // M_int) * f
        while state >= upper:
            renorm_buf.append(state & RENORM_MASK)
            state >>= RENORM_BITS
        # Encode: state' = (state // f) * M + cf + (state % f)
        state = (state // f) * M_int + cf + (state % f)
        if state >= state_max:
            raise RuntimeError(
                f"rANS state overflow: state={state}, M={M_int}, f={f}"
            )

    # Assemble: final state first (decoder reads it first), then u16s in REVERSE
    # of renorm_buf (because rANS decode pops them in the opposite order).
    out = io.BytesIO()
    out.write(struct.pack("<I", state))
    for u in reversed(renorm_buf):
        out.write(struct.pack("<H", u))
    return out.getvalue()


def decode_stream(
    data: bytes,
    n_symbols: int,
    freqs: np.ndarray,
    slot_to_symbol: np.ndarray,
    M: int,
) -> np.ndarray:
    """Inverse of encode_stream. Returns uint32 array of decoded symbols."""
    if n_symbols == 0:
        return np.zeros(0, dtype=np.uint32)

    if len(data) < 4:
        raise ValueError("rANS decode_stream: blob too short for final state")

    state = struct.unpack("<I", data[:4])[0]
    pos = 4
    n_data = len(data)
    M_int = int(M)
    freqs64 = freqs.astype(np.int64)
    cumfreqs = cumulative_from_freqs(freqs).astype(np.int64)

    out = np.empty(n_symbols, dtype=np.uint32)
    for i in range(n_symbols):
        slot = state & (M_int - 1)  # M is power of two
        s = int(slot_to_symbol[slot])
        out[i] = s
        f = int(freqs64[s])
        cf = int(cumfreqs[s])
        # state = f * (state // M) + slot - cf
        state = f * (state >> _log2_pow2(M_int)) + slot - cf
        # Renorm: while state below lower bound, pull in 16 bits
        while state < L and pos + 1 < n_data + 1:
            if pos + 2 > n_data:
                # Pad with zeros: decoder ran past available bytes (legitimate
                # when the encoder finished without needing all renorm slots).
                state = state << RENORM_BITS
                pos = n_data
                break
            u = struct.unpack("<H", data[pos:pos + 2])[0]
            pos += 2
            state = (state << RENORM_BITS) | u

    return out


def _log2_pow2(M: int) -> int:
    """Return log2(M); raises if M is not a power of two."""
    if M <= 0 or (M & (M - 1)) != 0:
        raise ValueError(f"M must be a power of two, got {M}")
    return M.bit_length() - 1
