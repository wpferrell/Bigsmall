"""Tensor-level routing: dynamically detect special structure and pick a codec.

Special patterns detected (architecture-agnostic):

  - lowcard:  tensor with <= MAX_LOWCARD_UNIQUE distinct raw-byte values.
              Detected by counting unique uint{8,16,32} values. Captures
              attn.bias-style lower-triangular masks WITHOUT requiring the
              GPT-2 specific name or 1024x1024 shape.

  - wpe_delta: 2D float tensor whose consecutive rows are highly correlated.
              Detected by computing per-row delta and checking that the
              delta std is materially smaller than the raw std. Captures
              learned position embeddings on any model.

  - tied:     two tensors share the same underlying storage (embed_tokens
              and lm_head are usually tied in transformer LMs). Detected by
              comparing md5 of raw bytes between same-shape same-dtype pairs.

  - generic:  any tensor without a detected pattern -> standard format codec.

The detection pass is O(N) and runs once per encode. All thresholds are
tunable here in one place.
"""
from __future__ import annotations

import hashlib
import numpy as np

# Threshold tunables
MAX_LOWCARD_UNIQUE = 16    # tensors with <=16 unique values use lowcard codec
WPE_DELTA_RATIO = 0.6      # row-delta std must be <= this fraction of raw std
WPE_MIN_ROWS = 16          # 2D tensors with fewer rows are not worth wpe_delta
WPE_MIN_TENSOR_BYTES = 16 * 1024  # only attempt on >= 16KB tensors


def _count_unique_fast(raw: bytes, item_bytes: int, cap: int = MAX_LOWCARD_UNIQUE + 1) -> int:
    """Fast unique-count with an early cap. Returns count clamped to <= cap."""
    n = len(raw) // item_bytes
    if n == 0:
        return 0
    np_dtype = {1: np.uint8, 2: np.uint16, 4: np.uint32}[item_bytes]
    arr = np.frombuffer(raw, dtype=np_dtype)
    # Sample first 4096 values for quick uniqueness check
    sample = arr[: min(n, 4096)]
    n_sample_unique = len(np.unique(sample))
    if n_sample_unique > cap:
        return cap
    # Confirm full
    full_unique = np.unique(arr)
    return min(len(full_unique), cap)


def _is_wpe_candidate(raw: bytes, item_bytes: int, shape: list[int]) -> bool:
    """Heuristic: 2D tensor with high row-row correlation."""
    if len(shape) != 2:
        return False
    rows, cols = shape
    if rows < WPE_MIN_ROWS or cols < 4:
        return False
    if len(raw) < WPE_MIN_TENSOR_BYTES:
        return False
    np_dtype = {1: np.uint8, 2: np.uint16, 4: np.uint32}[item_bytes]
    arr = np.frombuffer(raw, dtype=np_dtype).reshape(rows, cols)

    # Delta on a small sample (first 64 rows)
    n_samp = min(rows, 64)
    samp = arr[:n_samp].astype(np.int64)
    raw_var = float(samp.std())
    if raw_var == 0.0:
        return False
    delta = samp[1:] - samp[:-1]
    delta_var = float(delta.std())
    return delta_var <= WPE_DELTA_RATIO * raw_var


def _md5_hex(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def analyze_tensors(
    tensors: list[dict],
) -> tuple[list[dict], dict[int, str]]:
    """Run special-tensor detection across a model.

    Args:
        tensors: list of dicts with keys: name, shape, dtype, item_bytes, raw (bytes).

    Returns:
        (decisions, tied_master_map)
          decisions[i] = {"kind": "lowcard"|"wpe_delta"|"tied"|"generic",
                          "tied_to": int_index | None}
          tied_master_map: {follower_idx: master_idx}
    """
    n = len(tensors)
    decisions: list[dict] = [{"kind": "generic", "tied_to": None} for _ in range(n)]

    # Pass 1: tied detection (md5 by (item_bytes, shape) bucket for speed)
    md5_index: dict[tuple, int] = {}
    for i, t in enumerate(tensors):
        key = (t["item_bytes"], tuple(t["shape"]))
        # Only consider tying for non-trivial tensors (>= 1024 bytes)
        if len(t["raw"]) < 1024:
            continue
        h = _md5_hex(t["raw"])
        seen_key = (key, h)
        if seen_key in md5_index:
            master = md5_index[seen_key]
            decisions[i] = {"kind": "tied", "tied_to": master}
        else:
            md5_index[seen_key] = i

    # Pass 2: lowcard / wpe_delta on tensors not already tied
    for i, t in enumerate(tensors):
        if decisions[i]["kind"] != "generic":
            continue
        ib = t["item_bytes"]
        if ib not in (1, 2, 4):
            continue
        # lowcard check
        n_unique = _count_unique_fast(t["raw"], ib)
        if n_unique <= MAX_LOWCARD_UNIQUE and n_unique > 0:
            # Require tensor to be at least 256 bytes for the special codec to pay off
            if len(t["raw"]) >= 256:
                decisions[i] = {"kind": "lowcard", "tied_to": None}
                continue
        # wpe_delta check
        if _is_wpe_candidate(t["raw"], ib, t["shape"]):
            decisions[i] = {"kind": "wpe_delta", "tied_to": None}
            continue

    tied_master_map = {i: d["tied_to"] for i, d in enumerate(decisions) if d["kind"] == "tied"}
    return decisions, tied_master_map


def codec_for_format(fmt: str) -> str:
    """Map BigSmall format string to the standard codec name."""
    return {
        "fp32": "fp32_se_ac",
        "bf16": "bf16_se_ac",
        "fp16": "fp16_se_ac",
        "fp8":  "fp8_cat_ac",
        "fp4":  "fp4_cat_ac",
    }[fmt]
