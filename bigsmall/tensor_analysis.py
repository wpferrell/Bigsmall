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


# ----------------------------------------------------------------------------
# A5: sparsity-aware codec qualification scan.
# ----------------------------------------------------------------------------
#
# The bf16 joint-entropy codec is at its lower bound for "well-behaved" weight
# tensors but pays for the wide H(exp) of high-kurtosis distributions whose
# outliers populate many exponent values even though most weights are tiny.
# `compute_sparsity_stats()` runs a cheap O(n) scan (sampled if necessary) and
# returns the data the encoder needs to decide whether the A5 codec should
# be tried for a given BF16 tensor.
#
# Thresholds chosen from the entropy analysis in
# `research/compression_analysis_may2026.md`:
#   - typical matrix kurtosis: 0.2 - 1.2
#   - Qwen3 early MLP kurtosis: 4.5 - 17  (these benefit from A5)
#   - typical matrix near-zero%: 0.002 - 0.010
#   - Qwen3 early MLP near-zero%: 0.03 - 0.23


A5_MIN_ELEMENTS = 65_536
A5_KURTOSIS_THRESHOLD = 2.0   # excess kurtosis
A5_NEAR_ZERO_THRESHOLD_PCT = 0.05  # fraction of |w| < threshold
A5_NEAR_ZERO_ABS_THRESHOLD = 1e-6
A5_STATS_SAMPLE_CAP = 1_000_000  # 1 M elements is enough for kurtosis/skew


def compute_sparsity_stats(raw: bytes, dtype: str = "BF16",
                           sample_cap: int = A5_STATS_SAMPLE_CAP) -> dict:
    """Cheap scan: kurtosis_excess, near_zero_pct, mean_abs, qualifies_for_a5.

    Only BF16 is supported on the qualification side; other dtypes return
    `qualifies_for_a5=False` unconditionally. Large tensors are stride-sampled
    to `sample_cap` elements so the scan is O(min(n, sample_cap)).

    Returns a dict with:
        n_elements:     int
        kurtosis_excess: float (NaN if std == 0)
        near_zero_pct:  float in [0, 100]
        mean_abs:       float
        qualifies_for_a5: bool
    """
    if dtype not in ("BF16", "bfloat16"):
        return {
            "n_elements": 0,
            "kurtosis_excess": float("nan"),
            "near_zero_pct": float("nan"),
            "mean_abs": 0.0,
            "qualifies_for_a5": False,
        }
    u16 = np.frombuffer(raw, dtype=np.uint16)
    n = int(u16.size)
    if n < A5_MIN_ELEMENTS:
        return {
            "n_elements": n,
            "kurtosis_excess": float("nan"),
            "near_zero_pct": float("nan"),
            "mean_abs": 0.0,
            "qualifies_for_a5": False,
        }
    # Stride-sample for very large tensors. The stats (kurtosis, near-zero %,
    # mean abs) are all law-of-large-numbers stable at 1M elements.
    sample = u16
    if u16.size > sample_cap:
        step = max(1, u16.size // sample_cap)
        sample = u16[::step]
    # BF16 -> FP32 via leftshift into FP32 word
    u32 = sample.astype(np.uint32) << 16
    f = u32.view(np.float32)
    finite = f[np.isfinite(f)]
    if finite.size == 0:
        return {
            "n_elements": n,
            "kurtosis_excess": float("nan"),
            "near_zero_pct": 100.0,
            "mean_abs": 0.0,
            "qualifies_for_a5": False,
        }
    abs_f = np.abs(finite)
    mu = float(finite.mean())
    sd = float(finite.std())
    if sd == 0.0:
        kurt = float("nan")
    else:
        d = finite - mu
        m2 = float((d * d).mean())
        m4 = float((d ** 4).mean())
        kurt = (m4 / (m2 * m2)) - 3.0 if m2 > 0 else float("nan")
    near_zero_count = int((abs_f < A5_NEAR_ZERO_ABS_THRESHOLD).sum())
    near_zero_pct = 100.0 * near_zero_count / float(finite.size)
    mean_abs = float(abs_f.mean())

    qualifies = (
        (isinstance(kurt, float) and not (kurt != kurt) and kurt >= A5_KURTOSIS_THRESHOLD)
        or (near_zero_pct >= A5_NEAR_ZERO_THRESHOLD_PCT)
    )
    return {
        "n_elements": n,
        "kurtosis_excess": kurt,
        "near_zero_pct": near_zero_pct,
        "mean_abs": mean_abs,
        "qualifies_for_a5": bool(qualifies),
    }


# ============================================================================
# Deep entropy analysis (research / V3 codec planning)
# ============================================================================
#
# These helpers produce per-tensor Shannon entropies on the actual byte
# distributions a real BF16 codec sees:
#
#   raw_bits          - H over the 16-bit word as a whole
#   sign_bit_bits     - H of the 1-bit sign distribution
#   exp_bits          - H of the 8-bit exponent
#   mant_bits         - H of the 7-bit mantissa (unconditional)
#   H(mant|exp)       - what BigSmall codes for the mantissa
#   H(sign,exp)       - what BigSmall codes for the sign+exp joint alphabet
#   H(exp)            - what DFloat11 codes (sign + mantissa stored raw)
#
# Plus per-tensor compression ratios, distribution statistics, and layer
# classification.


_LAYER_KEYWORDS = (
    # ordered: first match wins
    ("embedding", ("embed", "wte", "wpe", "embed_tokens", "lm_head", "token_emb")),
    ("norm",      ("norm", "ln_", "_ln", "layernorm", "rms_norm", "layer_norm")),
    ("attention", ("attn", "attention", "q_proj", "k_proj", "v_proj", "o_proj",
                   "qkv", "c_attn", "self_attention", "self_attn")),
    ("mlp",       ("mlp", "ffn", "feed_forward", "up_proj", "down_proj",
                   "gate_proj", "c_fc", "c_proj", "fc1", "fc2", "wi", "wo")),
    ("bias",      ("bias",)),
)


def classify_layer(name: str) -> str:
    """Bucket a tensor name into a coarse layer-type label.

    Order matters: 'embedding' before 'norm' because lm_head/embed names are
    more specific. 'bias' is a fallback when nothing else matched.
    """
    n = name.lower()
    for label, keywords in _LAYER_KEYWORDS:
        if label == "bias":
            continue
        for kw in keywords:
            if kw in n:
                return label
    if "bias" in n:
        return "bias"
    return "other"


def _shannon_entropy_bits(counts: "np.ndarray") -> float:
    """Shannon entropy in bits given an integer count histogram."""
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    p = counts.astype(np.float64) / total
    nz = p[p > 0]
    return float(-(nz * np.log2(nz)).sum())


def _bf16_decompose(raw: bytes) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """Split BF16 raw bytes into (u16 word, sign, exp, mantissa) numpy arrays."""
    u16 = np.frombuffer(raw, dtype=np.uint16)
    sign = (u16 >> 15).astype(np.uint8)
    exp = ((u16 >> 7) & 0xFF).astype(np.uint8)
    mant = (u16 & 0x7F).astype(np.uint8)
    return u16, sign, exp, mant


def _fp16_decompose(raw: bytes) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """Split FP16 raw bytes into (u16 word, sign, exp, mantissa) numpy arrays."""
    u16 = np.frombuffer(raw, dtype=np.uint16)
    sign = (u16 >> 15).astype(np.uint8)
    exp = ((u16 >> 10) & 0x1F).astype(np.uint8)
    mant = (u16 & 0x3FF).astype(np.uint16)
    return u16, sign, exp, mant


def _fp32_decompose(raw: bytes) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """Split FP32 raw bytes into (u32 word, sign, exp, mantissa) numpy arrays."""
    u32 = np.frombuffer(raw, dtype=np.uint32)
    sign = (u32 >> 31).astype(np.uint8)
    exp = ((u32 >> 23) & 0xFF).astype(np.uint8)
    mant = (u32 & 0x7FFFFF).astype(np.uint32)
    return u32, sign, exp, mant


def _entropy_block_bf16(raw: bytes) -> dict:
    """Return a dict of entropy measurements for a BF16 byte buffer.

    All entropies are returned in BITS per element. The compressed-size
    estimates assume an ideal arithmetic coder (entropy is a lower bound).
    """
    n = len(raw) // 2
    if n == 0:
        return {
            "n_elements": 0,
            "raw_bits_per_el": 16.0,
        }

    u16 = np.frombuffer(raw, dtype=np.uint16)

    # Whole-word entropy via histogram. uint16 has 65536 bins, fast with bincount.
    word_hist = np.bincount(u16, minlength=65536)
    raw_bits = _shannon_entropy_bits(word_hist)

    # Derive sign / exp / mant histograms from the 16-bit histogram so we
    # never materialise three more uint8 arrays of size n.
    word_idx = np.arange(65536, dtype=np.uint16)
    sign_of = (word_idx >> 15).astype(np.uint8)
    exp_of = ((word_idx >> 7) & 0xFF).astype(np.uint8)
    mant_of = (word_idx & 0x7F).astype(np.uint8)

    sign_hist = np.bincount(sign_of, weights=word_hist, minlength=2)
    sign_bits = _shannon_entropy_bits(sign_hist)

    exp_hist = np.bincount(exp_of, weights=word_hist, minlength=256)
    exp_bits = _shannon_entropy_bits(exp_hist)

    mant_hist = np.bincount(mant_of, weights=word_hist, minlength=128)
    mant_bits_unconditional = _shannon_entropy_bits(mant_hist)

    # H(sign, exp): combined 9-bit alphabet (what BigSmall codes).
    se_idx = (sign_of.astype(np.uint16) << 8) | exp_of.astype(np.uint16)  # 0..511
    se_hist = np.bincount(se_idx, weights=word_hist, minlength=512)
    se_joint_bits = _shannon_entropy_bits(se_hist)

    # H(exp, mant): joint exponent+mantissa entropy.
    em_idx = (exp_of.astype(np.uint16) << 7) | mant_of.astype(np.uint16)  # 0..32767
    em_hist = np.bincount(em_idx, weights=word_hist, minlength=32768)
    h_em_joint = _shannon_entropy_bits(em_hist)
    # H(m|e) = H(e,m) - H(e)
    h_mant_given_exp = h_em_joint - exp_bits

    # H(word) via decomposition sanity:
    # H(s, e, m) = H(s, e) + H(m | s, e)  ; we approximate the latter with H(m|e)
    # (sign rarely changes mantissa distribution beyond noise -- this is checked
    # by the gap between raw_bits and (se_joint_bits + h_mant_given_exp))
    decomposed_bits = se_joint_bits + h_mant_given_exp

    # Theoretical compressed bits per element:
    #   BigSmall codes:  H(s,e) + H(m|e)
    #   DFloat11 codes:  1 (sign raw) + H(e) + 7 (mant raw)
    bigsmall_lower_bound_bits = se_joint_bits + h_mant_given_exp
    dfloat11_lower_bound_bits = 1.0 + exp_bits + 7.0

    return {
        "n_elements": n,
        "raw_bits_per_el": raw_bits,
        "sign_bits_per_el": sign_bits,
        "exp_bits_per_el": exp_bits,
        "mant_bits_per_el": mant_bits_unconditional,
        "H_mant_given_exp_bits_per_el": h_mant_given_exp,
        "H_sign_exp_joint_bits_per_el": se_joint_bits,
        "decomposed_bits_per_el": decomposed_bits,
        "bigsmall_lower_bound_bits_per_el": bigsmall_lower_bound_bits,
        "dfloat11_lower_bound_bits_per_el": dfloat11_lower_bound_bits,
        "bigsmall_theoretical_ratio_pct": 100.0 * bigsmall_lower_bound_bits / 16.0,
        "dfloat11_theoretical_ratio_pct": 100.0 * dfloat11_lower_bound_bits / 16.0,
    }


def _entropy_block_fp16(raw: bytes) -> dict:
    n = len(raw) // 2
    if n == 0:
        return {"n_elements": 0, "raw_bits_per_el": 16.0}
    u16 = np.frombuffer(raw, dtype=np.uint16)
    word_hist = np.bincount(u16, minlength=65536)
    raw_bits = _shannon_entropy_bits(word_hist)

    word_idx = np.arange(65536, dtype=np.uint16)
    sign_of = (word_idx >> 15).astype(np.uint8)
    exp_of = ((word_idx >> 10) & 0x1F).astype(np.uint8)
    mant_of = (word_idx & 0x3FF).astype(np.uint16)

    sign_hist = np.bincount(sign_of, weights=word_hist, minlength=2)
    sign_bits = _shannon_entropy_bits(sign_hist)
    exp_hist = np.bincount(exp_of, weights=word_hist, minlength=32)
    exp_bits = _shannon_entropy_bits(exp_hist)
    mant_hist = np.bincount(mant_of, weights=word_hist, minlength=1024)
    mant_bits_unconditional = _shannon_entropy_bits(mant_hist)
    se_idx = (sign_of.astype(np.uint16) << 5) | exp_of.astype(np.uint16)
    se_hist = np.bincount(se_idx, weights=word_hist, minlength=64)
    se_joint_bits = _shannon_entropy_bits(se_hist)

    em_idx = (exp_of.astype(np.uint16) << 10) | mant_of
    em_hist = np.bincount(em_idx, weights=word_hist, minlength=32768)
    h_em_joint = _shannon_entropy_bits(em_hist)
    h_mant_given_exp = h_em_joint - exp_bits

    bigsmall_lb = se_joint_bits + h_mant_given_exp
    dfloat11_lb = 1.0 + exp_bits + 10.0
    return {
        "n_elements": n,
        "raw_bits_per_el": raw_bits,
        "sign_bits_per_el": sign_bits,
        "exp_bits_per_el": exp_bits,
        "mant_bits_per_el": mant_bits_unconditional,
        "H_mant_given_exp_bits_per_el": h_mant_given_exp,
        "H_sign_exp_joint_bits_per_el": se_joint_bits,
        "decomposed_bits_per_el": bigsmall_lb,
        "bigsmall_lower_bound_bits_per_el": bigsmall_lb,
        "dfloat11_lower_bound_bits_per_el": dfloat11_lb,
        "bigsmall_theoretical_ratio_pct": 100.0 * bigsmall_lb / 16.0,
        "dfloat11_theoretical_ratio_pct": 100.0 * dfloat11_lb / 16.0,
    }


def _entropy_block_fp32(raw: bytes) -> dict:
    n = len(raw) // 4
    if n == 0:
        return {"n_elements": 0, "raw_bits_per_el": 32.0}
    u32, sign, exp, mant = _fp32_decompose(raw)
    sign_hist = np.bincount(sign, minlength=2)
    sign_bits = _shannon_entropy_bits(sign_hist)
    exp_hist = np.bincount(exp, minlength=256)
    exp_bits = _shannon_entropy_bits(exp_hist)
    # For mantissa (23 bits) the unconditional histogram is too large; sample
    # top 8 bits and use a single bincount over (exp, mant_top8) joint index.
    mant_top8 = (mant >> 15).astype(np.uint8)
    mant_top_hist = np.bincount(mant_top8, minlength=256)
    mant_bits_top = _shannon_entropy_bits(mant_top_hist)
    se = (sign.astype(np.uint16) << 8) | exp.astype(np.uint16)
    se_hist = np.bincount(se, minlength=512)
    se_joint_bits = _shannon_entropy_bits(se_hist)

    em_idx = (exp.astype(np.uint32) << 8) | mant_top8.astype(np.uint32)
    em_hist = np.bincount(em_idx, minlength=65536)
    h_em_top_joint = _shannon_entropy_bits(em_hist)
    h_mant_top_given_exp = h_em_top_joint - exp_bits

    # Total estimate: H(s,e) + H(m_top8|e) + 15 raw mantissa bits below top8
    bigsmall_lb = se_joint_bits + h_mant_top_given_exp + 15.0
    dfloat11_lb = 1.0 + exp_bits + 23.0
    return {
        "n_elements": n,
        "raw_bits_per_el": float("nan"),  # too expensive at 32-bit
        "sign_bits_per_el": sign_bits,
        "exp_bits_per_el": exp_bits,
        "mant_bits_per_el": mant_bits_top + 15.0,  # approximate
        "H_mant_given_exp_bits_per_el": h_mant_top_given_exp + 15.0,
        "H_sign_exp_joint_bits_per_el": se_joint_bits,
        "decomposed_bits_per_el": bigsmall_lb,
        "bigsmall_lower_bound_bits_per_el": bigsmall_lb,
        "dfloat11_lower_bound_bits_per_el": dfloat11_lb,
        "bigsmall_theoretical_ratio_pct": 100.0 * bigsmall_lb / 32.0,
        "dfloat11_theoretical_ratio_pct": 100.0 * dfloat11_lb / 32.0,
    }


def _entropy_block(raw: bytes, dtype: str) -> dict:
    if dtype == "BF16":
        return _entropy_block_bf16(raw)
    if dtype == "F16":
        return _entropy_block_fp16(raw)
    if dtype == "F32":
        return _entropy_block_fp32(raw)
    return {"n_elements": len(raw), "raw_bits_per_el": float("nan")}


def _distribution_stats_bf16(raw: bytes, sample_cap: int = 1_000_000) -> dict:
    """Compute kurtosis, skewness, and near-zero fraction from BF16 weights.

    On tensors above `sample_cap` elements we take an evenly-spaced sample so
    that 4th-moment calculations remain affordable.
    """
    if len(raw) < 4:
        return {"kurtosis": float("nan"), "skewness": float("nan"),
                "near_zero_pct": float("nan"), "abs_max": 0.0}
    # BF16 -> FP32 via leftshift into FP32 word
    u16 = np.frombuffer(raw, dtype=np.uint16)
    if u16.size > sample_cap:
        step = max(1, u16.size // sample_cap)
        u16 = u16[::step]
    u32 = (u16.astype(np.uint32) << 16)
    f = u32.view(np.float32)
    # Drop NaN/Inf so stats are not destroyed by sentinel values.
    finite = f[np.isfinite(f)]
    if finite.size == 0:
        return {"kurtosis": float("nan"), "skewness": float("nan"),
                "near_zero_pct": 100.0, "abs_max": 0.0}
    mu = float(finite.mean())
    sd = float(finite.std())
    if sd == 0.0:
        kurt = float("nan")
        skew = float("nan")
    else:
        d = (finite - mu)
        m2 = float((d * d).mean())
        m3 = float((d ** 3).mean())
        m4 = float((d ** 4).mean())
        skew = m3 / (m2 ** 1.5) if m2 > 0 else float("nan")
        kurt = (m4 / (m2 * m2)) - 3.0 if m2 > 0 else float("nan")
    near_zero = float((np.abs(finite) < 1e-6).sum()) / float(finite.size)
    abs_max = float(np.abs(finite).max())
    return {
        "kurtosis_excess": kurt,
        "skewness": skew,
        "near_zero_pct": 100.0 * near_zero,
        "abs_max": abs_max,
        "mean": mu,
        "std": sd,
    }


def _distribution_stats_fp16(raw: bytes, sample_cap: int = 1_000_000) -> dict:
    if len(raw) < 4:
        return {"kurtosis_excess": float("nan"), "skewness": float("nan"),
                "near_zero_pct": float("nan"), "abs_max": 0.0,
                "mean": 0.0, "std": 0.0}
    f16 = np.frombuffer(raw, dtype=np.float16)
    if f16.size > sample_cap:
        step = max(1, f16.size // sample_cap)
        f16 = f16[::step]
    f = f16.astype(np.float32)
    finite = f[np.isfinite(f)]
    if finite.size == 0:
        return {"kurtosis_excess": float("nan"), "skewness": float("nan"),
                "near_zero_pct": 100.0, "abs_max": 0.0,
                "mean": 0.0, "std": 0.0}
    mu = float(finite.mean()); sd = float(finite.std())
    if sd == 0.0:
        kurt = float("nan"); skew = float("nan")
    else:
        d = (finite - mu)
        m2 = float((d * d).mean()); m3 = float((d ** 3).mean()); m4 = float((d ** 4).mean())
        skew = m3 / (m2 ** 1.5) if m2 > 0 else float("nan")
        kurt = (m4 / (m2 * m2)) - 3.0 if m2 > 0 else float("nan")
    near_zero = float((np.abs(finite) < 1e-6).sum()) / float(finite.size)
    return {
        "kurtosis_excess": kurt, "skewness": skew,
        "near_zero_pct": 100.0 * near_zero,
        "abs_max": float(np.abs(finite).max()),
        "mean": mu, "std": sd,
    }


def _distribution_stats_fp32(raw: bytes, sample_cap: int = 1_000_000) -> dict:
    if len(raw) < 4:
        return {"kurtosis_excess": float("nan"), "skewness": float("nan"),
                "near_zero_pct": float("nan"), "abs_max": 0.0,
                "mean": 0.0, "std": 0.0}
    f = np.frombuffer(raw, dtype=np.float32)
    if f.size > sample_cap:
        step = max(1, f.size // sample_cap)
        f = f[::step]
    finite = f[np.isfinite(f)]
    if finite.size == 0:
        return {"kurtosis_excess": float("nan"), "skewness": float("nan"),
                "near_zero_pct": 100.0, "abs_max": 0.0,
                "mean": 0.0, "std": 0.0}
    mu = float(finite.mean()); sd = float(finite.std())
    if sd == 0.0:
        kurt = float("nan"); skew = float("nan")
    else:
        d = (finite - mu)
        m2 = float((d * d).mean()); m3 = float((d ** 3).mean()); m4 = float((d ** 4).mean())
        skew = m3 / (m2 ** 1.5) if m2 > 0 else float("nan")
        kurt = (m4 / (m2 * m2)) - 3.0 if m2 > 0 else float("nan")
    near_zero = float((np.abs(finite) < 1e-6).sum()) / float(finite.size)
    return {
        "kurtosis_excess": kurt, "skewness": skew,
        "near_zero_pct": 100.0 * near_zero,
        "abs_max": float(np.abs(finite).max()),
        "mean": mu, "std": sd,
    }


def _distribution_stats(raw: bytes, dtype: str) -> dict:
    if dtype == "BF16":
        return _distribution_stats_bf16(raw)
    if dtype == "F16":
        return _distribution_stats_fp16(raw)
    if dtype == "F32":
        return _distribution_stats_fp32(raw)
    return {"kurtosis_excess": float("nan"), "skewness": float("nan"),
            "near_zero_pct": float("nan"), "abs_max": 0.0,
            "mean": 0.0, "std": 0.0}


def _find_shards(model_dir):
    """Return ordered safetensors shard paths in a model directory."""
    from pathlib import Path
    import json as _json
    d = Path(model_dir)
    idx = d / "model.safetensors.index.json"
    if idx.exists():
        with open(idx, "r", encoding="utf-8") as f:
            data = _json.load(f)
        files = sorted(set(data["weight_map"].values()))
        return [d / s for s in files]
    single = d / "model.safetensors"
    if single.exists():
        return [single]
    return sorted(d.glob("*.safetensors"))


def deep_entropy_analysis(model_dir, output_json,
                          measure_compressed: bool = True,
                          tensor_limit: int = 0,
                          progress: bool = True) -> dict:
    """Run per-tensor entropy and distribution analysis on a HuggingFace model.

    Args:
        model_dir: path to a model directory containing safetensors shard(s).
        output_json: path to write the resulting JSON.
        measure_compressed: if True, also compress each tensor with BigSmall
            and record the achieved size. Adds significant runtime on large
            models; turn off for a quick pass.
        tensor_limit: if > 0, stop after this many tensors (useful for smoke
            tests). 0 means analyse everything.
        progress: print one line per tensor.

    Returns:
        The full analysis dict (also written to JSON).
    """
    from pathlib import Path
    import json as _json
    import time as _time

    from safetensors import safe_open
    from . import encoder

    model_dir = Path(model_dir)
    shards = _find_shards(model_dir)
    if not shards:
        raise FileNotFoundError(f"No safetensors files in {model_dir}")

    tensors_out: list[dict] = []
    t_start = _time.perf_counter()
    counter = 0

    for shard_idx, shard in enumerate(shards, start=1):
        with safe_open(str(shard), framework="pt") as f:
            keys = list(f.keys())
            for name in keys:
                t = f.get_tensor(name)
                # safetensors gives us a torch tensor; pull raw bytes.
                import torch as _torch
                arr = t.contiguous().view(_torch.uint8).cpu().numpy()
                raw = arr.tobytes()
                shape = list(t.shape)
                # Resolve the BigSmall-native format string from torch dtype.
                dtype_str = str(t.dtype).replace("torch.", "")
                fmt = {
                    "float32": "F32", "bfloat16": "BF16", "float16": "F16",
                    "float64": "F64", "int8": "I8", "int16": "I16",
                    "int32": "I32", "int64": "I64", "uint8": "U8",
                }.get(dtype_str, dtype_str.upper())

                ent = _entropy_block(raw, fmt)
                dist = _distribution_stats(raw, fmt)
                ent["dtype"] = fmt
                ent["shape"] = shape
                ent["raw_bytes"] = len(raw)
                ent["layer_type"] = classify_layer(name)
                ent["name"] = name
                ent["shard"] = shard.name
                ent.update(dist)

                if measure_compressed and fmt in ("BF16", "F16", "F32"):
                    # Spin up a tiny in-memory compress of this single tensor.
                    # We use the encoder's standalone path by saving a 1-tensor
                    # safetensors and compressing it.
                    import tempfile as _tempfile
                    from safetensors.torch import save_file as _save_file
                    with _tempfile.TemporaryDirectory() as td:
                        single_st = Path(td) / "t.safetensors"
                        _save_file({name: t.contiguous()}, str(single_st))
                        single_bs = Path(td) / "t.bs"
                        try:
                            encoder.compress(single_st, single_bs, mode="balanced",
                                             workers=1)
                            compressed_bytes = single_bs.stat().st_size
                        except Exception as e:
                            compressed_bytes = -1
                            ent["compression_error"] = str(e)
                    ent["bigsmall_compressed_bytes"] = compressed_bytes
                    if compressed_bytes > 0 and len(raw) > 0:
                        achieved_pct = 100.0 * compressed_bytes / len(raw)
                        ent["bigsmall_achieved_ratio_pct"] = achieved_pct
                        # Overhead vs theoretical: achieved_bits - lower_bound_bits
                        if "bigsmall_lower_bound_bits_per_el" in ent and ent["n_elements"]:
                            achieved_bits_per_el = 8.0 * compressed_bytes / ent["n_elements"]
                            ent["bigsmall_achieved_bits_per_el"] = achieved_bits_per_el
                            ent["bigsmall_overhead_bits_per_el"] = (
                                achieved_bits_per_el - ent["bigsmall_lower_bound_bits_per_el"]
                            )

                tensors_out.append(ent)
                counter += 1
                if progress:
                    lb = ent.get("bigsmall_lower_bound_bits_per_el")
                    lb_str = f"{lb:.3f}" if isinstance(lb, (int, float)) else "NA"
                    msg = f"[{counter}] {name} ({fmt}, {len(raw):,}B) lb={lb_str}/16"
                    if "bigsmall_achieved_bits_per_el" in ent:
                        msg += f" got={ent['bigsmall_achieved_bits_per_el']:.3f}/16"
                    print(msg, flush=True)
                if tensor_limit and counter >= tensor_limit:
                    break
        if tensor_limit and counter >= tensor_limit:
            break

    elapsed = _time.perf_counter() - t_start

    # Aggregate
    by_layer: dict[str, dict] = {}
    total_raw = 0
    total_lb_bits = 0.0
    total_dfloat11_lb_bits = 0.0
    total_achieved = 0
    achieved_count = 0
    for e in tensors_out:
        n = e.get("n_elements", 0)
        if not n:
            continue
        total_raw += int(e["raw_bytes"])
        total_lb_bits += float(e.get("bigsmall_lower_bound_bits_per_el", 0.0)) * n
        total_dfloat11_lb_bits += float(e.get("dfloat11_lower_bound_bits_per_el", 0.0)) * n
        if "bigsmall_compressed_bytes" in e and e["bigsmall_compressed_bytes"] > 0:
            total_achieved += int(e["bigsmall_compressed_bytes"])
            achieved_count += 1

        lt = e["layer_type"]
        b = by_layer.setdefault(lt, {
            "tensors": 0, "raw_bytes": 0, "lb_bits": 0.0,
            "dfloat11_lb_bits": 0.0, "achieved_bytes": 0,
            "n_elements": 0,
        })
        b["tensors"] += 1
        b["n_elements"] += n
        b["raw_bytes"] += int(e["raw_bytes"])
        b["lb_bits"] += float(e.get("bigsmall_lower_bound_bits_per_el", 0.0)) * n
        b["dfloat11_lb_bits"] += float(e.get("dfloat11_lower_bound_bits_per_el", 0.0)) * n
        if "bigsmall_compressed_bytes" in e and e["bigsmall_compressed_bytes"] > 0:
            b["achieved_bytes"] += int(e["bigsmall_compressed_bytes"])

    def _agg_pct(b):
        if b["raw_bytes"] == 0 or b["n_elements"] == 0:
            return None
        lb_bytes = b["lb_bits"] / 8.0
        df_lb_bytes = b["dfloat11_lb_bits"] / 8.0
        out = {
            "tensors": b["tensors"],
            "raw_bytes": b["raw_bytes"],
            "bigsmall_lower_bound_pct": 100.0 * lb_bytes / b["raw_bytes"],
            "dfloat11_lower_bound_pct": 100.0 * df_lb_bytes / b["raw_bytes"],
        }
        if b["achieved_bytes"] > 0:
            out["bigsmall_achieved_pct"] = 100.0 * b["achieved_bytes"] / b["raw_bytes"]
            out["overhead_pct"] = out["bigsmall_achieved_pct"] - out["bigsmall_lower_bound_pct"]
        return out

    summary = {
        "model_dir": str(model_dir),
        "total_tensors": len(tensors_out),
        "total_raw_bytes": total_raw,
        "elapsed_seconds": elapsed,
        "aggregate": {
            "bigsmall_lower_bound_pct": (100.0 * (total_lb_bits / 8.0) / total_raw)
                if total_raw > 0 else None,
            "dfloat11_lower_bound_pct": (100.0 * (total_dfloat11_lb_bits / 8.0) / total_raw)
                if total_raw > 0 else None,
        },
        "by_layer_type": {k: _agg_pct(v) for k, v in by_layer.items()},
    }
    if achieved_count > 0:
        summary["aggregate"]["bigsmall_achieved_pct"] = (
            100.0 * total_achieved / total_raw if total_raw > 0 else None
        )
        summary["aggregate"]["overhead_pct"] = (
            summary["aggregate"]["bigsmall_achieved_pct"]
            - summary["aggregate"]["bigsmall_lower_bound_pct"]
        )

    out = {
        "summary": summary,
        "tensors": tensors_out,
    }
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as fh:
        _json.dump(out, fh, indent=2, default=lambda o: None)

    if progress:
        print(f"Wrote {output_json} ({len(tensors_out)} tensors, {elapsed:.1f}s)", flush=True)
    return out
