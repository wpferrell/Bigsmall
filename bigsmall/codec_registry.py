"""Per-tensor codec auto-selection registry.

Replaces the encoder's fixed `format -> codec` dispatch with a registry-based
"try all candidates, keep the smallest" model.

Design constraints (from `B4_CLAUDE.md`):

- `auto_select_codec` MUST NEVER produce a blob larger than the previous fixed
  dispatch would have.  Enforced structurally: the previous default codec for
  each dtype is always the first entry in the candidate list, and we return
  the smallest blob across all attempts.

- It MUST NOT raise.  Any individual codec failure is caught and we fall
  through to the next candidate.

- It is deterministic.  Ties go to the earliest candidate in the list (first
  one wins), so the same input always picks the same codec.

- The `.bs` on-disk format is unchanged.  A new optional `codec_stats` header
  key is the only difference; old readers ignore unknown header keys.

- Tiny-tensor handling stays in `encoder.py` (it short-circuits to raw bytes
  before auto-select runs).  Tied-weight and special-codec (lowcard /
  wpe_delta) handling also stays in `encoder.py`; auto-select is only used
  for the "generic float tensor" path.
"""
from __future__ import annotations

from typing import Callable, Optional

from . import tensor_analysis as ta
from .codecs import bf16, bf16_sparsity, fp2_residual, fp32, fp16, fp8, fp4, generic


# Registry of (encode_fn, decode_fn) keyed by codec name.
# encode_fn signature: (raw: bytes, **ctx) -> (blob: bytes, extras: dict)
# decode_fn signature: (blob: bytes, extras: dict, n_weights: int) -> bytes
_REGISTRY: dict[str, tuple[Callable, Callable]] = {}


def register_codec(name: str, encode_fn: Callable, decode_fn: Callable) -> None:
    """Register a codec.  Overwrites any prior registration for `name`."""
    _REGISTRY[name] = (encode_fn, decode_fn)


def get_codec(name: str) -> Optional[tuple[Callable, Callable]]:
    return _REGISTRY.get(name)


def registered_names() -> list[str]:
    return list(_REGISTRY.keys())


# ---- Built-in codec wrappers --------------------------------------------------
# Each wrapper accepts `**_` so future codecs can receive extra context (dtype,
# shape, layer-type hints) without breaking older entries in the registry.

def _enc_bf16(raw, **_):
    return bf16.encode(raw)


def _enc_bf16_sparsity(raw, threshold_word=None, **_):
    return bf16_sparsity.encode(raw, threshold_word=threshold_word)


def _enc_fp2_residual(raw, **_):
    return fp2_residual.encode(raw)


def _enc_fp32(raw, **_):
    return fp32.encode(raw)


def _enc_fp16(raw, **_):
    return fp16.encode(raw)


def _enc_fp8(raw, **_):
    return fp8.encode(raw)


def _enc_fp4(raw, **_):
    return fp4.encode(raw)


def _enc_zstd(raw, **_):
    return generic.encode_zstd(raw)


def _dec_bf16(blob, extras, n_weights):
    return bf16.decode(blob, extras, n_weights)


def _dec_bf16_sparsity(blob, extras, n_weights):
    return bf16_sparsity.decode(blob, extras, n_weights)


def _dec_fp2_residual(blob, extras, n_weights):
    return fp2_residual.decode(blob, extras, n_weights)


def _dec_fp32(blob, extras, n_weights):
    return fp32.decode(blob, extras, n_weights)


def _dec_fp16(blob, extras, n_weights):
    return fp16.decode(blob, extras, n_weights)


def _dec_fp8(blob, extras, n_weights):
    return fp8.decode(blob, extras, n_weights)


def _dec_fp4(blob, extras, n_weights):
    return fp4.decode(blob, extras, n_weights)


def _dec_zstd(blob, extras, n_weights):
    return generic.decode_zstd(blob, extras)


register_codec("bf16_se_ac", _enc_bf16, _dec_bf16)
register_codec("bf16_sparsity_v1", _enc_bf16_sparsity, _dec_bf16_sparsity)
register_codec("fp2_residual_v1", _enc_fp2_residual, _dec_fp2_residual)
register_codec("fp32_se_ac", _enc_fp32, _dec_fp32)
register_codec("fp16_se_ac", _enc_fp16, _dec_fp16)
register_codec("fp8_cat_ac", _enc_fp8, _dec_fp8)
register_codec("fp4_cat_ac", _enc_fp4, _dec_fp4)
register_codec("zstd", _enc_zstd, _dec_zstd)


# Per-format ordered candidate lists.  Order matters: the first entry MUST be
# the codec that was used by the pre-B4 fixed dispatch for that format, so the
# "smallest-wins" rule cannot regress the file size.
CODEC_CANDIDATES: dict[str, list[str]] = {
    "bf16": ["bf16_se_ac", "bf16_sparsity_v1", "fp2_residual_v1", "zstd"],
    "fp32": ["fp32_se_ac", "zstd"],
    "fp16": ["fp16_se_ac", "zstd"],
    "fp8":  ["fp8_cat_ac", "zstd"],
    "fp4":  ["fp4_cat_ac", "zstd"],
    "raw":  ["zstd"],
}


def _fp2_residual_qualifies(raw: bytes, fmt: str, tensor_name: str) -> bool:
    """Gate for the FP2+residual candidate.

    Session A showed the technique only beats plain bf16 on attention and
    mlp tensors of meaningful size. Norm scales and embeddings either don't
    have the value clustering that lets FP2 hit useful levels, or they're
    too small for the 2-bit-index overhead to amortise.
    """
    if fmt != "bf16":
        return False
    if not fp2_residual.expected_to_beat_bf16(raw):
        return False
    # Layer-type gate: parse from tensor name. Attention and mlp weight
    # tensors carry the matching substring; everything else (norms, embeddings,
    # biases) skips the candidate so the AC pass cost stays bounded.
    return ta.classify_layer(tensor_name) in ("attention", "mlp")


def auto_select_codec(raw: bytes, fmt: str, dtype: str,
                      tensor_name: str = "",
                      shape: tuple = (),
                      item_bytes: int = 0,
                      enable_a5: bool = True,
                      enable_fp2_residual: bool = True) -> tuple[bytes, str, dict]:
    """Try every candidate codec for `fmt` and return the smallest blob.

    Args:
        raw:         tensor bytes (little-endian).
        fmt:         BigSmall format string (e.g. "bf16", "fp32", "raw").
        dtype:       safetensors dtype string (e.g. "BF16", "F32").  Needed by
                     the A5 sparsity scanner.
        tensor_name: passed through to codec wrappers for future use; the
                     built-in codecs ignore it.
        shape:       same — ignored by built-ins.
        item_bytes:  same — ignored by built-ins.
        enable_a5:   if False, skip the A5 sparsity candidate entirely (used
                     by the `enable_a5=False` opt-out on `compress()`).
        enable_fp2_residual: if False, skip the V4 B1 FP2+residual candidate.

    Returns:
        (best_blob, best_codec_name, best_extras).  Always returns something —
        if every candidate raises, falls back to plain zstd of the raw bytes.
    """
    candidates = CODEC_CANDIDATES.get(fmt, ["zstd"])

    # Decide once whether the A5 sparsity candidate is allowed.  Even when the
    # sparsity gate qualifies, the candidate may still produce a larger blob
    # than plain bf16 — in that case the "smallest wins" rule rejects it.
    skip_a5 = True
    a5_threshold_word: Optional[int] = None
    if fmt == "bf16" and enable_a5 and "bf16_sparsity_v1" in candidates:
        try:
            stats = ta.compute_sparsity_stats(raw, dtype=dtype)
        except Exception:
            stats = {"qualifies_for_a5": False}
        if stats.get("qualifies_for_a5"):
            try:
                a5_threshold_word = bf16_sparsity.choose_threshold_word(raw)
                skip_a5 = False
            except Exception:
                skip_a5 = True

    # FP2 residual gate (Session B B1). The "smallest wins" rule below
    # guarantees we never regress; the gate just stops wasting CPU on
    # tensors that can't benefit.
    skip_fp2_residual = True
    if (enable_fp2_residual and fmt == "bf16"
            and "fp2_residual_v1" in candidates):
        try:
            skip_fp2_residual = not _fp2_residual_qualifies(raw, fmt, tensor_name)
        except Exception:
            skip_fp2_residual = True

    best_size: Optional[int] = None
    best_name: Optional[str] = None
    best_blob: Optional[bytes] = None
    best_extras: Optional[dict] = None

    for name in candidates:
        if name == "bf16_sparsity_v1" and skip_a5:
            continue
        if name == "fp2_residual_v1" and skip_fp2_residual:
            continue
        pair = _REGISTRY.get(name)
        if pair is None:
            continue
        encode_fn, _decode_fn = pair
        ctx: dict = {}
        if name == "bf16_sparsity_v1":
            ctx["threshold_word"] = a5_threshold_word
        try:
            blob, extras = encode_fn(raw, **ctx)
        except Exception:
            continue
        size = len(blob)
        # Strictly less than ensures the FIRST candidate (the historical
        # default) wins on ties -> deterministic and no churn for callers
        # that grep the resulting codec name.
        if best_size is None or size < best_size:
            best_size = size
            best_name = name
            best_blob = blob
            best_extras = extras or {}

    if best_blob is None:
        # Pathological — every candidate failed.  Last-ditch fallback so we
        # never raise out of this function.
        blob, extras = generic.encode_zstd(raw)
        return blob, "zstd", extras
    return best_blob, best_name, best_extras


class CodecStats:
    """Tiny accumulator used by the encoder to count codec usages per run."""

    __slots__ = ("counts",)

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}

    def record(self, codec_name: str) -> None:
        self.counts[codec_name] = self.counts.get(codec_name, 0) + 1

    def as_dict(self) -> dict[str, int]:
        return dict(self.counts)
