"""Float format detection and dtype helpers."""
import numpy as np

# .bs container format version. The container.py reader accepts any value in
# `BS_SUPPORTED_FORMAT_VERSIONS`; the writer emits `BS_FORMAT_VERSION` by
# default. v1 is the original layout shipped through 2.2.x. v2 reserves room
# for future codec features (shared probability tables, row-delta transforms,
# sparsity masks, QKV block references) without forcing existing v1 files to
# be re-encoded -- a v2 decoder reads v1 files transparently.
BS_FORMAT_VERSION_V1 = 1
BS_FORMAT_VERSION_V2 = 2
# Default writer version stays at v1 until a release ships an actual v2-only
# codec feature (shared probability tables, row-delta, sparsity, QKV dedup).
# That keeps every model produced by 2.3.0 readable by any 2.0.x consumer
# still pinned in the wild. The reader already accepts both.
BS_FORMAT_VERSION = BS_FORMAT_VERSION_V1
BS_SUPPORTED_FORMAT_VERSIONS = frozenset({BS_FORMAT_VERSION_V1, BS_FORMAT_VERSION_V2})

# Mapping of safetensors / torch dtype names to BigSmall format keys
SAFETENSORS_TO_FORMAT = {
    "F32":  "fp32",
    "F16":  "fp16",
    "BF16": "bf16",
    "F8_E4M3": "fp8",
    "F8_E5M2": "fp8",
    "F4":   "fp4",
}

FORMAT_TO_NUMPY = {
    "fp32": np.uint32,  # raw byte view
    "bf16": np.uint16,
    "fp16": np.uint16,
    "fp8":  np.uint8,
    "fp4":  np.uint8,   # we store FP4 as uint8 indices (unpacked) for codec simplicity
}

FORMAT_BYTES_PER_WEIGHT = {
    "fp32": 4,
    "bf16": 2,
    "fp16": 2,
    "fp8":  1,
    "fp4":  0.5,
}


def detect_format_from_dtype(dtype_str: str) -> str:
    """Detect BigSmall format from a safetensors / numpy / torch dtype string.

    Accepts strings like 'F32', 'BF16', 'F16', 'float32', 'bfloat16', 'torch.float32'.
    Returns one of: 'fp32', 'bf16', 'fp16', 'fp8', 'fp4'.
    """
    s = dtype_str
    if s in SAFETENSORS_TO_FORMAT:
        return SAFETENSORS_TO_FORMAT[s]
    s_low = s.lower().replace("torch.", "")
    if s_low in ("float32", "f32", "fp32"):
        return "fp32"
    if s_low in ("float16", "f16", "fp16", "half"):
        return "fp16"
    if s_low in ("bfloat16", "bf16"):
        return "bf16"
    if "float8" in s_low or s_low in ("fp8",):
        return "fp8"
    if "float4" in s_low or s_low in ("fp4",):
        return "fp4"
    raise ValueError(f"Unknown dtype for BigSmall: {dtype_str}")


def is_float_dtype(dtype_str: str) -> bool:
    try:
        detect_format_from_dtype(dtype_str)
        return True
    except ValueError:
        return False
