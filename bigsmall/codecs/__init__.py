"""BigSmall codecs - per-format encoder/decoder implementations."""
from . import bf16, fp32, fp16, fp8, fp4, special, generic

CODEC_REGISTRY = {
    "bf16_se_ac":  bf16,
    "fp16_se_ac":  fp16,
    "fp32_se_ac":  fp32,
    "fp8_cat_ac":  fp8,
    "fp4_cat_ac":  fp4,
    "special":     special,
    "zstd":        generic,
    "blosc2_shuffle_zstd": generic,
}


def get_codec(name):
    if name not in CODEC_REGISTRY:
        raise KeyError(f"Unknown codec: {name}")
    return CODEC_REGISTRY[name]
