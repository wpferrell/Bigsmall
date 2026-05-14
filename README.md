# BigSmall

Lossless neural network weight compression. One package, every float format,
every model. Compress big models so they fit (or load faster) on small hardware.

## Why BigSmall

BigSmall is **lossless**, not quantization. After decompression the weights are
bit-for-bit identical to the original (md5-verified on every shard). You get the
same inference outputs as the uncompressed model — no quality degradation, no
fine-tune drift, no surprise accuracy regression on long-tail prompts.

Existing tools force a tradeoff: ZipNN (~83% ratio, FP32 only), DFloat11 (~68%,
BF16 only, ~2× slower inference at batch=1), ZipServ (~70%, BF16 only, H100-style
GPUs only). BigSmall covers FP32/BF16/FP16/FP8/FP4 in a single package, hits the
proven mathematical floor on each format, and adds no inference overhead — you
decompress once at load time and run at native speed.

It is the right tool when reproducibility, fine-tuning, or production quality
matters. Ollama-style 4-bit quantization gives you a smaller, **worse** version of
the model. BigSmall gives you a smaller version of **the same** model.

## Install

```bash
pip install bigsmall
```

Optional extras: `pip install "bigsmall[torch]"`, `[hf]`, `[diffusion]`, `[vllm]`, `[all]`.

Requirements: Python 3.9+, PyTorch 2.0+, safetensors, numpy, zstandard, constriction.

## Quick start

```python
import bigsmall

# 1. Compress a safetensors model
bigsmall.compress("model.safetensors", "model.bs")

# 2. Load it back as a torch state_dict (one line, works on local path or HF repo)
sd = bigsmall.from_pretrained("./model.bs")
my_model.load_state_dict(sd, strict=False)

# 3. Stream layer-by-layer for models bigger than RAM
with bigsmall.StreamingLoader("model.bs", device="cuda") as L:
    non_layer = L.load_non_layer_tensors()
    for i, layer in L.iter_layers():
        ...  # one layer's tensors in memory at a time
```

## Benchmarks

All results are bit-exact md5-verified lossless. Source safetensors → `.bs`.

| Model | Format | Source | Compressed | Ratio |
|-------|--------|--------|------------|-------|
| GPT-2 117M | FP32 | 548 MB | 414 MB | **75.53%** |
| GPT-2 117M | BF16 | 274 MB | 165 MB | **60.11%** |
| GPT-2 117M | FP16 | 274 MB | 215 MB | **78.50%** |
| Mistral 7B Instruct v0.3 (shard) | BF16 | 4.55 GB | 2.98 GB | **65.56%** |
| Llama 3.1 8B (shard) | BF16 | 1.17 GB | 768 MB | **65.73%** |
| Qwen 2.5 14B (shard) | BF16 | 1.70 GB | 1.12 GB | **65.75%** |
| Stable Diffusion 1.5 VAE | FP32 | 335 MB | 278 MB | **83.20%** |
| Stable Diffusion 1.5 UNet | FP16 | 1.72 GB | 1.48 GB | **85.92%** |

Delta compression (fine-tune vs base) on GPT-2 with a simulated fine-tune:
**6.95%** of source — fine-tunes ship as tiny diffs against the base.

Streaming peak RAM on GPT-2 117M is **29.6%** lower than full load. The win
grows with depth: a 70B BF16 model normally needs 132 GB to load; with the
streaming loader the peak is `non_layer + one layer` ≈ a few GB.

## CLI

```bash
# Compress / decompress
bigsmall compress model.safetensors                  # balanced (default)
bigsmall compress model.safetensors --storage        # max ratio, slow decode
bigsmall compress model.safetensors --inference      # fastest decode
bigsmall decompress model.bs -o /path/to/output.safetensors

# Info / verify / benchmark
bigsmall info model.bs
bigsmall verify model.bs
bigsmall benchmark model.safetensors

# Delta compression
bigsmall compress finetune.safetensors --base base.safetensors -o delta.bs
bigsmall decompress delta.bs --base base.safetensors -o reconstructed.safetensors
```

## Python API

```python
import bigsmall

# Standard compress / decompress
bigsmall.compress("model.safetensors", "model.bs", mode="balanced")
tensors = bigsmall.decompress("model.bs")           # dict[str, np.ndarray]
torch_tensors = bigsmall.load("model.bs", device="cuda")

# Inspect a .bs file
info = bigsmall.info("model.bs")
print(info["ratio_pct"], info["format"], info["tensor_count"])

# Verify
ok = bigsmall.verify("model.bs")

# Delta compression
bigsmall.compress_delta("ft.safetensors", "base.safetensors", "delta.bs")
tensors = bigsmall.decompress_delta("delta.bs", "base.safetensors")

# Hub integration
bigsmall.compress_for_hub("gpt2", output_dir="./gpt2_bs")
bigsmall.upload_to_hub("./gpt2_bs", "user/gpt2-bigsmall")
sd = bigsmall.from_pretrained("user/gpt2-bigsmall")
```

## HuggingFace integration

```python
from bigsmall.integrations.huggingface import from_pretrained, install_hook

# Drop-in loader that returns a transformers model
from transformers import AutoModelForCausalLM
model = from_pretrained("model.bs", model_class=AutoModelForCausalLM,
                        config_dir="/path/to/hf/model_dir")

# Or patch safetensors globally so any from_pretrained call understands .bs
install_hook()
```

## vLLM integration

```python
from bigsmall.integrations.vllm import decompress_to_temp, get_loader_class

# Portable: decompress to temp dir, then point vLLM at it
out_dir = decompress_to_temp("model.bs", config_dir="/path/to/hf_dir")

# Or use the BigSmallModelLoader subclass directly (vLLM 0.4+)
LoaderClass = get_loader_class()
```

## Diffusion model support

```python
from bigsmall.integrations.diffusion import (
    compress_diffusion, decompress_diffusion, load_pipeline, is_diffusion_model
)

compress_diffusion("unet.safetensors", "unet.bs")
pipe = load_pipeline("unet.bs", config_dir="/path/to/diffusers_dir")
```

## Container format

`.bs` files are self-describing:

| Bytes | Field |
|-------|-------|
| 0..3  | Magic `BGSM` |
| 4..5  | Version (uint16, currently 1) |
| 6..9  | Header JSON length (uint32) |
| 10..  | Header JSON (utf-8) |
| ...   | Concatenated compressed blobs |

Header JSON encodes per-tensor `name`, `shape`, `dtype`, `codec`, `special`,
`compressed_bytes`, `offset`, `md5`, and any codec-specific extras.

## Codecs

| Format | Codec | Notes |
|--------|-------|-------|
| FP32   | per-tensor (sign,exp) AC + zstd byte-plane mantissa | 75-83% ratio |
| BF16   | per-tensor (sign,exp) AC + per-(exp) mantissa AC | 60-66% ratio |
| FP16   | per-tensor (sign,exp) AC + per-(exp) mantissa AC | 77-86% ratio |
| FP8    | per-tensor Categorical AC on byte stream | 71-72% ratio |
| FP4    | per-tensor Categorical AC on 4-bit indices | 30% ratio (huge savings) |

Special tensors (auto-detected, architecture-agnostic):
- **lowcard**: tensors with ≤16 unique values (e.g. attention masks) → tiny lookup table
- **wpe_delta**: 2D embeddings with high row-row correlation → delta + blosc2
- **tied**: tensors with identical bytes (embed_tokens / lm_head) → stored once

## Paper

Technical paper with full research records and floor proofs across all five
float formats: **coming soon (arXiv preprint in preparation)**.

## License

Apache 2.0. See `LICENSE`.
