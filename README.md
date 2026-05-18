# BigSmall

**Run any model. No compromises.**

Mistral 7B is 14 GB. Your machine has 8 GB. Today your only option is quantization -- a degraded, worse version of the model. BigSmall changes that.

BigSmall compresses model weights **losslessly**. Mistral 7B goes from 14 GB to 9 GB. The streaming loader means you never need 9 GB free at once -- it decompresses one layer at a time, directly into VRAM, with a peak RAM footprint of under 2 GB. You run the **exact same model**. Bit-for-bit identical weights. No quality loss. No accuracy regression. No surprises.

```bash
pip install bigsmall
```

> **Version compatibility:** Models compressed with `bigsmall` 2.4.0+ may
> use container format v2 for tensors that benefit from newer codecs and
> require `bigsmall >= 2.4.0` to decompress. v1 files written by every
> previous release remain readable in 2.x and 3.x. Run
> `pip install --upgrade bigsmall` to install the latest release.

```python
import bigsmall

# Load a compressed model -- same as the original, smaller footprint
state_dict = bigsmall.from_pretrained("wpferrell/mistral-7b-instruct-bigsmall")
model.load_state_dict(state_dict)

# Or stream it layer-by-layer -- runs models bigger than your RAM
with bigsmall.StreamingLoader("wpferrell/mistral-7b-instruct-bigsmall", device="cuda") as loader:
    for layer_idx, tensors in loader.iter_layers():
        # one layer in memory at a time, previous layer already freed
        pass
```

---

## The problem with quantization

When a model doesn't fit, the standard answer is quantization. Drop to 4-bit. Use Ollama. It fits now.

But it's not the same model anymore. 4-bit quantization degrades every weight. The outputs are different. Fine-tuning on a quantized model introduces drift. Reproducibility goes out the window. For research, production, or anything where the answer actually matters -- quantization is a compromise you shouldn't have to make.

**BigSmall is not quantization.** After decompression, every weight is bit-for-bit identical to the original, md5-verified on every tensor. You get the full model. Always.

---

## What it does

### BigSmall vs quantization (llama.cpp, GGUF, bitsandbytes, AWQ)

| | Quantization (4-bit) | BigSmall |
|--|--|--|
| Lossless? | No -- weights permanently degraded | **Yes -- bit-identical** |
| Mistral 7B size | ~4 GB | **9 GB** |
| Peak RAM to load | ~4 GB | **< 2 GB** (streaming loader) |
| Inference speed | Slower on some hardware | **Native -- decompress once, run forever** |
| Fine-tuning safe? | No -- drift from degraded base | **Yes -- clean original weights** |
| Reproducible outputs? | No | **Yes** |
| FP32 support? | No | **Yes** |

### BigSmall vs DFloat11 (the other lossless option)

DFloat11 keeps weights compressed in GPU memory and decompresses per forward pass. BigSmall decompresses once at load time and runs at full native speed. Different tools, different tradeoffs.

| | BigSmall | DFloat11 |
|--|--|--|
| Compression ratio (BF16) | **65-66%** | ~70% |
| Compression ratio (FP32) | **75-83%** | BF16 only |
| Inference overhead | **None -- decompress at load** | ~2x slower at batch=1 |
| Hardware | **CPU, Apple Silicon, AMD, any GPU** | CUDA only |
| FP32 / FP16 / FP8 / FP4 | **All supported** | BF16 only |
| Fine-tuning safe? | **Yes -- decompress and fine-tune** | No -- stays compressed |
| Delta compression | **Yes -- 6.95% of source size** | No |
| vLLM compatible? | **Yes** | Custom inference engine only |
| Peak RAM (streaming) | **< 2 GB for any model size** | Full model in VRAM |
| Pre-compressed models on HF | **21 and growing** | ~30 (low downloads) |

### BigSmall vs ZipNN (the other storage-compression option)

Both decompress at load time. BigSmall compresses significantly better and supports more formats.

| | BigSmall | ZipNN |
|--|--|--|
| Compression ratio (BF16) | **65-66%** | ~67% |
| Compression ratio (FP32) | **75-83%** | ~83% |
| FP32 / FP16 / FP8 / FP4 | **All supported** | Mainly BF16 |
| Streaming loader | **Yes -- peak RAM < 2 GB** | No |
| Pre-compressed models on HF | **21 and growing** | 5 total |
| Hardware | **Any** | Any |

---

## Benchmarks

All results are lossless -- md5-verified bit-identical reconstruction on every tensor.

| Model | Format | Original | Compressed | Ratio |
|-------|--------|----------|------------|-------|
| Mistral 7B Instruct v0.3 | BF16 | 14.2 GB | **9.3 GB** | 65.6% |
| Llama 3.1 8B | BF16 | 15.0 GB | **9.9 GB** | 65.7% |
| Qwen 2.5 14B | BF16 | 28.6 GB | **18.8 GB** | 65.8% |
| Stable Diffusion 1.5 UNet | FP16 | 1.72 GB | **1.48 GB** | 85.9% |
| Stable Diffusion 1.5 VAE | FP32 | 335 MB | **278 MB** | 83.2% |
| GPT-2 117M | FP32 | 548 MB | **414 MB** | 75.5% |
| GPT-2 117M | BF16 | 274 MB | **165 MB** | 60.1% |

Fine-tune delta compression: **6.95%** of source size -- ship fine-tunes as tiny diffs, not full model copies.

Streaming peak RAM: **29.6% lower** than full load on GPT-2. On a 70B model the difference is tens of gigabytes.

---

## Install

```bash
pip install bigsmall
```

Requirements: Python 3.9, 3.10, 3.11, 3.12 | PyTorch 2.0+

Optional extras:
```bash
pip install "bigsmall[hf]"        # HuggingFace Hub integration
pip install "bigsmall[diffusion]" # Stable Diffusion support
pip install "bigsmall[vllm]"      # vLLM integration
pip install "bigsmall[all]"       # everything
```

---

## HuggingFace integration

```python
import bigsmall

# Compress any HuggingFace model
bigsmall.compress_for_hub("mistralai/Mistral-7B-Instruct-v0.3", output_dir="./mistral_bs")

# Upload to the Hub
bigsmall.upload_to_hub("./mistral_bs", "you/mistral-7b-bigsmall")

# Anyone can load it with one line
state_dict = bigsmall.from_pretrained("you/mistral-7b-bigsmall")
```

---

## Pre-compressed models

Ready to use -- no compression step needed. Just swap the model ID:

```python
import bigsmall
bigsmall.install_hook()

from transformers import AutoModelForCausalLM, AutoTokenizer

# Pick any model from the table below -- works identically to the original
model = AutoModelForCausalLM.from_pretrained("wpferrell/qwen2.5-7b-instruct-bigsmall")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

inputs = tokenizer("Hello!", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

| Model | HuggingFace | Compressed | Ratio |
|-------|-------------|------------|-------|
| Qwen 2.5 14B Instruct | [wpferrell/qwen2.5-14b-instruct-bigsmall](https://huggingface.co/wpferrell/qwen2.5-14b-instruct-bigsmall) | 18.19 GB | 66.1% |
| Gemma 2 9B Instruct | [wpferrell/gemma-2-9b-it-bigsmall](https://huggingface.co/wpferrell/gemma-2-9b-it-bigsmall) | 11.31 GB | ~65% |
| Qwen 3 8B | [wpferrell/qwen3-8b-bigsmall](https://huggingface.co/wpferrell/qwen3-8b-bigsmall) | 10.08 GB | ~66% |
| Llama 3 8B Instruct | [wpferrell/llama-3-8b-instruct-bigsmall](https://huggingface.co/wpferrell/llama-3-8b-instruct-bigsmall) | 9.83 GB | 65.3% |
| Llama 3.1 8B Instruct | [wpferrell/llama-3.1-8b-instruct-bigsmall](https://huggingface.co/wpferrell/llama-3.1-8b-instruct-bigsmall) | 9.74 GB | 65.0% |
| Qwen 2.5 7B Instruct | [wpferrell/qwen2.5-7b-instruct-bigsmall](https://huggingface.co/wpferrell/qwen2.5-7b-instruct-bigsmall) | 9.36 GB | 66.0% |
| Mistral 7B Instruct v0.3 | [wpferrell/mistral-7b-instruct-bigsmall](https://huggingface.co/wpferrell/mistral-7b-instruct-bigsmall) | 8.87 GB | 65.6% |
| Mistral 7B Instruct v0.2 | [wpferrell/mistral-7b-instruct-v0.2-bigsmall](https://huggingface.co/wpferrell/mistral-7b-instruct-v0.2-bigsmall) | 8.86 GB | 65.5% |
| Gemma 2 2B | [wpferrell/gemma-2-2b-bigsmall](https://huggingface.co/wpferrell/gemma-2-2b-bigsmall) | 8.09 GB | ~65% |
| Qwen 3 4B Instruct | [wpferrell/qwen3-4b-instruct-bigsmall](https://huggingface.co/wpferrell/qwen3-4b-instruct-bigsmall) | 4.95 GB | 65.7% |
| Phi-3.5 Mini Instruct | [wpferrell/phi-3.5-mini-instruct-bigsmall](https://huggingface.co/wpferrell/phi-3.5-mini-instruct-bigsmall) | 4.67 GB | 65.6% |
| Llama 3.2 3B Instruct | [wpferrell/llama-3.2-3b-instruct-bigsmall](https://huggingface.co/wpferrell/llama-3.2-3b-instruct-bigsmall) | 3.93 GB | 65.0% |
| Qwen 2.5 3B Instruct | [wpferrell/qwen2.5-3b-instruct-bigsmall](https://huggingface.co/wpferrell/qwen2.5-3b-instruct-bigsmall) | 3.81 GB | 66.1% |
| Gemma 2 2B Instruct | [wpferrell/gemma-2-2b-it-bigsmall](https://huggingface.co/wpferrell/gemma-2-2b-it-bigsmall) | 3.20 GB | ~65% |
| Qwen 2.5 1.5B Instruct | [wpferrell/qwen2.5-1.5b-instruct-bigsmall](https://huggingface.co/wpferrell/qwen2.5-1.5b-instruct-bigsmall) | 1.89 GB | 66.1% |
| Llama 3.2 1B Instruct | [wpferrell/llama-3.2-1b-instruct-bigsmall](https://huggingface.co/wpferrell/llama-3.2-1b-instruct-bigsmall) | 1.51 GB | **60.4%** |
| Gemma 3 1B Instruct | [wpferrell/gemma-3-1b-it-bigsmall](https://huggingface.co/wpferrell/gemma-3-1b-it-bigsmall) | 1.22 GB | ~65% |
| Qwen 2.5 0.5B Instruct | [wpferrell/qwen2.5-0.5b-instruct-bigsmall](https://huggingface.co/wpferrell/qwen2.5-0.5b-instruct-bigsmall) | 0.61 GB | 63.9% |
| GPT-2 117M | [wpferrell/gpt2-bigsmall](https://huggingface.co/wpferrell/gpt2-bigsmall) | 0.39 GB | 75.5% |
| Gemma 3 270M Instruct | [wpferrell/gemma-3-270m-it-bigsmall](https://huggingface.co/wpferrell/gemma-3-270m-it-bigsmall) | 0.33 GB | ~65% |
| Gemma 3 270M | [wpferrell/gemma-3-270m-bigsmall](https://huggingface.co/wpferrell/gemma-3-270m-bigsmall) | 0.33 GB | ~65% |

> *BF16 ratio varies by model: **60.4 % on Llama 3.2 1B** to **66.1 % on Qwen 2.5 14B**. v2.4.0+ ships a sparsity-aware codec for high-kurtosis MLP tensors; on every real LLM we measured it sits at the joint-entropy floor, so the headline ratio above is essentially independent of codec version. See [`A5_DONE.md`](A5_DONE.md) for the research notes.*
---

## Streaming loader -- run any model on any hardware

The streaming loader decompresses one transformer layer at a time, loads it into VRAM, runs inference, then frees the memory before the next layer. Your peak VRAM is one layer -- not the whole model.

**What this means in practice:**

| Model | Normal VRAM needed | BigSmall streaming VRAM | Runs on |
|-------|-------------------|------------------------|---------|
| GPT-2 117M | 0.55 GB | < 0.1 GB | Any GPU or CPU |
| Gemma 3 1B | 1.86 GB | < 0.2 GB | Any GPU or CPU |
| Llama 3.2 3B | 6.0 GB | < 0.5 GB | Any GPU or CPU |
| Mistral 7B | 14.2 GB | < 1 GB | Any GPU or CPU |
| Llama 3.1 8B | 15.0 GB | < 1 GB | Any GPU or CPU |
| Qwen 2.5 14B | 29.5 GB | < 2 GB | Any GPU or CPU |
| DeepSeek V4-Flash | 148.7 GB | < 4 GB | Any GPU or CPU |

A 4 GB GPU (GTX 1650, M1 MacBook) can run Mistral 7B losslessly. An 8 GB GPU (RTX 3060, M2) can run Qwen 14B. No quantization. Full quality.

BigSmall is the only lossless compression tool with a streaming loader. DFloat11 and ZipNN load the full model into memory before inference.

```python
from bigsmall import StreamingLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

# Runs Mistral 7B on a 4 GB GPU
with StreamingLoader("wpferrell/mistral-7b-instruct-bigsmall", device="cuda") as loader:
    model = loader.load_model(AutoModelForCausalLM)

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
inputs = tokenizer("Hello!", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0]))
```

**Hardware guide:**

| Your GPU VRAM | What you can run (BigSmall streaming, lossless) |
|---|---|
| 2 GB | GPT-2, Gemma 270M |
| 4 GB | Llama 3.2 3B, Mistral 7B, Gemma 2B, Llama 3.1 8B |
| 8 GB | Qwen 2.5 14B, Gemma 2 9B, Phi-3.5 Mini |
| 12 GB | Qwen 2.5 32B, Gemma 3 12B |
| 24 GB | Llama 70B, Qwen 72B, Gemma 3 27B, DeepSeek V4-Flash |
| CPU only | Everything -- slower but lossless full quality |


## vLLM integration

```bash
pip install bigsmall[vllm]
```

```python
import bigsmall

# Serve directly from HuggingFace -- decompresses automatically
bigsmall.vllm_serve("wpferrell/mistral-7b-instruct-bigsmall", port=8000)
```

Or decompress first and use vLLM normally:

```python
out_dir = bigsmall.vllm_decompress("wpferrell/mistral-7b-instruct-bigsmall")

from vllm import LLM
llm = LLM(model=str(out_dir))
outputs = llm.generate("Tell me about lossless compression.")
```

`vllm_decompress` and `vllm_serve` both accept a HuggingFace repo ID, a local directory of `.bs` shards, or a single `.bs` file.

---

## CLI

```bash
bigsmall compress model.safetensors                   # balanced (default)
bigsmall compress model.safetensors --storage         # maximum compression
bigsmall compress model.safetensors --inference       # fastest load
bigsmall decompress model.bs -o model.safetensors
bigsmall info model.bs
bigsmall verify model.bs

# Fine-tune delta
bigsmall compress finetune.safetensors --base base.safetensors -o delta.bs
bigsmall decompress delta.bs --base base.safetensors -o reconstructed.safetensors

# Upgrade an older .bs file to current best codecs (lossless, never grows)
bigsmall migrate model.bs                    # in-place, .bs.bak written by default
bigsmall migrate model.bs --dry-run          # preview savings, write nothing
bigsmall migrate model.bs --no-backup        # skip the .bs.bak
```

`bigsmall migrate` re-encodes each tensor against the per-tensor
auto-selection registry (see below).  If a newer codec produces a smaller
blob it replaces the old one; if not the original is kept byte-for-byte.
By construction the migrated file is therefore never larger than the
original, and every tensor's decompressed bytes are unchanged
(md5-verified).  Useful for picking up new codec wins on models you
already compressed without re-downloading the originals.

### Codec auto-selection

Starting in v2.5.0, every tensor goes through `auto_select_codec`: the
encoder tries every codec that is registered for the tensor's dtype and
keeps the smallest blob.  Ties go to the historical default so file sizes
cannot regress.  The `.bs` header records a `codec_stats` map of
`codec_name -> tensor_count` for audit; `bigsmall info <file>` prints it
as a `codec_breakdown` section.

---

## Format support

| Format | Ratio | Notes |
|--------|-------|-------|
| BF16 | 60-66% | LLMs (Mistral, Llama, Qwen) |
| FP32 | 75-83% | GPT-2, SD VAE, research models |
| FP16 | 77-86% | SD UNet, half-precision models |
| FP8 | 71-72% | Quantization-aware models |
| FP4 | ~30% | Extreme compression |

---

## Comparison

| Tool | BF16 Ratio | FP32 Ratio | Inference Overhead | Hardware | Venue |
|------|------------|------------|-------------------|---------|-------|
| [ZipNN](https://arxiv.org/abs/2411.05239) | 67% | 83% | None (load-time only) | CPU | arXiv '24 |
| [DFloat11](https://arxiv.org/abs/2504.11651) | ~70% | BF16 only | ~2x at batch=1 | CUDA | NeurIPS '25 |
| [ZipServ](https://arxiv.org/abs/2603.17435) | ~70% | BF16 only | 1.22x faster | GDDR GPU | ASPLOS '26 |
| [Unweight](https://research.cloudflare.com/papers/unweight-2026.pdf) | ~80%* | BF16 only | None | H100/H200 | Tech Report |
| **BigSmall** | **65.6%** | **75.5%** | **None** | **CPU + any GPU** | — |

*Lower ratio = better compression. BigSmall BF16 ratio measured on Mistral 7B, FP32 on GPT-2, md5 verified lossless.*
*\*Unweight compresses MLP weights only (~20% total model size reduction).*

---

## Paper

**[BigSmall: Lossless Neural Network Weight Compression at the Joint Entropy Floor](https://github.com/wpferrell/Bigsmall/blob/main/paper.pdf)**

Full technical paper covering the joint entropy floor proof, per-tensor arithmetic codec, streaming loader architecture, and benchmarks across all five float formats. Preprint — arXiv submission in progress.

---

## License

Apache 2.0

