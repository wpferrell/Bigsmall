[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20279248.svg)](https://doi.org/10.5281/zenodo.20279248)

# BigSmall

**Lossless neural network weight compression. 34% smaller files, bit-identical weights.**

BigSmall losslessly compresses model weights. Mistral 7B goes from 14 GB to 9 GB. Bit-for-bit identical reconstruction, md5-verified on every tensor. No quantization, no accuracy regression, no surprises — the same model, smaller.

```bash
pip install bigsmall
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("wpferrell/mistral-7b-instruct-bigsmall")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
```

---

## Install

```
pip install bigsmall

# From source
git clone https://github.com/wpferrell/Bigsmall
cd Bigsmall
pip install -e .
```

Requirements: Python 3.9+ | PyTorch 2.0+
Works on CPU, NVIDIA (CUDA), AMD (ROCm), Apple Silicon (MPS).

Optional extras:
```
pip install "bigsmall[hf]"        # HuggingFace Hub integration
pip install "bigsmall[diffusion]" # Stable Diffusion support
pip install "bigsmall[vllm]"      # vLLM integration
pip install "bigsmall[all]"       # everything
```

---

## Three ways to use BigSmall

### 1. Use a pre-compressed model from HuggingFace

Pick any model from the table below and load it like a standard HuggingFace model:

```python
import bigsmall
bigsmall.install_hook()

from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("wpferrell/qwen2.5-7b-instruct-bigsmall")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
```

### 2. Compress your own model

```python
import bigsmall

bigsmall.compress_for_hub("mistralai/Mistral-7B-Instruct-v0.3", output_dir="./mistral_bs")
bigsmall.upload_to_hub("./mistral_bs", "you/mistral-7b-bigsmall")
```

Or from the CLI:
```
bigsmall compress model.safetensors -o model.bs
bigsmall decompress model.bs -o model.safetensors
```

### 3. Run compressed models with low VRAM

`BigSmallStreamingModel` decompresses one layer at a time, loads it into VRAM, runs inference, then frees it. Peak VRAM is one layer — not the whole model.

```python
from bigsmall import BigSmallStreamingModel

model = BigSmallStreamingModel.from_pretrained("wpferrell/mistral-7b-instruct-bigsmall")
```

A 4 GB GPU can run Mistral 7B losslessly. An 8 GB GPU can run Qwen 14B. No quantization. Full quality.

---

## Pre-compressed models

| Model | Original Size | Compressed Size | Ratio | HuggingFace |
|-------|---------------|------------------|-------|-------------|
| Qwen 2.5 14B Instruct | 29.5 GB | 18.19 GB | 66.1% | [link](https://huggingface.co/wpferrell/qwen2.5-14b-instruct-bigsmall) |
| Gemma 2 9B Instruct | 17.2 GB | 11.31 GB | 65.7% | [link](https://huggingface.co/wpferrell/gemma-2-9b-it-bigsmall) |
| Qwen 3 8B | 15.26 GB | 10.08 GB | 66.0% | [link](https://huggingface.co/wpferrell/qwen3-8b-bigsmall) |
| Llama 3 8B Instruct | 15.0 GB | 9.83 GB | 65.7% | [link](https://huggingface.co/wpferrell/llama-3-8b-instruct-bigsmall) |
| Llama 3.1 8B Instruct | 15.0 GB | 9.74 GB | 65.0% | [link](https://huggingface.co/wpferrell/llama-3.1-8b-instruct-bigsmall) |
| Qwen 2.5 7B Instruct | 15.2 GB | 9.36 GB | 66.0% | [link](https://huggingface.co/wpferrell/qwen2.5-7b-instruct-bigsmall) |
| Mistral 7B Instruct v0.3 | 14.2 GB | 8.87 GB | 65.6% | [link](https://huggingface.co/wpferrell/mistral-7b-instruct-bigsmall) |
| Mistral 7B Instruct v0.2 | 14.2 GB | 8.86 GB | 65.5% | [link](https://huggingface.co/wpferrell/mistral-7b-instruct-v0.2-bigsmall) |
| Gemma 2 2B | 9.8 GB | 8.09 GB | 82.6% | [link](https://huggingface.co/wpferrell/gemma-2-2b-bigsmall) |
| Gemma 3 4B Instruct | 8.01 GB | 5.23 GB | 65.3% | [link](https://huggingface.co/wpferrell/gemma-3-4b-it-bigsmall) |
| Qwen 3 4B Instruct | 7.5 GB | 4.95 GB | 65.7% | [link](https://huggingface.co/wpferrell/qwen3-4b-instruct-bigsmall) |
| Phi-3.5 Mini Instruct | 7.2 GB | 4.67 GB | 65.6% | [link](https://huggingface.co/wpferrell/phi-3.5-mini-instruct-bigsmall) |
| Llama 3.2 3B Instruct | 5.9 GB | 3.93 GB | 65.0% | [link](https://huggingface.co/wpferrell/llama-3.2-3b-instruct-bigsmall) |
| Qwen 2.5 3B Instruct | 5.8 GB | 3.81 GB | 65.7% | [link](https://huggingface.co/wpferrell/qwen2.5-3b-instruct-bigsmall) |
| Gemma 2 2B Instruct | 4.9 GB | 3.20 GB | 65.7% | [link](https://huggingface.co/wpferrell/gemma-2-2b-it-bigsmall) |
| Qwen 2.5 1.5B Instruct | 2.9 GB | 1.89 GB | 66.1% | [link](https://huggingface.co/wpferrell/qwen2.5-1.5b-instruct-bigsmall) |
| Llama 3.2 1B Instruct | 2.3 GB | 1.51 GB | **60.4%** | [link](https://huggingface.co/wpferrell/llama-3.2-1b-instruct-bigsmall) |
| Gemma 3 1B Instruct | 1.9 GB | 1.22 GB | 65.7% | [link](https://huggingface.co/wpferrell/gemma-3-1b-it-bigsmall) |
| Qwen 2.5 0.5B Instruct | 0.9 GB | 0.61 GB | 63.9% | [link](https://huggingface.co/wpferrell/qwen2.5-0.5b-instruct-bigsmall) |
| GPT-2 117M | 548 MB | 414 MB | 75.5% | [link](https://huggingface.co/wpferrell/gpt2-bigsmall) |
| Gemma 3 270M Instruct | 0.5 GB | 0.33 GB | 65.7% | [link](https://huggingface.co/wpferrell/gemma-3-270m-it-bigsmall) |
| Gemma 3 270M | 0.5 GB | 0.33 GB | 65.7% | [link](https://huggingface.co/wpferrell/gemma-3-270m-bigsmall) |

All entries lossless, bit-identical, md5-verified. Requires `bigsmall >= 3.0.0`.

---

## What's new in v3.x

- **Streaming compression** — compress 70B-class models with under 5 GB of working RAM.
- **GPU-accelerated decode** — NVIDIA (CUDA), AMD (ROCm), Apple Silicon (MPS).
- **KV cache compression** — reduce KV cache memory pressure during inference.
- **Low-VRAM streaming inference** — `BigSmallStreamingModel` uses up to 12x less VRAM than standard loading by decompressing one layer at a time.
- **CLI tools**: `bigsmall verify`, `bigsmall stat`, `bigsmall diff`, `bigsmall benchmark`, `bigsmall migrate`.
- **Codec auto-selection** — every tensor goes through the per-tensor registry; the smallest codec wins and ratios cannot regress.

---

## CLI

```
bigsmall compress model.safetensors        # compress
bigsmall decompress model.bs               # decompress
bigsmall info model.bs                     # show tensor + codec breakdown
bigsmall verify model.bs                   # full integrity check
bigsmall verify model.bs --fast            # header-only check (seconds)
bigsmall stat model.bs                     # per-tensor stats
bigsmall diff a.bs b.bs                    # structural diff
bigsmall benchmark model.bs                # ratio + speed + per-layer breakdown
bigsmall migrate model.bs                  # re-encode against current codec registry
```

`bigsmall migrate` re-encodes each tensor through the per-tensor auto-selection
registry. If a newer codec produces a smaller blob it replaces the old one; if
not, the original is kept byte-for-byte. The migrated file is therefore never
larger than the original, and every tensor's decompressed bytes are unchanged
(md5-verified).

---

## HuggingFace integration

```python
import bigsmall

bigsmall.compress_for_hub("mistralai/Mistral-7B-Instruct-v0.3", output_dir="./mistral_bs")
bigsmall.upload_to_hub("./mistral_bs", "you/mistral-7b-bigsmall")

state_dict = bigsmall.from_pretrained("you/mistral-7b-bigsmall")
```

---

## vLLM integration

```
pip install bigsmall[vllm]
```

```python
import bigsmall

bigsmall.vllm_serve("wpferrell/mistral-7b-instruct-bigsmall", port=8000)
```

Or decompress first and use vLLM normally:

```python
out_dir = bigsmall.vllm_decompress("wpferrell/mistral-7b-instruct-bigsmall")

from vllm import LLM
llm = LLM(model=str(out_dir))
outputs = llm.generate("Tell me about lossless compression.")
```

`vllm_decompress` and `vllm_serve` both accept a HuggingFace repo ID, a local
directory of `.bs` shards, or a single `.bs` file.

---

## Comparison

| Tool | BF16 Ratio | FP32 Ratio | Inference Overhead | Hardware |
|------|------------|------------|--------------------|----------|
| [ZipNN](https://arxiv.org/abs/2411.05239) | 67% | 83% | None (load-time only) | CPU |
| [DFloat11](https://arxiv.org/abs/2504.11651) | ~70% | BF16 only | ~2x at batch=1 | CUDA |
| **BigSmall** | **65.6%** | **75.5%** | **None** | **CPU + any GPU** |

Lower ratio = better compression. BigSmall BF16 ratio measured on Mistral 7B, FP32 on GPT-2. md5-verified lossless.

---

## Paper

**[BigSmall: Lossless Neural Network Weight Compression at the Joint Entropy Floor](https://github.com/wpferrell/Bigsmall/blob/main/paper.pdf)**

---

## License

BigSmall v3.1.0 and below: Apache License 2.0
BigSmall v3.2.0 and above: [Elastic License 2.0 (ELv2)](LICENSE)

Free for personal, academic, and internal commercial use.
For managed services or SaaS: wpferrell@gmail.com

See [LICENSING.md](LICENSING.md) for details.
