# BigSmall

**Run any model. No compromises.**

Mistral 7B is 14 GB. Your machine has 8 GB. Today your only option is quantization -- a degraded, worse version of the model. BigSmall changes that.

BigSmall compresses model weights **losslessly**. Mistral 7B goes from 14 GB to 9 GB. The streaming loader means you never need 9 GB free at once -- it decompresses one layer at a time, directly into VRAM, with a peak RAM footprint of under 2 GB. You run the **exact same model**. Bit-for-bit identical weights. No quality loss. No accuracy regression. No surprises.

```bash
pip install bigsmall
```

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

| | Quantization (4-bit) | BigSmall |
|--|--|--|
| Lossless? | No -- weights degraded | Yes -- bit-identical |
| Mistral 7B size | ~4 GB | **9 GB** |
| Peak RAM to load | ~4 GB | **< 2 GB** (streaming) |
| Inference speed | Slower on some hardware | Native (decompress once) |
| Fine-tuning safe? | No -- drift from quantized base | Yes -- clean base |
| Reproducible? | No | Yes |

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

Ready to use -- no compression step needed:

| Model | HuggingFace | Original | Compressed |
|-------|-------------|----------|------------|
| Mistral 7B Instruct v0.3 | [wpferrell/mistral-7b-instruct-bigsmall](https://huggingface.co/wpferrell/mistral-7b-instruct-bigsmall) | 14.2 GB | 9.3 GB |
| GPT-2 117M | [wpferrell/gpt2-bigsmall](https://huggingface.co/wpferrell/gpt2-bigsmall) | 548 MB | 414 MB |

---

## Streaming loader

The streaming loader lets you run models that don't fit in RAM or VRAM. It decompresses one transformer layer at a time, directly into the target device, and frees the previous layer before loading the next. Peak memory is `embeddings + one layer` -- typically under 2 GB even for 7B models.

```python
with bigsmall.StreamingLoader("wpferrell/mistral-7b-instruct-bigsmall", device="cuda") as loader:
    print(f"{loader.layer_count()} layers")

    # Load embeddings and non-layer tensors upfront (small)
    base = loader.load_non_layer_tensors()

    # Stream layers one at a time
    for layer_idx, layer_tensors in loader.iter_layers():
        # Previous layer already freed from memory
        # layer_tensors is on device, ready to use
        pass
```

---

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
```

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
