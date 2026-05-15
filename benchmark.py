"""
BigSmall benchmark script â€” reproducible head-to-head compression benchmarks.

Usage:
    python benchmark.py --model gpt2
    python benchmark.py --model mistral
    python benchmark.py --model qwen7b
    python benchmark.py --model qwen14b
    python benchmark.py --all

Output: formatted table + benchmark_results.json
"""
import argparse
import json
import os
import sys
import time
import tempfile
import hashlib
import tracemalloc
from pathlib import Path

# Make sure local bigsmall is importable
sys.path.insert(0, str(Path(__file__).parent))

import bigsmall
from huggingface_hub import snapshot_download

MODELS = {
    "gpt2": {
        "hf_id": "openai-community/gpt2",
        "description": "GPT-2 117M (FP32)",
    },
    "gpt2-bf16": {
        "hf_id": "openai-community/gpt2",
        "description": "GPT-2 117M (BF16 â€” convert after download)",
        "convert_bf16": True,
    },
    "mistral": {
        "hf_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "description": "Mistral 7B Instruct v0.3 (BF16)",
    },
    "qwen7b": {
        "hf_id": "Qwen/Qwen2.5-7B-Instruct",
        "description": "Qwen 2.5 7B Instruct (BF16)",
    },
    "qwen14b": {
        "hf_id": "Qwen/Qwen2.5-14B-Instruct",
        "description": "Qwen 2.5 14B Instruct (BF16)",
    },
}


def get_dir_size(path):
    total = 0
    for f in Path(path).rglob("*.safetensors"):
        total += f.stat().st_size
    return total


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def benchmark_model(model_key, cache_dir=None):
    cfg = MODELS[model_key]
    print(f"\n{'='*60}")
    print(f"Benchmarking: {cfg['description']}")
    print(f"{'='*60}")

    cache_dir = Path(cache_dir or tempfile.gettempdir()) / f"bigsmall_bench_{model_key}"
    orig_dir = cache_dir / "original"
    comp_dir = cache_dir / "compressed"
    orig_dir.mkdir(parents=True, exist_ok=True)
    comp_dir.mkdir(parents=True, exist_ok=True)

    # Download
    if not any(orig_dir.glob("*.safetensors")):
        print(f"Downloading {cfg['hf_id']}...")
        snapshot_download(cfg["hf_id"], local_dir=str(orig_dir),
                          ignore_patterns=["*.bin", "*.pt", "flax_model*", "tf_model*"])
    else:
        print("Using cached download.")

    orig_bytes = get_dir_size(orig_dir)
    print(f"Original size: {orig_bytes / 1e9:.2f} GB")

    # Compress
    print("Compressing...")
    t0 = time.perf_counter()
    bigsmall.compress_for_hub(str(orig_dir), str(comp_dir), progress=False)
    compress_time = time.perf_counter() - t0

    comp_bytes = sum(f.stat().st_size for f in comp_dir.rglob("*.bs"))
    ratio = comp_bytes / orig_bytes * 100
    print(f"Compressed size: {comp_bytes / 1e9:.2f} GB ({ratio:.1f}%) in {compress_time:.1f}s")

    # Decompress
    print("Decompressing (measuring time)...")
    decomp_dir = cache_dir / "decompressed"
    decomp_dir.mkdir(exist_ok=True)
    t0 = time.perf_counter()
    bigsmall.from_pretrained(str(comp_dir), progress=False)
    decompress_time = time.perf_counter() - t0
    print(f"Decompression time: {decompress_time:.1f}s")

    # Verify losslessness
    print("Verifying losslessness...")
    all_verified = True
    for bs_file in sorted(comp_dir.glob("*.bs")):
        ok = bigsmall.verify(str(bs_file))
        if not ok:
            print(f"  FAIL: {bs_file.name}")
            all_verified = False
    print(f"Lossless: {'YES' if all_verified else 'NO - FAILURE'}")

    # Streaming peak RAM
    print("Measuring streaming peak RAM...")
    tracemalloc.start()
    loader = bigsmall.StreamingLoader(str(comp_dir), device="cpu")
    loader.load_non_layer_tensors()
    peak_per_layer = 0
    for _, tensors in loader.iter_layers(progress=False):
        current, peak = tracemalloc.get_traced_memory()
        peak_per_layer = max(peak_per_layer, current)
        break  # one layer is enough to measure
    tracemalloc.stop()
    streaming_peak_mb = peak_per_layer / 1e6

    result = {
        "model": model_key,
        "description": cfg["description"],
        "original_gb": round(orig_bytes / 1e9, 2),
        "compressed_gb": round(comp_bytes / 1e9, 2),
        "ratio_pct": round(ratio, 1),
        "compress_time_s": round(compress_time, 1),
        "decompress_time_s": round(decompress_time, 1),
        "lossless": all_verified,
        "streaming_peak_mb": round(streaming_peak_mb, 0),
    }
    return result


def print_table(results):
    print(f"\n{'='*90}")
    print("BIGSMALL BENCHMARK RESULTS")
    print(f"{'='*90}")
    header = f"{'Model':<25} {'Orig':>8} {'Comp':>8} {'Ratio':>7} {'Comp(s)':>8} {'Decomp(s)':>10} {'Peak RAM':>10} {'Lossless':>9}"
    print(header)
    print("-" * 90)
    for r in results:
        print(f"{r['description']:<25} {r['original_gb']:>7.2f}G {r['compressed_gb']:>7.2f}G "
              f"{r['ratio_pct']:>6.1f}% {r['compress_time_s']:>8.1f} {r['decompress_time_s']:>10.1f} "
              f"{r['streaming_peak_mb']:>8.0f}MB {'YES' if r['lossless'] else 'NO':>9}")
    print(f"{'='*90}")


def main():
    parser = argparse.ArgumentParser(description="BigSmall compression benchmark")
    parser.add_argument("--model", choices=list(MODELS.keys()) + ["all"], default="gpt2")
    parser.add_argument("--cache-dir", default=None, help="Directory to cache downloads")
    parser.add_argument("--output", default="benchmark_results.json")
    args = parser.parse_args()

    models_to_run = list(MODELS.keys()) if args.model == "all" else [args.model]

    results = []
    for model_key in models_to_run:
        try:
            result = benchmark_model(model_key, cache_dir=args.cache_dir)
            results.append(result)
        except Exception as e:
            print(f"ERROR benchmarking {model_key}: {e}")
            results.append({"model": model_key, "error": str(e)})

    print_table([r for r in results if "error" not in r])

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
