"""HuggingFace-streaming helpers added in v3.9.0.

Public entry points:

  - `compress_from_hub(repo_id, output_path, ...)`:
        Compress a HuggingFace model without keeping a full local copy of
        the raw safetensors. Each shard is downloaded to the HF cache via
        `hf_hub_download` (which streams in chunks under the hood) and
        then run through `compress_streaming`, so peak RAM stays at one
        tensor.

  - `decompress_layers(bs_path, layer_indices, ...)`:
        Convenience wrapper around `StreamingLoader` that returns only the
        named layers (and any tensors that depend on them via tied_ref).
        Useful for partial fine-tuning, layer analysis, and early-exit
        inference.

NOTE on "truly streaming from HF CDN without disk": that would require
custom HTTP range-request handling and re-implementing safetensors
loading on top of network buffers. The current implementation downloads
each shard to the standard HF cache (typical ~5-15 GB) before
compressing. This keeps the user-facing behaviour simple (a single
function call) and matches what `huggingface_hub` already does under
the hood.
"""
from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Optional


def compress_from_hub(
    repo_id: str,
    output_path: str | Path,
    *,
    token: Optional[str] = None,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    workers: Optional[int] = None,  # unused — streaming compress is serial
    progress: bool = True,
    enable_a5: bool = True,
    prefer_speed: bool = False,
) -> str:
    """Compress a HuggingFace model directly into a `.bs` shard set.

    Streams each shard through `compress_streaming` (peak RAM ≈ one tensor).
    Requires the `huggingface_hub` package.

    Args:
        repo_id: HF repo identifier, e.g. ``"meta-llama/Meta-Llama-3-8B"``.
        output_path: directory to write the resulting `.bs` shards into.
        token: optional HF token; falls back to `HF_TOKEN` env var.
        revision: optional git revision (commit, branch, or tag).
        cache_dir: forwarded to `hf_hub_download`; defaults to HF default.
        workers: kept for API parity with `compress()`; ignored — streaming
            mode is intentionally serial.
        progress: show tqdm progress per shard.
        enable_a5 / prefer_speed: forwarded to `auto_select_codec`.

    Returns:
        The output_path as a string.
    """
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as e:
        raise ImportError(
            "compress_from_hub requires huggingface_hub. "
            "Install with: pip install huggingface_hub"
        ) from e

    from .encoder import compress_streaming

    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    if token is None:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    api = HfApi()
    try:
        info = api.repo_info(repo_id, revision=revision, token=token)
    except Exception as e:
        raise RuntimeError(
            f"compress_from_hub: could not access repo {repo_id!r} "
            f"(token set: {token is not None}). Original error: {e}"
        ) from e

    safetensors_files = [
        s.rfilename for s in info.siblings
        if s.rfilename.endswith(".safetensors")
    ]
    if not safetensors_files:
        raise RuntimeError(
            f"compress_from_hub: {repo_id!r} contains no .safetensors files. "
            "BigSmall only compresses safetensors-format models."
        )

    for i, fname in enumerate(safetensors_files):
        if progress:
            print(f"[{i+1}/{len(safetensors_files)}] Downloading {fname}...", flush=True)
        local = hf_hub_download(
            repo_id=repo_id, filename=fname,
            token=token, revision=revision, cache_dir=cache_dir,
        )
        if progress:
            print(f"  Compressing {fname}...", flush=True)
        bs_name = Path(fname).stem + ".bs"
        bs_path = out_dir / bs_name
        compress_streaming(
            local, bs_path, progress=progress,
            enable_a5=enable_a5, prefer_speed=prefer_speed,
        )
        # Hint GC after each shard so caches and per-shard buffers are freed.
        gc.collect()

    return str(out_dir)


def decompress_layers(
    bs_path: str | Path,
    layer_indices: list[int],
    *,
    device: str = "cpu",
    dtype=None,
    include_non_layer: bool = False,
) -> dict[str, "object"]:
    """Decompress only the named transformer layers from a `.bs` file or shard set.

    Args:
        bs_path: path to a single `.bs` file OR a directory containing
            ``bigsmall.index.json`` plus shards.
        layer_indices: integer layer indices to load. Names that don't carry
            a recognisable layer index (embeddings, lm_head, final norm)
            are returned only when ``include_non_layer=True``.
        device: target device for the returned torch tensors.
        dtype: optional torch dtype override.
        include_non_layer: if True, also include non-layer tensors
            (embeddings, lm_head, final norm). Default False — used
            primarily for analysis / partial fine-tuning of body layers.

    Returns:
        Dict ``{tensor_name: torch.Tensor}``.
    """
    from .streaming import StreamingLoader

    out: dict[str, object] = {}
    with StreamingLoader(bs_path, device=device, dtype=dtype) as loader:
        if include_non_layer:
            out.update(loader.load_non_layer_tensors())
        for li in layer_indices:
            try:
                out.update(loader.load_layer(li))
            except IndexError:
                raise ValueError(
                    f"decompress_layers: layer {li} not present in this model "
                    f"(have {loader.layer_count()} layers)"
                )
    return out
