"""BigSmall encoder: compress safetensors -> .bs container.

Public functions:
    compress(src, dst, mode="balanced") -> str (output path)
    compress_delta(finetune, base, dst, mode="balanced") -> str
"""
from __future__ import annotations

import hashlib
import io
import os
import platform
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
from safetensors import safe_open

from . import codecs as codec_pkg
from .codecs import bf16, fp32, fp16, fp8, fp4, special as special_codec, generic
from . import container, formats, tensor_analysis as ta


def _maybe_tqdm(iterable, *, total=None, desc="", disable=False):
    """Wrap an iterable in tqdm if available; otherwise return it unchanged."""
    if disable:
        return iterable
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, total=total, desc=desc, unit="tensor", dynamic_ncols=True)


_FORMAT_CODECS = {
    "fp32": fp32,
    "bf16": bf16,
    "fp16": fp16,
    "fp8":  fp8,
    "fp4":  fp4,
}


# Tensors at or below this byte count are stored uncompressed (codec="raw").
# Header bytes spent on a codec stub dominate the body for tiny bias / norm
# entries, so the round-trip costs us bytes once you account for the JSON
# overhead in the .bs header. 512 keeps norm scales (typically 4 KiB or more
# at fp32) on the codec path while shortcutting biases that are O(hidden) at
# bf16, i.e. a few hundred bytes.
RAW_TINY_THRESHOLD = 512


def _default_workers() -> int:
    """Pick a sane default worker count.

    Platform default:
      Windows  -> 1 (multiprocessing here costs more than it saves and the
                     spawn-import side-effects regularly break user setups)
      Linux/macOS -> min(cpu_count, 8) using fork-style multiprocessing

    Override at any time with the BIGSMALL_WORKERS environment variable.
    """
    env = os.environ.get("BIGSMALL_WORKERS")
    if env is not None:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    if platform.system() == "Windows":
        return 1
    return min(os.cpu_count() or 1, 8)


def _encode_worker(job: tuple) -> tuple[int, bytes, str, dict, str | None]:
    """Module-level worker: encode one tensor.

    Picklable so it works with ProcessPoolExecutor on Windows (spawn).

    Args (job tuple):
        idx, kind, fmt, raw, item_bytes, shape

    Returns:
        (idx, blob, codec_name, extras, special_label)
    """
    idx, kind, fmt, raw, item_bytes, shape = job
    # Special codecs come first
    if kind == "lowcard":
        try:
            blob, extras = special_codec.encode_lowcard(raw, item_bytes)
            extras = {**extras, "special_kind": "lowcard"}
            return idx, blob, "special", extras, "lowcard"
        except Exception:
            kind = "generic"  # fall through to generic
    if kind == "wpe_delta":
        try:
            blob, extras = special_codec.encode_wpe_delta(raw, item_bytes, shape)
            extras = {**extras, "special_kind": "wpe_delta"}
            return idx, blob, "special", extras, "wpe_delta"
        except Exception:
            kind = "generic"

    # Generic / format codec
    if fmt == "raw":
        blob, extras = generic.encode_zstd(raw)
        return idx, blob, "zstd", extras, None
    mod = _FORMAT_CODECS[fmt]
    blob, extras = mod.encode(raw)
    codec_name = ta.codec_for_format(fmt)
    return idx, blob, codec_name, extras, None


def _detect_dominant_format(reader_keys, reader_dtypes) -> str:
    """Pick the format used by the bulk of model weights.

    We pick the format that covers the largest total byte count among float
    tensors. Mixed-precision models still get a single 'format' label in the
    header but each tensor's actual codec is chosen per-tensor below.
    """
    counts: dict[str, int] = {}
    for k, d in zip(reader_keys, reader_dtypes):
        try:
            f = formats.detect_format_from_dtype(d)
        except ValueError:
            continue
        counts[f] = counts.get(f, 0) + 1
    if not counts:
        return "bf16"  # default
    return max(counts, key=counts.get)


def _load_tensor_raw(f, key: str) -> tuple[bytes, list[int], str]:
    """Load tensor as raw little-endian bytes from safetensors file handle.

    safetensors stores bytes directly; we use get_slice to avoid a torch dep
    when possible, but fall back to get_tensor.
    """
    t = f.get_tensor(key)
    # Detect dtype name
    dtype_name = str(t.dtype)
    if "torch" in dtype_name:
        dtype_name = dtype_name.replace("torch.", "")
    # Convert to numpy view of raw bytes
    if hasattr(t, "untyped_storage"):
        # torch tensor path
        try:
            import torch
            raw = t.contiguous().view(torch.uint8).cpu().numpy().tobytes()
        except Exception:
            # fallback: numpy via float intermediate
            raw = t.contiguous().cpu().numpy().tobytes()
    else:
        raw = t.tobytes()
    shape = list(t.shape)
    return raw, shape, dtype_name


def _safetensors_dtype_name(t) -> str:
    """Return canonical dtype string for a safetensors tensor."""
    s = str(t.dtype)
    return s.replace("torch.", "")


def _gather_tensors(src: str | Path) -> tuple[list[dict], dict]:
    """Read all tensors from a safetensors file as raw bytes plus metadata.

    Returns (tensors_list, header_meta) where each tensor dict has:
        name, shape, dtype, item_bytes, raw, fmt
    """
    src = str(src)
    tensors: list[dict] = []
    meta: dict = {}

    with safe_open(src, framework="pt") as f:
        try:
            meta = f.metadata() or {}
        except Exception:
            meta = {}
        keys = list(f.keys())
        for k in keys:
            t = f.get_tensor(k)
            dtype_str = _safetensors_dtype_name(t)
            try:
                fmt = formats.detect_format_from_dtype(dtype_str)
            except ValueError:
                # Non-float tensor (e.g. int) - skip BigSmall handling, store raw
                fmt = "raw"
            shape = list(t.shape)
            # Get raw bytes
            try:
                import torch
                raw = t.contiguous().view(torch.uint8).cpu().numpy().tobytes()
            except Exception:
                raw = bytes(t.cpu().numpy().tobytes())
            item_bytes = len(raw) // max(1, int(np.prod(shape) if shape else 1))
            if not shape:
                item_bytes = len(raw)
            tensors.append({
                "name": k,
                "shape": shape,
                "dtype": dtype_str,
                "item_bytes": item_bytes,
                "raw": raw,
                "fmt": fmt,
            })
    return tensors, meta


def _detect_model_type(tensor_names: list[str]) -> str:
    """Return 'diffusion' if names look like a diffusion model, else 'llm'."""
    name_blob = " ".join(tensor_names[:200]).lower()
    diffusion_markers = ("unet", "vae", "controlnet", "double_block", "transformer_block",
                         "x_embedder", "joint_blocks", "time_embed")
    if any(m in name_blob for m in diffusion_markers):
        return "diffusion"
    return "llm"


def _encode_tensor_block(t: dict, fmt: str) -> tuple[bytes, str, dict, str | None]:
    """Encode a single non-tied non-special tensor.

    Returns (blob, codec_name, extras, special_kind=None).
    """
    if fmt == "raw":
        # Non-float tensor: zstd
        blob, extras = generic.encode_zstd(t["raw"])
        return blob, "zstd", extras, None
    codec_name = ta.codec_for_format(fmt)
    mod = _FORMAT_CODECS[fmt]
    blob, extras = mod.encode(t["raw"])
    return blob, codec_name, extras, None


def _encode_special(t: dict, kind: str) -> tuple[bytes, str, dict]:
    if kind == "lowcard":
        blob, extras = special_codec.encode_lowcard(t["raw"], t["item_bytes"])
        extras = {**extras, "special_kind": "lowcard"}
        return blob, "special", extras
    if kind == "wpe_delta":
        blob, extras = special_codec.encode_wpe_delta(t["raw"], t["item_bytes"], t["shape"])
        extras = {**extras, "special_kind": "wpe_delta"}
        return blob, "special", extras
    raise ValueError(f"Unknown special kind: {kind}")


def compress(src: str | Path, dst: str | Path, mode: str = "balanced",
             workers: Optional[int] = None, progress: bool = True,
             exclude_names: Optional[set[str]] = None) -> str:
    """Compress a safetensors file into a .bs container.

    Args:
        src: path to .safetensors
        dst: output path (.bs)
        mode: 'storage' | 'balanced' | 'inference' (currently same codec, different
              future hooks for chunking strategy)
        workers: number of parallel worker processes for per-tensor encoding.
                 None (default) -> min(cpu_count, 8) or BIGSMALL_WORKERS env var.
                 1 -> serial encoding (no process pool overhead).
        exclude_names: tensor names to skip entirely (used by `compress_for_hub`
                       for cross-shard tied-weight deduplication). Excluded
                       tensors do not appear in the resulting .bs file at all;
                       reconstruction is the caller's responsibility (via
                       `bigsmall.index.json:duplicate_map`).

    Returns: dst as string.
    """
    src = Path(src); dst = Path(dst)
    tensors, st_meta = _gather_tensors(src)
    if exclude_names:
        tensors = [t for t in tensors if t["name"] not in exclude_names]

    # Pick dominant format for the header label
    dominant_fmt = "bf16"
    counts: dict[str, int] = {}
    for t in tensors:
        if t["fmt"] != "raw":
            counts[t["fmt"]] = counts.get(t["fmt"], 0) + 1
    if counts:
        dominant_fmt = max(counts, key=counts.get)

    # Run special-tensor analysis (only on float tensors)
    decisions, tied_map = ta.analyze_tensors(tensors)
    model_type = _detect_model_type([t["name"] for t in tensors])

    if workers is None:
        workers = _default_workers()

    # Pre-compute md5 + build the list of encode jobs (parallelizable indices only).
    raw_md5s: list[str] = [hashlib.md5(t["raw"]).hexdigest() for t in tensors]

    jobs: list[tuple] = []
    encoded: dict[int, tuple[bytes, str, dict, str | None]] = {}

    for i, t in enumerate(tensors):
        kind = decisions[i]["kind"]
        if kind == "tied":
            master_idx = decisions[i]["tied_to"]
            encoded[i] = (b"", "tied_ref", {"tied_to": tensors[master_idx]["name"]}, "tied")
            continue
        if len(t["raw"]) < RAW_TINY_THRESHOLD:
            encoded[i] = (t["raw"], "raw", {"n_bytes": len(t["raw"])}, None)
            continue
        jobs.append((i, kind, t["fmt"], t["raw"], t["item_bytes"], t["shape"]))

    pbar = None
    if progress:
        try:
            from tqdm.auto import tqdm
            pbar = tqdm(total=len(tensors), desc="compress", unit="tensor",
                        dynamic_ncols=True)
        except ImportError:
            pbar = None

    raw_bytes_seen = 0
    compressed_bytes_seen = 0

    def _account_progress(idx, blob, t):
        nonlocal raw_bytes_seen, compressed_bytes_seen
        if pbar is None:
            return
        raw_bytes_seen += len(t["raw"])
        compressed_bytes_seen += len(blob)
        ratio = (compressed_bytes_seen / raw_bytes_seen * 100.0) if raw_bytes_seen else 0.0
        pbar.set_postfix_str(f"{t['name'][:48]} ratio={ratio:.1f}%")
        pbar.update(1)

    # Account tied entries (no compute) upfront. Raw-tiny entries skip the
    # worker pool entirely; account them here too so the progress bar matches
    # the actual write order.
    for i, t in enumerate(tensors):
        if decisions[i]["kind"] == "tied":
            _account_progress(i, b"", t)
        elif i in encoded:
            blob, codec_name, _extras, _special = encoded[i]
            if codec_name == "raw":
                _account_progress(i, blob, t)

    if workers <= 1 or len(jobs) <= 1:
        # Serial path - avoid process pool overhead for tiny models / tests
        for job in jobs:
            idx, blob, codec_name, extras, special_label = _encode_worker(job)
            encoded[idx] = (blob, codec_name, extras, special_label)
            _account_progress(idx, blob, tensors[idx])
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for fut in as_completed([pool.submit(_encode_worker, j) for j in jobs]):
                idx, blob, codec_name, extras, special_label = fut.result()
                encoded[idx] = (blob, codec_name, extras, special_label)
                _account_progress(idx, blob, tensors[idx])

    if pbar is not None:
        pbar.close()

    # Assemble container in original tensor order so .bs layout is deterministic
    data_buf = io.BytesIO()
    header_tensors: list[dict] = []
    offset = 0
    for i, t in enumerate(tensors):
        blob, codec_name, extras, special_label = encoded[i]
        data_buf.write(blob)
        header_tensors.append({
            "name": t["name"],
            "shape": t["shape"],
            "dtype": t["dtype"],
            "codec": codec_name,
            "special": special_label,
            "compressed_bytes": len(blob),
            "offset": offset,
            "md5": raw_md5s[i],
            "extra": extras or None,
        })
        offset += len(blob)

    header = {
        "format": dominant_fmt,
        "mode": mode,
        "model_type": model_type,
        "base_model": None,
        "tensor_count": len(tensors),
        "tensors": header_tensors,
        "safetensors_metadata": st_meta or None,
    }
    container.write_container(dst, header, data_buf.getvalue())
    return str(dst)


# ---------------------- Delta compression -----------------------------------


def _delta_worker(job: tuple) -> tuple[int, bytes, str, dict, str | None]:
    """Module-level worker for delta encoding. Picklable."""
    idx, is_delta, payload, fmt = job
    if is_delta:
        blob, extras = generic.encode_zstd(payload, level=19)
        extras = {**extras, "is_xor_delta": True}
        return idx, blob, "zstd_xor_delta", extras, "delta"
    if fmt == "raw":
        blob, extras = generic.encode_zstd(payload)
        return idx, blob, "zstd", extras, None
    mod = _FORMAT_CODECS[fmt]
    blob, extras = mod.encode(payload)
    return idx, blob, ta.codec_for_format(fmt), extras, None


def compress_delta(finetune_src: str | Path, base_src: str | Path,
                   dst: str | Path, mode: str = "balanced",
                   workers: Optional[int] = None,
                   progress: bool = True) -> str:
    """Compress a fine-tuned model as the delta against a base model.

    XOR base raw bytes with finetune raw bytes per tensor. The XOR has many
    leading zero bytes when weights are similar -> compresses far better.

    Tensors present only in the finetune are stored standalone (no XOR).
    Tensors with mismatched shape/dtype between base and finetune are stored
    standalone (no XOR).
    """
    finetune_src = Path(finetune_src); base_src = Path(base_src); dst = Path(dst)
    base_tensors, _ = _gather_tensors(base_src)
    base_index = {t["name"]: i for i, t in enumerate(base_tensors)}

    ft_tensors, st_meta = _gather_tensors(finetune_src)

    # Pick dominant format
    counts: dict[str, int] = {}
    for t in ft_tensors:
        if t["fmt"] != "raw":
            counts[t["fmt"]] = counts.get(t["fmt"], 0) + 1
    dominant_fmt = max(counts, key=counts.get) if counts else "bf16"

    if workers is None:
        workers = _default_workers()

    raw_md5s: list[str] = [hashlib.md5(t["raw"]).hexdigest() for t in ft_tensors]

    # Build jobs - precompute XOR delta payload sequentially (cheap, just memory)
    jobs: list[tuple] = []
    for i, t in enumerate(ft_tensors):
        is_delta = False
        payload = t["raw"]
        if t["name"] in base_index:
            bt = base_tensors[base_index[t["name"]]]
            if (bt["shape"] == t["shape"] and bt["dtype"] == t["dtype"]
                    and len(bt["raw"]) == len(t["raw"])):
                a = np.frombuffer(t["raw"], dtype=np.uint8)
                b = np.frombuffer(bt["raw"], dtype=np.uint8)
                payload = (a ^ b).tobytes()
                is_delta = True
        jobs.append((i, is_delta, payload, t["fmt"]))

    encoded: dict[int, tuple[bytes, str, dict, str | None]] = {}

    pbar = None
    if progress:
        try:
            from tqdm.auto import tqdm
            pbar = tqdm(total=len(jobs), desc="delta", unit="tensor",
                        dynamic_ncols=True)
        except ImportError:
            pbar = None

    raw_bytes_seen = 0
    compressed_bytes_seen = 0

    def _account(idx, blob):
        nonlocal raw_bytes_seen, compressed_bytes_seen
        if pbar is None:
            return
        t = ft_tensors[idx]
        raw_bytes_seen += len(t["raw"])
        compressed_bytes_seen += len(blob)
        ratio = (compressed_bytes_seen / raw_bytes_seen * 100.0) if raw_bytes_seen else 0.0
        pbar.set_postfix_str(f"{t['name'][:48]} ratio={ratio:.1f}%")
        pbar.update(1)

    if workers <= 1 or len(jobs) <= 1:
        for job in jobs:
            idx, blob, codec_name, extras, special_label = _delta_worker(job)
            encoded[idx] = (blob, codec_name, extras, special_label)
            _account(idx, blob)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for fut in as_completed([pool.submit(_delta_worker, j) for j in jobs]):
                idx, blob, codec_name, extras, special_label = fut.result()
                encoded[idx] = (blob, codec_name, extras, special_label)
                _account(idx, blob)

    if pbar is not None:
        pbar.close()

    data_buf = io.BytesIO()
    header_tensors: list[dict] = []
    offset = 0
    for i, t in enumerate(ft_tensors):
        blob, codec_name, extras, special_label = encoded[i]
        data_buf.write(blob)
        header_tensors.append({
            "name": t["name"],
            "shape": t["shape"],
            "dtype": t["dtype"],
            "codec": codec_name,
            "special": special_label,
            "compressed_bytes": len(blob),
            "offset": offset,
            "md5": raw_md5s[i],
            "extra": extras or None,
        })
        offset += len(blob)

    header = {
        "format": dominant_fmt,
        "mode": mode,
        "model_type": "delta",
        "base_model": str(base_src),
        "tensor_count": len(ft_tensors),
        "tensors": header_tensors,
        "safetensors_metadata": st_meta or None,
    }
    container.write_container(dst, header, data_buf.getvalue())
    return str(dst)
