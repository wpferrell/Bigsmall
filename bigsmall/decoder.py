"""BigSmall decoder: .bs container -> tensors.

Public functions:
    decompress(src, dst=None) -> dict[str, np.ndarray]
        If dst is given, also writes a safetensors file at that path.
    decompress_delta(delta_src, base_src, dst=None) -> dict[str, np.ndarray]
    load(src, device='cpu') -> dict[str, torch.Tensor]
"""
from __future__ import annotations

import io
import struct
from pathlib import Path
from typing import Optional

import numpy as np

from . import container, formats
from .codecs import (
    bf16, bf16_rans, bf16_tans, bf16_sparsity, bf16_parallel, fp2_residual,
    fp32, fp16, fp8, fp4,
    special as special_codec, generic,
)
from .exceptions import BigSmallVersionError


_FORMAT_CODECS = {
    "fp32": fp32,
    "bf16": bf16,
    "fp16": fp16,
    "fp8":  fp8,
    "fp4":  fp4,
}

_DTYPE_TO_NUMPY = {
    "F32":  np.float32, "float32": np.float32,
    "F16":  np.float16, "float16": np.float16,
    "BF16": None,        "bfloat16": None,  # numpy has no native bf16 - return uint16 view
    "F8_E4M3": np.uint8, "float8_e4m3fn": np.uint8,
    "F8_E5M2": np.uint8, "float8_e5m2":  np.uint8,
    "I64": np.int64, "int64": np.int64, "I32": np.int32, "int32": np.int32,
    "I16": np.int16, "int16": np.int16, "I8": np.int8, "int8": np.int8,
    "U8": np.uint8, "uint8": np.uint8,
    "BOOL": np.bool_, "bool": np.bool_,
}


def _decode_blob(t: dict, blob: bytes) -> bytes:
    """Decode one tensor blob to raw bytes."""
    extras = t.get("extra") or {}
    codec = t["codec"]
    n_weights = int(np.prod(t["shape"])) if t["shape"] else 1

    if codec == "tied_ref":
        return None  # caller must resolve from master tensor
    if codec == "raw":
        return blob  # tensor stored uncompressed (tiny tensor short-circuit)
    if codec == "bf16_se_rans":
        return bf16_rans.decode(blob, extras, n_weights)
    if codec == "bf16_se_tans":
        return bf16_tans.decode(blob, extras, n_weights)
    if codec == "bf16_sparsity_v1":
        return bf16_sparsity.decode(blob, extras, n_weights)
    if codec == "fp2_residual_v1":
        return fp2_residual.decode(blob, extras, n_weights)
    if codec == "bf16_parallel_v1":
        # GPU-kernel infrastructure (opt-in via gpu_optimised=True at compress).
        # The decoder picks CPU/GPU automatically based on `bigsmall.kernels.use_gpu()`.
        from . import kernels as _kernels
        return _kernels.decode_bf16_parallel(blob, extras, n_weights)
    if codec == "special":
        # n_bytes is total tensor byte length
        # item_bytes from extras
        ib = extras.get("item_bytes", 1)
        n_bytes = n_weights * ib
        return special_codec.decode(blob, extras, n_bytes)
    if codec == "zstd":
        return generic.decode_zstd(blob, extras)
    if codec.endswith("_se_ac") or codec in ("fp8_cat_ac", "fp4_cat_ac"):
        # Format codec
        fmt = codec.split("_")[0]
        return _FORMAT_CODECS[fmt].decode(blob, extras, n_weights)
    if codec == "zstd_xor_delta":
        # Returns the XOR delta bytes - caller XORs with base
        return generic.decode_zstd(blob, extras)
    # Unknown codec name -- almost always means the file was produced by a
    # newer bigsmall release that introduced a codec this build doesn't
    # know how to decode. Surface the actionable upgrade message.
    from . import __version__ as _bs_version
    raise BigSmallVersionError(
        required="newer-than-installed",
        installed=_bs_version,
        detail=f"file references unknown codec {codec!r}",
    )


def decompress(src: str | Path, dst: Optional[str | Path] = None,
               progress: bool = True) -> dict[str, np.ndarray]:
    """Decompress a .bs file to a dict of {name: numpy ndarray}.

    If dst is given, also writes a safetensors file at that path.
    progress=True (default) shows a tqdm progress bar if tqdm is installed.
    """
    src = Path(src)
    header, data_offset = container.read_header(src)

    # Read all blobs
    with open(src, "rb") as f:
        f.seek(data_offset)
        all_data = f.read()

    out: dict[str, np.ndarray] = {}
    raw_by_name: dict[str, bytes] = {}

    pbar = None
    if progress:
        try:
            from tqdm.auto import tqdm
            pbar = tqdm(total=len(header["tensors"]), desc="decompress",
                        unit="tensor", dynamic_ncols=True)
        except ImportError:
            pbar = None

    # First pass: decode non-tied tensors
    for t in header["tensors"]:
        if t["codec"] == "tied_ref":
            continue
        blob = all_data[t["offset"]:t["offset"] + t["compressed_bytes"]]
        raw = _decode_blob(t, blob)
        raw_by_name[t["name"]] = raw
        if pbar is not None:
            pbar.set_postfix_str(t["name"][:48])
            pbar.update(1)

    # Second pass: resolve tied refs
    for t in header["tensors"]:
        if t["codec"] == "tied_ref":
            extras = t.get("extra") or {}
            master = extras["tied_to"]
            raw_by_name[t["name"]] = raw_by_name[master]
            if pbar is not None:
                pbar.update(1)

    # Third pass: convert raw bytes to numpy arrays of correct dtype/shape
    for t in header["tensors"]:
        raw = raw_by_name[t["name"]]
        out[t["name"]] = _raw_to_numpy(raw, t["dtype"], t["shape"])

    if pbar is not None:
        pbar.close()

    if dst is not None:
        _write_safetensors(dst, out, header)
    return out


def decompress_delta(delta_src: str | Path, base_src: str | Path,
                     dst: Optional[str | Path] = None,
                     progress: bool = True) -> dict[str, np.ndarray]:
    """Decompress a delta .bs file using a base safetensors or .bs file.

    Returns the reconstructed fine-tuned tensors as a dict.
    """
    delta_src = Path(delta_src); base_src = Path(base_src)
    # Load base raw bytes per tensor
    base_raw = _load_base_raw(base_src)

    header, data_offset = container.read_header(delta_src)
    with open(delta_src, "rb") as f:
        f.seek(data_offset)
        all_data = f.read()

    out: dict[str, np.ndarray] = {}

    pbar = None
    if progress:
        try:
            from tqdm.auto import tqdm
            pbar = tqdm(total=len(header["tensors"]), desc="delta-decompress",
                        unit="tensor", dynamic_ncols=True)
        except ImportError:
            pbar = None

    for t in header["tensors"]:
        blob = all_data[t["offset"]:t["offset"] + t["compressed_bytes"]]
        extras = t.get("extra") or {}
        if t["codec"] == "zstd_xor_delta":
            delta_bytes = generic.decode_zstd(blob, extras)
            base_bytes = base_raw[t["name"]]
            if len(base_bytes) != len(delta_bytes):
                raise ValueError(f"Delta size mismatch for {t['name']}")
            a = np.frombuffer(base_bytes, dtype=np.uint8)
            b = np.frombuffer(delta_bytes, dtype=np.uint8)
            raw = (a ^ b).tobytes()
        else:
            raw = _decode_blob(t, blob)
        out[t["name"]] = _raw_to_numpy(raw, t["dtype"], t["shape"])
        if pbar is not None:
            pbar.set_postfix_str(t["name"][:48])
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    if dst is not None:
        _write_safetensors(dst, out, header)
    return out


def _load_base_raw(base_src: Path) -> dict[str, bytes]:
    """Load raw bytes per tensor from a safetensors or .bs file."""
    base_src = Path(base_src)
    if base_src.suffix == ".bs":
        # Decompress base .bs and return raw bytes
        header, data_offset = container.read_header(base_src)
        with open(base_src, "rb") as f:
            f.seek(data_offset)
            data = f.read()
        raw_by_name: dict[str, bytes] = {}
        for t in header["tensors"]:
            if t["codec"] == "tied_ref":
                continue
            blob = data[t["offset"]:t["offset"] + t["compressed_bytes"]]
            raw_by_name[t["name"]] = _decode_blob(t, blob)
        # Resolve tied
        for t in header["tensors"]:
            if t["codec"] == "tied_ref":
                extras = t.get("extra") or {}
                raw_by_name[t["name"]] = raw_by_name[extras["tied_to"]]
        return raw_by_name
    # safetensors
    from safetensors import safe_open
    raw_by_name = {}
    with safe_open(str(base_src), framework="pt") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            try:
                import torch
                raw = t.contiguous().view(torch.uint8).cpu().numpy().tobytes()
            except Exception:
                raw = bytes(t.cpu().numpy().tobytes())
            raw_by_name[k] = raw
    return raw_by_name


def _raw_to_numpy(raw: bytes, dtype_str: str, shape: list[int]) -> np.ndarray:
    """Convert raw bytes + dtype + shape to numpy array.

    For BF16, numpy has no native dtype - we return a uint16 view of the bytes
    (caller can reinterpret as bfloat16 in torch via .view(torch.bfloat16)).
    """
    np_dtype = _DTYPE_TO_NUMPY.get(dtype_str)
    if np_dtype is None:
        # bf16 -> uint16 view
        np_dtype = np.uint16
    arr = np.frombuffer(raw, dtype=np_dtype)
    # Defensive: empty shape -> scalar
    if not shape:
        return arr
    return arr.reshape(shape)


def _write_safetensors(dst: Path, tensors: dict[str, np.ndarray], header: dict) -> None:
    """Write tensors back to a .safetensors file using torch.save_file."""
    import torch
    from safetensors.torch import save_file

    torch_tensors: dict[str, torch.Tensor] = {}
    for t_meta in header["tensors"]:
        name = t_meta["name"]
        dtype_str = t_meta["dtype"]
        arr = tensors[name]
        torch_tensors[name] = _numpy_to_torch(arr, dtype_str)
    save_file(torch_tensors, str(dst))


def _numpy_to_torch(arr: np.ndarray, dtype_str: str):
    """Convert decoded numpy array to torch tensor with correct dtype.

    BF16 round-trip uses uint16 -> torch.uint16 -> view bfloat16 to preserve bytes.
    """
    import torch
    if dtype_str in ("BF16", "bfloat16"):
        # arr is uint16
        t = torch.from_numpy(arr.astype(np.uint16, copy=False).copy())
        return t.view(torch.bfloat16)
    if dtype_str in ("F8_E4M3", "float8_e4m3fn"):
        t = torch.from_numpy(arr.astype(np.uint8, copy=False).copy())
        return t.view(torch.float8_e4m3fn) if hasattr(torch, "float8_e4m3fn") else t
    if dtype_str in ("F8_E5M2", "float8_e5m2"):
        t = torch.from_numpy(arr.astype(np.uint8, copy=False).copy())
        return t.view(torch.float8_e5m2) if hasattr(torch, "float8_e5m2") else t
    # Standard dtypes: numpy -> torch
    t = torch.from_numpy(arr.copy())
    return t


def load(src: str | Path, device: str = "cpu", progress: bool = True) -> dict:
    """Decompress and return tensors as torch.Tensors on the specified device."""
    import torch
    raw_dict = decompress(src, progress=progress)
    header, _ = container.read_header(Path(src))
    name_to_dtype = {t["name"]: t["dtype"] for t in header["tensors"]}
    out = {}
    for name, arr in raw_dict.items():
        out[name] = _numpy_to_torch(arr, name_to_dtype[name]).to(device)
    return out
