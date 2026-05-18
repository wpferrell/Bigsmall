"""GPU acceleration kernels for BigSmall.

Currently exposes one entry point:

    decode_bf16_parallel(blob, extras, n_weights) -> bytes

Auto-detects at decode time:
  - CUDA C extension (`bigsmall.kernels.ac_cuda`) if compiled and CUDA available.
  - Triton (`bigsmall.kernels.ac_triton`) if installed and CUDA available.
  - CPU fallback (`bigsmall.codecs.bf16_parallel.decode`) always works.

The fallback path is *always* correct — GPU kernels are optional speedups.
Environment overrides:
  BIGSMALL_FORCE_CPU=1   — disable GPU even if available (for benchmarking)
  BIGSMALL_FORCE_TRITON=1 — prefer Triton over CUDA-C even if both present
"""
from __future__ import annotations

import os
from typing import Optional


_GPU_BACKEND: Optional[str] = None   # None = not yet probed, "" = CPU only,
                                     # "cuda_c" or "triton" otherwise
_GPU_DECODE_FN = None                # populated on first probe


def _force_cpu() -> bool:
    return os.environ.get("BIGSMALL_FORCE_CPU", "").lower() in ("1", "true", "yes")


def _prefer_triton() -> bool:
    return os.environ.get("BIGSMALL_FORCE_TRITON", "").lower() in ("1", "true", "yes")


def _probe_backend() -> None:
    """Decide which GPU backend (if any) to use. Cached after first call."""
    global _GPU_BACKEND, _GPU_DECODE_FN
    if _GPU_BACKEND is not None:
        return

    if _force_cpu():
        _GPU_BACKEND = ""
        return

    # CUDA availability gate
    try:
        import torch
        if not torch.cuda.is_available():
            _GPU_BACKEND = ""
            return
    except Exception:
        _GPU_BACKEND = ""
        return

    # Try CUDA-C extension first (unless Triton preference set)
    if not _prefer_triton():
        try:
            from . import ac_cuda as _ac_cuda  # type: ignore
            _GPU_BACKEND = "cuda_c"
            _GPU_DECODE_FN = _ac_cuda.decode
            return
        except Exception:
            pass

    # Fall back to Triton
    try:
        from . import ac_triton as _ac_triton  # type: ignore
        _GPU_BACKEND = "triton"
        _GPU_DECODE_FN = _ac_triton.decode
        return
    except Exception:
        pass

    _GPU_BACKEND = ""


def use_gpu() -> bool:
    """Return True if a GPU backend is available and will be used."""
    _probe_backend()
    return bool(_GPU_BACKEND)


def backend_name() -> str:
    _probe_backend()
    return _GPU_BACKEND or "cpu"


def decode_bf16_parallel(blob: bytes, extras: dict, n_weights: int) -> bytes:
    """Decode a `bf16_parallel_v1` blob to raw BF16 bytes.

    Picks the fastest available backend (CUDA C ext > Triton > CPU). The
    CPU fallback is always correct and is used silently when no GPU
    backend is available.
    """
    _probe_backend()
    if _GPU_BACKEND and _GPU_DECODE_FN is not None:
        try:
            return _GPU_DECODE_FN(blob, extras, n_weights)
        except Exception:
            # If the GPU path errors at runtime, fall back to CPU rather than
            # crashing the load. We log via warnings on first occurrence so the
            # user knows their GPU path is broken.
            import warnings
            warnings.warn(
                f"BigSmall {_GPU_BACKEND} decoder raised; falling back to CPU. "
                "Set BIGSMALL_FORCE_CPU=1 to silence this warning.",
                RuntimeWarning,
            )
    from ..codecs import bf16_parallel
    return bf16_parallel.decode(blob, extras, n_weights)
