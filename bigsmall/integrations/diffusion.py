"""Diffusion model support: 4D conv tensor handling, FLUX/SDXL/DiT auto-detect.

The standard BigSmall encoder works on raw bytes regardless of tensor rank,
so 4D conv weights compress correctly out of the box. This module provides:

  - is_diffusion_model(safetensors_path) -> bool
  - compress_diffusion(src, dst, mode="balanced") -> str
        Wrapper around encoder.compress that sets model_type="diffusion".
  - load_pipeline(bs_path, pipe_class) -> diffusers pipeline

The actual codec path is identical - 4D tensors get flattened to bytes by the
codec which already handles arbitrary shapes. This module exists so users have
a clear diffusion-aware entry point.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional

from .. import encoder, decoder, container


_DIFFUSION_MARKERS = (
    "unet", "vae", "controlnet", "joint_blocks", "double_block",
    "transformer_block", "x_embedder", "time_embed", "time_text_embed",
    "context_embedder", "single_blocks",
)


def is_diffusion_model(safetensors_path: str | Path) -> bool:
    """Return True if the safetensors file looks like a diffusion model."""
    from safetensors import safe_open
    with safe_open(str(safetensors_path), framework="pt") as f:
        keys = list(f.keys())[:200]
    name_blob = " ".join(keys).lower()
    return any(m in name_blob for m in _DIFFUSION_MARKERS)


def compress_diffusion(src: str | Path, dst: str | Path, mode: str = "balanced") -> str:
    """Compress a diffusion model. Auto-handles 4D conv tensors."""
    out = encoder.compress(src, dst, mode=mode)
    # Patch model_type in header if not already set
    header, _ = container.read_header(Path(out))
    if header.get("model_type") != "diffusion":
        # Re-write header with corrected model_type
        with open(out, "rb") as f:
            f.seek(0)
            from struct import unpack
            magic = f.read(4)
            version, = unpack("<H", f.read(2))
            hdr_len, = unpack("<I", f.read(4))
            f.read(hdr_len)
            data = f.read()
        header["model_type"] = "diffusion"
        container.write_container(out, header, data)
    return out


def decompress_diffusion(bs_path: str | Path, dst: str | Path | None = None):
    """Decompress a diffusion .bs back to safetensors (or dict)."""
    return decoder.decompress(bs_path, dst)


def load_pipeline(bs_path: str | Path, pipe_class=None,
                  config_dir: Optional[str | Path] = None, **kwargs):
    """Load a diffusers pipeline from a .bs file by decompressing to temp dir.

    pipe_class: e.g. diffusers.StableDiffusionPipeline. If None, attempt
                AutoPipelineForText2Image.
    """
    bs_path = Path(bs_path)
    if config_dir is None:
        config_dir = bs_path.parent

    tmp = Path(tempfile.mkdtemp(prefix="bigsmall_diff_"))
    st_path = tmp / "model.safetensors"
    decoder.decompress(bs_path, st_path)

    # Copy config files
    for f in Path(config_dir).iterdir():
        if f.name == "model.safetensors" or f.suffix == ".bs":
            continue
        if f.is_file():
            try:
                shutil.copy2(f, tmp / f.name)
            except OSError:
                pass

    if pipe_class is None:
        try:
            from diffusers import AutoPipelineForText2Image
            pipe_class = AutoPipelineForText2Image
        except ImportError:
            raise ImportError("diffusers is required for load_pipeline")

    return pipe_class.from_pretrained(str(tmp), **kwargs)
