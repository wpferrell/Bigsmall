"""vLLM integration: BigSmallModelLoader.

vLLM uses pluggable model loaders. The interface evolves between vLLM versions,
so this module provides three layers of integration:

1) decompress_to_temp(bs_path) -> tempdir   - works with any vLLM version,
   transparently presents a normal HF directory containing model.safetensors.

2) BigSmallModelLoader (vLLM ModelLoader subclass) - direct in-memory loader
   for vLLM versions that support custom loaders. Only imported lazily because
   vLLM is a large dependency.

3) bigsmall_vllm_serve(bs_path, **kwargs) - convenience launcher that prepares
   a temp dir and starts an OpenAI-compatible vLLM API server.

Reference: https://github.com/scitix/ZipServ-vLLM
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional

from .. import decoder, container


def decompress_to_temp(bs_path: str | Path,
                       config_dir: Optional[str | Path] = None,
                       output_dir: Optional[str | Path] = None) -> Path:
    """Decompress a .bs file into a HF-style directory.

    Returns the path to the directory containing model.safetensors plus configs.
    Suitable to pass to vLLM's `model=...` parameter.
    """
    bs_path = Path(bs_path)
    if config_dir is None:
        config_dir = bs_path.parent
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="bigsmall_vllm_"))
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    st_path = output_dir / "model.safetensors"
    decoder.decompress(bs_path, st_path)

    # Copy non-tensor config files
    for f in Path(config_dir).iterdir():
        if f.name == "model.safetensors" or f.suffix == ".bs":
            continue
        if f.is_file():
            try:
                shutil.copy2(f, output_dir / f.name)
            except OSError:
                pass
    return output_dir


def get_loader_class():
    """Return a BigSmallModelLoader class compatible with the installed vLLM.

    vLLM's ModelLoader API has changed; we attempt to subclass the modern
    interface (vllm.model_executor.model_loader.base_loader.BaseModelLoader).
    Falls back to a simple wrapper if unavailable.
    """
    try:
        from vllm.model_executor.model_loader.base_loader import BaseModelLoader  # type: ignore
        from vllm.config import LoadConfig, ModelConfig  # type: ignore
    except Exception as e:
        raise ImportError(
            "vLLM is not installed or the loader API has changed. "
            "Use bigsmall.integrations.vllm.decompress_to_temp() to materialise "
            "a HF directory and pass it to vLLM as `model=`. "
            f"Original error: {e}"
        )

    class BigSmallModelLoader(BaseModelLoader):
        """vLLM ModelLoader that decompresses .bs files on the fly."""

        def __init__(self, load_config: LoadConfig):  # type: ignore
            super().__init__(load_config)

        def download_model(self, model_config: ModelConfig) -> None:  # type: ignore
            return None

        def load_weights(self, model, model_config: ModelConfig):  # type: ignore
            bs_path = Path(model_config.model)
            if bs_path.is_dir():
                bs_files = list(bs_path.glob("*.bs"))
                if not bs_files:
                    raise FileNotFoundError(f"No .bs file in {bs_path}")
                bs_path = bs_files[0]
            tensors = decoder.load(bs_path, device="cpu")
            # Standard HF-style state_dict load
            try:
                from vllm.model_executor.model_loader.utils import (
                    process_weights_after_loading,  # type: ignore
                )
            except Exception:
                process_weights_after_loading = None

            # Use vLLM's weight loading utilities if present
            try:
                from vllm.model_executor.model_loader.weight_utils import (
                    default_weight_loader,  # type: ignore
                )
            except Exception:
                default_weight_loader = None

            params_dict = dict(model.named_parameters())
            for name, tensor in tensors.items():
                if name not in params_dict:
                    continue
                param = params_dict[name]
                if default_weight_loader is not None:
                    default_weight_loader(param, tensor)
                else:
                    with __import__("torch").no_grad():
                        param.data.copy_(tensor)
            if process_weights_after_loading is not None:
                process_weights_after_loading(model, model_config, target_device="cuda")
            return model

    return BigSmallModelLoader


def bigsmall_vllm_serve(bs_path: str | Path,
                        config_dir: Optional[str | Path] = None,
                        port: int = 8000,
                        **server_kwargs: Any) -> None:
    """Convenience: decompress to temp dir then start a vLLM OpenAI server.

    This is the most portable way to get vLLM serving a .bs model regardless
    of the loader API version - it just feeds vLLM a normal HF directory.
    """
    out_dir = decompress_to_temp(bs_path, config_dir=config_dir)
    try:
        # vLLM CLI entrypoint
        import subprocess, sys
        cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
               "--model", str(out_dir), "--port", str(port)]
        for k, v in server_kwargs.items():
            cmd.append(f"--{k.replace('_', '-')}")
            cmd.append(str(v))
        print("Launching:", " ".join(cmd))
        subprocess.run(cmd, check=True)
    finally:
        # leave decompressed dir on disk for inspection by default
        pass
