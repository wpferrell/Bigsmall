"""vLLM integration for BigSmall.

Two top-level helpers (also exposed as ``bigsmall.vllm_decompress`` and
``bigsmall.vllm_serve``) plus the low-level loader class.

Quick start::

    import bigsmall

    # Option 1: Serve directly from HuggingFace
    bigsmall.vllm_serve("wpferrell/mistral-7b-instruct-bigsmall", port=8000)

    # Option 2: Decompress first, then use vLLM normally
    out_dir = bigsmall.vllm_decompress("wpferrell/mistral-7b-instruct-bigsmall")
    from vllm import LLM
    llm = LLM(model=str(out_dir))

    # Option 3: Use BigSmallModelLoader directly (advanced)
    from bigsmall.integrations.vllm import get_loader_class
    from vllm import LLM
    BigSmallLoader = get_loader_class()
    llm = LLM(
        model="wpferrell/mistral-7b-instruct-bigsmall",
        load_format="auto",
        model_loader_extras={"loader_cls": BigSmallLoader},
    )

Both ``vllm_decompress`` and ``vllm_serve`` accept either:
    - A local path to a ``.bs`` file
    - A local directory containing ``.bs`` shards + ``bigsmall.index.json``
    - A HuggingFace repo ID (e.g. ``"wpferrell/mistral-7b-instruct-bigsmall"``)

Multi-shard models are materialised as a proper sharded safetensors directory
with ``model.safetensors.index.json`` alongside the per-shard files.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional

from .. import decoder, hub, hub_index


def _resolve_input(path_or_repo: str | Path) -> Path:
    """Return a local Path for a .bs file, a directory of .bs shards, or a HF repo ID.

    HF repo IDs are downloaded via ``hub._download_bigsmall_repo`` and the local
    snapshot directory is returned.
    """
    s = str(path_or_repo)
    if hub._is_repo_id(s):
        print(f"[bigsmall] downloading {s} from HuggingFace Hub", flush=True)
        return hub._download_bigsmall_repo(s)
    p = Path(path_or_repo)
    if not p.exists():
        raise FileNotFoundError(f"Not a local path or HF repo ID: {path_or_repo}")
    return p


def _safetensors_shard_name(shard_idx: int, total: int) -> str:
    if total == 1:
        return "model.safetensors"
    return f"model-{shard_idx:05d}-of-{total:05d}.safetensors"


def _write_safetensors(state_dict: dict, dst: Path) -> None:
    """Write a dict of {name: np.ndarray|torch.Tensor} to a safetensors file."""
    from safetensors.torch import save_file
    import torch
    tensors = {}
    for name, arr in state_dict.items():
        if isinstance(arr, torch.Tensor):
            tensors[name] = arr.contiguous().cpu()
        else:
            # numpy array - convert through torch, handling bf16 (uint16) views
            if hasattr(arr, "dtype") and str(arr.dtype) == "uint16":
                # bf16 stored as uint16 - reinterpret as bfloat16
                t = torch.from_numpy(arr.copy()).view(torch.bfloat16)
            else:
                t = torch.from_numpy(arr.copy())
            tensors[name] = t.contiguous()
    save_file(tensors, str(dst))


def _copy_config_files(src_dir: Path, out_dir: Path) -> None:
    """Copy non-tensor config files from src_dir to out_dir."""
    for f in src_dir.iterdir():
        if not f.is_file():
            continue
        if f.suffix in (".bs", ".safetensors"):
            continue
        if f.name in ("bigsmall.index.json", "model.safetensors.index.json"):
            continue
        try:
            shutil.copy2(f, out_dir / f.name)
        except OSError:
            pass


def decompress_to_temp(path_or_repo: str | Path,
                       config_dir: Optional[str | Path] = None,
                       output_dir: Optional[str | Path] = None) -> Path:
    """Materialise a BigSmall model as a HF-style safetensors directory.

    Args:
        path_or_repo: one of
            - a HuggingFace repo ID like ``"wpferrell/mistral-7b-instruct-bigsmall"``
            - a directory containing ``.bs`` shards plus ``bigsmall.index.json``
            - a single ``.bs`` file
        config_dir: optional directory to copy config / tokenizer files from.
            Defaults to the parent of ``path_or_repo`` (single file) or the
            directory itself (multi-shard / repo snapshot).
        output_dir: where to write the safetensors output. Defaults to a fresh
            ``tempfile.mkdtemp()``.

    Returns:
        Path to the directory containing ``model.safetensors`` (single shard) or
        ``model.safetensors.index.json`` + sharded safetensors files, plus all
        tokenizer / config files. Suitable to pass to vLLM's ``model=...`` arg.

    Examples::

        out = bigsmall.vllm_decompress("wpferrell/mistral-7b-instruct-bigsmall")
        from vllm import LLM
        llm = LLM(model=str(out))
    """
    local = _resolve_input(path_or_repo)

    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="bigsmall_vllm_"))
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(output_dir)

    # Single .bs file
    if local.is_file():
        cfg_dir = Path(config_dir) if config_dir is not None else local.parent
        print(f"[bigsmall] decompressing {local.name} -> {output_dir}", flush=True)
        tensors = decoder.decompress(local)
        _write_safetensors(tensors, output_dir / "model.safetensors")
        _copy_config_files(cfg_dir, output_dir)
        return output_dir

    if not local.is_dir():
        raise FileNotFoundError(f"Path not found: {local}")

    # Multi-shard directory (preferred: has bigsmall.index.json)
    cfg_dir = Path(config_dir) if config_dir is not None else local
    index_path = local / hub_index.INDEX_FILENAME
    if index_path.exists():
        index = hub_index.read_index(local)
        shards = hub_index.shard_paths_from_index(local, index=index)
    else:
        shards = sorted(local.glob("*.bs"))
        if not shards:
            raise FileNotFoundError(f"No .bs files in {local}")

    total = len(shards)
    weight_map: dict[str, str] = {}
    total_bytes = 0

    for i, shard in enumerate(shards, start=1):
        print(f"[bigsmall] decompressing shard {i}/{total}: {shard.name}", flush=True)
        tensors = decoder.decompress(shard)
        st_name = _safetensors_shard_name(i, total)
        st_path = output_dir / st_name
        _write_safetensors(tensors, st_path)
        total_bytes += st_path.stat().st_size
        for name in tensors:
            weight_map[name] = st_name

    if total > 1:
        # Write the HF-style sharded index alongside the safetensors shards
        st_index = {
            "metadata": {"total_size": total_bytes},
            "weight_map": weight_map,
        }
        with open(output_dir / "model.safetensors.index.json", "w", encoding="utf-8") as f:
            json.dump(st_index, f, indent=2)

    _copy_config_files(cfg_dir, output_dir)
    return output_dir


def get_loader_class():
    """Return a BigSmallModelLoader class compatible with the installed vLLM.

    vLLM's ModelLoader API has changed across versions; we attempt to subclass
    the modern interface (``vllm.model_executor.model_loader.base_loader.BaseModelLoader``).
    For maximum portability, prefer :func:`decompress_to_temp` and pass the
    resulting directory as ``model=`` to vLLM directly.
    """
    try:
        from vllm.model_executor.model_loader.base_loader import BaseModelLoader  # type: ignore
        from vllm.config import LoadConfig, ModelConfig  # type: ignore
    except Exception as e:
        raise ImportError(
            "vLLM is not installed or the loader API has changed. "
            "Use bigsmall.vllm_decompress() to materialise a HF directory and "
            "pass it to vLLM as `model=`. "
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
                # Multi-shard: merge all into one state_dict
                tensors: dict = {}
                for bf in sorted(bs_files):
                    tensors.update(decoder.load(bf, device="cpu"))
            else:
                tensors = decoder.load(bs_path, device="cpu")

            try:
                from vllm.model_executor.model_loader.utils import (
                    process_weights_after_loading,  # type: ignore
                )
            except Exception:
                process_weights_after_loading = None

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
                    import torch
                    with torch.no_grad():
                        param.data.copy_(tensor)
            if process_weights_after_loading is not None:
                process_weights_after_loading(model, model_config, target_device="cuda")
            return model

    return BigSmallModelLoader


def bigsmall_vllm_serve(path_or_repo: str | Path,
                        config_dir: Optional[str | Path] = None,
                        port: int = 8000,
                        output_dir: Optional[str | Path] = None,
                        **server_kwargs: Any) -> None:
    """Decompress a BigSmall model and launch a vLLM OpenAI-compatible server.

    Args:
        path_or_repo: HF repo ID, directory of ``.bs`` shards, or single ``.bs`` file.
        config_dir: optional directory to copy tokenizer / config files from.
        port: port for the OpenAI-compatible server.
        output_dir: where to write the materialised safetensors directory.
        **server_kwargs: forwarded as ``--key value`` to ``vllm.entrypoints.openai.api_server``.

    Examples::

        # Serve directly from HuggingFace
        bigsmall.vllm_serve("wpferrell/mistral-7b-instruct-bigsmall", port=8000)

        # Serve a local directory of .bs shards
        bigsmall.vllm_serve("./mistral_bs", port=8000, tensor_parallel_size=2)
    """
    out_dir = decompress_to_temp(path_or_repo,
                                 config_dir=config_dir,
                                 output_dir=output_dir)
    import subprocess, sys
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
           "--model", str(out_dir), "--port", str(port)]
    for k, v in server_kwargs.items():
        cmd.append(f"--{k.replace('_', '-')}")
        cmd.append(str(v))
    print("Launching:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
