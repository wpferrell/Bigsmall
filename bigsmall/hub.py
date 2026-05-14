"""BigSmall HuggingFace Hub integration.

High-level functions for the round-trip:
    compress_for_hub(source, output_dir)  - compress any HF model dir or repo
    upload_to_hub(output_dir, repo_id)    - push compressed shards to the Hub
    from_pretrained(repo_or_path)         - download + decompress to state_dict

The `from_pretrained` here returns a torch state_dict, suitable for
`model.load_state_dict(...)`. For a more turnkey loader that returns a
transformers model object, see `bigsmall.integrations.huggingface.from_pretrained`.
"""
from __future__ import annotations

import glob
import os
import shutil
from pathlib import Path
from typing import Optional

from . import container, decoder, encoder, hub_index


# ---------------------- helpers ---------------------------------------------


def _is_repo_id(s: str) -> bool:
    """Heuristic: HF repo IDs are not existing local paths and contain no path
    separators (Windows backslash, no leading slash/dot). Accepts both legacy
    single-name IDs ("gpt2") and namespaced IDs ("user/model")."""
    if Path(s).exists():
        return False
    if "\\" in s:
        return False
    if s.startswith(".") or s.startswith("/") or s.startswith("~"):
        return False
    if ":" in s:  # drive letters like "C:" - clearly a path
        return False
    # Reject patterns that look like filenames (have suffix)
    if Path(s).suffix in {".safetensors", ".bs", ".json", ".bin"}:
        return False
    return True


def _materialise_source(source: str | Path) -> Path:
    """Return a local directory for `source`, downloading from the Hub if needed."""
    s = str(source)
    if _is_repo_id(s):
        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise ImportError(
                "compress_for_hub with a repo ID requires `huggingface_hub` "
                "(pip install huggingface_hub)."
            ) from e
        local_dir = snapshot_download(
            repo_id=s,
            allow_patterns=[
                "*.safetensors", "*.json", "*.txt", "*.model",
                "tokenizer*", "vocab*", "merges*", "special_tokens_map*",
            ],
        )
        return Path(local_dir)
    p = Path(source)
    if p.is_file():
        return p.parent
    if p.is_dir():
        return p
    raise FileNotFoundError(f"Source not found: {source}")


def _find_safetensors_shards(directory: Path) -> list[Path]:
    """Return all safetensors shards in directory.

    If a model.safetensors.index.json exists we use its file list to respect
    sharded layouts. Otherwise we glob *.safetensors directly.
    """
    idx = directory / "model.safetensors.index.json"
    if idx.exists():
        import json
        with open(idx, "r", encoding="utf-8") as f:
            data = json.load(f)
        shard_files = sorted(set(data["weight_map"].values()))
        return [directory / s for s in shard_files]
    single = directory / "model.safetensors"
    if single.exists():
        return [single]
    return sorted(directory.glob("*.safetensors"))


def _shard_output_name(src_shard: Path, shard_idx: int, total: int) -> str:
    """Map a safetensors shard filename to a .bs shard filename.

    Single-shard models become `model.bs`. Multi-shard models become
    `model-00001-of-NNNNN.bs` mirroring the HF convention.
    """
    if total == 1:
        return "model.bs"
    return f"model-{shard_idx:05d}-of-{total:05d}.bs"


# ---------------------- public API ------------------------------------------


def compress_for_hub(source: str | Path,
                     output_dir: str | Path,
                     mode: str = "balanced",
                     workers: Optional[int] = None,
                     include_configs: bool = True,
                     overwrite: bool = False) -> str:
    """Compress an entire HF model (local dir or repo ID) for Hub upload.

    Args:
        source: local model directory OR HF repo ID ("mistralai/Mistral-7B-Instruct-v0.3").
        output_dir: directory to write .bs shards + bigsmall.index.json.
        mode: bigsmall codec mode (storage|balanced|inference). Currently informational.
        workers: per-tensor worker count for encoding. None -> default.
        include_configs: copy config.json / tokenizer files alongside the .bs files.
        overwrite: clear output_dir first if it already exists.

    Returns:
        Absolute path to output_dir.
    """
    src_dir = _materialise_source(source)
    out_dir = Path(output_dir)
    if out_dir.exists():
        if not overwrite and any(out_dir.iterdir()):
            # Leave existing files in place; we'll overwrite per-file below.
            pass
        if overwrite:
            shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shards = _find_safetensors_shards(src_dir)
    if not shards:
        raise FileNotFoundError(f"No *.safetensors files found under {src_dir}")

    total = len(shards)
    bs_paths: list[Path] = []
    for i, shard in enumerate(shards, start=1):
        out_name = _shard_output_name(shard, i, total)
        dst = out_dir / out_name
        print(f"[bigsmall] compressing shard {i}/{total}: {shard.name} -> {out_name}", flush=True)
        encoder.compress(shard, dst, mode=mode, workers=workers)
        bs_paths.append(dst)

    hub_index.write_index(out_dir, bs_paths)

    if include_configs:
        for name in os.listdir(src_dir):
            sp = src_dir / name
            if not sp.is_file():
                continue
            if sp.suffix == ".safetensors":
                continue
            if name == "model.safetensors.index.json":
                continue
            dst = out_dir / name
            if dst.exists() and not overwrite:
                continue
            try:
                shutil.copy2(sp, dst)
            except OSError:
                pass

    return str(out_dir.resolve())


def upload_to_hub(output_dir: str | Path,
                  repo_id: str,
                  private: bool = False,
                  commit_message: str = "Upload BigSmall compressed model",
                  token: Optional[str] = None) -> str:
    """Upload a compress_for_hub() output directory to HuggingFace Hub.

    Creates the repo if it doesn't exist.

    Args:
        output_dir: directory containing .bs shards + bigsmall.index.json.
        repo_id: target HF repo ID, e.g. "wpferrell/mistral-7b-bigsmall".
        private: create as private if the repo doesn't exist.
        commit_message: commit message for the upload.
        token: HF auth token; falls back to env HF_TOKEN / cached login.

    Returns:
        The repo URL.
    """
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError as e:
        raise ImportError(
            "upload_to_hub requires `huggingface_hub` (pip install huggingface_hub)."
        ) from e

    out_dir = Path(output_dir)
    if not (out_dir / hub_index.INDEX_FILENAME).exists():
        raise FileNotFoundError(
            f"No bigsmall.index.json in {out_dir}. Run compress_for_hub first."
        )

    create_repo(repo_id, repo_type="model", private=private, token=token, exist_ok=True)

    api = HfApi(token=token)
    api.upload_folder(
        folder_path=str(out_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
        allow_patterns=["*.bs", "*.json", "*.txt", "*.model", "tokenizer*", "vocab*", "merges*"],
    )
    return f"https://huggingface.co/{repo_id}"


def _download_bigsmall_repo(repo_id: str, cache_dir: Optional[str] = None) -> Path:
    """Use huggingface_hub.snapshot_download to fetch .bs + index + configs."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError(
            "from_pretrained with a repo ID requires `huggingface_hub`."
        ) from e
    local = snapshot_download(
        repo_id=repo_id,
        cache_dir=cache_dir,
        allow_patterns=["*.bs", "*.json", "*.txt", "*.model",
                        "tokenizer*", "vocab*", "merges*", "special_tokens_map*"],
    )
    return Path(local)


def from_pretrained(repo_or_path: str | Path,
                    device: str = "cpu",
                    cache_dir: Optional[str] = None,
                    show_progress: bool = True) -> dict:
    """Download (if needed) and decompress a BigSmall model into a torch state_dict.

    Args:
        repo_or_path: HF repo ID ("user/name") OR a local path. The local path
                      may be a directory (multi-shard with bigsmall.index.json)
                      OR a single .bs file.
        device: device for the returned torch tensors.
        cache_dir: optional cache override (defaults to HF_HOME).
        show_progress: print per-shard progress.

    Returns:
        dict[str, torch.Tensor] - state_dict suitable for model.load_state_dict().
    """
    import torch  # noqa: F401  (imported for early failure if torch missing)

    s = str(repo_or_path)
    if _is_repo_id(s):
        local = _download_bigsmall_repo(s, cache_dir=cache_dir)
    else:
        local = Path(repo_or_path)

    if local.is_file():
        if show_progress:
            print(f"[bigsmall] decompressing {local.name}", flush=True)
        return decoder.load(local, device=device)

    if not local.is_dir():
        raise FileNotFoundError(f"Path not found: {local}")

    index_path = local / hub_index.INDEX_FILENAME
    if index_path.exists():
        index = hub_index.read_index(local)
        shards = hub_index.shard_paths_from_index(local, index=index)
    else:
        shards = sorted(local.glob("*.bs"))
        if not shards:
            raise FileNotFoundError(
                f"No bigsmall.index.json and no .bs files in {local}"
            )

    state_dict: dict = {}
    total = len(shards)
    for i, shard in enumerate(shards, start=1):
        if show_progress:
            print(f"[bigsmall] decompressing shard {i}/{total}: {shard.name}", flush=True)
        part = decoder.load(shard, device=device)
        # Detect duplicate keys across shards (HF disallows this; we error loudly)
        dup = set(part) & set(state_dict)
        if dup:
            raise ValueError(f"Duplicate tensor names across shards: {sorted(dup)[:5]}...")
        state_dict.update(part)

    return state_dict
