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

import base64
import glob
import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Optional

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


def _scan_cross_shard_duplicates(shards: list[Path]) -> dict[str, dict]:
    """Scan every tensor across `shards` for md5-identical duplicates.

    The first occurrence (by shard order, then tensor order within a shard) of
    each (shape, dtype, md5) key is the master; later occurrences are recorded
    as duplicates pointing to the master.

    Returns:
        `{dup_name: {"master": master_name}}` -- the same shape used by
        `bigsmall.index.json:duplicate_map`. Empty if no duplicates.
    """
    import hashlib
    from safetensors import safe_open

    seen: dict[tuple, str] = {}
    duplicates: dict[str, dict] = {}
    for shard in shards:
        with safe_open(str(shard), framework="pt") as f:
            for name in f.keys():
                t = f.get_tensor(name)
                shape = tuple(t.shape)
                dtype = str(t.dtype).replace("torch.", "")
                try:
                    import torch
                    raw = t.contiguous().view(torch.uint8).cpu().numpy().tobytes()
                except Exception:
                    raw = bytes(t.cpu().numpy().tobytes())
                if len(raw) < 1024:
                    # Tiny tensors are below the threshold where saving a copy
                    # offsets the duplicate-map overhead. Match the in-shard
                    # tied-detection threshold from tensor_analysis.
                    continue
                key = (shape, dtype, hashlib.md5(raw).hexdigest())
                if key in seen:
                    duplicates[name] = {"master": seen[key]}
                else:
                    seen[key] = name
    return duplicates


def compress_for_hub(source: str | Path,
                     output_dir: str | Path,
                     mode: str = "balanced",
                     workers: Optional[int] = None,
                     include_configs: bool = True,
                     overwrite: bool = False,
                     dedupe_cross_shard: bool = True) -> str:
    """Compress an entire HF model (local dir or repo ID) for Hub upload.

    Args:
        source: local model directory OR HF repo ID ("mistralai/Mistral-7B-Instruct-v0.3").
        output_dir: directory to write .bs shards + bigsmall.index.json.
        mode: bigsmall codec mode (storage|balanced|inference). Currently informational.
        workers: per-tensor worker count for encoding. None -> default.
        include_configs: copy config.json / tokenizer files alongside the .bs files.
        overwrite: clear output_dir first if it already exists.
        dedupe_cross_shard: scan all tensors across all shards for md5
            duplicates (e.g. lm_head tied to embed_tokens) and store only one
            copy. The duplicate aliases live in `bigsmall.index.json`.

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

    duplicate_map: dict[str, dict] = {}
    if dedupe_cross_shard:
        duplicate_map = _scan_cross_shard_duplicates(shards)
        if duplicate_map:
            print(
                f"[bigsmall] found {len(duplicate_map)} cross-shard tied tensors; "
                f"will dedupe ({sorted(duplicate_map)[:3]}...)",
                flush=True,
            )

    exclude_set = set(duplicate_map.keys())

    total = len(shards)
    bs_paths: list[Path] = []
    for i, shard in enumerate(shards, start=1):
        out_name = _shard_output_name(shard, i, total)
        dst = out_dir / out_name
        print(f"[bigsmall] compressing shard {i}/{total}: {shard.name} -> {out_name}", flush=True)
        encoder.compress(shard, dst, mode=mode, workers=workers,
                         exclude_names=exclude_set)
        bs_paths.append(dst)

    hub_index.write_index(out_dir, bs_paths, duplicate_map=duplicate_map)

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


_UPLOAD_PATTERNS = (
    "*.bs", "*.json", "*.txt", "*.model",
    "tokenizer*", "vocab*", "merges*", "special_tokens_map*",
    "README.md", "README",
)


def _iter_upload_files(out_dir: Path) -> list[Path]:
    """Return the local files in `out_dir` that match the upload allowlist."""
    import fnmatch
    selected: list[Path] = []
    for p in sorted(out_dir.iterdir()):
        if not p.is_file():
            continue
        if any(fnmatch.fnmatch(p.name, pat) for pat in _UPLOAD_PATTERNS):
            selected.append(p)
    return selected


def _remote_file_sizes(api, repo_id: str) -> dict[str, int]:
    """Return {filename: size_bytes} for files already in the HF repo. Empty if missing."""
    try:
        info = api.repo_info(repo_id=repo_id, repo_type="model", files_metadata=True)
    except Exception:
        return {}
    out: dict[str, int] = {}
    for s in getattr(info, "siblings", None) or []:
        name = getattr(s, "rfilename", None)
        size = getattr(s, "size", None)
        if name is not None and size is not None:
            out[name] = int(size)
    return out


def _cleanup_dirs(*dirs: Path | str | None) -> None:
    for d in dirs:
        if d is None:
            continue
        p = Path(d)
        if p.exists() and p.is_dir():
            shutil.rmtree(p, ignore_errors=True)


def upload_to_hub(output_dir: str | Path,
                  repo_id: str,
                  private: bool = False,
                  commit_message: str = "Upload BigSmall compressed model",
                  token: Optional[str] = None,
                  cleanup: bool = False,
                  source_dir: Optional[str | Path] = None) -> str:
    """Upload a compress_for_hub() output directory to HuggingFace Hub.

    Creates the repo if it doesn't exist. Resumable: files already on the Hub
    with matching size are skipped.

    Args:
        output_dir: directory containing .bs shards + bigsmall.index.json.
        repo_id: target HF repo ID, e.g. "wpferrell/mistral-7b-bigsmall".
        private: create as private if the repo doesn't exist.
        commit_message: commit message for the upload.
        token: HF auth token; falls back to env HF_TOKEN / cached login.
        cleanup: if True, delete `output_dir` (and `source_dir` if provided) on success.
        source_dir: optional source model dir to also remove on cleanup.

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
    remote_sizes = _remote_file_sizes(api, repo_id)

    local_files = _iter_upload_files(out_dir)
    to_upload: list[Path] = []
    for f in local_files:
        local_size = f.stat().st_size
        remote = remote_sizes.get(f.name)
        if remote is not None and remote == local_size:
            print(f"[bigsmall] already uploaded: {f.name}", flush=True)
            continue
        to_upload.append(f)

    for f in to_upload:
        print(f"[bigsmall] uploading {f.name} ({f.stat().st_size:,} bytes)", flush=True)
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=f.name,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"{commit_message}: {f.name}",
        )

    if cleanup:
        _cleanup_dirs(out_dir, source_dir)

    return f"https://huggingface.co/{repo_id}"


def upload_to_hub_lfs(local_dir: str | Path,
                      repo_id: str,
                      token: Optional[str] = None,
                      commit_message: str = "Upload BigSmall compressed model (LFS)",
                      private: bool = False,
                      cleanup: bool = False,
                      source_dir: Optional[str | Path] = None) -> str:
    """Push a BigSmall output directory using git LFS rather than the Python API.

    Workaround for the HF Python upload API's tendency to drop the connection
    during the LFS finalization phase on files larger than ~2 GB. Uses
    `git clone` + `git lfs track *.bs` + `git push` instead.

    Requires `git` and `git-lfs` in PATH, plus an HF write token.

    Args:
        local_dir: directory containing .bs shards + bigsmall.index.json.
        repo_id: target HF repo ID, e.g. "wpferrell/mistral-7b-bigsmall".
        token: HF write token. Falls back to env HF_TOKEN.
        commit_message: commit message for the push.
        private: create the repo as private if it doesn't exist yet.
        cleanup: if True, delete `local_dir` (and `source_dir` if given) on success.
        source_dir: optional source model dir to also remove on cleanup.

    Returns:
        The repo URL.
    """
    try:
        from huggingface_hub import create_repo
    except ImportError as e:
        raise ImportError(
            "upload_to_hub_lfs requires `huggingface_hub` (pip install huggingface_hub)."
        ) from e

    local_dir = Path(local_dir)
    if not (local_dir / hub_index.INDEX_FILENAME).exists():
        raise FileNotFoundError(
            f"No bigsmall.index.json in {local_dir}. Run compress_for_hub first."
        )

    tok = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not tok:
        token_file = Path.home() / ".huggingface" / "token"
        if token_file.exists():
            tok = token_file.read_text(encoding="utf-8").strip() or None
    if not tok:
        raise RuntimeError(
            "upload_to_hub_lfs needs an HF write token via `token=`, env HF_TOKEN, "
            "or ~/.huggingface/token."
        )

    # Make sure the repo exists.
    create_repo(repo_id, repo_type="model", private=private, token=tok, exist_ok=True)

    # Clone via tokenized URL into a sibling staging directory.
    import tempfile
    workdir = Path(tempfile.mkdtemp(prefix="bigsmall_lfs_"))
    clone_url = f"https://user:{tok}@huggingface.co/{repo_id}"
    auth_header = "Authorization: Basic " + base64.b64encode(
        f"user:{tok}".encode("utf-8")
    ).decode("ascii")

    def run(cmd: list[str], cwd: Optional[Path] = None,
            extra_env: Optional[dict[str, str]] = None) -> None:
        env = os.environ.copy()
        if extra_env:
            env.update(extra_env)
        print(f"[bigsmall lfs] $ {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)

    try:
        run(["git", "clone", clone_url, str(workdir)])
        run(["git", "lfs", "install"], cwd=workdir)
        run(["git", "lfs", "track", "*.bs"], cwd=workdir)

        # Copy our compressed payload into the clone (overwriting).
        for f in _iter_upload_files(local_dir):
            shutil.copy2(f, workdir / f.name)
        # Ensure .gitattributes (created by `lfs track`) is committed.
        gitattr = workdir / ".gitattributes"
        if gitattr.exists():
            run(["git", "add", ".gitattributes"], cwd=workdir)

        run(["git", "add", "--all"], cwd=workdir)

        # Skip the commit if nothing changed.
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=str(workdir),
            capture_output=True, text=True, check=True,
        )
        if status.stdout.strip():
            run(["git", "commit", "-m", commit_message], cwd=workdir)
            run([
                "git",
                "-c", f"http.extraHeader={auth_header}",
                "push", "origin", "HEAD",
            ], cwd=workdir)
        else:
            print("[bigsmall lfs] working tree clean - nothing to push", flush=True)
    finally:
        # Best-effort cleanup of the staging clone.
        shutil.rmtree(workdir, ignore_errors=True)

    if cleanup:
        _cleanup_dirs(local_dir, source_dir)

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
                    show_progress: bool = True,
                    progress: Optional[bool] = None) -> dict:
    """Download (if needed) and decompress a BigSmall model into a torch state_dict.

    Args:
        repo_or_path: HF repo ID ("user/name") OR a local path. The local path
                      may be a directory (multi-shard with bigsmall.index.json)
                      OR a single .bs file.
        device: device for the returned torch tensors.
        cache_dir: optional cache override (defaults to HF_HOME).
        show_progress: kept for backwards compatibility. If `progress` is None,
                       this value is used to enable/disable tqdm bars.
        progress: explicit override for tqdm bars. None = follow show_progress.

    Returns:
        dict[str, torch.Tensor] - state_dict suitable for model.load_state_dict().
    """
    import torch  # noqa: F401  (imported for early failure if torch missing)

    if progress is None:
        progress = show_progress

    s = str(repo_or_path)
    if _is_repo_id(s):
        if show_progress:
            print(f"[bigsmall] downloading {s} from HuggingFace Hub", flush=True)
        local = _download_bigsmall_repo(s, cache_dir=cache_dir)
    else:
        local = Path(repo_or_path)

    if local.is_file():
        if show_progress:
            print(f"[bigsmall] decompressing {local.name}", flush=True)
        return decoder.load(local, device=device, progress=progress)

    if not local.is_dir():
        raise FileNotFoundError(f"Path not found: {local}")

    index_path = local / hub_index.INDEX_FILENAME
    duplicate_map: dict[str, dict] = {}
    if index_path.exists():
        index = hub_index.read_index(local)
        shards = hub_index.shard_paths_from_index(local, index=index)
        duplicate_map = (index.get("metadata") or {}).get("duplicate_map") or {}
    else:
        shards = sorted(local.glob("*.bs"))
        if not shards:
            raise FileNotFoundError(
                f"No bigsmall.index.json and no .bs files in {local}"
            )

    state_dict: dict = {}
    total = len(shards)

    shard_bar = None
    if progress and total > 1:
        try:
            from tqdm.auto import tqdm
            shard_bar = tqdm(total=total, desc="shards", unit="shard",
                             dynamic_ncols=True, position=0)
        except ImportError:
            shard_bar = None

    for i, shard in enumerate(shards, start=1):
        if shard_bar is None and show_progress:
            print(f"[bigsmall] decompressing shard {i}/{total}: {shard.name}", flush=True)
        part = decoder.load(shard, device=device, progress=progress)
        # Detect duplicate keys across shards (HF disallows this; we error loudly)
        dup = set(part) & set(state_dict)
        if dup:
            raise ValueError(f"Duplicate tensor names across shards: {sorted(dup)[:5]}...")
        state_dict.update(part)
        if shard_bar is not None:
            shard_bar.set_postfix_str(shard.name)
            shard_bar.update(1)

    if shard_bar is not None:
        shard_bar.close()

    # Materialise cross-shard tied weights -- the duplicate tensor lives only
    # in the index, not in any shard. Aliasing the master tensor (not copying)
    # preserves storage sharing if the master is a torch tensor.
    for dup_name, info in duplicate_map.items():
        master_name = info.get("master") if isinstance(info, dict) else None
        if not master_name or master_name not in state_dict:
            continue
        if dup_name in state_dict:
            continue
        state_dict[dup_name] = state_dict[master_name]

    return state_dict
