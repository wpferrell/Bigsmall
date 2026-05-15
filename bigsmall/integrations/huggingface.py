"""HuggingFace integration: from_pretrained() hook.

Two usage patterns:

1) Drop-in replacement loader:
   from bigsmall.integrations.huggingface import from_pretrained
   model = from_pretrained("path/to/model.bs", model_class=AutoModelForCausalLM)

2) Transparent monkey-patch (v2 enhanced):
   import bigsmall
   bigsmall.install_hook()
   model = AutoModelForCausalLM.from_pretrained("wpferrell/mistral-7b-instruct-bigsmall")

   With the hook installed, transformers.AutoModel(.*).from_pretrained() now
   recognises BigSmall repos (those containing bigsmall.index.json) and
   transparently downloads the .bs shards, decompresses them, and returns
   the loaded model. Repos without bigsmall.index.json fall through to the
   normal HuggingFace loader.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

from .. import decoder, container


def from_pretrained(bs_path: str | Path, model_class=None, *, config_dir=None,
                    safetensors_target: str | Path | None = None, **kwargs):
    """Load a HuggingFace model from a BigSmall .bs file.

    Args:
        bs_path: path to a .bs container.
        model_class: optional transformers model class (e.g. AutoModelForCausalLM).
        config_dir: directory containing config.json + tokenizer files. If None,
                    we look in the same directory as bs_path.
        safetensors_target: optional path for the temp safetensors. Defaults
                    to a temporary file that is deleted after load.
        **kwargs: forwarded to model_class.from_pretrained.

    Returns:
        Loaded model.
    """
    bs_path = Path(bs_path)
    if config_dir is None:
        config_dir = bs_path.parent

    if safetensors_target is not None:
        target_dir = Path(safetensors_target).parent
        target_dir.mkdir(parents=True, exist_ok=True)
        st_path = Path(safetensors_target)
        decoder.decompress(bs_path, st_path)
        for f in Path(config_dir).iterdir():
            if f.name == "model.safetensors":
                continue
            try:
                (target_dir / f.name).symlink_to(f.resolve())
            except (OSError, NotImplementedError):
                if f.is_file():
                    shutil.copy2(f, target_dir / f.name)
        load_dir = target_dir
    else:
        tmp = tempfile.mkdtemp(prefix="bigsmall_hf_")
        st_path = Path(tmp) / "model.safetensors"
        decoder.decompress(bs_path, st_path)
        for f in Path(config_dir).iterdir():
            if f.name == "model.safetensors":
                continue
            try:
                if f.is_file():
                    shutil.copy2(f, Path(tmp) / f.name)
            except OSError:
                pass
        load_dir = Path(tmp)

    if model_class is None:
        from transformers import AutoModel
        model_class = AutoModel
    return model_class.from_pretrained(str(load_dir), **kwargs)


# ---------------------- Transparent AutoModel hook --------------------------

_HOOK_STATE = {
    "installed": False,
    "orig_safetensors_load_file": None,
    "patched_classes": [],          # list of (cls, "from_pretrained", original_fn)
}


def _is_bigsmall_repo_or_dir(pretrained_name_or_path) -> bool:
    """Detect whether the argument points at a BigSmall-style model.

    True if:
      - it's a local directory containing bigsmall.index.json, OR
      - it's a local directory containing one or more .bs files, OR
      - it's an HF repo ID whose root listing includes bigsmall.index.json.
    """
    s = str(pretrained_name_or_path)
    p = Path(s)
    if p.is_dir():
        if (p / "bigsmall.index.json").exists():
            return True
        if any(p.glob("*.bs")):
            return True
        return False
    if p.is_file() and p.suffix == ".bs":
        return True

    # Heuristic: treat as repo id if not a local path and not absurdly long
    if "\\" in s or s.startswith(".") or s.startswith("/") or s.startswith("~"):
        return False
    if ":" in s and len(s) > 2 and s[1] == ":":  # drive letter
        return False
    if Path(s).suffix in {".safetensors", ".bs", ".json", ".bin"}:
        return False

    # Hit the Hub to check for bigsmall.index.json
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return False
    try:
        api = HfApi()
        files = api.list_repo_files(repo_id=s, repo_type="model")
    except Exception:
        return False
    return "bigsmall.index.json" in files or any(f.endswith(".bs") for f in files)


def _materialise_bigsmall_repo_to_dir(repo_or_path) -> Path:
    """Download (if remote) and prepare a directory containing decompressed
    safetensors + config files, suitable to hand to HF from_pretrained().
    """
    from .. import hub as bs_hub
    from .. import hub_index

    s = str(repo_or_path)
    p = Path(s)
    if p.is_file() and p.suffix == ".bs":
        # Single shard file - decompress next to it
        local = p.parent
        shards = [p]
        config_dir = p.parent
    elif p.is_dir():
        local = p
        config_dir = p
        if (p / "bigsmall.index.json").exists():
            idx = hub_index.read_index(p)
            shards = hub_index.shard_paths_from_index(p, index=idx)
        else:
            shards = sorted(p.glob("*.bs"))
    else:
        # Remote repo - snapshot_download
        local = bs_hub._download_bigsmall_repo(s)
        config_dir = local
        idx_path = local / "bigsmall.index.json"
        if idx_path.exists():
            idx = hub_index.read_index(local)
            shards = hub_index.shard_paths_from_index(local, index=idx)
        else:
            shards = sorted(local.glob("*.bs"))

    if not shards:
        raise FileNotFoundError(
            f"No .bs files found in {local}; not a BigSmall repo"
        )

    out_dir = Path(tempfile.mkdtemp(prefix="bigsmall_hook_"))

    # Decompress shards. For each .bs shard write a parallel .safetensors shard
    # named so HF picks them up: model.safetensors or model-00001-of-NNNNN.safetensors.
    total = len(shards)
    weight_map: dict[str, str] = {}
    total_size = 0
    for i, sh in enumerate(shards, start=1):
        if total == 1:
            st_name = "model.safetensors"
        else:
            st_name = f"model-{i:05d}-of-{total:05d}.safetensors"
        st_path = out_dir / st_name
        decoder.decompress(sh, st_path)
        total_size += st_path.stat().st_size
        # Read tensor names from .bs header (cheap - no decompression)
        header, _ = container.read_header(sh)
        for t in header["tensors"]:
            weight_map[t["name"]] = st_name

    # Write a safetensors index for multi-shard models so HF loads all shards.
    if total > 1:
        idx_out = {
            "metadata": {"total_size": total_size},
            "weight_map": weight_map,
        }
        with open(out_dir / "model.safetensors.index.json", "w", encoding="utf-8") as f:
            json.dump(idx_out, f, indent=2)

    # Copy non-weight config / tokenizer files alongside.
    for f in Path(config_dir).iterdir():
        if not f.is_file():
            continue
        name = f.name
        if name.endswith(".bs"):
            continue
        if name == "bigsmall.index.json":
            continue
        if name == "model.safetensors.index.json":
            # We wrote our own above; skip the original (it points at .safetensors
            # shards the source repo doesn't actually have).
            continue
        try:
            shutil.copy2(f, out_dir / name)
        except OSError:
            pass

    return out_dir


def _make_patched_from_pretrained(cls, orig_from_pretrained):
    """Build a replacement classmethod that routes BigSmall repos through us."""
    def patched(pretrained_model_name_or_path, *args, **kwargs):
        try:
            is_bs = _is_bigsmall_repo_or_dir(pretrained_model_name_or_path)
        except Exception:
            is_bs = False
        if is_bs:
            local_dir = _materialise_bigsmall_repo_to_dir(pretrained_model_name_or_path)
            return orig_from_pretrained(str(local_dir), *args, **kwargs)
        return orig_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
    patched.__name__ = "from_pretrained"
    patched.__qualname__ = f"{cls.__name__}.from_pretrained"
    return classmethod(patched)


_AUTO_CLASS_NAMES = (
    "AutoModel",
    "AutoModelForCausalLM",
    "AutoModelForSeq2SeqLM",
    "AutoModelForMaskedLM",
    "AutoModelForSequenceClassification",
    "AutoModelForTokenClassification",
    "AutoModelForQuestionAnswering",
)


def install_hook():
    """Install BigSmall transparent hooks.

    After calling this once, the following work transparently with BigSmall
    repos (those containing bigsmall.index.json or .bs files):

        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            "wpferrell/mistral-7b-instruct-bigsmall"
        )

    Also patches safetensors.torch.load_file so callers that point it at a .bs
    file get the decompressed tensors back.

    Returns:
        A dict with the original objects, in case the caller wants to restore.
    """
    if _HOOK_STATE["installed"]:
        return _HOOK_STATE

    # Patch safetensors.torch.load_file (existing v1 behaviour).
    try:
        import safetensors.torch as st_torch
        orig_load_file = st_torch.load_file

        def patched_load_file(filename, device="cpu"):
            p = Path(filename)
            if p.suffix == ".bs":
                return decoder.load(p, device=device)
            return orig_load_file(filename, device=device)

        st_torch.load_file = patched_load_file
        _HOOK_STATE["orig_safetensors_load_file"] = orig_load_file
    except ImportError:
        pass

    # Patch transformers AutoModel.* from_pretrained.
    try:
        import transformers
    except ImportError:
        transformers = None

    if transformers is not None:
        for class_name in _AUTO_CLASS_NAMES:
            cls = getattr(transformers, class_name, None)
            if cls is None:
                continue
            orig = cls.from_pretrained
            patched_cm = _make_patched_from_pretrained(cls, orig)
            cls.from_pretrained = patched_cm
            _HOOK_STATE["patched_classes"].append((cls, "from_pretrained", orig))

    _HOOK_STATE["installed"] = True
    return _HOOK_STATE


def uninstall_hook():
    """Restore the original safetensors.load_file and AutoModel.from_pretrained."""
    if not _HOOK_STATE["installed"]:
        return

    orig_load_file = _HOOK_STATE["orig_safetensors_load_file"]
    if orig_load_file is not None:
        try:
            import safetensors.torch as st_torch
            st_torch.load_file = orig_load_file
        except ImportError:
            pass
        _HOOK_STATE["orig_safetensors_load_file"] = None

    for cls, attr, orig in _HOOK_STATE["patched_classes"]:
        setattr(cls, attr, orig)
    _HOOK_STATE["patched_classes"] = []
    _HOOK_STATE["installed"] = False
