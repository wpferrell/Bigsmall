"""HuggingFace integration: from_pretrained() hook.

Two usage patterns:

1) Drop-in replacement loader:
   from bigsmall.integrations.huggingface import from_pretrained
   model = from_pretrained("path/to/model.bs", model_class=AutoModelForCausalLM)

2) Monkey-patch transformers safetensors loader to recognise .bs:
   bigsmall.integrations.huggingface.install_hook()
   model = AutoModelForCausalLM.from_pretrained("path/with/model.bs")
"""
from __future__ import annotations

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

    # Decompress to a temporary directory that mirrors a HF model dir
    if safetensors_target is not None:
        target_dir = Path(safetensors_target).parent
        target_dir.mkdir(parents=True, exist_ok=True)
        st_path = Path(safetensors_target)
        decoder.decompress(bs_path, st_path)
        # Symlink config files (or copy if not allowed)
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


def install_hook():
    """Monkey-patch transformers/safetensors load to transparently handle .bs.

    Limited scope: redirects torch.load / safetensors.torch.load_file when given
    a .bs path, by decompressing on the fly to a temp safetensors and forwarding.
    """
    try:
        import safetensors.torch as st_torch
    except ImportError:
        raise ImportError("safetensors is required for install_hook()")

    orig_load_file = st_torch.load_file

    def patched_load_file(filename, device="cpu"):
        p = Path(filename)
        if p.suffix == ".bs":
            return decoder.load(p, device=device)
        return orig_load_file(filename, device=device)

    st_torch.load_file = patched_load_file
    return orig_load_file  # so caller can restore if needed
