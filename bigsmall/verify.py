"""md5 verification: round-trip a .bs file or compare against a source safetensors."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from . import container
from .decoder import decompress


def verify(src: str | Path, source_safetensors: str | Path | None = None) -> bool:
    """Verify a .bs file.

    If source_safetensors is provided, compare each decoded tensor's md5 to the
    md5 of the source. Otherwise verify each tensor's decoded bytes match the
    md5 stored in the .bs header.
    """
    src = Path(src)
    header, _ = container.read_header(src)
    out = decompress(src)

    all_ok = True
    if source_safetensors is not None:
        from safetensors import safe_open
        import torch
        with safe_open(str(source_safetensors), framework="pt") as f:
            for name in f.keys():
                t = f.get_tensor(name)
                src_md5 = hashlib.md5(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
                arr = out.get(name)
                if arr is None:
                    print(f"  MISSING {name}")
                    all_ok = False
                    continue
                dec_md5 = hashlib.md5(arr.tobytes()).hexdigest()
                if src_md5 != dec_md5:
                    print(f"  MISMATCH {name} src={src_md5} dec={dec_md5}")
                    all_ok = False
        return all_ok

    # Header md5 verification
    for t_meta in header["tensors"]:
        arr = out[t_meta["name"]]
        dec_md5 = hashlib.md5(arr.tobytes()).hexdigest()
        if dec_md5 != t_meta["md5"]:
            print(f"  MISMATCH {t_meta['name']} hdr={t_meta['md5']} dec={dec_md5}")
            all_ok = False
    return all_ok
